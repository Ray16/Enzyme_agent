"""Define a design pocket around the WHOLE bound complex.

The pocket is the shell around *everything the chemistry involves at once* --
substrate, reaction intermediate, cofactor, catalytic metals -- taken together,
not around one ligand at a time.  That matters because a per-ligand definition
silently decides the answer: a scan whose mutable set is "first shell of the
substrate" cannot contain a cofactor-coordinating residue, so it cannot find
one, and the resulting "only these two positions score" reads as a result when
it is a property of the selection.

Two knobs, both with a reason:

``cutoff``
    Heavy-atom to heavy-atom distance from a residue to the complex.  A residue
    is in the pocket if ANY of its heavy atoms is within ``cutoff`` of ANY heavy
    atom of ANY admitted component.

``link_cutoff`` / connectivity
    Not every modelled non-polymer entity is part of the site.  A predicted or
    crystallographic pose routinely carries an ion parked on a surface acidic
    patch tens of angstroms from the active site; admitting its shell imports
    positions defined by a misplaced atom rather than by the chemistry.  So
    components are admitted by CONNECTIVITY: start from an anchor (by default
    the largest component -- the substrate or intermediate), and keep any
    component within ``link_cutoff`` of something already admitted.  Excluded
    components are reported with their distance, never dropped silently.

Nothing here knows the name of any ligand, cofactor or enzyme.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..structure import Component, Structure, load_structure, min_distance


@dataclass(frozen=True)
class Position:
    """One pocket residue and its relationship to the complex."""

    chain: str
    resnum: int
    resname: str
    wt_aa: str
    d_complex: float                     # min distance to the admitted complex
    distances: dict[str, float]          # component id -> min distance
    contacts: tuple[str, ...]            # components within `cutoff`

    @property
    def membership(self) -> str:
        """Which components this residue lines, as a stable string.

        ``"A+B"`` for a residue touching both; the single id when it touches one.
        Deliberately derived rather than drawn from a fixed vocabulary -- the old
        ``both|mvap_only|atp_only`` tags only exist for a two-ligand system.
        """
        return "+".join(self.contacts) if self.contacts else "none"


@dataclass(frozen=True)
class ExcludedComponent:
    component: str
    resname: str
    n_atoms: int
    d_to_site: float
    reason: str


@dataclass
class Pocket:
    structure_path: str
    cutoff: float
    link_cutoff: float
    anchor: str
    admitted: tuple[str, ...]
    excluded: tuple[ExcludedComponent, ...]
    positions: list[Position] = field(default_factory=list)

    def resnums(self, chain: str | None = None) -> list[int]:
        return [p.resnum for p in self.positions if chain is None or p.chain == chain]

    def to_rows(self) -> list[dict]:
        """Flat rows for a TSV: one per position, one column per component."""
        rows = []
        for p in self.positions:
            row = {
                "chain": p.chain,
                "resnum": p.resnum,
                "resname": p.resname,
                "wt_aa": p.wt_aa,
                "d_complex": round(p.d_complex, 3),
                "membership": p.membership,
                "n_contacts": len(p.contacts),
            }
            for cid in self.admitted:
                row[f"d_{cid}"] = round(p.distances.get(cid, float("inf")), 3)
            rows.append(row)
        return rows


def _select_components(structure: Structure, components: list[str] | None) -> list[Component]:
    if components is None:
        return list(structure.components)
    chosen, missing = [], []
    for cid in components:
        c = structure.component(cid)
        if c is None:
            missing.append(cid)
        elif c not in chosen:
            chosen.append(c)
    if missing:
        have = sorted({c.id for c in structure.components})
        raise ValueError(f"no such component(s) {missing} in {structure.path}; have {have}")
    return chosen


def admit_complex(
    comps: list[Component],
    link_cutoff: float = 5.0,
    anchor: str | None = None,
) -> tuple[list[Component], list[ExcludedComponent], str]:
    """Grow the site outward from an anchor; return (admitted, excluded, anchor_id).

    Connectivity rather than a flat distance-to-anchor test, so a metal bridging
    substrate and cofactor is admitted through whichever it is close to.
    """
    if not comps:
        return [], [], ""
    if anchor is None:
        anchor_c = max(comps, key=lambda c: (c.n_atoms, c.id))
    else:
        matches = [c for c in comps if c.id == anchor or c.chain == anchor]
        if not matches:
            raise ValueError(f"anchor {anchor!r} is not among the selected components "
                             f"{[c.id for c in comps]}")
        anchor_c = matches[0]

    admitted = [anchor_c]
    pending = [c for c in comps if c is not anchor_c]
    # Track why each outsider stayed out: its closest approach to the grown site.
    best: dict[str, float] = {c.id: float("inf") for c in pending}
    grew = True
    while grew:
        grew = False
        for c in list(pending):
            d = min(min_distance(c.xyz, a.xyz) for a in admitted)
            best[c.id] = min(best[c.id], d)
            if d <= link_cutoff:
                admitted.append(c)
                pending.remove(c)
                grew = True

    excluded = tuple(
        ExcludedComponent(
            component=c.id, resname=c.resname, n_atoms=c.n_atoms,
            d_to_site=round(best[c.id], 3),
            reason=f"{best[c.id]:.2f} A from the site, beyond link_cutoff={link_cutoff} A",
        )
        for c in sorted(pending, key=lambda c: c.id)
    )
    return admitted, list(excluded), anchor_c.id


def define_pocket(
    structure: str | Structure,
    cutoff: float = 5.0,
    components: list[str] | None = None,
    anchor: str | None = None,
    link_cutoff: float = 5.0,
    require_connected: bool = True,
    protein_chain: str | None = None,
) -> Pocket:
    """Residues lining the bound complex.

    Args:
        structure: path to a PDB, or an already-parsed ``Structure``.
        cutoff: residue-to-complex heavy-atom cutoff, in angstrom.
        components: component ids (usually chain ids) to treat as the complex.
            ``None`` takes every non-water non-polymer entity in the file.
        anchor: component the site is grown from. ``None`` picks the largest,
            which is the substrate/intermediate in every ordinary case.
        link_cutoff: how close a component must come to the growing site to join it.
        require_connected: set ``False`` to admit every selected component
            regardless of distance (you have already curated the list).
        protein_chain: restrict positions to one chain. ``None`` scans all.
    """
    st = load_structure(structure) if isinstance(structure, str) else structure
    comps = _select_components(st, components)
    if not comps:
        raise ValueError(
            f"{st.path}: no non-polymer components found, so there is no complex to "
            "define a pocket around. Pass `components` explicitly if the ligand is "
            "recorded as ATOM rather than HETATM."
        )

    if require_connected:
        admitted, excluded, anchor_id = admit_complex(comps, link_cutoff, anchor)
    else:
        admitted, excluded, anchor_id = comps, [], (anchor or comps[0].id)

    union = np.vstack([c.xyz for c in admitted])
    positions: list[Position] = []
    for r in st.residues:
        if protein_chain is not None and r.chain != protein_chain:
            continue
        d_complex = min_distance(r.xyz, union)
        if d_complex > cutoff:
            continue
        dists = {c.id: min_distance(r.xyz, c.xyz) for c in admitted}
        contacts = tuple(cid for cid in (c.id for c in admitted) if dists[cid] <= cutoff)
        positions.append(
            Position(chain=r.chain, resnum=r.resnum, resname=r.resname, wt_aa=r.aa1,
                     d_complex=d_complex, distances=dists, contacts=contacts)
        )

    positions.sort(key=lambda p: (p.chain, p.resnum))
    return Pocket(
        structure_path=st.path, cutoff=cutoff, link_cutoff=link_cutoff,
        anchor=anchor_id, admitted=tuple(c.id for c in admitted),
        excluded=tuple(excluded), positions=positions,
    )
