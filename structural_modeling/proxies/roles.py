"""Resolve GENERIC catalytic roles from a cofolded structure.

A structural proxy is only transferable if it is defined over *roles* --
nucleophile, electrophile, leaving group, charge centre -- rather than over named
atoms. ``PA``/``O3A``/``adenine`` are adenylation vocabulary; a proxy written in
those terms cannot be tried on a hydrolase. This module is the translation layer:
chemistry in, roles out.

Role resolution is keyed on the MECHANISM CLASS (what the transfer chemistry is),
never on the enzyme, family or pathway. Mechanism class comes from the reasoning
contract / ``proxy_policy`` and is data, not code: adding a class means adding a
resolver to ``RESOLVERS``, no change to any proxy in ``library.py``.

Chain handling is composition-based, so the same resolver works regardless of the
order chains happen to appear in a YAML. That matters: the amide-synthetase
diversity panel and the literature panel use different chain orders for the same
chemistry, and index-keyed code silently read the wrong pair (``angle_gate_proxy``
reports an ATP-ligand ipTM under the name ``mean_iptm`` for exactly this reason).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

COVALENT_CUT = 1.6          # heavy-atom bond cutoff, Angstrom
P_O_BOND = 1.75             # P-O bond is longer than C-O


@dataclass(frozen=True)
class ReactiveSite:
    """One pose's catalytic geometry, in generic role terms.

    nucleophile  attacking atom
    electrophile atom under attack
    leaving      atom/centroid departing along the breaking bond (may be None)
    charge_centre atom that accumulates negative charge in the transition state
                  (carbonyl O for acyl transfer, a non-bridging O for phosphoryl);
                  the target of oxyanion-hole-style stabilisation
    frame        other substituents on the electrophile, defining its local plane
    ideal_angle  mechanism's ideal nucleophile-electrophile-leaving angle (deg)
    """

    nucleophile: np.ndarray
    electrophile: np.ndarray
    leaving: np.ndarray | None = None
    charge_centre: np.ndarray | None = None
    frame: tuple[np.ndarray, ...] = field(default_factory=tuple)
    ideal_angle: float = 180.0

    @property
    def attack_distance(self) -> float:
        return float(np.linalg.norm(self.nucleophile - self.electrophile))


@dataclass(frozen=True)
class Pose:
    """Parsed structure: protein scaffold plus ligand chains, element-annotated."""

    ca: np.ndarray                       # protein CA, for ensemble superposition
    protein: np.ndarray                  # protein heavy atoms
    donors: np.ndarray                   # protein H-bond donor N/O
    bases: np.ndarray                    # Arg/Lys/His sidechain N (general base/acid)
    ligands: dict[str, list[tuple[str, str, np.ndarray]]]   # chain -> [(name, element, xyz)]

    def chain_with(self, element: str) -> str | None:
        for ch, atoms in self.ligands.items():
            if any(e == element for _n, e, _x in atoms):
                return ch
        return None

    def lone_atom_chain(self, element: str) -> str | None:
        for ch, atoms in self.ligands.items():
            if len(atoms) == 1 and atoms[0][1] == element:
                return ch
        return None


_DONOR_SC = {
    "ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"},
    "TYR": {"OH"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "HIS": {"ND1", "NE2"},
    "TRP": {"NE1"}, "CYS": {"SG"},
}
_BASE_SC = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"}}


def parse_pose(path: str) -> Pose:
    """Read a PDB into a Pose. Stdlib + numpy only; no Biopython dependency."""
    ca, prot, don, bas = [], [], [], []
    lig: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            name = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21]
            elem = (line[76:78].strip() or name[0]).upper()
            if elem == "H":
                continue
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if rec == "HETATM":
                lig.setdefault(chain, []).append((name, elem, xyz))
                continue
            prot.append(xyz)
            if name == "CA":
                ca.append(xyz)
            if name == "N" or (resn in _DONOR_SC and name in _DONOR_SC[resn]):
                don.append(xyz)
            if resn in _BASE_SC and name in _BASE_SC[resn]:
                bas.append(xyz)//
    def arr(v):
        return np.array(v) if v else np.zeros((0, 3))

    return Pose(ca=arr(ca), protein=arr(prot), donors=arr(don), bases=arr(bas), ligands=lig)


def _bonded(centre: np.ndarray, atoms, element: str, cut: float):
    return [x for _n, e, x in atoms if e == element and np.linalg.norm(centre - x) < cut]


def resolve_substitution_at_phosphorus(pose: Pose, leaving_atom: str = "O3A",
                                       phosphorus: str = "PA") -> ReactiveSite | None:
    """In-line (SN2-like) displacement at a phosphorus centre.

    Generic over which phosphate is attacked: the caller passes the phosphorus and
    the bridging leaving oxygen, both of which ``proxy_policy`` derives from the
    reaction's nucleotide products. Nucleophile = the carboxylate/hydroxyl O of the
    non-phosphate ligand closest to that phosphorus.
    """
    cof = pose.chain_with("P")
    if cof is None:
        return None
    named = {n: x for n, _e, x in pose.ligands[cof]}
    if phosphorus not in named:
        return None
    elec = named[phosphorus]
    leaving = named.get(leaving_atom)

    best = None
    for ch, atoms in pose.ligands.items():
        if ch == cof or len(atoms) < 2:
            continue
        for _n, e, cx in atoms:
            if e != "C":
                continue
            oxy = _bonded(cx, atoms, "O", COVALENT_CUT)
            if len(oxy) < 2:                       # a carboxylate, not an ester/alcohol
                continue
            o = min(oxy, key=lambda z: np.linalg.norm(z - elec))
            d = float(np.linalg.norm(o - elec))
            if best is None or d < best[0]:
                best = (d, o)
    if best is None:
        return None

    # charge centre: a non-bridging O on the attacked phosphorus
    nonbridge = [x for _n, e, x in pose.ligands[cof]
                 if e == "O" and np.linalg.norm(x - elec) < P_O_BOND
                 and (leaving is None or not np.allclose(x, leaving))]
    frame = tuple(x for _n, e, x in pose.ligands[cof]
                  if e == "O" and np.linalg.norm(x - elec) < P_O_BOND)
    return ReactiveSite(nucleophile=best[1], electrophile=elec, leaving=leaving,
                        charge_centre=nonbridge[0] if nonbridge else None,
                        frame=frame, ideal_angle=180.0)


def resolve_acyl_addition_elimination(pose: Pose) -> ReactiveSite | None:
    """Nucleophilic addition at an activated acyl carbon (ester/anhydride/thioester).

    Electrophile = the carbon bearing BOTH a carbonyl O and an ester O that is bonded
    to the activating group (here P; generalises to any heteroatom leaving group).
    Nucleophile = the nearest N/O/S of a ligand chain that is not the acyl donor.
    Ideal trajectory is Burgi-Dunitz (~107 deg), not in-line.
    """
    cof = pose.chain_with("P")
    if cof is None:
        return None
    atoms = pose.ligands[cof]
    phos = [x for _n, e, x in atoms if e == "P"]

    elec = carbonyl = ester = alkyl = None
    for _n, e, cx in atoms:
        if e != "C":
            continue
        oxy = _bonded(cx, atoms, "O", COVALENT_CUT)
        if len(oxy) < 2:
            continue
        est = [o for o in oxy if any(np.linalg.norm(o - p) < P_O_BOND for p in phos)]
        car = [o for o in oxy if not any(np.linalg.norm(o - p) < P_O_BOND for p in phos)]
        if est and car:
            elec, carbonyl, ester = cx, car[0], est[0]
            nb = [x for _n2, e2, x in atoms
                  if e2 == "C" and not np.allclose(x, cx)
                  and np.linalg.norm(cx - x) < 1.7]
            alkyl = nb[0] if nb else None
            break
    if elec is None:
        return None

    cand = []
    for ch, lig in pose.ligands.items():
        if ch == cof or (len(lig) == 1 and lig[0][1] == "MG"):
            continue
        cand += [x for _n, e, x in lig if e in ("N", "O", "S")]
    if not cand:
        return None
    nuc = min(cand, key=lambda z: np.linalg.norm(z - elec))

    frame = tuple(z for z in (carbonyl, ester, alkyl) if z is not None)
    return ReactiveSite(nucleophile=nuc, electrophile=elec, leaving=ester,
                        charge_centre=carbonyl, frame=frame, ideal_angle=107.0)


#: Mechanism class -> resolver. Keyed on CHEMISTRY, never on enzyme or family.
RESOLVERS: dict[str, Callable[..., ReactiveSite | None]] = {
    "substitution_at_phosphorus": resolve_substitution_at_phosphorus,
    "acyl_addition_elimination": resolve_acyl_addition_elimination,
}


def resolve(pose: Pose, mechanism_class: str, **kwargs) -> ReactiveSite | None:
    try:
        resolver = RESOLVERS[mechanism_class]
    except KeyError as exc:
        raise KeyError(
            f"no resolver for mechanism class {mechanism_class!r}; "
            f"known: {sorted(RESOLVERS)}"
        ) from exc
    return resolver(pose, **kwargs)
