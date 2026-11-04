"""Residue-level structure parsing for the design tools.

``structural_modeling_agent.proxies.parse_pose`` already reads a PDB, but it throws the
residue numbering away -- it answers "where are the atoms" for geometry scoring,
not "which residue is this" for mutagenesis.  Saturation design needs the
numbering (every mutation is applied by residue index), so this reads the same
files and keeps it.

Stdlib + numpy, matching the rest of the structural stack: no Biopython, so the
design tools stay importable in a bare agent env.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # Common protonation/alternate names that are still the same residue.
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E", "LYN": "K", "MSE": "M",
}

#: Solvent is never part of the pocket-defining complex.  Kept narrow on purpose:
#: anything not listed here is treated as a real component, because "is this
#: molecule a cofactor or a crystallisation additive" is chemistry knowledge the
#: caller has and this module does not.
WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "TIP", "TIP3", "SOL"})


@dataclass(frozen=True)
class Residue:
    """One polymer residue, with every heavy atom it owns."""

    chain: str
    resnum: int
    resname: str
    aa1: str
    xyz: np.ndarray                      # (n_atoms, 3)
    icode: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resnum, self.icode)


@dataclass(frozen=True)
class Component:
    """One non-polymer entity: a substrate, an intermediate, a cofactor, an ion.

    ``id`` is the chain id when that chain holds exactly one non-polymer residue
    -- the one-ligand-per-chain convention cofolding tools emit, and the one a
    caller naming components by hand will reach for.  When a chain holds several
    (a ligand plus its ions, say), ids fall back to ``chain:RESNAME<resnum>`` so
    they stay unique and addressable.
    """

    id: str
    chain: str
    resname: str
    resnum: int
    xyz: np.ndarray                      # (n_atoms, 3)
    elements: tuple[str, ...] = ()

    @property
    def n_atoms(self) -> int:
        return int(self.xyz.shape[0])


@dataclass
class Structure:
    path: str
    residues: list[Residue] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)

    # -- lookups -----------------------------------------------------------
    def component(self, cid: str) -> Component | None:
        for c in self.components:
            if c.id == cid or c.chain == cid:
                return c
        return None

    def protein_chains(self) -> tuple[str, ...]:
        seen: list[str] = []
        for r in self.residues:
            if r.chain not in seen:
                seen.append(r.chain)
        return tuple(seen)

    def sequence(self, chain: str | None = None) -> str:
        """One-letter sequence, index i == residue ``i+1``.

        Asserts gap-free 1-based numbering, because every mutation downstream is
        applied by string index: a silent gap here corrupts every variant built
        from it.  Raises rather than returning something plausible.
        """
        chain = chain or (self.protein_chains() or ("",))[0]
        rs = [r for r in self.residues if r.chain == chain]
        if not rs:
            raise ValueError(f"{self.path}: no protein residues in chain {chain!r}")
        icoded = [r for r in rs if r.icode]
        if icoded:
            raise ValueError(
                f"{self.path}: chain {chain!r} has insertion codes "
                f"(e.g. {icoded[0].resnum}{icoded[0].icode}); residue number is not a "
                "sequence index, so mutations cannot be applied positionally"
            )
        nums = [r.resnum for r in rs]
        if nums[0] != 1:
            raise ValueError(
                f"{self.path}: chain {chain!r} numbering starts at {nums[0]}, expected 1"
            )
        if nums != list(range(1, nums[-1] + 1)):
            gaps = [b for a, b in zip(nums, nums[1:]) if b != a + 1]
            raise ValueError(
                f"{self.path}: chain {chain!r} has a gap in residue numbering before "
                f"{gaps[:5]}"
            )
        unknown = [r.resnum for r in rs if r.aa1 == "X"]
        if unknown:
            raise ValueError(
                f"{self.path}: chain {chain!r} has non-standard residues at {unknown[:5]}"
            )
        return "".join(r.aa1 for r in rs)


def _element(line: str, name: str) -> str:
    return (line[76:78].strip() or name.strip()[:1]).upper()


def load_structure(path: str | Path, keep_water: bool = False) -> Structure:
    """Parse a PDB into polymer residues and non-polymer components.

    Hydrogens are dropped: shells are defined on heavy atoms, and whether a file
    carries hydrogens is a property of the preparation pipeline, not of the
    pocket.  Alternate locations other than the first are dropped too, so an
    altloc does not silently double a residue's atom count.
    """
    p = Path(path).expanduser()
    text = p.read_text()

    residues: dict[tuple[str, int, str], dict] = {}
    het: dict[tuple[str, str, int], dict] = {}
    seen_alt: set[tuple] = set()

    for line in text.splitlines():
        rec = line[:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        name = line[12:16]
        elem = _element(line, name)
        if elem == "H" or elem == "D":
            continue
        altloc = line[16]
        chain = line[21]
        resname = line[17:20].strip()
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26].strip()
        atom_key = (chain, resnum, icode, resname, name.strip())
        if altloc not in (" ", ""):
            if atom_key in seen_alt:
                continue
            seen_alt.add(atom_key)
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))

        # MSE is a selenomethionine: a HETATM record that is really polymer.
        is_polymer = rec == "ATOM  " or resname == "MSE"
        if is_polymer:
            k = (chain, resnum, icode)
            d = residues.setdefault(
                k, {"resname": resname, "aa1": THREE_TO_ONE.get(resname, "X"), "xyz": []}
            )
            d["xyz"].append(xyz)
        else:
            if not keep_water and resname in WATER_RESNAMES:
                continue
            k2 = (chain, resname, resnum)
            d2 = het.setdefault(k2, {"xyz": [], "elements": []})
            d2["xyz"].append(xyz)
            d2["elements"].append(elem)

    res_list = [
        Residue(chain=c, resnum=n, resname=d["resname"], aa1=d["aa1"],
                xyz=np.asarray(d["xyz"], dtype=float), icode=i)
        for (c, n, i), d in sorted(residues.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    ]

    # One het residue in a chain -> the chain id IS the component id.
    per_chain: dict[str, int] = {}
    for (c, _rn, _num) in het:
        per_chain[c] = per_chain.get(c, 0) + 1
    comps = []
    for (c, resname, num), d in sorted(het.items()):
        cid = c if per_chain[c] == 1 else f"{c}:{resname}{num}"
        comps.append(Component(id=cid, chain=c, resname=resname, resnum=num,
                               xyz=np.asarray(d["xyz"], dtype=float),
                               elements=tuple(d["elements"])))

    return Structure(path=str(p), residues=res_list, components=comps)


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum heavy-atom--heavy-atom distance between two point sets (A)."""
    if a.size == 0 or b.size == 0:
        return float("inf")
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min()))
