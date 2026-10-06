"""Minimal PDB reader for ML/MM input validation.

Deliberately separate from ``proxies.parse_pose``, which drops hydrogens and
keeps only the atoms the geometric proxies need. An ML/MM system stands or falls
on exactly what that parser throws away: hydrogens, the element column, and the
per-residue identity that fixes the net charge. So this module keeps everything
and refuses to guess.

Fail-closed on the element column. ``mlmm`` ships ``add-elem-info`` to repair a
PDB whose columns 77-78 are blank; silently inferring the element from the atom
name instead is how a Boltz ligand's ``CA`` (a carbon, Boltz's RDKit-order name)
becomes a calcium, which then changes the electron count and the net charge
without anything erroring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Atomic numbers, for the electron-parity check. Only the elements an enzyme
# active site can legitimately contain -- an unexpected symbol must raise rather
# than contribute zero electrons.
Z = {
    "H": 1, "D": 1, "LI": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11,
    "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17, "K": 19, "CA": 20,
    "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30, "SE": 34,
    "BR": 35, "MO": 42, "I": 53, "W": 74,
}

WATER_RESNAMES = frozenset({"HOH", "WAT", "H2O", "SOL", "TIP", "TIP3", "OPC"})
# OPC/TIP4P extra points carry charge in MM but are not atoms. They must be kept
# for an MM point-charge field and stripped before anything counts electrons --
# conflating the two put a net +380 where -4 belonged in a previous campaign.
VIRTUAL_SITE_NAMES = frozenset({"EPW", "EP", "LP", "MW"})


class StructureError(ValueError):
    """Raised when a structure cannot be read well enough to be validated."""


@dataclass(frozen=True)
class Atom:
    serial: int          # 1-based, as written in the file (mlmm freeze lists are 1-based)
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    element: str         # upper-case, verified against Z
    xyz: np.ndarray
    occupancy: float
    altloc: str

    @property
    def is_hydrogen(self) -> bool:
        return self.element in ("H", "D")

    @property
    def is_virtual(self) -> bool:
        return self.name.strip().upper() in VIRTUAL_SITE_NAMES

    @property
    def residue_key(self) -> tuple[str, int, str, str]:
        return (self.chain, self.resseq, self.icode, self.resname)

    @property
    def atomic_number(self) -> int:
        return Z[self.element]


@dataclass
class Structure:
    """Every atom of one ML/MM candidate system, hydrogens and virtual sites included."""

    atoms: list[Atom]
    path: str | None = None
    _by_residue: dict[tuple[str, int, str, str], list[Atom]] | None = field(
        default=None, repr=False, compare=False
    )

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def coords(self) -> np.ndarray:
        return np.array([a.xyz for a in self.atoms], dtype=float)

    @property
    def real(self) -> list[Atom]:
        """Atoms that carry electrons: everything except MM virtual sites."""
        return [a for a in self.atoms if not a.is_virtual]

    @property
    def heavy(self) -> list[Atom]:
        return [a for a in self.real if not a.is_hydrogen]

    def residues(self) -> dict[tuple[str, int, str, str], list[Atom]]:
        if self._by_residue is None:
            out: dict[tuple[str, int, str, str], list[Atom]] = {}
            for a in self.atoms:
                out.setdefault(a.residue_key, []).append(a)
            self._by_residue = out
        return self._by_residue

    def by_resname(self, *resnames: str) -> list[Atom]:
        want = {r.upper() for r in resnames}
        return [a for a in self.atoms if a.resname.upper() in want]

    def waters(self) -> list[Atom]:
        return [a for a in self.atoms if a.resname.upper() in WATER_RESNAMES]

    def electron_count(self, net_charge: int) -> int:
        """Total electrons. Virtual sites excluded; they are not atoms."""
        return sum(a.atomic_number for a in self.real) - int(net_charge)

    def select(self, name: str, resseq: int | None = None,
               resname: str | None = None, chain: str | None = None) -> Atom:
        """Resolve exactly one atom, or raise.

        Ambiguity is an error rather than a first-match, because a reaction
        coordinate silently anchored to the wrong copy of a two-copy ligand
        produces a plausible-looking barrier for the wrong chemistry.
        """
        hits = [
            a for a in self.atoms
            if a.name.strip().upper() == name.strip().upper()
            and (resseq is None or a.resseq == resseq)
            and (resname is None or a.resname.upper() == resname.upper())
            and (chain is None or a.chain == chain)
        ]
        if not hits:
            raise StructureError(
                f"no atom {name!r} (resseq={resseq}, resname={resname}, chain={chain}) "
                f"in {self.path or '<structure>'}"
            )
        if len(hits) > 1:
            where = ", ".join(f"{a.chain}/{a.resname}{a.resseq}/{a.name}" for a in hits[:5])
            raise StructureError(
                f"atom {name!r} is ambiguous ({len(hits)} matches: {where}); "
                "qualify it with resseq/resname/chain"
            )
        return hits[0]


def read_pdb(path: str, *, require_element: bool = True,
             keep_altloc: str = "A") -> Structure:
    """Read a PDB into a `Structure`.

    ``require_element`` fails closed on a blank element column -- repair the file
    with ``mlmm add-elem-info`` rather than relaxing this.

    Alternate locations other than ``keep_altloc`` (and blank) are dropped, which
    is what ``mlmm fix-altloc`` does; a system carrying both altlocs has two atoms
    at one site and will not parameterize.
    """
    atoms: list[Atom] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.startswith(("ATOM", "HETATM")):
                continue
            altloc = line[16]
            if altloc.strip() and altloc != keep_altloc:
                continue
            element = line[76:78].strip().upper()
            name = line[12:16].strip()
            if not element:
                if require_element:
                    raise StructureError(
                        f"{path}:{lineno}: blank element column for atom {name!r}. "
                        "Run `mlmm add-elem-info` first; inferring the element from "
                        "the atom name silently corrupts the electron count."
                    )
                element = _element_from_name(name)
            if element not in Z:
                raise StructureError(
                    f"{path}:{lineno}: unknown element {element!r} for atom {name!r}. "
                    "Add it to structure.Z if it is genuinely part of the system."
                )
            try:
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                )
            except ValueError as exc:
                raise StructureError(f"{path}:{lineno}: bad coordinates") from exc
            try:
                occ = float(line[54:60])
            except ValueError:
                occ = 1.0
            atoms.append(Atom(
                serial=len(atoms) + 1,
                name=name,
                resname=line[17:20].strip(),
                chain=line[21],
                resseq=int(line[22:26]),
                icode=line[26],
                element=element,
                xyz=xyz,
                occupancy=occ,
                altloc=altloc,
            ))
    if not atoms:
        raise StructureError(f"{path}: no ATOM/HETATM records")
    return Structure(atoms=atoms, path=path)


def _element_from_name(name: str) -> str:
    """Last-resort element guess. Only reached with require_element=False."""
    stripped = name.strip().lstrip("0123456789")
    if len(stripped) >= 2 and stripped[:2].upper() in Z:
        return stripped[:2].upper()
    return stripped[:1].upper()


def neighbours(structure: Structure, centre: Atom, cutoff: float,
               *, heavy_only: bool = False,
               exclude_same_residue: bool = False) -> list[tuple[Atom, float]]:
    """Atoms within ``cutoff`` A of ``centre``, nearest first, excluding itself."""
    pool = structure.heavy if heavy_only else structure.real
    out: list[tuple[Atom, float]] = []
    for a in pool:
        if a.serial == centre.serial:
            continue
        if exclude_same_residue and a.residue_key == centre.residue_key:
            continue
        d = float(np.linalg.norm(a.xyz - centre.xyz))
        if d <= cutoff:
            out.append((a, d))
    out.sort(key=lambda t: t[1])
    return out


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in degrees, b at the vertex."""
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        raise StructureError("degenerate angle: coincident atoms")
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
