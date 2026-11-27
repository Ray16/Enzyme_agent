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
import shlex
from pathlib import Path

import numpy as np

_TWO_LETTER_ELEMENTS = frozenset({
    "MG", "MN", "ZN", "FE", "CA", "CU", "CO", "NI", "NA", "CL", "BR",
    "SE", "MO", "HG", "CD", "AL", "SI", "LI", "BE", "PT", "AU", "AG",
})

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
    atom_names: tuple[str, ...] = ()
    elements: tuple[str, ...] = ()

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
    atom_names: tuple[str, ...] = ()

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


def _element(line: str, name: str, resname: str = "", is_het: bool = False) -> str:
    col = line[76:78].strip().upper()
    if col:
        return col
    stripped = name.strip().upper()
    # Two-letter ions/metals in PDB files with missing element columns (e.g. MG, ZN, FE, HG).
    if is_het and stripped[:2] in _TWO_LETTER_ELEMENTS and (
        resname.upper() == stripped[:2] or stripped in _TWO_LETTER_ELEMENTS
    ):
        return stripped[:2]
    # Standard 4-char PDB atom name: leading digit (e.g. "1HB") or space-padded.
    letters = "".join(ch for ch in stripped if ch.isalpha())
    return letters[:1] if letters else ""


def _assemble_structure(
    path: Path,
    residues: dict[tuple[str, int, str], dict],
    het: dict[tuple[str, str, int], dict],
) -> Structure:
    res_list = [
        Residue(
            chain=c, resnum=n, resname=d["resname"], aa1=d["aa1"],
            xyz=np.asarray(d["xyz"], dtype=float), icode=i,
            atom_names=tuple(d["atom_names"]), elements=tuple(d["elements"]),
        )
        for (c, n, i), d in sorted(residues.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    ]

    # One het residue in a chain -> the chain id IS the component id.
    per_chain: dict[str, int] = {}
    for (c, _rn, _num) in het:
        per_chain[c] = per_chain.get(c, 0) + 1
    comps = []
    for (c, resname, num), d in sorted(het.items()):
        cid = c if per_chain[c] == 1 else f"{c}:{resname}{num}"
        comps.append(Component(
            id=cid, chain=c, resname=resname, resnum=num,
            xyz=np.asarray(d["xyz"], dtype=float),
            elements=tuple(d["elements"]),
            atom_names=tuple(d["atom_names"]),
        ))

    return Structure(path=str(path), residues=res_list, components=comps)


def _load_mmcif(p: Path, text: str, keep_water: bool) -> Structure:
    """Parse model 1 of an mmCIF file's _atom_site loop (stdlib + numpy)."""
    lines = text.splitlines()
    headers: list[str] = []
    rows: list[list[str]] = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        s = lines[i].strip()
        if s == "loop_":
            i += 1
            cols: list[str] = []
            while i < n_lines and lines[i].strip().startswith("_"):
                cols.append(lines[i].strip().split()[0])
                i += 1
            if cols and all(c.startswith("_atom_site.") for c in cols):
                headers = cols
                while i < n_lines:
                    raw = lines[i].strip()
                    if not raw or raw.startswith("#"):
                        i += 1
                        continue
                    if raw == "loop_" or raw.startswith("_") or raw.startswith("data_"):
                        break
                    toks = shlex.split(raw)
                    if len(toks) >= len(headers):
                        rows.append(toks[:len(headers)])
                    i += 1
                break
            continue
        i += 1

    if not headers:
        return Structure(path=str(p), residues=[], components=[])

    col = {h: idx for idx, h in enumerate(headers)}

    def _get(row: list[str], *names: str, default: str = "") -> str:
        for nm in names:
            idx = col.get(nm)
            if idx is not None and idx < len(row):
                val = row[idx]
                if val not in (".", "?"):
                    return val
        return default

    residues: dict[tuple[str, int, str], dict] = {}
    het: dict[tuple[str, str, int], dict] = {}
    seen_alt: set[tuple] = set()
    first_model: str | None = None

    for row in rows:
        rec = _get(row, "_atom_site.group_PDB", default="ATOM").upper()
        if rec not in ("ATOM", "HETATM"):
            continue
        model_num = _get(row, "_atom_site.pdbx_PDB_model_num", default="1")
        if first_model is None:
            first_model = model_num
        elif model_num != first_model:
            continue
        name = _get(row, "_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        elem = _get(row, "_atom_site.type_symbol", default=name[:1]).upper()
        if elem in ("H", "D"):
            continue
        altloc = _get(row, "_atom_site.label_alt_id")
        chain = _get(row, "_atom_site.auth_asym_id", "_atom_site.label_asym_id", default="A")
        resname = _get(row, "_atom_site.auth_comp_id", "_atom_site.label_comp_id").upper()
        seq_s = _get(row, "_atom_site.auth_seq_id", "_atom_site.label_seq_id")
        try:
            resnum = int(seq_s)
        except ValueError:
            resnum = 1
        icode = _get(row, "_atom_site.pdbx_PDB_ins_code")
        atom_key = (chain, resnum, icode, resname, name)
        if altloc:
            if atom_key in seen_alt:
                continue
            seen_alt.add(atom_key)
        try:
            xyz = (
                float(_get(row, "_atom_site.Cartn_x")),
                float(_get(row, "_atom_site.Cartn_y")),
                float(_get(row, "_atom_site.Cartn_z")),
            )
        except ValueError:
            continue

        is_polymer = rec == "ATOM" or resname == "MSE"
        if is_polymer:
            k = (chain, resnum, icode)
            d = residues.setdefault(
                k, {"resname": resname, "aa1": THREE_TO_ONE.get(resname, "X"),
                    "xyz": [], "atom_names": [], "elements": []}
            )
            d["xyz"].append(xyz)
            d["atom_names"].append(name)
            d["elements"].append(elem)
        else:
            if not keep_water and resname in WATER_RESNAMES:
                continue
            k2 = (chain, resname, resnum)
            d2 = het.setdefault(k2, {"xyz": [], "elements": [], "atom_names": []})
            d2["xyz"].append(xyz)
            d2["elements"].append(elem)
            d2["atom_names"].append(name)

    return _assemble_structure(p, residues, het)


def load_structure(path: str | Path, keep_water: bool = False) -> Structure:
    """Parse a PDB or mmCIF into polymer residues and non-polymer components.

    Hydrogens are dropped: shells are defined on heavy atoms, and whether a file
    carries hydrogens is a property of the preparation pipeline, not of the
    pocket.  Alternate locations other than the first are dropped too, so an
    altloc does not silently double a residue's atom count.  For multi-model
    files, only the first model is parsed.
    """
    p = Path(path).expanduser()
    text = p.read_text()
    if p.suffix.lower() in (".cif", ".mmcif") or "_atom_site." in text[:4096]:
        return _load_mmcif(p, text, keep_water=keep_water)

    residues: dict[tuple[str, int, str], dict] = {}
    het: dict[tuple[str, str, int], dict] = {}
    seen_alt: set[tuple] = set()

    for line in text.splitlines():
        rec = line[:6]
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM  ", "HETATM") or len(line) < 54:
            continue
        name = line[12:16]
        resname = line[17:20].strip()
        is_polymer = rec == "ATOM  " or resname == "MSE"
        elem = _element(line, name, resname=resname, is_het=not is_polymer)
        if elem == "H" or elem == "D":
            continue
        altloc = line[16]
        chain = line[21]
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

        if is_polymer:
            k = (chain, resnum, icode)
            d = residues.setdefault(
                k, {"resname": resname, "aa1": THREE_TO_ONE.get(resname, "X"),
                    "xyz": [], "atom_names": [], "elements": []}
            )
            d["xyz"].append(xyz)
            d["atom_names"].append(name.strip())
            d["elements"].append(elem)
        else:
            if not keep_water and resname in WATER_RESNAMES:
                continue
            k2 = (chain, resname, resnum)
            d2 = het.setdefault(k2, {"xyz": [], "elements": [], "atom_names": []})
            d2["xyz"].append(xyz)
            d2["elements"].append(elem)
            d2["atom_names"].append(name.strip())

    return _assemble_structure(p, residues, het)


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum heavy-atom--heavy-atom distance between two point sets (A)."""
    if a.size == 0 or b.size == 0:
        return float("inf")
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min()))
