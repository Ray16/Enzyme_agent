"""Read molecule structure files and convert to canonical SMILES via RDKit.

Supported formats: .mol, .sdf/.sd, .mol2, .pdb, .smi/.smiles, .inchi
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("enzyme_agent")

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    Chem = None  # type: ignore
    logger.warning("RDKit not available — molecule file parsing unavailable")

# Maximum molecules to read from a multi-molecule SDF
_SDF_CAP = 20

_EXTENSION_MAP: dict[str, str] = {
    ".mol": "mol",
    ".sdf": "sdf",
    ".sd": "sdf",
    ".mol2": "mol2",
    ".pdb": "pdb",
    ".smi": "smiles",
    ".smiles": "smiles",
    ".inchi": "inchi",
}


def detect_format(file_path: str | Path) -> str | None:
    """Return format name from file extension, or None if unsupported."""
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_MAP.get(ext)


def _mol_name(mol, file_path: Path, index: int = 0) -> str | None:
    """Extract molecule name with priority: _Name prop > filename stem > None."""
    if mol is not None and mol.HasProp("_Name"):
        name = mol.GetProp("_Name").strip()
        if name:
            return name
    if index == 0:
        return Path(file_path).stem
    return None


def _mol_to_record(mol, file_path: Path, fmt: str, index: int = 0) -> dict | None:
    """Convert an RDKit mol to a record dict, or None if conversion fails."""
    if mol is None:
        return None
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        return None
    if not smiles:
        return None
    return {
        "smiles": smiles,
        "name": _mol_name(mol, file_path, index),
        "index": index,
        "format": fmt,
    }


def read_molecules(file_path: str | Path) -> list[dict]:
    """Read all molecules from a structure file.

    Returns a list of dicts: {smiles, name, index, format}.
    Raises ValueError if the format is unsupported or RDKit is missing.
    """
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit is not installed — cannot parse molecule files")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    fmt = detect_format(file_path)
    if fmt is None:
        supported = ", ".join(sorted(_EXTENSION_MAP.keys()))
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported: {supported}")

    records: list[dict] = []

    if fmt == "mol":
        mol = Chem.MolFromMolFile(str(file_path), sanitize=True, removeHs=True)
        rec = _mol_to_record(mol, file_path, fmt)
        if rec:
            records.append(rec)

    elif fmt == "sdf":
        supplier = Chem.SDMolSupplier(str(file_path), sanitize=True, removeHs=True)
        for i, mol in enumerate(supplier):
            if i >= _SDF_CAP:
                logger.warning("SDF file has >%d molecules; capped at %d", _SDF_CAP, _SDF_CAP)
                break
            rec = _mol_to_record(mol, file_path, fmt, index=i)
            if rec:
                records.append(rec)

    elif fmt == "mol2":
        mol = Chem.MolFromMol2File(str(file_path), sanitize=True, removeHs=True)
        rec = _mol_to_record(mol, file_path, fmt)
        if rec:
            records.append(rec)

    elif fmt == "pdb":
        mol = Chem.MolFromPDBFile(str(file_path), sanitize=True, removeHs=True)
        rec = _mol_to_record(mol, file_path, fmt)
        if rec:
            records.append(rec)

    elif fmt == "smiles":
        text = file_path.read_text().strip()
        for i, line in enumerate(text.splitlines()):
            if i >= _SDF_CAP:
                break
            parts = line.strip().split()
            if not parts:
                continue
            mol = Chem.MolFromSmiles(parts[0])
            rec = _mol_to_record(mol, file_path, fmt, index=i)
            if rec:
                # Use second column as name if present
                if len(parts) > 1:
                    rec["name"] = parts[1]
                records.append(rec)

    elif fmt == "inchi":
        text = file_path.read_text().strip()
        for i, line in enumerate(text.splitlines()):
            if i >= _SDF_CAP:
                break
            inchi_str = line.strip()
            if not inchi_str:
                continue
            mol = Chem.MolFromInchi(inchi_str)
            rec = _mol_to_record(mol, file_path, fmt, index=i)
            if rec:
                records.append(rec)

    if not records:
        raise ValueError(f"No valid molecules could be parsed from {file_path}")

    return records


def read_single_molecule(file_path: str | Path) -> dict:
    """Read the first molecule from a structure file.

    Returns a dict: {smiles, name, index, format}.
    Warns if an SDF contains multiple molecules.
    """
    records = read_molecules(file_path)
    if len(records) > 1:
        logger.warning(
            "File %s contains %d molecules; using the first one (%s)",
            file_path, len(records), records[0].get("name", "unnamed"),
        )
    return records[0]
