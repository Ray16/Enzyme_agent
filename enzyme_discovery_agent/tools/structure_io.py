"""LLM-callable tool for loading molecule structure files during agent execution."""

from __future__ import annotations

from enzyme_discovery_agent.tool_registry import tool


@tool(
    description=(
        "Load a molecule structure file (.mol, .sdf, .mol2, .pdb, .smi, .inchi) "
        "and convert to SMILES. For multi-molecule SDF files, returns up to 20 molecules. "
        "Use molecule_index to select a specific molecule from a multi-molecule file."
    ),
    category="chemistry",
    parallel_safe=True,
    params={
        "file_path": {
            "type": "string",
            "description": "Path to the molecule structure file",
        },
        "molecule_index": {
            "type": "integer",
            "description": (
                "Index of a specific molecule to return from a multi-molecule file "
                "(0-based). Use -1 to return all molecules (default)."
            ),
        },
    },
)
def load_structure_file(file_path: str, molecule_index: int = -1) -> dict:
    """Load a molecule from a structure file and return SMILES."""
    try:
        from enzyme_discovery_agent.molecule_input import read_molecules
    except RuntimeError as e:
        return {
            "status": "error",
            "summary": str(e),
            "data": None,
            "suggestions": ["Install RDKit: conda install -c conda-forge rdkit"],
        }

    try:
        records = read_molecules(file_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return {
            "status": "error",
            "summary": str(e),
            "data": None,
            "suggestions": [
                "Check that the file path is correct and the format is supported "
                "(.mol, .sdf, .mol2, .pdb, .smi, .inchi)."
            ],
        }

    # Select specific molecule if requested
    if molecule_index >= 0:
        if molecule_index >= len(records):
            return {
                "status": "error",
                "summary": (
                    f"Molecule index {molecule_index} out of range "
                    f"(file has {len(records)} molecules)"
                ),
                "data": None,
                "suggestions": [f"Use an index between 0 and {len(records) - 1}."],
            }
        rec = records[molecule_index]
        return {
            "status": "success",
            "summary": f"Loaded molecule: {rec['name'] or 'unnamed'} ({rec['smiles']})",
            "data": {
                "smiles": rec["smiles"],
                "name": rec["name"],
                "format": rec["format"],
                "file_path": file_path,
            },
        }

    # Single molecule
    if len(records) == 1:
        rec = records[0]
        return {
            "status": "success",
            "summary": f"Loaded molecule: {rec['name'] or 'unnamed'} ({rec['smiles']})",
            "data": {
                "smiles": rec["smiles"],
                "name": rec["name"],
                "format": rec["format"],
                "file_path": file_path,
            },
        }

    # Multi-molecule file
    molecules = [
        {"smiles": r["smiles"], "name": r["name"], "index": r["index"]}
        for r in records
    ]
    return {
        "status": "success",
        "summary": f"Loaded {len(records)} molecules from {file_path}",
        "data": {
            "molecules": molecules,
            "format": records[0]["format"],
            "file_path": file_path,
        },
        "suggestions": [
            "Use molecule_index to select a specific molecule for analysis."
        ],
    }
