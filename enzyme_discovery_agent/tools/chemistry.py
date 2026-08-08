"""Chemistry tools: SMILES validation, molecular properties, fingerprint similarity.

No hardcoded mappings — the LLM decides what to search for.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("enzyme_agent")

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    Chem = None  # type: ignore
    logger.warning("RDKit not available — chemistry tools will return errors")

from enzyme_discovery_agent.tool_registry import tool

_RDKIT_MISSING = {
    "status": "error",
    "summary": "RDKit is not installed — chemistry tools unavailable",
    "data": None,
    "suggestions": ["Install RDKit: conda install -c conda-forge rdkit"],
}


@tool(
    description="Validate a SMILES string. Returns True if valid, False otherwise.",
    category="chemistry",
    parallel_safe=True,
    params={
        "smiles": {"type": "string", "description": "SMILES string to validate"},
    },
)
def validate_smiles(smiles: str) -> dict:
    """Check if a SMILES string is valid and return canonical form."""
    if not _HAS_RDKIT:
        return _RDKIT_MISSING
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "status": "success",
            "summary": f"Invalid SMILES: {smiles}",
            "data": {"valid": False, "smiles": smiles},
            "suggestions": ["Check for syntax errors in the SMILES string."],
        }
    canonical = Chem.MolToSmiles(mol)
    return {
        "status": "success",
        "summary": f"Valid SMILES: {canonical}",
        "data": {"valid": True, "canonical_smiles": canonical, "input_smiles": smiles},
    }


@tool(
    description="Compute molecular properties (MW, logP, HBD, HBA, etc.) from a SMILES string.",
    category="chemistry",
    parallel_safe=True,
    params={
        "smiles": {"type": "string", "description": "SMILES string"},
    },
)
def get_molecular_properties(smiles: str) -> dict:
    """Compute basic molecular properties from SMILES."""
    if not _HAS_RDKIT:
        return _RDKIT_MISSING
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "status": "error",
            "summary": f"Invalid SMILES: {smiles}",
            "data": None,
            "suggestions": ["Validate the SMILES first with validate_smiles."],
        }
    props = {
        "smiles": smiles,
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
    }
    return {
        "status": "success",
        "summary": f"MW={props['molecular_weight']}, LogP={props['logp']}, Formula={props['formula']}",
        "data": props,
    }


@tool(
    description=(
        "Compute Tanimoto similarity between two molecules using Morgan "
        "fingerprints (ECFP4, radius=2, 2048 bits). Returns a float [0, 1]."
    ),
    category="chemistry",
    parallel_safe=True,
    params={
        "smiles1": {"type": "string", "description": "First SMILES string"},
        "smiles2": {"type": "string", "description": "Second SMILES string"},
    },
)
def compute_fingerprint_similarity(smiles1: str, smiles2: str) -> dict:
    """Tanimoto similarity between two molecules using Morgan fingerprints."""
    if not _HAS_RDKIT:
        return _RDKIT_MISSING
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        invalid = []
        if mol1 is None:
            invalid.append(smiles1)
        if mol2 is None:
            invalid.append(smiles2)
        return {
            "status": "error",
            "summary": f"Invalid SMILES: {', '.join(invalid)}",
            "data": {"similarity": 0.0},
            "suggestions": ["Check SMILES validity with validate_smiles."],
        }
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return {
        "status": "success",
        "summary": f"Tanimoto similarity: {sim:.3f}",
        "data": {"similarity": round(sim, 4), "smiles1": smiles1, "smiles2": smiles2},
    }
