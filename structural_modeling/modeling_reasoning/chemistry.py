#!/usr/bin/env python3
"""Boundary re-verification of ligand chemistry with RDKit.

The discovery handoff already carries RDKit facts produced upstream by
``modeling_handoff.characterize_smiles``. This module re-derives them at the
resolution boundary so a modeling SMILES the reasoning layer *changed* (e.g. a
protonation choice) is verified here rather than trusted as an asserted boolean.
It intentionally mirrors the upstream schema so a resolved ``modeling_representation``
stays drop-in compatible with the schema-2.0 handoff.

Requires the base environment's RDKit (``~/miniforge3/bin/python``); the
``ragmosis`` env does not have RDKit.
"""
from __future__ import annotations

from typing import Any


def characterize_smiles(smiles: str) -> dict[str, Any]:
    """Return representation-sensitive charge and stereochemistry facts.

    Mirrors ``modeling_handoff.characterize_smiles`` so resolved substrate
    representations validate identically against the handoff contract.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"valid": False, "input_smiles": smiles}
    centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True, includeCIP=True)
    assigned = [{"atom_index": i, "cip": label} for i, label in centers if label != "?"]
    unassigned = [i for i, label in centers if label == "?"]
    if not centers:
        stereo_status = "not_applicable"
    elif unassigned:
        stereo_status = "unspecified"
    else:
        stereo_status = "specified"
    return {
        "valid": True,
        "input_smiles": smiles,
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        ),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "stereochemistry": {
            "status": stereo_status,
            "assigned_centers": assigned,
            "unassigned_atom_indices": unassigned,
        },
    }


def resolve_ligand_atom(smiles: str, smarts: str) -> dict[str, Any]:
    """Resolve a reaction-coordinate ligand atom role onto a SMILES via SMARTS.

    Returns the number of matches and the reactive atom index (first atom of the
    SMARTS) of each match. The caller abstains when ``match_count`` is 0.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    pattern = Chem.MolFromSmarts(smarts) if smarts else None
    if molecule is None or pattern is None:
        return {"match_count": 0, "reactive_atom_indices": [], "smarts": smarts}
    matches = molecule.GetSubstructMatches(pattern)
    return {
        "match_count": len(matches),
        "reactive_atom_indices": [m[0] for m in matches],
        "smarts": smarts,
    }


def resolve_cofactor_atom(ccd: str | None, selector: str, cofactor_ccd_atoms: dict) -> str | None:
    """Resolve a cofactor atom role (e.g. alpha_phosphate) to a PDB CCD atom name."""
    return (cofactor_ccd_atoms.get(selector) or {}).get(ccd)


def substrate_errors(modeling_smiles: str | None, representation: dict[str, Any]) -> list[str]:
    """Execution-blocking chemistry errors for one modeled substrate."""
    errors: list[str] = []
    if not modeling_smiles:
        errors.append("modeling SMILES is missing")
        return errors
    if not representation.get("valid"):
        errors.append("modeling SMILES does not parse in RDKit")
        return errors
    if representation.get("stereochemistry", {}).get("status") == "unspecified":
        errors.append("modeling stereochemistry is unspecified")
    return errors
