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
    # An ordinary phosphate diester monoanion has two resonance-equivalent
    # nonbridging oxygens. A chosen P=O/O- drawing must not invent P chirality.
    def resonance_equivalent_oxygens(index):
        atom=molecule.GetAtomWithIdx(index)
        if atom.GetAtomicNum()!=15:return False
        terminal=[n for n in atom.GetNeighbors() if n.GetAtomicNum()==8 and n.GetDegree()==1 and n.GetIsotope()==0]
        return (len(terminal)>=2 and any(n.GetFormalCharge()==-1 for n in terminal)
                and any(molecule.GetBondBetweenAtoms(index,n.GetIdx()).GetBondType()==Chem.BondType.DOUBLE for n in terminal))
    centers=[(index,label) for index,label in centers if not (label=='?' and resonance_equivalent_oxygens(index))]
    assigned = [{"atom_index": i, "cip": label} for i, label in centers if label != "?"]
    unassigned = [i for i, label in centers if label == "?"]
    if not centers:
        stereo_status = "not_applicable"
    elif unassigned:
        stereo_status = "unspecified"
    else:
        stereo_status = "specified"
    # Functional-group protonation counts for mechanistic charge validation
    cooh_pat = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    coo_pat = Chem.MolFromSmarts("[CX3](=O)[O-1]")
    neutral_amine_pat = Chem.MolFromSmarts("[NX3;H2,H1;+0;!$(NC=O);!$(N=*);!$(N[a])]")
    prot_amine_pat = Chem.MolFromSmarts("[NX4;H3,H2;+1;!$(NC=O);!$(N[a])]")
    any_neutral_nuc_amine = Chem.MolFromSmarts("[NX3;H2,H1;+0;!$(NC=O);!$(N=*)]")
    protonation_summary = {
        "protonated_carboxylic_acids": len(molecule.GetSubstructMatches(cooh_pat)) if cooh_pat else 0,
        "deprotonated_carboxylates": len(molecule.GetSubstructMatches(coo_pat)) if coo_pat else 0,
        "neutral_aliphatic_amines": len(molecule.GetSubstructMatches(neutral_amine_pat)) if neutral_amine_pat else 0,
        "protonated_aliphatic_amines": len(molecule.GetSubstructMatches(prot_amine_pat)) if prot_amine_pat else 0,
        "neutral_nucleophilic_amines": len(molecule.GetSubstructMatches(any_neutral_nuc_amine)) if any_neutral_nuc_amine else 0,
    }
    return {
        "valid": True,
        "input_smiles": smiles,
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        ),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "protonation_summary": protonation_summary,
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

    Also reports whether the distinct matched atoms are SYMMETRY-EQUIVALENT
    (one topological equivalence class under RDKit's canonical ranking with ties
    unbroken).  Multiple matches are normally refused, because choosing one
    arbitrarily scores a different reaction than the one being modelled -- but
    that reasoning does not hold when the sites are homotopic.  A symmetric
    bifunctional substrate genuinely has two identical reactive centres, which
    is the usual shape of a polymer-derived monomer, and either choice scores
    the same reaction.  The distinction is chemical, not cosmetic: in a
    symmetric biscarbamate the two carbamates are interchangeable, whereas in an
    unsymmetrically substituted one they sit in different electronic
    environments and hydrolyse differently, so that case must still be refused.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    pattern = Chem.MolFromSmarts(smarts) if smarts else None
    if molecule is None or pattern is None:
        return {"match_count": 0, "reactive_atom_indices": [], "smarts": smarts,
                "distinct_atom_indices": [], "symmetry_equivalent": False,
                "symmetry_classes": []}
    matches = molecule.GetSubstructMatches(pattern)
    indices = [m[0] for m in matches]
    distinct = sorted(set(indices))
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=False))
    classes = sorted({ranks[i] for i in distinct})
    return {
        "match_count": len(matches),
        "reactive_atom_indices": indices,
        "distinct_atom_indices": distinct,
        "symmetry_classes": classes,
        "symmetry_equivalent": len(classes) == 1,
        "smarts": smarts,
    }


def resolve_cofactor_atom(ccd: str | None, selector: str, cofactor_ccd_atoms: dict) -> str | None:
    """Resolve a cofactor atom role (e.g. alpha_phosphate) to a PDB CCD atom name."""
    return (cofactor_ccd_atoms.get(selector) or {}).get(ccd)


def substrate_errors(
    modeling_smiles: str | None,
    representation: dict[str, Any],
    *,
    role: str | None = None,
    source_smiles: str | None = None,
    source_representation: dict[str, Any] | None = None,
    assay_pH: float | None = None,
    expected_formal_charge: int | None = None,
    allow_all_neutral_polyamine: bool = False,
) -> list[str]:
    """Execution-blocking stereochemistry and pH/role-aware charge errors for one modeled substrate."""
    from collections import Counter

    errors: list[str] = []
    if not modeling_smiles:
        errors.append("modeling SMILES is missing")
        return errors
    if not representation.get("valid"):
        errors.append("modeling SMILES does not parse in RDKit")
        return errors

    stereo = representation.get("stereochemistry") or {}
    if stereo.get("status") == "unspecified":
        errors.append("modeling stereochemistry is unspecified")

    # 1. Stereochemistry preservation check against source_smiles (catches R/S inversion or dropped centers)
    role_l = str(role or "").lower()
    src_rep = source_representation
    if src_rep is None and source_smiles:
        src_rep = characterize_smiles(source_smiles)
    if isinstance(src_rep, dict) and src_rep.get("valid"):
        src_centers = (src_rep.get("stereochemistry") or {}).get("assigned_centers") or []
        mod_centers = stereo.get("assigned_centers") or []
        if src_centers:
            src_cips = Counter(c.get("cip") for c in src_centers if c.get("cip"))
            mod_cips = Counter(c.get("cip") for c in mod_centers if c.get("cip"))
            if "intermediate" not in role_l:
                if src_cips != mod_cips:
                    errors.append(
                        f"modeling stereochemistry does not match source stereoisomer "
                        f"(source CIP={dict(src_cips)}, modeling CIP={dict(mod_cips)})"
                    )
            elif not mod_cips:
                errors.append(
                    f"source stereocenters {dict(src_cips)} were lost in modeling intermediate"
                )

    # 2. Explicit formal charge check when declared
    if expected_formal_charge is not None:
        actual_charge = representation.get("formal_charge")
        if actual_charge != int(expected_formal_charge):
            errors.append(
                f"formal charge mismatch: got {actual_charge}, expected {int(expected_formal_charge)}"
            )

    # 3. Role- and pH-aware protonation validation
    ph = 7.4 if assay_pH is None else float(assay_pH)
    prot = representation.get("protonation_summary")
    if not isinstance(prot, dict):
        prot = (characterize_smiles(modeling_smiles).get("protonation_summary") or {})

    # Free carboxylic acids (pKa ~4-5) must be deprotonated (-COO-) at physiological/assay pH >= 6.0
    n_cooh = int(prot.get("protonated_carboxylic_acids") or 0)
    if ph >= 6.0 and n_cooh > 0:
        errors.append(
            f"protonated carboxylic acid (-COOH) present at pH {ph:g} (count={n_cooh}); "
            "free carboxylates should be modeled as deprotonated (-COO-)"
        )

    # Amine nucleophiles must have an unprotonated -NH2/-NH- to attack; at pH <= 9.5,
    # spectator aliphatic amines on diamines/polyamines should remain protonated (-NH3+).
    if ("amine" in role_l or "nucleophile" in role_l) and "intermediate" not in role_l:
        n_nuc = int(prot.get("neutral_nucleophilic_amines") or 0)
        n_aliph_neut = int(prot.get("neutral_aliphatic_amines") or 0)
        n_aliph_prot = int(prot.get("protonated_aliphatic_amines") or 0)
        if n_nuc == 0 and n_aliph_prot > 0:
            errors.append(
                "amine nucleophile has no neutral unprotonated amine (-NH2/-NH-) for nucleophilic attack; "
                "protonated -NH3+ cannot act as a nucleophile"
            )
        if (not allow_all_neutral_polyamine and ph <= 9.5
                and (n_aliph_neut + n_aliph_prot) >= 2 and n_aliph_prot == 0):
            errors.append(
                f"polyamine nucleophile at pH {ph:g} has all {n_aliph_neut} aliphatic amines unprotonated; "
                "model one attacking amine as neutral (-NH2) and spectator aliphatic amine(s) as protonated (-NH3+)"
            )

    return errors
