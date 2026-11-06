"""Starting-state protonation bookkeeping, independent of pKa predictions."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from structural_modeling_agent.sampling.protonation import validate_protonation_review


def prepared(state="HIE"):
    symbols = ["N", "N", "Mg"]
    positions = [[0, 0, 0], [2.4, 0, 0], [-2.1, 0, 0]]
    names = ["ND1", "NE2", "MG"]
    if state in ("HID", "HIP"):
        symbols.append("H"); positions.append([-1.02, 0, 0]); names.append("HD1")
    if state in ("HIE", "HIP"):
        symbols.append("H"); positions.append([3.42, 0, 0]); names.append("HE2")
    atoms = Atoms(symbols, positions=positions)
    atoms.new_array("atomnames", np.array(names))
    report = {
        "assay_conditions": {"pH": 7.5, "pH_source": "synthetic fixture; not an assay assignment"},
        "key_residue_review": {
            "coverage_evidence": "synthetic one-site fixture",
            "histidines": [{"residue_id": "A:123", "state": state, "ND1": 0, "NE2": 1,
                            "role": "metal ligand", "rationale": "synthetic test assignment",
                            "evidence": ["fixture"], "considered_states": ["HID", "HIE", "HIP"]}]
        }
    }
    return atoms, report


@pytest.mark.parametrize("state", ["HID", "HIE", "HIP"])
def test_explicit_histidine_states_match_hydrogen_geometry(state):
    atoms, report = prepared(state)
    assert validate_protonation_review(atoms, report)["histidines_checked"] == 1


def test_swapped_tautomer_is_rejected():
    atoms, report = prepared("HIE")
    report["key_residue_review"]["histidines"][0]["state"] = "HID"
    with pytest.raises(ValueError, match="hydrogens disagree"):
        validate_protonation_review(atoms, report)


def test_protonated_metal_donor_is_rejected():
    atoms, report = prepared("HID")
    site = report["key_residue_review"]["histidines"][0]
    site["metal_bindings"] = [{"nitrogen": "ND1", "metal_index": 2, "max_distance_A": 2.6}]
    with pytest.raises(ValueError, match="metal-donor.*proton"):
        validate_protonation_review(atoms, report)


def test_unprotonated_metal_donor_and_geometry_are_checked():
    atoms, report = prepared("HIE")
    site = report["key_residue_review"]["histidines"][0]
    site["metal_bindings"] = [{"nitrogen": "ND1", "metal_index": 2, "max_distance_A": 2.6}]
    validate_protonation_review(atoms, report)
    atoms.positions[2] = [-5, 0, 0]
    with pytest.raises(ValueError, match="coordination is absent"):
        validate_protonation_review(atoms, report)


def test_missing_review_or_atom_identity_fails_closed():
    atoms, report = prepared()
    incomplete = copy.deepcopy(report)
    del incomplete["key_residue_review"]
    with pytest.raises(ValueError, match="coverage"):
        validate_protonation_review(atoms, incomplete)
    del atoms.arrays["atomnames"]
    with pytest.raises(ValueError, match="retain ND1/NE2"):
        validate_protonation_review(atoms, report)


def test_no_histidines_requires_a_reason_and_assay_ph_is_explicit():
    atoms, report = prepared()
    report["key_residue_review"]["histidines"] = []
    with pytest.raises(ValueError, match="reason no key histidines"):
        validate_protonation_review(atoms, report)
    report["key_residue_review"]["no_key_histidines_reason"] = "Fixture coverage reviewed"
    validate_protonation_review(atoms, report)
    report["assay_conditions"]["pH"] = None
    with pytest.raises(ValueError, match="assay pH"):
        validate_protonation_review(atoms, report)
