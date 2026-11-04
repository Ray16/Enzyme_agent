"""Independent setup evidence, missingness and joint physical checks."""
import copy
import hashlib

import pytest

from structural_modeling_agent.validation import (entity_map_for, file_digest, validate_ensemble,
                                            validate_setup)


def modeling_record(tmp_path):
    source = tmp_path / "reference.txt"
    source.write_text("Synthetic reference: intact construct AG, one complete single-chain site, no cofactors.\n"
                      "Synthetic tolerances are specified only for these unit tests.\n")
    evidence = [{"kind": "primary_literature", "source_id": "synthetic:reference",
                 "path": str(source), "sha256": file_digest(source),
                 "quote": "Synthetic reference: intact construct AG, one complete single-chain site, no cofactors.",
                 "reviewed": True, "rationale": "Synthetic reference for this test's declared expectations."}]
    substrate = {"component_index": 1, "role": "reactant", "include_in_model": True,
                 "modeling_smiles": "CC=O", "review_status": "reviewed"}
    chain = {"source": "full_sequence"}
    expected = {
        "assembly": {"protomer_chain_counts": [1], "active_sites": 1, "site_context": "complete"},
        "enzyme_state": {"state": "intact", "chains": [chain], "precursor_sha256": hashlib.sha256(b"AG").hexdigest()},
        "cofactors": {"complete_for_step": True, "species": []},
        "substrates": {"components": [substrate]},
        "reaction_step": {"index": 1, "mechanism": "synthetic_transfer"},
    }
    atom = lambda chain, number, name: {"chain": chain, "residue": number, "atom": name}
    return {"status": "ready", "method": "cofold", "unresolved": [],
            "enzyme_state": {"state": "intact", "chains": [chain]}, "cofactors": [],
            "target_substrates": [substrate], "modeled_step": expected["reaction_step"],
            "geometry": {"step": 1},
            "validation_plan": {"schema_version": "1.0", "setup": {
                key: {"expected": copy.deepcopy(value), "evidence": copy.deepcopy(evidence)}
                for key, value in expected.items()},
                "poses": {"expected_predictions": 1, "minimum_passing_predictions": 1,
                          "evidence": evidence,
                          "sites": [{"id": "site1", "substrate_chain": "B",
                                     "nucleophile": atom("A", "1", "N"),
                                     "electrophile": atom("B", "1", "C1"),
                                     "angle_reference": atom("B", "1", "O1"),
                                     "distance_range_A": [2.5, 4.], "angle_range_deg": [170., 180.]}],
                          "ligand_environment": {"contact_radius_A": 5., "clash_distance_A": 1.,
                                                 "minimum_contact_atoms": 1, "maximum_clashing_atoms": 0}}}}


def execution_for(record):
    return {"accession": "SYNTHETIC1", "modeling_input": record,
            "entity_map": entity_map_for([{"sequence": "AG", "msa": "unused"}],
                                         record["cofactors"], record["target_substrates"])}


def write_pose(path, displacement=0., cif=False):
    import gemmi
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    protein = gemmi.Chain("A")
    for number, name in [(1, "ALA"), (2, "GLY")]:
        residue = gemmi.Residue()
        residue.name, residue.seqid, residue.het_flag = name, gemmi.SeqId(number, " "), "A"
        labels = ("N", "CA", "C", "O", "CB") if name == "ALA" else ("N", "CA", "C", "O")
        for i, label in enumerate(labels):
            atom = gemmi.Atom()
            atom.name, atom.element = label, gemmi.Element("N" if label == "N" else "O" if label == "O" else "C")
            atom.pos = (gemmi.Position(0, (number-1)*3+i, 0) if i < 3 else
                        gemmi.Position(0, (number-1)*3+2, 1.2) if label == "O" else
                        gemmi.Position(-1.5, (number-1)*3+1, 0))
            residue.add_atom(atom)
        protein.add_residue(residue)
    model.add_chain(protein)
    ligand = gemmi.Chain("B")
    residue = gemmi.Residue()
    residue.name, residue.seqid, residue.het_flag = "LIG", gemmi.SeqId(1, " "), "H"
    for label, element, xyz in [("C1", "C", (3+displacement, 0, 0)),
                                ("C2", "C", (3+displacement, 1.5, 0)),
                                ("O1", "O", (4.2+displacement, 0, 0))]:
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = label, gemmi.Element(element), gemmi.Position(*xyz)
        residue.add_atom(atom)
    ligand.add_residue(residue)
    model.add_chain(ligand)
    structure.add_model(model)
    if cif:
        structure.make_mmcif_document().write_file(str(path))
    else:
        structure.write_pdb(str(path))
    return path


def test_setup_recomputes_evidence_and_decisions(tmp_path):
    record = modeling_record(tmp_path)
    assert validate_setup(record, "AG")["status"] == "pass"
    assert validate_setup(record, "AA")["status"] == "fail"
    record["enzyme_state"]["chains"] *= 2
    assert validate_setup(record, "AG")["status"] == "fail"


def test_unknown_is_not_supported_or_negative(tmp_path):
    record = modeling_record(tmp_path)
    del record["validation_plan"]["setup"]["assembly"]
    assert validate_setup(record, "AG")["status"] == "abstain"
    record = modeling_record(tmp_path)
    record["validation_plan"]["setup"]["assembly"]["evidence"][0]["derived_from_prediction"] = True
    assert validate_setup(record, "AG")["status"] == "abstain"
    source = tmp_path / "reference.txt"
    source.write_text("changed after review")
    assert validate_setup(record, "AG")["status"] == "fail"


def test_cofactor_identity_count_and_per_site_scaling(tmp_path):
    record = modeling_record(tmp_path)
    record["validation_plan"]["setup"]["assembly"]["expected"]["active_sites"] = 2
    expected = record["validation_plan"]["setup"]["cofactors"]["expected"]
    expected["species"] = [{"ccd": "MG", "role": "catalytic_metal", "count_per_active_site": 2}]
    record["cofactors"] = [{"ccd": "MG", "role": "catalytic_metal", "count": 4}]
    assert validate_setup(record, "AG")["status"] == "pass"
    record["cofactors"][0]["count"] = 2
    assert validate_setup(record, "AG")["status"] == "fail"
    record["cofactors"][0].update(ccd="MN", count=4)
    assert validate_setup(record, "AG")["status"] == "fail"
    record["cofactors"] = []
    assert validate_setup(record, "AG")["status"] == "fail"


@pytest.mark.parametrize("cif", [False, True])
def test_pose_inventory_and_joint_attack_checks(tmp_path, cif):
    record = modeling_record(tmp_path)
    path = write_pose(tmp_path / ("pose.cif" if cif else "pose.pdb"), cif=cif)
    report = validate_ensemble([path], execution_for(record))
    assert report["status"] == "pass", report
    assert report["n_passing"] == 1
    site = next(c for c in report["poses"][0]["checks"] if c["check"] == "site.site1")
    assert site["distance_A"] == pytest.approx(3.)
    assert site["angle_deg"] == pytest.approx(180.)
    record["validation_plan"]["poses"]["sites"][0]["angle_range_deg"] = [90., 120.]
    assert validate_ensemble([path], execution_for(record))["status"] == "fail"


def test_bad_poses_and_incomplete_ensembles_are_distinguished(tmp_path):
    record = modeling_record(tmp_path)
    path = write_pose(tmp_path / "pose.pdb", displacement=30)
    assert validate_ensemble([path], execution_for(record))["status"] == "fail"
    record["validation_plan"]["poses"]["expected_predictions"] = 2
    assert validate_ensemble([path], execution_for(record))["status"] == "abstain"
    assert validate_ensemble([], execution_for(record))["status"] == "abstain"
    assert validate_ensemble([path, path], execution_for(record))["status"] == "fail"


def test_missing_reactive_atom_abstains_instead_of_zero(tmp_path):
    record = modeling_record(tmp_path)
    record["validation_plan"]["poses"]["sites"][0]["nucleophile"]["atom"] = "ABSENT"
    path = write_pose(tmp_path / "pose.pdb")
    report = validate_ensemble([path], execution_for(record))
    assert report["status"] == "abstain"
    assert report["n_unresolved"] == 1


def test_all_modeled_sites_and_cofactors_need_checks(tmp_path):
    record = modeling_record(tmp_path)
    path = write_pose(tmp_path / "pose.pdb")
    record["validation_plan"]["setup"]["assembly"]["expected"]["active_sites"] = 2
    assert validate_ensemble([path], execution_for(record))["status"] == "abstain"


def test_assembly_prediction_cannot_justify_itself():
    from structural_modeling_agent.modeling_reasoning.modeling_unit import UnitEvidence, assess
    report = assess(UnitEvidence("SYNTHETIC", liganded_other_chain_fraction=.9,
                                 derived_from_docked_poses=True), 10)
    assert report["unit"] == "undetermined"
    assert report["evidence_rank_used"] is None


def test_metal_identity_and_declared_coordination_are_checked(tmp_path):
    import gemmi
    record = modeling_record(tmp_path)
    record["cofactors"] = [{"ccd": "MG", "role": "catalytic_metal", "count": 1}]
    plan = record["validation_plan"]["poses"]
    plan["sites"][0]["substrate_chain"] = "C"
    for role in ("electrophile", "angle_reference"):
        plan["sites"][0][role]["chain"] = "C"
    center = {"chain": "B", "residue": "1", "atom": "MG"}
    donor = {"chain": "A", "residue": "1", "atom": "N"}
    plan["distance_checks"] = [{"id": "metal_contact", "a": center, "b": donor, "range_A": [1.8, 2.4]}]
    plan["coordination_checks"] = [{"id": "metal_site", "center": center, "element": "MG",
                                    "donors": [donor], "distance_range_A": [1.8, 2.4], "donor_count_range": [1, 1]}]
    path = write_pose(tmp_path / "pose.pdb")
    structure = gemmi.read_structure(str(path))
    structure[0]["B"].name = "C"
    chain, residue, atom = gemmi.Chain("B"), gemmi.Residue(), gemmi.Atom()
    residue.name, residue.seqid, residue.het_flag = "MG", gemmi.SeqId(1, " "), "H"
    atom.name, atom.element, atom.pos = "MG", gemmi.Element("Mg"), gemmi.Position(-2.1, 0, 0)
    residue.add_atom(atom)
    chain.add_residue(residue)
    structure[0].add_chain(chain)
    structure.write_pdb(str(path))
    report = validate_ensemble([path], execution_for(record))
    assert report["status"] == "pass", report
    structure[0]["B"][0][0].pos.x = -20
    structure.write_pdb(str(path))
    report = validate_ensemble([path], execution_for(record))
    assert report["status"] == "fail"
    assert any(c["check"] == "coordination.metal_site" and c["status"] == "fail" for c in report["poses"][0]["checks"])


def test_required_interface_can_be_present_or_separated(tmp_path):
    import gemmi
    record = modeling_record(tmp_path)
    path = write_pose(tmp_path / "pose.pdb")
    structure = gemmi.read_structure(str(path))
    second = structure[0]["A"].clone()
    second.name = "C"
    for residue in second:
        for atom in residue:
            atom.pos.x -= 3
    structure[0].add_chain(second)
    structure.write_pdb(str(path))
    execution = execution_for(record)
    execution["entity_map"].append({"chain": "C", "kind": "protein", "sequence": "AG"})
    assert validate_ensemble([path], execution)["status"] == "abstain"
    record["validation_plan"]["poses"]["interfaces"] = [
        {"id": "contact", "chains": ["A", "C"], "contact_radius_A": 4., "minimum_residue_pairs": 1}]
    assert validate_ensemble([path], execution)["status"] == "pass"
    for residue in structure[0]["C"]:
        for atom in residue:
            atom.pos.x -= 30
    structure.write_pdb(str(path))
    assert validate_ensemble([path], execution)["status"] == "fail"


def test_msa_query_and_fragment_identity(tmp_path):
    from structural_modeling_agent.validation import validate_msa
    path = tmp_path / "query.csv"
    path.write_text("key,sequence\n0,AG\n")
    assert validate_msa(path, "AG")["sha256"] == file_digest(path)
    with pytest.raises(ValueError, match="does not match"):
        validate_msa(path, "AA")
