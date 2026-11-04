"""Frozen definitions, transfer scope, and discovery evidence integration."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from structural_modeling_agent.integration import (definition_digest, apply_bundle,
    frozen_geometry_score, frozen_linear_score, validate_bundle)
from enzyme_discovery_agent.tools.structural_evidence import get_structural_evaluation
from enzyme_discovery_agent.import_structural_evidence import import_structural_evidence


def bundle():
    definition = {"kind": "standardized_linear", "required_poses": 10,
                  "feature_definition_id": "test_v1", "features": ["a", "b"],
                  "means": [2., 3.], "scales": [2., 4.], "coefficients": [.2, -.3],
                  "intercept": -.4, "score_units": "log10 product concentration"}
    return {"schema_version": "1.0", "bundle_id": "test_bundle_v1", "definition": definition,
            "definition_sha256": definition_digest(definition), "sources": [],
            "calibration": {"training_ids": ["training_member"], "validation_status": "exploratory",
                "scope": {"step_index": 1, "mechanisms": ["declared"], "families": ["reference"],
                          "assembly": "reference_assembly"}}}


def candidate():
    return {"accession": "candidate", "discovery_rank": 7, "mechanism": "declared",
            "family": "reference", "step_index": 1, "assembly": "reference_assembly",
            "feature_definition_id": "test_v1", "n_poses": 10,
            "features": {"a": 4., "b": 7., "interface_jaccard_min": .8}}


def test_frozen_application_reuses_training_scaling_and_preserves_rank(tmp_path):
    b, c = bundle(), candidate()
    before = copy.deepcopy(b)
    result = apply_bundle(b, [c], tmp_path)["candidates"][0]
    assert result["score"] == pytest.approx(-.5)
    assert result["discovery_rank"] == 7 and result["status"] == "scored"
    assert result["standardized_features"] == {"a": 1., "b": 1.}
    assert result["definition_integrity"] == "verified"
    assert result["input_provenance"] == "unverified"
    assert result["calibration_source_integrity"] == "unverified"
    assert b == before


@pytest.mark.parametrize("key,value,expected", [
    ("family", "other", "family_transfer_unavailable"),
    ("assembly", "other", "incompatible_assembly"),
    ("step_index", 2, "incompatible_step"),
    ("mechanism", None, "unresolved_mechanism")])
def test_unsupported_transfer_does_not_force_a_score(tmp_path, key, value, expected):
    c = {**candidate(), key: value}
    result = apply_bundle(bundle(), [c], tmp_path)["candidates"][0]
    assert result["status"] == "not_applicable" and result["score"] is None
    assert result["transfer_status"] == expected


def test_missing_features_wrong_sampling_and_definition_drift_abstain(tmp_path):
    c = candidate()
    del c["features"]["b"]
    assert apply_bundle(bundle(), [c], tmp_path)["candidates"][0]["status"] == "unavailable"
    c = {**candidate(), "n_poses": 50}
    assert apply_bundle(bundle(), [c], tmp_path)["candidates"][0]["status"] == "unavailable"
    b = bundle()
    b["definition"]["coefficients"][0] = 100.
    with pytest.raises(ValueError, match="fingerprint"):
        validate_bundle(b)


def test_declared_feature_source_matches_values_and_fingerprint(tmp_path):
    from structural_modeling_agent.integration import digest
    c = candidate()
    source = tmp_path / "features.json"
    source.write_text(json.dumps({key: c[key] for key in ("accession", "feature_definition_id", "n_poses", "features")}))
    c.update(feature_source=str(source), feature_source_sha256=digest(source))
    result = apply_bundle(bundle(), [c], tmp_path)["candidates"][0]
    assert result["status"] == "scored"
    assert result["input_provenance"] == "source_content_verified"
    changed = {**c, "features": {**c["features"], "a": 9}}
    assert apply_bundle(bundle(), [changed], tmp_path)["candidates"][0]["status"] == "unavailable"
    source.write_text('{}')
    assert apply_bundle(bundle(), [c], tmp_path)["candidates"][0]["status"] == "unavailable"


def test_geometry_matches_the_existing_frozen_score():
    from exp_data.amide_synthetase.code.continuous_score import geometry_score
    d = np.linspace(2.5, 12., 50)
    a = np.linspace(20., 179., 50)
    definition = {"required_poses": 50, "parameters": {"distance_midpoint_A": 3.5,
        "distance_softness_A": .25, "angular_width_deg": 30., "regularization": .1}}
    actual = frozen_geometry_score(d, a, definition)
    score, total, effective = geometry_score(d, a)
    assert actual["score"] == pytest.approx(float(score), abs=1e-12)
    assert actual["distance_support_sum"] == pytest.approx(float(total), abs=1e-12)
    assert actual["effective_weight_count"] == pytest.approx(float(effective), abs=1e-12)


def test_new_manifest_is_readable_by_the_agent_and_imported_without_normalizing(tmp_path):
    report = apply_bundle(bundle(), [candidate()], tmp_path)
    manifest = tmp_path / "candidate_proxy_evidence.json"
    manifest.write_text(json.dumps(report))
    result = get_structural_evaluation("candidate", str(tmp_path))
    assert result["data"]["candidate"]["score"] == pytest.approx(-.5)
    run = tmp_path / "run"
    run.mkdir()
    (run / "candidate_registry.json").write_text(json.dumps({"candidates": {"candidate": {
        "accession": "candidate", "metadata": {}, "statuses": [{"status": "ranked"}]}}}))
    output = import_structural_evidence(run, manifest)
    assert output.name == "structural_proxy_review.json"
    registry = json.loads((run / "candidate_registry.json").read_text())
    assert registry["candidates"]["candidate"]["metadata"]["structural_proxy_evidence"]["score"] == pytest.approx(-.5)


def test_reference_template_reader_accepts_path_objects(tmp_path):
    gemmi = pytest.importorskip("gemmi")
    from structural_modeling_agent.proxies.ntn_ensemble import ca_cif
    structure = gemmi.Structure()
    model, chain, residue = gemmi.Model("1"), gemmi.Chain("A"), gemmi.Residue()
    residue.name, residue.seqid = "ALA", gemmi.SeqId(1, " ")
    atom = gemmi.Atom()
    atom.name, atom.pos = "CA", gemmi.Position(1., 2., 3.)
    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    path = tmp_path / "reference.cif"
    structure.make_mmcif_document().write_file(str(path))
    assert ca_cif(path)["A"][1].tolist() == [1., 2., 3.]


def test_current_example_uses_the_fixed_angle_metric():
    from structural_modeling_agent.integration_examples import campaign_configs
    from structural_modeling_agent.integration import export_bundle
    definition = export_bundle(campaign_configs()["amide_activation"])["definition"]
    assert definition["kind"] == "proximity_weighted_attack_angle"
    assert definition["metric_id"] == "inverse_distance_attack_angle_v1"
    assert definition["score_units"] == "degrees"
    assert definition["fitted_parameters"] == []


def test_structural_import_cannot_reopen_a_rejected_candidate(tmp_path):
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry, discovery_status
    run = tmp_path / "run"
    update_candidate_registry(run, "candidate", "rejected")
    report = apply_bundle(bundle(), [candidate()], tmp_path)
    manifest = tmp_path / "evidence.json"
    manifest.write_text(json.dumps(report))
    import_structural_evidence(run, manifest)
    assert discovery_status(load_candidate_registry(run)["candidates"]["candidate"]) == "rejected"


def test_current_angle_application_uses_real_coordinates_and_reports_distance(tmp_path):
    import gemmi
    import math
    def pose(filename, distance, angle):
        structure, model = gemmi.Structure(), gemmi.Model("1")
        xyz = (distance * math.cos(math.radians(angle)), distance * math.sin(math.radians(angle)))
        groups = [("A", "ATP", [("PA", "P", (0, 0, 0)), ("O3A", "O", (1, 0, 0))]),
                  ("B", "LIG", [("O1", "O", (*xyz, 0)), ("C1", "C", (*xyz, 1.3)), ("O2", "O", (*xyz, 2.5))])]
        for chain_id, name, atoms in groups:
            chain, residue = gemmi.Chain(chain_id), gemmi.Residue()
            residue.name, residue.seqid, residue.het_flag = name, gemmi.SeqId(1, " "), "H"
            for label, element, coordinates in atoms:
                atom = gemmi.Atom()
                atom.name, atom.element, atom.pos = label, gemmi.Element(element), gemmi.Position(*coordinates)
                residue.add_atom(atom)
            chain.add_residue(residue)
            model.add_chain(chain)
        structure.add_model(model)
        path = tmp_path / filename
        structure.write_pdb(str(path))
        return str(path)
    definition = {"kind": "proximity_weighted_attack_angle", "required_poses": 2,
        "metric_id": "inverse_distance_attack_angle_v1", "score_units": "degrees", "fitted_parameters": [],
        "phosphate_by_mechanism": {"declared": "PA"}}
    b = bundle()
    b.update(definition=definition, definition_sha256=definition_digest(definition))
    c = {**candidate(), "phosphate": "PA", "prediction_files": [pose("first.pdb", 2, 90), pose("second.pdb", 4, 180)]}
    result = apply_bundle(b, [c], tmp_path)["candidates"][0]
    assert result["status"] == "scored", result
    assert result["score"] == pytest.approx(120)
    assert result["harmonic_distance_A"] == pytest.approx(8/3)
    assert len(result["source_files"]) == 2
