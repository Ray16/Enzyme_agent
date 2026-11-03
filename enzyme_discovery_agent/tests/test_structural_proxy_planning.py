"""Agent access to the generic catalog and pre-label declaration checks."""
from enzyme_discovery_agent.tools.structural_proxy_planning import (
    get_structural_proxy_guidance, validate_structural_proxy_plan,
    inspect_experimental_proxy_bundle)
from enzyme_discovery_agent.structural_proxy_planning import load_proxy_catalog


def test_all_five_categories_are_available_without_activity_labels():
    result = get_structural_proxy_guidance("declared mechanism", "declared step", "matched endpoint", ["reactive_roles"])
    assert result["status"] == "success"
    assert len({d["category"] for d in result["data"]["descriptors"]}) == 5
    entries = {d["id"]: d for d in result["data"]["descriptors"]}
    assert entries["approach_distance"]["ready_from_declared_inputs"]
    assert not entries["protein_substrate_iptm"]["ready_from_declared_inputs"]
    assert entries["reaction_barrier"]["execution"] == "unavailable"


def test_plan_requires_rationale_and_a_declared_direction():
    p = {"schema_version": "1.0", "mechanism": "declared mechanism", "reaction_step": "declared step",
         "assay_endpoint": "matched endpoint", "context": {}, "hypotheses": [{
            "proxy_id": "approach_distance", "rationale": "Reactive groups must approach",
            "expected_direction": "-", "evidence_ids": ["inference:mechanism"]}]}
    assert validate_structural_proxy_plan(p)["data"]["valid"]
    p["hypotheses"][0]["expected_direction"] = None
    assert not validate_structural_proxy_plan(p)["data"]["valid"]
    p["hypotheses"][0]["expected_direction"] = "-"
    p["hypotheses"].append(dict(p["hypotheses"][0]))
    assert not validate_structural_proxy_plan(p)["data"]["valid"]


def test_catalog_does_not_claim_transferred_descriptors_are_validated():
    descriptors = load_proxy_catalog()["descriptors"]
    assert all(d["evidence_status"] in ("candidate_descriptor", "confidence_baseline") for d in descriptors)


def test_frozen_bundle_inspection_checks_definition_and_omits_training_identities(tmp_path):
    import hashlib
    import json
    definition = {"kind": "test", "features": ["descriptor"]}
    checksum = hashlib.sha256(json.dumps(definition, sort_keys=True, allow_nan=False).encode()).hexdigest()
    path = tmp_path / "bundle.json"
    bundle = {"schema_version": "1.0", "bundle_id": "test", "definition": definition,
              "definition_sha256": checksum, "calibration": {"training_ids": ["training_member"],
              "validation_status": "exploratory"}, "sources": []}
    path.write_text(json.dumps(bundle))
    result = inspect_experimental_proxy_bundle(str(path))
    assert result["status"] == "success"
    assert result["data"]["calibration"]["n_training_enzymes"] == 1
    assert "training_ids" not in result["data"]["calibration"]
    bundle["definition"]["features"] = ["changed"]
    path.write_text(json.dumps(bundle))
    assert inspect_experimental_proxy_bundle(str(path))["status"] == "error"
