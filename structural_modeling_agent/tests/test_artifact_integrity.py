import json

from structural_modeling_agent.integrity import proxy_integrity
from structural_modeling_agent.validation import file_digest, fingerprint


def test_proxy_integrity_detects_changed_or_new_sidecar_and_plan(tmp_path):
    prediction = tmp_path / "item_model_0.pdb"
    prediction.write_text("synthetic prediction")
    sidecar = tmp_path / "confidence_item_model_0.json"
    sidecar.write_text('{}')
    plan = {"schema_version": "1.0"}
    result = {"plan": plan, "plan_sha256": fingerprint(plan),
              "source_files": [{"path": str(prediction), "sha256": file_digest(prediction)}],
              "confidence_sources": [{"path": str(sidecar), "sha256": file_digest(sidecar)}]}
    artifact = tmp_path / "evaluation.json"
    artifact.write_text(json.dumps(result))
    assert proxy_integrity(plan, result, artifact)["status"] == "verified"
    assert proxy_integrity({"schema_version": "changed"}, result, artifact)["status"] == "unverified"
    sidecar.write_text('{"changed": true}')
    assert proxy_integrity(plan, result, artifact)["status"] == "unverified"
    sidecar.unlink()
    result["confidence_sources"] = []
    artifact.write_text(json.dumps(result))
    assert proxy_integrity(plan, result, artifact)["status"] == "verified"
    sidecar.write_text('{}')
    assert proxy_integrity(plan, result, artifact)["status"] == "unverified"
