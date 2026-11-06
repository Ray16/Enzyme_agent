import hashlib
import json

from structural_modeling_agent.sampling.toolchain import inspect_toolchain


def test_missing_or_changed_evidence_revokes_recorded_scope(tmp_path):
    evidence = tmp_path / "report.json"
    evidence.write_text('{"status":"passed","scope":"software only"}')
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps({"schema": "processivity-toolchain-1.0",
        "checks": [{"id": "fixture", "evidence": [{"path": evidence.name,
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]}],
        "environments": [{"python": "/unavailable/python"}]}))
    report = inspect_toolchain(manifest)
    assert report["recorded_evidence_current"]
    assert not report["environments"][0]["interpreter_available"]
    assert not report["execution_authorized"] and not report["chemistry_validated"]
    evidence.write_text("changed")
    assert not inspect_toolchain(manifest)["recorded_evidence_current"]
    evidence.unlink()
    assert not inspect_toolchain(manifest)["recorded_evidence_current"]


def test_agent_exposes_inventory_without_starting_a_simulation(tmp_path):
    from test_modeling_agent import make_agent
    agent = make_agent(tmp_path)
    report = agent.dispatch("inspect_processivity_toolchain", {})
    assert report["schema"] == "processivity-toolchain-1.0"
    assert report["checks"]
    assert report["recorded_evidence_current"]
    assert not report["live_hardware_tested_now"] and not report["execution_authorized"]
