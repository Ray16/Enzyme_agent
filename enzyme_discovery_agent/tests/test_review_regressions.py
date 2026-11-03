"""Provider changes, controller ownership and immutable shared-code snapshots."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from enzyme_discovery_agent.tools import mechanism_families as mf
from enzyme_discovery_agent import provenance
from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.state_store import controller_lock


def test_failover_result_is_not_cached_as_the_original_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ENZYME_AGENT_CACHE_DIR", str(tmp_path))
    client = SimpleNamespace(active_label="primary")
    def derive(*args, **kwargs):
        client.active_label = "fallback"
        return [{"name": "Synthetic fallback family"}]
    with patch.object(mf.config, "ENABLE_MECHANISM_FAMILY_CACHE", True), patch.object(mf, "family_passes_constraints", return_value=(True, "")):
        with patch.object(mf, "_derive_once", side_effect=derive):
            mf.derive_mechanism_families("synthetic transfer", "CCO", client, reflect=False, max_attempts=1)
        with patch.object(mf, "_derive_once", return_value=[{"name": "Synthetic primary family"}]) as fresh:
            result = mf.derive_mechanism_families("synthetic transfer", "CCO", SimpleNamespace(active_label="primary"), reflect=False, max_attempts=1)
        assert fresh.call_count == 1
        assert result[0]["name"] == "Synthetic primary family"


def test_discovery_refuses_second_controller_before_dispatch(tmp_path):
    agent = EnzymeAgent.__new__(EnzymeAgent)
    agent.run_dir = tmp_path
    with controller_lock(tmp_path), patch.object(agent, "_run") as run, patch.object(agent, "_close_logging"):
        with pytest.raises(BlockingIOError):
            agent.run("synthetic transformation")
        run.assert_not_called()


def test_shared_snapshot_covers_assets_and_preserves_old_code(tmp_path, monkeypatch):
    root = tmp_path / "source"
    package = root / "enzyme_discovery_agent"
    output = package / "custom_output"
    package.mkdir(parents=True)
    (package / "agent.py").write_text("version = 1\n")
    sibling = root / "structural_modeling_agent" / "knowledge"
    sibling.mkdir(parents=True)
    (sibling / "contract.json").write_text('{"version": 1}')
    (root / "pyproject.toml").write_text("[project]\nname='synthetic'\n")
    excluded = package / "runs" / "do_not_copy"
    excluded.mkdir(parents=True)
    (excluded / "result.json").write_text("{}")
    monkeypatch.setattr(provenance, "_REPO_ROOT", root)
    monkeypatch.setattr(provenance, "_PACKAGE_DIR", package)
    first = provenance.snapshot_source_code(output)
    old = output / first["source_snapshot_dir"]
    assert (old / "_dependencies/structural_modeling_agent/knowledge/contract.json").is_file()
    assert not (old / "runs").exists()
    (package / "agent.py").write_text("version = 2\n")
    second = provenance.snapshot_source_code(output)
    assert first["source_sha256"] != second["source_sha256"]
    assert first["source_files"] == second["source_files"]
    assert (old / "agent.py").read_text() == "version = 1\n"
    assert (output / second["source_snapshot_dir"] / "agent.py").read_text() == "version = 2\n"
