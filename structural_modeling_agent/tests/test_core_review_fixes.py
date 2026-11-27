"""Regression tests for core structural_modeling_agent review fixes."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from structural_modeling_agent.agent import ModelingAgent
from structural_modeling_agent.controller.dispatch_handlers import dispatch_tool
from structural_modeling_agent.io_utils import file_digest
from test_modeling_agent import Message, make_agent


def test_authorized_evidence_supports_directories_when_allowed(tmp_path):
    agent = make_agent(tmp_path)
    art_dir = agent.run_dir / "artifacts" / "channels_demo"
    art_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="must be a file"):
        agent._authorized_evidence(art_dir)
    assert agent._authorized_evidence(art_dir, allow_dir=True) == art_dir.resolve()

    outside = tmp_path / "unauthorized_dir"
    outside.mkdir()
    with pytest.raises(PermissionError):
        agent._authorized_evidence(outside, allow_dir=True)


def test_task_evidence_paths_authorizes_dir_fields(tmp_path):
    supplied_dir = tmp_path / "mlmm_out"
    supplied_dir.mkdir()
    task = {
        "target": {"name": "test_target"},
        "candidates": [
            {
                "accession": "SYNTH1",
                "sequence": "MKTAYIAK",
                "evidence": {"results_dir": str(supplied_dir)},
            }
        ],
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    agent = ModelingAgent(task_path=task_path, run_dir=tmp_path / "run")
    assert agent._authorized_evidence(supplied_dir, allow_dir=True) == supplied_dir.resolve()


def test_unloaded_domain_tool_returns_recoverable_skill_hint(tmp_path):
    class Client:
        def __init__(self):
            self.turn = 0

        def call(self, **kwargs):
            self.turn += 1
            if self.turn == 1:
                calls = [("analyze_substrate_contacts", {"mode": "pose", "pose_path": "/nonexistent"})]
            else:
                calls = [("finish_task", {"outcome": "inconclusive", "rationale": "Recovered after skill hint."})]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=Message(calls))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
            )

    agent = make_agent(tmp_path, client=Client())
    report = agent.run(max_turns=3, token_budget=1000)
    assert report["status"] == "inconclusive"
    tool_msgs = [json.loads(m["content"]) for m in agent.messages if m.get("role") == "tool"]
    first_summary = json.loads(tool_msgs[0]["summary"])
    assert first_summary["status"] == "blocked"
    assert "load_skill" in first_summary["reason"]


def test_loading_multiple_skills_does_not_overflow_context_allowance(tmp_path):
    agent = make_agent(tmp_path)
    from structural_modeling_agent.controller.skills import list_skills

    for skill in list_skills():
        agent.dispatch("load_skill", {"name": skill["name"]})
    context = agent._context()
    assert "Active domain playbooks:" in context[0]["content"]


def test_file_digest_cache_detects_modifications(tmp_path):
    big = tmp_path / "large.txt"
    big.write_bytes(b"A" * 8192)
    h1 = file_digest(big)
    assert file_digest(big) == h1
    big.write_bytes(b"B" * 8193)
    h2 = file_digest(big)
    assert h2 != h1


def test_dispatch_tool_delegates_to_unified_registry(tmp_path):
    agent = make_agent(tmp_path)
    res = dispatch_tool(agent, "load_skill", {"name": "cofold-setup"})
    assert res["name"] == "cofold-setup"


def test_family_cofold_reexports_pose_analysis():
    from structural_modeling_agent.family import cofold, pose_analysis

    assert cofold.analyze_outputs is pose_analysis.analyze_outputs
    assert cofold._holo_seating_metrics is pose_analysis._holo_seating_metrics
    assert cofold.DEFAULT_SAMPLES == 50
    assert cofold.DEFAULT_SEED == 42
