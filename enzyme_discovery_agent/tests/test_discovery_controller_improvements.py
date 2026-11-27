"""Regression tests for discovery controller, review, and caching improvements."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from enzyme_discovery_agent import config
from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.gates import GateResult
from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.models import target_review as review
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.prompts import sections


def _register_verified_candidate(root, accession, *, reactions=None, ec_numbers=None, status="evaluated"):
    update_candidate_registry(
        root,
        accession,
        status,
        {
            "protein_name": f"Synthetic enzyme {accession}",
            "catalytic_activities": list(reactions or []),
            "ec_numbers": list(ec_numbers or []),
            "claim_validation": {
                "promotion_eligible": status == "evaluated",
                "claims": [
                    {"claim_type": kind, "subject": accession, "verdict": "supported"}
                    for kind in ("accession_identity", "record_active")
                ],
            },
        },
    )


def test_review_yield_does_not_trigger_forced_reflection(tmp_path, monkeypatch):
    """A single completed target-review batch should yield cleanly without reflection spam."""
    monkeypatch.setattr(config, "AUTO_TARGET_REVIEW", True)
    monkeypatch.setattr(config, "TARGET_REVIEW_BATCHES_BEFORE_DECISION", 1)
    monkeypatch.setattr(config, "ADAPTIVE_REASONING", False)

    with patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}):
        agent = EnzymeAgent(output_dir=tmp_path, token_budget=1000)
        update_candidate_registry(agent.run_dir, "SYNTH_S1", "screened", {"protein_name": "Synthetic catalyst"})

        def assess(*args, **kwargs):
            return {"assessed": [], "errors": [], "unassessed": []}

        def decide(**kwargs):
            agent.llm.resource_stop_reason = "Synthetic decision reached"
            return None

        with patch.object(review, "review_queue", return_value=[{"accession": "SYNTH_READY"}]), \
             patch.object(review, "run_review", side_effect=assess), \
             patch.object(agent.llm, "call", side_effect=decide), \
             patch("enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status", return_value={"required": False}), \
             patch("enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot", return_value=("checkpoint", 0)):
            agent.run("Synthetic transformation", substrate_smiles="CCO")

    events = read_events(agent.run_dir)
    assert not any(e.get("type") == "reflection_scheduled" for e in events)


def test_pre_checked_gates_do_not_redundantly_recheck_post_llm(tmp_path, monkeypatch):
    """Gates already checked pre-LLM are not redundantly re-checked post-LLM on non-eval turns."""
    monkeypatch.setattr(config, "AUTO_TARGET_REVIEW", False)
    monkeypatch.setattr(config, "ADAPTIVE_REASONING", False)

    with patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}):
        agent = EnzymeAgent(output_dir=tmp_path, token_budget=1000)
        calls = {"lit": 0}

        def count_lit(*args, **kwargs):
            calls["lit"] += 1
            return False

        agent._literature_harvest_gate.should_fire = Mock(side_effect=count_lit)
        for gate in (agent._mechanism_family_gate, agent._family_survey_gate,
                     agent._evidence_selection_gate, agent._systematic_gate,
                     agent._eval_gate, agent._expansion_gate):
            gate.should_fire = Mock(return_value=False)

        class _Msg:
            content = ""
            tool_calls = [
                {"id": "tc1", "type": "function", "function": {"name": "search_pubmed", "arguments": '{"query": "synthetic"}'}}
            ]
            def model_dump(self):
                return {"role": "assistant", "content": "", "tool_calls": self.tool_calls}

        def fake_call(**kwargs):
            return type("Resp", (), {
                "choices": [type("Ch", (), {"message": _Msg()})()],
                "usage": type("U", (), {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10})(),
            })()

        def fake_exec(name, args, iteration):
            agent.llm.resource_stop_reason = "done after one turn"
            return {"status": "success", "data": []}

        with patch.object(agent.llm, "call", side_effect=fake_call), \
             patch.object(agent, "_execute_single_tool", side_effect=fake_exec), \
             patch("enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status", return_value={"required": False}), \
             patch("enzyme_discovery_agent.models.recovery.next_recovery_action", return_value=None), \
             patch("enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot", return_value=("checkpoint", 0)):
            agent.run("Synthetic transformation", substrate_smiles="CCO")

    # Checked once pre-LLM; skipped post-LLM in the same turn.
    assert calls["lit"] == 1


def test_review_queue_skips_deterministic_partner_exclusions(tmp_path):
    """Recorded incompatible partner classes skip LLM target review unless explicitly requested."""
    (tmp_path / "target_context.json").write_text(
        json.dumps({"substrate_smiles": "O=C(O)CCC(=O)O.NCCCCN", "target_reasoning": {"reaction_direction": "synthesis"}})
    )
    _register_verified_candidate(
        tmp_path,
        "SYNTH_EXCL",
        reactions=["L-aspartate + L-glutamine + ATP + H2O = L-asparagine + L-glutamate + AMP + diphosphate + H(+)"],
        ec_numbers=["6.3.5.4"],
    )
    _register_verified_candidate(
        tmp_path,
        "SYNTH_COMPAT",
        reactions=["putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+)"],
        ec_numbers=["6.3.1.11"],
    )
    _register_verified_candidate(
        tmp_path,
        "SYNTH_UNANNOT",
        reactions=[],
        ec_numbers=[],
        status="evidence_pending",
    )

    queued = [row["accession"] for row in review.review_queue(tmp_path, limit=10)]
    assert "SYNTH_EXCL" not in queued
    assert "SYNTH_COMPAT" in queued
    assert "SYNTH_UNANNOT" in queued

    # Explicit accessions or force=True still allow reviewing the excluded candidate.
    forced = [row["accession"] for row in review.review_queue(tmp_path, limit=10, accessions={"SYNTH_EXCL"})]
    assert forced == ["SYNTH_EXCL"]


def test_revision_with_outcome_interpretation_auto_binds_outcome_evidence_id(tmp_path):
    """Explicit revises + outcome_interpretation binds the verified outcome evidence ID cleanly."""
    for i in range(20):
        update_candidate_registry(
            tmp_path,
            f"SYNTH_{i}",
            "evaluated",
            {
                "screening_family_id": "SYNTH_FOLD",
                "claim_validation": {"promotion_eligible": True},
                "substrate_similarity": {
                    "accommodation": {
                        "unaccommodated_targets": ["SYNTH_ROLE"],
                        "all_target_substrates_accommodated": False,
                    }
                },
            },
        )
    ctx = decisions.decision_context(tmp_path)
    obs_id = next(row["id"] for row in ctx["observations"] if row["kind"] != "user_target")
    base = {
        "hypothesis": "Family A may accommodate all target roles.",
        "alternative": "Family A lacks a counterpart for one target role.",
        "scope": "search_strategy",
        "prediction": "Focused lookup will locate a compatible member.",
        "contrasting_prediction": "Lookup will show the same missing role.",
        "why_this_test": "Separates native similarity from full-partner compatibility.",
        "revision_trigger": "Revise if a compatible member is retrieved.",
        "evidence_refs": [obs_id],
        "tool_name": "inspect_synthetic_evidence",
        "arguments": {},
    }
    decision_id = decisions.start_decision(tmp_path, base, {"inspect_synthetic_evidence"})
    outcome_ev = decisions.finish_decision(
        tmp_path, decision_id, {"status": "success", "summary": "Synthetic result", "data": {"ok": True}}
    )

    # Without outcome_interpretation and without outcome_ev in evidence_refs -> still rejected.
    with pytest.raises(ValueError, match="recorded outcome"):
        decisions.validate_plan(tmp_path, {**base, "revises": decision_id}, {"inspect_synthetic_evidence"})

    # With explicit revises + outcome_interpretation -> auto-binds verified outcome_ev.
    revised = decisions.validate_plan(
        tmp_path,
        {
            **base,
            "revises": decision_id,
            "belief_update": "retain",
            "outcome_interpretation": "The synthetic lookup returned a positive observation consistent with the hypothesis.",
        },
        {"inspect_synthetic_evidence"},
    )
    assert outcome_ev in revised["evidence_refs"]
    assert revised.get("outcome_citation_inferred_from_revises") is True


def test_literature_observations_cache_invalidates_on_new_artifact(tmp_path):
    """Stat-keyed cache returns cached observations and refreshes when a new artifact arrives."""
    lit_dir = tmp_path / "artifacts" / "literature"
    lit_dir.mkdir(parents=True)
    (lit_dir / "search_1.json").write_text(
        json.dumps({"result": {"data": [{"title": "Synthetic Paper One", "doi": "10.0000/one"}]}})
    )
    first = decisions._literature_observations(tmp_path)
    assert len(first) == 1
    assert first[0]["facts"]["title"] == "Synthetic Paper One"

    # Second call hits cache and returns same result.
    second = decisions._literature_observations(tmp_path)
    assert len(second) == 1

    # Adding a second artifact invalidates the stat signature.
    (lit_dir / "search_2.json").write_text(
        json.dumps({"result": {"data": [{"title": "Synthetic Paper Two", "doi": "10.0000/two"}]}})
    )
    third = decisions._literature_observations(tmp_path)
    assert len(third) == 2


def test_prompt_sections_hygiene():
    assert "semantic re-ranking" not in sections.ROLE.lower()
    assert "\n  5. Search by the protein's described function" in sections.SEARCH_GUIDANCE
    assert "call evaluate_candidate 15 times" not in sections.CANDIDATE_EVALUATION_GUIDANCE
