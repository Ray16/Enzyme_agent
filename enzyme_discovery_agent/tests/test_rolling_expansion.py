"""Expansion priority, rolling work, and budget recovery without live services."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.gates import AgentState
from enzyme_discovery_agent.gates.ec_triage import ECTriageGate
from enzyme_discovery_agent.gates.homolog_expansion import HomologExpansionGate
from enzyme_discovery_agent.gates.systematic_coverage import SystematicCoverageGate
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent.models.candidate_registry import (
    update_candidate_registry, load_candidate_registry, register_structural_homologs,
)
from enzyme_discovery_agent.models.expansion import select_expansion_seeds, record_seed_search
from enzyme_discovery_agent.models.rolling import new_evaluation_work, record_evaluation_batch
from enzyme_discovery_agent.tools import embedding_map, esm_expansion, esm_atlas_expansion, structural_homolog_search


def add(root, accession, fit, status="evaluated", **metadata):
    update_candidate_registry(root, accession, status, {"substrate_fit": fit, **metadata})


def candidates(root):
    return load_candidate_registry(root)["candidates"]


class TestExpansionPriority(unittest.TestCase):
    def test_later_high_fit_candidate_beats_early_entries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "A00001", .1)
            add(root, "Z00001", .9)
            self.assertEqual(select_expansion_seeds(candidates(root), 1, root), ["Z00001"])
            self.assertEqual(embedding_map._select_promising(candidates(root), 1, root), ["Z00001"])

    def test_evaluation_fit_overrides_old_survey_fit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "A00001", .99, substrate_similarity={"property_comparisons": [
                {"functional_group_similarity": .1, "mw_similarity": .1}]})
            add(root, "Z00001", .1, substrate_similarity={"property_comparisons": [
                {"functional_group_similarity": .9, "mw_similarity": .9}]})
            self.assertEqual(select_expansion_seeds(candidates(root), 1), ["Z00001"])

    def test_no_unevaluated_fallback_or_contradicted_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "A00001", 1, "discovered", source_tool="family_substrate_fit:IPR000001")
            add(root, "A00002", 1, "screened")
            add(root, "A00003", 1, critical_warnings=["SEVERE SIZE MISMATCH"])
            add(root, "A00004", 1, claim_validation={"promotion_eligible": False})
            add(root, "A00005", 1, evaluation_data={"critical_warnings": ["BOND TYPE MISMATCH"]})
            add(root, "A00006", 1)
            update_candidate_registry(root, "A00006", "reopened")
            self.assertEqual(select_expansion_seeds(candidates(root), 10), [])

    def test_modeling_status_preserves_evaluated_eligibility(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "A00001", .9)
            update_candidate_registry(root, "A00001", "structurally_evaluated")
            self.assertEqual(select_expansion_seeds(candidates(root), 1), ["A00001"])

    def test_complete_saved_ranking_is_honored_but_new_candidates_compete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "A00001", .8)
            add(root, "A00002", .7)
            (root / "pass1_ranking.json").write_text(json.dumps([
                {"accession": "A00002", "rank": 1}, {"accession": "A00001", "rank": 2}]))
            self.assertEqual(select_expansion_seeds(candidates(root), 1, root), ["A00002"])
            add(root, "Z00001", .99)
            self.assertEqual(select_expansion_seeds(candidates(root), 1, root), ["Z00001"])

    def test_new_ties_are_stable_and_nonpositive_limit_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "Z00001", .5)
            add(root, "A00001", .5)
            self.assertEqual(select_expansion_seeds(candidates(root), 2), ["A00001", "Z00001"])
            self.assertEqual(select_expansion_seeds(candidates(root), 0), [])

    def test_expansion_tools_defer_before_network_or_embedding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "P12345", 1, "discovered", source_tool="family_substrate_fit:IPR000001")
            with patch("enzyme_discovery_agent.config.ENABLE_CANDIDATE_MAP", True), \
                 patch.object(embedding_map, "_run_embed_subprocess") as map_embed, \
                 patch.object(esm_expansion, "embed_sequences") as expand_embed, \
                 patch.object(structural_homolog_search, "_fetch_alphafold_structure") as fold_fetch, \
                 patch.object(esm_atlas_expansion, "_fetch_alphafold_structure") as atlas_fetch:
                for function in (embedding_map.map_candidate_neighbors, esm_expansion.expand_candidates_esm,
                                 structural_homolog_search.search_structural_homologs,
                                 esm_atlas_expansion.search_esm_atlas_homologs):
                    self.assertEqual(function(accession="P12345", run_dir=str(root))["status"], "skipped")
                self.assertEqual(embedding_map.map_candidate_neighbors(run_dir=str(root))["status"], "skipped")
                self.assertEqual(esm_expansion.expand_candidates_esm(run_dir=str(root))["status"], "skipped")
                for mocked in (map_embed, expand_embed, fold_fetch, atlas_fetch):
                    mocked.assert_not_called()


class TestRollingWork(unittest.TestCase):
    def test_expansion_surfaces_the_decision_to_the_agent_without_running_a_search(self):
        # The gate must not expand on its own: it ranks the supported seeds and
        # hands the choice to the model, naming the tools to act with.
        with tempfile.TemporaryDirectory() as d, \
                patch("enzyme_discovery_agent.config.MAX_STRUCTURAL_EXPANSION_SEEDS", 1):
            root = Path(d)
            add(root, "A00001", .1)
            add(root, "Z00001", .9)
            state = AgentState(root, critique_injected=True)
            gate = HomologExpansionGate()
            fake = Mock()
            with patch("enzyme_discovery_agent.tools.structural_homolog_search."
                       "search_structural_homologs", fake):
                result = gate.execute(state, [])
            fake.assert_not_called()
            self.assertTrue(result.intercept)
            advisory = result.new_messages[0]["content"]
            # Strongest seed is offered; the agent decides, and the tools are named.
            self.assertIn("Z00001", advisory)
            self.assertIn("search_structural_homologs", advisory)
            self.assertIn("set_family_focus", advisory)

    def test_expansion_waits_for_its_reserved_cadence_then_becomes_due(self):
        from enzyme_discovery_agent.models.event_log import append_event
        from enzyme_discovery_agent.models.expansion import expansion_due
        with tempfile.TemporaryDirectory() as d, \
                patch("enzyme_discovery_agent.config.EXPANSION_EVAL_INTERVAL", 3):
            root = Path(d)
            add(root, "Z00001", .9)
            state = AgentState(root, critique_injected=True)
            gate = HomologExpansionGate()
            # No evaluation has happened yet: breadth does not pre-empt it.
            self.assertFalse(expansion_due(root, 3))
            self.assertFalse(gate.should_fire(state, False, []))
            for _ in range(3):
                append_event(root, "gate", gate="evaluation_backlog")
            # After a full interval of evaluation, the reserved pass is due.
            self.assertTrue(expansion_due(root, 3))
            self.assertTrue(gate.should_fire(state, False, []))
            # It never pre-empts the model's own tool calls on their turn.
            self.assertFalse(gate.should_fire(state, True, []))
            # Once it fires, the next pass waits another full interval.
            append_event(root, "gate", gate="homolog_expansion")
            self.assertFalse(expansion_due(root, 3))

    def test_literature_selects_strongest_in_each_family_and_revisits_new_top(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "A00001", .1, interpro_family_id="IPR000001")
            add(root, "Z00001", .9, interpro_family_id="IPR000001")
            add(root, "A00002", .8, interpro_family_id="IPR000002")
            gate = SystematicCoverageGate()
            self.assertEqual(gate._families(root), [("IPR000001", "Z00001"), ("IPR000002", "A00002")])
            for _, acc in gate._families(root):
                record_seed_search(root, "sequence_literature", acc, {"status": "success"})
            self.assertEqual(SystematicCoverageGate()._families(root), [])
            add(root, "P12345", .99, interpro_family_id="IPR000001")
            self.assertEqual(gate._families(root), [("IPR000001", "P12345")])

    def test_new_candidates_rearm_evaluation_despite_completed_gate_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "A00001", .1, "discovered")
            state = AgentState(root, candidate_count=1, eval_gate_fired=True)
            gate = ECTriageGate()
            self.assertTrue(gate.should_fire(state, False, []))
            record_evaluation_batch(root)
            self.assertFalse(gate.should_fire(state, False, []))
            add(root, "Z00001", .9, "discovered")
            self.assertTrue(gate.should_fire(state, False, []))
            record_evaluation_batch(root)
            self.assertFalse(new_evaluation_work(root))
            update_candidate_registry(root, "Z00001", None, {"literature_evidence": {"new": "evidence"}})
            self.assertTrue(new_evaluation_work(root))

    def test_retrieving_a_known_homolog_does_not_reset_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            add(root, "P12345", .9)
            register_structural_homologs(root, {"data": {"homologs": [{"accession": "P12345"}]}})
            self.assertEqual(candidates(root)["P12345"]["statuses"][-1]["status"], "evaluated")


class TestBudgetRecovery(unittest.TestCase):
    def test_daily_quota_checkpoint_resumes_on_the_saved_google_model(self):
        class Message:
            content = "continue"
            tool_calls = None

            def model_dump(self):
                return {"role": "assistant", "content": self.content}

        error = RuntimeError("quota exceeded")
        error.status_code = 429
        error.body = {"error": {"details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        ]}}
        response = SimpleNamespace(choices=[SimpleNamespace(message=Message())],
                                   usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        provider = Mock()
        provider.with_options.return_value = provider
        provider.chat.completions.create.side_effect = [error, response]
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.config.GEMINI_API_KEY", "test-key"), \
             patch("enzyme_discovery_agent.config.GEMINI_REQUESTS_PER_MINUTE", 0), \
             patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}):
            first = EnzymeAgent(model="gemini:test", output_dir=d, max_iterations=1)
            text = first.run("target", substrate_smiles="CCO")
            root = first.run_dir
            self.assertIn("daily quota", text)
            self.assertEqual(provider.chat.completions.create.call_count, 1)
            self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "budget_exhausted")
            resumed = EnzymeAgent(resume_run_dir=root, token_budget=10)
            self.assertEqual(resumed.model_name, "gemini:test")
            resumed.run("")
            self.assertEqual(provider.chat.completions.create.call_count, 2)
            self.assertEqual(json.loads(run_file(root, "token_usage.json").read_text())["total_tokens"], 10)

    def test_full_run_and_resume_stop_on_tokens_not_turns(self):
        class Message:
            content = "continue"
            tool_calls = None

            def model_dump(self):
                return {"role": "assistant", "content": self.content}

        class Provider:
            def __init__(self):
                self.chat = self
                self.completions = self
                self.calls = 0

            def with_options(self, **kwargs):
                return self

            def create(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(choices=[SimpleNamespace(message=Message())],
                                       usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))

        provider = Provider()
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                 {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]), \
             patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}):
            first = EnzymeAgent(model="test", output_dir=d, max_iterations=1, token_budget=20)
            first.run("target", substrate_smiles="CCO")
            root = first.run_dir
            self.assertEqual(provider.calls, 2)
            self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "budget_exhausted")
            self.assertEqual(json.loads((root / "agent_state" / "transcript.json").read_text())["iteration"], 2)
            resumed = EnzymeAgent(model="test", resume_run_dir=root, max_iterations=1, token_budget=10)
            resumed.run("")
            self.assertEqual(provider.calls, 3)
            self.assertEqual(json.loads(run_file(root, "token_usage.json").read_text())["total_tokens"], 30)
            self.assertEqual(json.loads((root / "agent_state" / "transcript.json").read_text())["iteration"], 3)

    def test_elapsed_time_budget_is_independent_of_tokens(self):
        agent = EnzymeAgent.__new__(EnzymeAgent)
        agent.runtime_budget_seconds = 1
        agent._budget_started = time.monotonic() - 2
        agent.token_budget = None
        self.assertIn("Elapsed-time", agent._budget_reason())

    def test_internal_llm_call_stops_before_provider_request(self):
        client = LLMClient.__new__(LLMClient)
        client._active_idx = 0
        client.backends = [object()]
        client.budget_check = lambda: "budget reached"
        client._call_backend = Mock()
        self.assertIsNone(client.call([]))
        self.assertTrue(client.budget_exhausted)
        client._call_backend.assert_not_called()

    def test_budget_report_is_local_and_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "manifest.json").write_text(json.dumps({"status": "running", "query": "target"}))
            (root / "target_context.json").write_text(json.dumps({"query": "target", "target_profile": {}}))
            add(root, "P12345", .9)
            add(root, "P23456", .8, "discovered")
            record_seed_search(root, "foldseek", "P12345", {"status": "success"})
            agent = EnzymeAgent.__new__(EnzymeAgent)
            agent.run_dir = root
            agent._gate_state = AgentState(root)
            agent._query = "target"
            agent.model_name = "test"
            agent.llm = Mock()
            agent.log = Mock()
            agent._log_event = Mock()
            agent.token_budget = 10
            agent.runtime_budget_seconds = None
            agent._budget_start_tokens = 0
            agent._budget_started = time.monotonic()
            usage = TokenUsage(model="test")
            usage.total_tokens = 10
            text = agent._finish_budget_checkpoint([{"role": "user", "content": "target"}], 7, usage, "budget reached")
            self.assertIn("interim report", text)
            self.assertNotIn("| P12345 |", text)
            self.assertIn("P12345",(root/'candidate_assessment_queue.csv').read_text())
            self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "budget_exhausted")
            checkpoint = json.loads((root / "agent_state" / "budget_checkpoint.json").read_text())
            self.assertEqual(checkpoint["pending_candidates"], ["P23456"])
            agent.llm.assert_not_called()
            self.assertEqual(agent.llm.mock_calls, [])
            agent._resume_requested = True
            restored = agent._load_resume_payload()
            self.assertEqual(restored["iteration"], 7)
            self.assertEqual(restored["messages"][-1]["content"], "target")
            self.assertEqual(HomologExpansionGate()._seeds(root), [])


if __name__ == "__main__":
    unittest.main()


def test_evaluation_batches_prefer_under_sampled_labels_without_discarding():
    from enzyme_discovery_agent.models.rolling import diversify_evaluation_batch
    candidates = {
        'a0': {'statuses': [{'status': 'evaluated'}], 'metadata': {'gene_names': ['alpha']}},
        'a1': {'statuses': [{'status': 'screened'}], 'metadata': {'gene_names': ['alpha']}},
        'a2': {'statuses': [{'status': 'screened'}], 'metadata': {'gene_names': ['alpha']}},
        'b1': {'statuses': [{'status': 'screened'}], 'metadata': {'gene_names': ['beta']}},
        'b2': {'statuses': [{'status': 'screened'}], 'metadata': {'gene_names': ['beta']}},
        'c1': {'statuses': [{'status': 'screened'}], 'metadata': {'gene_names': ['gamma']}},
    }
    ordered = ['a1', 'a2', 'b1', 'b2', 'c1']
    assert diversify_evaluation_batch(candidates, ordered, 2) == ['b1', 'c1']
    assert set(diversify_evaluation_batch(candidates, ordered, 10)) == set(ordered)


def test_evaluation_diversity_preserves_priority_on_equal_coverage():
    from enzyme_discovery_agent.models.rolling import diversify_evaluation_batch
    candidates = {'a': {'statuses': [{'status': 'screened'}], 'metadata': {}},
                  'b': {'statuses': [{'status': 'screened'}], 'metadata': {}}}
    assert diversify_evaluation_batch(candidates, ['b', 'a'], 1) == ['b']
