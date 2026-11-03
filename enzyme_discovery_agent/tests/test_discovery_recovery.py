"""Research continues through an idle model using grounded, resumable tasks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.gates import AgentState
from enzyme_discovery_agent.gates.literature_harvest import LiteratureHarvestGate
from enzyme_discovery_agent.models.event_log import append_event
from enzyme_discovery_agent.models.recovery import next_recovery_action, record_recovery_attempt


class RecoveryTest(unittest.TestCase):
    def seed_family(self, root):
        (root / "mechanism_families.json").write_text(json.dumps([{
            "name": "grounded test hypothesis", "promise": "high",
            "literature_terms": ["enzyme acid amine ligation"], "descriptors": [],
        }]))

    def test_negative_search_closes_task_across_resume_and_new_hypothesis_adds_work(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.seed_family(root)
            action = next_recovery_action(root, {"search_pubmed"})
            record_recovery_attempt(root, action, "started")
            record_recovery_attempt(root, action, "empty")
            self.assertIsNone(next_recovery_action(root, {"search_pubmed"}))
            families = json.loads((root / "mechanism_families.json").read_text())
            families[0]["literature_terms"].append("second distinct hypothesis")
            (root / "mechanism_families.json").write_text(json.dumps(families))
            self.assertEqual(next_recovery_action(root, {"search_pubmed"})["args"]["query"], "second distinct hypothesis")

    def test_successful_model_search_is_not_repeated_and_error_gets_one_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.seed_family(root)
            action = next_recovery_action(root, {"search_pubmed"})
            for _ in range(2):
                record_recovery_attempt(root, action, "started")
                record_recovery_attempt(root, action, "error")
            self.assertIsNone(next_recovery_action(root, {"search_pubmed"}))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.seed_family(root)
            append_event(root, "tool_call", tool="search_pubmed", args={"query": "enzyme acid amine ligation"}, status="success")
            self.assertIsNone(next_recovery_action(root, {"search_pubmed"}))
            self.assertIsNone(next_recovery_action(root, {"get_notebook"}))

    def test_controller_searches_unresolved_task_when_model_only_polls_notebook(self):
        class Message:
            content = None
            tool_calls = [SimpleNamespace(id="read", function=SimpleNamespace(name="get_notebook", arguments="{}"))]
            def model_dump(self):
                return {"role": "assistant", "content": None, "tool_calls": [{"id": "read", "type": "function",
                        "function": {"name": "get_notebook", "arguments": "{}"}}]}
        provider = Mock()
        provider.with_options.return_value = provider
        provider.chat.completions.create.side_effect = lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=Message())], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        schemas = [{"type": "function", "function": {"name": name, "parameters": {}}}
                   for name in ("get_notebook", "search_pubmed")]
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
             patch("enzyme_discovery_agent.agent.get_all_tools", return_value=schemas), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                 {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]), \
             patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}):
            agent = EnzymeAgent(model="test", output_dir=d)
            self.seed_family(agent.run_dir)
            performed = []
            def execute(name, args, iteration):
                performed.append(name)
                append_event(agent.run_dir, "tool_call", tool=name, args=args, status="empty")
                return {"status": "empty", "data": []}
            with patch.object(agent, "_execute_single_tool", side_effect=execute):
                report = agent.run("target", substrate_smiles="CCO")
            self.assertEqual(performed.count("search_pubmed"), 1)
            self.assertEqual(provider.chat.completions.create.call_count, 5)
            self.assertIn("queued recovery searches exhausted", report)


class ResearchCadenceTest(unittest.TestCase):
    def test_complementary_search_and_citations_run_despite_changing_extraction_output(self):
        from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
        from enzyme_discovery_agent.models.event_log import read_events

        class Message:
            content = None
            tool_calls = [SimpleNamespace(id="extract", function=SimpleNamespace(
                name="extract_reaction_mechanism", arguments='{"enzyme":"Synthetic catalyst"}'))]

            def model_dump(self):
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "extract", "type": "function", "function": {
                        "name": "extract_reaction_mechanism", "arguments": '{"enzyme":"Synthetic catalyst"}'}}]}

        provider = Mock()
        provider.with_options.return_value = provider
        provider.chat.completions.create.side_effect = lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=Message())], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        schemas = [{"type": "function", "function": {"name": name, "parameters": {}}}
                   for name in ("extract_reaction_mechanism", "search_pubmed", "follow_paper_citations")]
        citation = {"tool": "follow_paper_citations", "args": {"doi": "10.test/primary"},
                    "reason": "Follow retrieved study", "selection": "citation", "lane": "citation"}
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
             patch("enzyme_discovery_agent.agent.get_all_tools", return_value=schemas), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                 {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]), \
             patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}), \
             patch("enzyme_discovery_agent.models.literature_strategy.citation_actions", return_value=[citation]), \
             patch("enzyme_discovery_agent.config.DISCOVERY_RESEARCH_INTERVAL", 3):
            agent = EnzymeAgent(model="test", output_dir=d, token_budget=80)
            (agent.run_dir / "agent_state/mechanism_families.json").write_text(json.dumps([{
                "name": "Synthetic catalytic family", "promise": "low", "literature_terms": ["synthetic reaction scope"]}]))
            performed = []

            def execute(name, args, iteration):
                performed.append((name, provider.chat.completions.create.call_count))
                if name == "extract_reaction_mechanism":
                    # A model can keep changing a claimed mechanism. Even if this
                    # passes progress detection, it must not starve other routes.
                    update_candidate_registry(agent.run_dir, "P12345", "rejected", {
                        "mechanism_evidence": {"status": "ok", "facts": {"chemical_step": str(len(performed))}}})
                append_event(agent.run_dir, "tool_call", tool=name, args=args, status="empty")
                return {"status": "empty", "data": []}

            with patch.object(agent, "_execute_single_tool", side_effect=execute):
                agent.run("target", substrate_smiles="CCO")
            research = [(name, turn) for name, turn in performed if name != "extract_reaction_mechanism"]
            self.assertEqual([name for name, _ in research], ["search_pubmed", "follow_paper_citations"])
            self.assertLessEqual(research[0][1], 3)
            self.assertLessEqual(research[1][1], 6)
            events = read_events(agent.run_dir)
            self.assertEqual(sum(e["type"] == "discovery_recovery" for e in events), 2)
            checkpoint = json.loads((agent.run_dir / "agent_state/transcript.json").read_text())
            self.assertGreaterEqual(checkpoint["research_progress"]["model_turns_since_recovery"], 2)

    def test_resume_preserves_research_debt_for_current_and_legacy_checkpoints(self):
        for saved_progress in ({"model_turns_since_recovery": 4}, {}):
            with self.subTest(saved_progress=saved_progress), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "agent_state").mkdir()
                (root / "manifest.json").write_text(json.dumps({"status": "budget_exhausted", "query": "target"}))
                (root / "target_context.json").write_text(json.dumps({"target_profile": {}}))
                (root / "agent_state/transcript.json").write_text(json.dumps({
                    "iteration": 6, "messages": [{"role": "user", "content": "target"}],
                    "research_progress": saved_progress}))
                append_event(root, "assistant_decision")
                append_event(root, "discovery_recovery")
                for _ in range(4):
                    append_event(root, "assistant_decision")
                agent = EnzymeAgent.__new__(EnzymeAgent)
                agent.run_dir, agent._resume_requested = root, True
                agent._gate_state, agent.model_name = AgentState(root), "test"
                agent._load_resume_payload()
                self.assertEqual(agent._model_turns_since_recovery, 4)


class RollingHarvestTest(unittest.TestCase):
    def test_new_papers_rearm_harvest_but_duplicates_do_not(self):
        with tempfile.TemporaryDirectory() as d:
            state = AgentState(Path(d), accumulated_papers=[{"doi": f"10.test/{i}", "title": f"Study {i}"} for i in range(20)])
            gate = LiteratureHarvestGate(llm_client=SimpleNamespace(budget_exhausted=False))
            with patch("enzyme_discovery_agent.tools.fulltext.seed_mechanism_literature"), \
                 patch("enzyme_discovery_agent.tools.fulltext.mine_and_register_fulltext_accessions", return_value=[]):
                result = gate.execute(state)
            state.literature_harvest_done = result.state_updates["literature_harvest_done"]
            # An unchanged pool still has work until its unread backlog has
            # been attempted. Mock the completed download attempts explicitly.
            from enzyme_discovery_agent.models.literature_reading import save_reading_state
            save_reading_state(state.run_dir, {"downloads": {f"10.test/{i}": {"attempts": 2} for i in range(20)}, "extractions": {}})
            self.assertFalse(gate.should_fire(state))
            state.accumulated_papers.append(dict(state.accumulated_papers[0]))
            self.assertFalse(gate.should_fire(state))
            state.accumulated_papers.append({"doi": "10.test/new", "title": "New experiment"})
            self.assertTrue(gate.should_fire(state))

    def test_quota_interrupted_harvest_leaves_batch_pending(self):
        with tempfile.TemporaryDirectory() as d:
            state = AgentState(Path(d), accumulated_papers=[{"title": f"Study {i}"} for i in range(20)])
            gate = LiteratureHarvestGate(llm_client=SimpleNamespace(budget_exhausted=True))
            with patch("enzyme_discovery_agent.tools.fulltext.seed_mechanism_literature"), \
                 patch("enzyme_discovery_agent.tools.fulltext.mine_and_register_fulltext_accessions", return_value=[]):
                gate.execute(state)
            self.assertFalse((state.run_dir / "agent_state" / "literature_harvest_batch.json").exists())
            self.assertTrue(gate.should_fire(state))


if __name__ == "__main__":
    unittest.main()
