"""Guard against dormant discovery gates and empty-notebook polling loops."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.gates.evidence_selection import EvidenceSelectionGate
from enzyme_discovery_agent.gates.literature_harvest import LiteratureHarvestGate
from enzyme_discovery_agent.gates.mechanism_family_coverage import MechanismFamilyGate
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.models.event_log import append_event
from enzyme_discovery_agent.models.progress import research_signature
from enzyme_discovery_agent.models.modeling_handoff import characterize_smiles, is_execution_ready


class TestResearchProgress(unittest.TestCase):
    def test_failed_review_defers_uncertainty_and_releases_ready_evaluation(self):
        with tempfile.TemporaryDirectory() as d, \
             patch('enzyme_discovery_agent.agent.register_builtin_tools'), \
             patch('enzyme_discovery_agent.agent.get_all_tools',return_value=[]), \
             patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}), \
             patch('enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status',return_value={'required':False}), \
             patch('enzyme_discovery_agent.models.research_decisions.review_due',return_value=True):
            agent=EnzymeAgent(output_dir=d,token_budget=100)
            update_candidate_registry(agent.run_dir,'SYNTH_READY','screened',{'protein_name':'Synthetic catalyst'})
            agent._invalid_research_decisions=2
            def evaluate(*args,**kwargs):
                agent.llm.resource_stop_reason='Synthetic queue release verified'
                return GateResult(intercept=True)
            with patch.object(agent._gate_pipeline.gates[2],'execute',side_effect=evaluate) as performed, \
                 patch.object(agent.llm,'call',side_effect=AssertionError('Failed review starved the ready queue')), \
                 patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',0)):
                agent.run('synthetic transformation',substrate_smiles='CCO')
            performed.assert_called_once()
            state=json.loads((agent.run_dir/'agent_state/research_decisions/state.json').read_text())
            assert state['grounding_deferrals']==1 and not state['awaiting_interpretation']

    def test_ready_evaluations_preempt_more_literature_after_a_bounded_review(self):
        with tempfile.TemporaryDirectory() as d, \
             patch('enzyme_discovery_agent.agent.register_builtin_tools'), \
             patch('enzyme_discovery_agent.agent.get_all_tools',return_value=[]), \
             patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}), \
             patch('enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status',return_value={'required':False}), \
             patch('enzyme_discovery_agent.models.research_decisions.review_due',return_value=False):
            agent=EnzymeAgent(output_dir=d,token_budget=100)
            update_candidate_registry(agent.run_dir,'SYNTH_READY','screened',{'protein_name':'Synthetic catalyst'})
            agent._literature_harvest_gate.should_fire=Mock(side_effect=AssertionError('Literature blocked ready evaluation'))
            def evaluate(*args,**kwargs):
                update_candidate_registry(agent.run_dir,'SYNTH_READY','evaluated',{'claim_validation':{'promotion_eligible':True}})
                agent.llm.resource_stop_reason='Synthetic rolling evaluation verified'
                return GateResult(intercept=True,state_updates={'eval_gate_fired':True})
            ready=agent._gate_pipeline.gates[2]
            with patch.object(ready,'execute',side_effect=evaluate) as performed, \
                 patch.object(agent.llm,'call',side_effect=AssertionError('Model work preceded the ready queue')), \
                 patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',1)):
                agent.run('synthetic transformation',substrate_smiles='CCO')
            performed.assert_called_once()
            agent._literature_harvest_gate.should_fire.assert_not_called()

    def test_automated_batches_yield_to_a_model_action_without_discarding_it(self):
        class Message:
            content = None
            tool_calls = [SimpleNamespace(id="action", function=SimpleNamespace(
                name="search_pubmed", arguments=json.dumps({"query": "new evidence route"})))]

            def model_dump(self):
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "action", "type": "function", "function": {
                        "name": "search_pubmed", "arguments": json.dumps({"query": "new evidence route"})}}]}

        provider = Mock()
        provider.with_options.return_value = provider
        provider.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=Message())],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        schemas = [{"type": "function", "function": {"name": name, "parameters": {}}}
                   for name in ("search_pubmed", "get_notebook", "update_notebook")]
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
             patch("enzyme_discovery_agent.agent.get_all_tools", return_value=schemas), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                 {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]), \
             patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}), \
             patch("enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status", return_value={'required':False}), \
             patch("enzyme_discovery_agent.config.MAX_AUTOMATED_BATCHES_BEFORE_REFLECTION", 3), \
             patch("enzyme_discovery_agent.models.recovery.next_recovery_action", return_value=None):
            agent = EnzymeAgent(model="test", output_dir=d, token_budget=20)
            gate = Mock()
            gate.should_fire.return_value = True
            gate.execute.side_effect = [GateResult(intercept=True)] * 3 + [AssertionError("gate starved model")]
            agent._literature_harvest_gate = gate
            agent._early_gates = [("literature_harvest", gate)]
            def take_action(*args):
                agent.token_budget = 10  # Stop after the chosen action has actually executed.
                return {"status": "success", "data": []}
            with patch.object(agent, "_execute_single_tool", side_effect=take_action) as execute:
                agent.run("target", substrate_smiles="CCO")
            self.assertEqual(gate.execute.call_count, 3)
            execute.assert_called_once_with("search_pubmed", {"query": "new evidence route"}, 1)
            # The schema list is stable across turns for cache reuse; the per-turn
            # restriction is now carried by the permitted directive and the allowlist
            # that gates execution, not by withholding schemas.
            offered = provider.chat.completions.create.call_args.kwargs["tools"]
            self.assertIn("search_pubmed", [t["function"]["name"] for t in offered])
            self.assertEqual(agent._offered_direct_tools, {"search_pubmed"})
            sent = provider.chat.completions.create.call_args.kwargs["messages"]
            self.assertIn("PERMITTED THIS TURN: search_pubmed", sent[-1]["content"])
            self.assertEqual(agent._automatic_evidence_batches, 0)

    def test_notebook_write_becomes_available_after_initial_discovery_turns(self):
        class Message:
            content = None

            def __init__(self, name, args):
                self.tool_calls = [SimpleNamespace(id="action", function=SimpleNamespace(
                    name=name, arguments=json.dumps(args)))]

            def model_dump(self):
                function = self.tool_calls[0].function
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "action", "type": "function",
                    "function": {"name": function.name, "arguments": function.arguments},
                }]}

        provider = Mock()
        provider.with_options.return_value = provider
        turn = [0]
        def response(**kwargs):
            turn[0] += 1
            message = (Message("search_pubmed", {"query": f"study {turn[0]}"}) if turn[0] <= 3
                       else Message("update_notebook", {"section": "decisions", "content": "Evidence summary"}))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                                   usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        provider.chat.completions.create.side_effect = response
        schemas = [{"type": "function", "function": {"name": name, "parameters": {}}}
                   for name in ("search_pubmed", "update_notebook")]
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
             patch("enzyme_discovery_agent.agent.get_all_tools", return_value=schemas), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                 {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]), \
             patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}):
            agent = EnzymeAgent(model="test", output_dir=d, token_budget=50)
            performed = []
            def execute(name, args, iteration):
                performed.append(name)
                if name == "search_pubmed":
                    agent._accumulated_papers.append({"title": args["query"]})
                    append_event(agent.run_dir,'tool_call',tool=name,args=args,status='success',count=1)
                return {"status": "success", "data": {}}
            with patch.object(agent, "_execute_single_tool", side_effect=execute):
                agent.run("target", substrate_smiles="CCO")
            self.assertEqual(performed[-1], "update_notebook")
            self.assertEqual(agent._notebook_tool_cooldown, 0)

    def test_repeated_notebook_requests_checkpoint_without_burning_the_quota(self):
        class Message:
            content = None
            tool_calls = [SimpleNamespace(id="read", function=SimpleNamespace(
                name="get_notebook", arguments="{}"))]

            def model_dump(self):
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "read", "type": "function",
                    "function": {"name": "get_notebook", "arguments": "{}"},
                }]}

        provider = Mock()
        provider.with_options.return_value = provider
        provider.chat.completions.create.side_effect = lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=Message())],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        schema = {"type": "function", "function": {"name": "get_notebook", "parameters": {}}}
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
             patch("enzyme_discovery_agent.agent.get_all_tools", return_value=[schema]), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                 {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]), \
             patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}):
            agent = EnzymeAgent(model="test", output_dir=d)
            with patch.object(agent, "_execute_single_tool", return_value={
                "status": "success", "summary": "Notebook retrieved (0 total chars)", "data": {},
            }) as execute:
                text = agent.run("target", substrate_smiles="CCO")
            self.assertEqual(provider.chat.completions.create.call_count, 3)
            self.assertEqual(execute.call_count, 2)  # hidden tool request is rejected on the third turn
            self.assertIn("Discovery stalled", text)
            self.assertEqual(json.loads((agent.run_dir / "manifest.json").read_text())["status"], "discovery_incomplete")
            self.assertTrue((agent.run_dir / "agent_state" / "progress_checkpoint.json").exists())

    def test_notebook_activity_and_timestamps_are_not_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P12345", "evaluated", {"protein_name": "Example enzyme"})
            before = research_signature(root, [])
            update_candidate_registry(root, "P12345", None, {"source_tool": "read", "last_updated": "later"})
            (root / "notebook.json").write_text(json.dumps({"sections": {"decisions": "draft reasoning"}}))
            append_event(root, "tool_call", tool="get_notebook", args={}, status="success")
            self.assertEqual(research_signature(root, []), before)

    def test_evaluation_or_new_papers_reset_stagnation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            agent = EnzymeAgent.__new__(EnzymeAgent)
            agent.run_dir = root
            agent._accumulated_papers = []
            agent._last_research_signature = None
            agent._consecutive_no_progress_turns = 0
            agent._record_research_progress()
            for _ in range(3):
                agent._consecutive_no_tool_turns = 0  # a read-only call must not reset progress
                agent._record_research_progress(count_turn=True)
            self.assertFalse(agent._should_continue(8))
            update_candidate_registry(root, "P12345", "evaluated", {"protein_name": "Example enzyme"})
            self.assertTrue(agent._record_research_progress())
            self.assertEqual(agent._consecutive_no_progress_turns, 0)
            agent._accumulated_papers = [{"title": "Experimental reaction study", "doi": "10.test/study"}]
            self.assertTrue(agent._record_research_progress())
            agent._accumulated_papers.append(dict(agent._accumulated_papers[0]))
            self.assertFalse(agent._record_research_progress())

    def test_weak_mechanism_rewrites_are_not_new_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P12345", "evaluated", {"protein_name": "Synthetic catalyst"})
            before = research_signature(root, [])
            for status in ("low_confidence", "partial", "error"):
                update_candidate_registry(root, "P12345", None, {"mechanism_evidence": {
                    "status": status, "facts": {"chemical_step": status}, "papers_tried": [status]}})
                self.assertEqual(research_signature(root, []), before)
            update_candidate_registry(root, "P12345", None, {"mechanism_evidence": {
                "status": "ok", "paper": {"doi": "10.test/source"},
                "facts": {"chemical_step": "source-supported step"}}})
            self.assertNotEqual(research_signature(root, []), before)

    def test_novel_negative_search_counts_but_repeat_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            before = research_signature(root, [])
            append_event(root, "tool_call", tool="search_pubmed", args={"query": "example reaction"}, status="empty")
            searched = research_signature(root, [])
            self.assertNotEqual(searched, before)
            append_event(root, "tool_call", tool="search_pubmed", args={"query": "example reaction"}, status="empty")
            self.assertEqual(research_signature(root, []), searched)


class TestEvidenceReadiness(unittest.TestCase):
    def test_mechanism_derivation_uses_chemistry_not_search_round_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "target_reasoning.json").write_text(json.dumps({"bond_types": ["ester"]}))
            state = AgentState(root, lit_search_count=0)
            self.assertTrue(MechanismFamilyGate(llm_client=object()).should_fire(state))
            state.mechanism_families_done = True
            self.assertFalse(MechanismFamilyGate(llm_client=object()).should_fire(state))

    def test_harvest_and_registration_can_fire_after_two_search_rounds(self):
        with tempfile.TemporaryDirectory() as d:
            state = AgentState(Path(d), lit_search_count=2,
                               accumulated_papers=[{"title": f"Study {i}"} for i in range(20)])
            self.assertTrue(LiteratureHarvestGate(llm_client=object()).should_fire(state))
            self.assertTrue(EvidenceSelectionGate().should_fire(state))

    def test_parsable_structure_must_contain_its_declared_reactive_group(self):
        def specification(smiles):
            return {"status": "ready", "target_substrates": [{
                "modeling_smiles": smiles, "modeling_representation": characterize_smiles(smiles),
                "reactive_atom": "carbonyl carbon",
            }]}
        ready, errors = is_execution_ready(specification("CCO"))
        self.assertFalse(ready)
        self.assertIn("modeling representation lacks its declared carbonyl reactive group", errors)
        self.assertTrue(is_execution_ready(specification("CC(=O)O"))[0])


if __name__ == "__main__":
    unittest.main()
