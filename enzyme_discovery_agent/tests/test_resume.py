"""Offline tests for interrupted-run checkpoint restoration."""

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.gates import AgentState
from enzyme_discovery_agent.gates import GateResult


class TestResumeCheckpoint(unittest.TestCase):
    def _agent(self, root: Path) -> EnzymeAgent:
        agent = EnzymeAgent.__new__(EnzymeAgent)
        agent.run_dir = root
        agent.model_name = "test-model"
        agent._resume_requested = True
        agent._gate_state = AgentState(run_dir=root)
        agent._search_queries = []
        agent._search_log = []
        return agent

    def test_restores_messages_iteration_tokens_and_gate_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_state").mkdir()
            (root / "manifest.json").write_text(json.dumps({
                "status": "running", "query": "find enzyme",
            }))
            (root / "target_context.json").write_text(json.dumps({
                "query": "find enzyme", "target_profile": {},
            }))
            messages = [{"role": "user", "content": "find enzyme"}]
            (root / "agent_state" / "transcript.json").write_text(json.dumps({
                "iteration": 4, "messages": messages,
            }))
            events = [
                {"type": "gate", "gate": "literature_harvest"},
                {"type": "gate", "gate": "pre_evaluation"},
                {"type": "llm_call", "iteration": 4, "prompt_tokens": 10,
                 "completion_tokens": 3, "total_tokens": 13},
            ]
            (root / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n"
            )
            agent = self._agent(root)
            payload = agent._load_resume_payload()
            self.assertEqual(payload["messages"], messages)
            self.assertEqual(payload["iteration"], 4)
            self.assertEqual(payload["token_usage"].total_tokens, 13)
            self.assertTrue(agent._gate_state.literature_harvest_done)
            self.assertTrue(agent._gate_state.eval_gate_fired)

    def test_refuses_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_state").mkdir()
            (root / "manifest.json").write_text(json.dumps({"status": "completed"}))
            (root / "agent_state" / "transcript.json").write_text(json.dumps({
                "iteration": 1, "messages": [],
            }))
            with self.assertRaisesRegex(RuntimeError, "cannot be resumed"):
                self._agent(root)._load_resume_payload()

    def test_legacy_failed_run_can_resume_with_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_state").mkdir()
            (root / "manifest.json").write_text(json.dumps({"status": "failed", "query": "target"}))
            (root / "target_context.json").write_text(json.dumps({"query": "target", "target_profile": {}}))
            (root / "agent_state" / "transcript.json").write_text(json.dumps({
                "iteration": 3, "messages": [{"role": "user", "content": "target"}],
            }))
            self.assertEqual(self._agent(root)._load_resume_payload()["iteration"], 3)

    def test_resume_resets_terminal_idle_streak_but_keeps_evidence_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_state").mkdir()
            (root / "manifest.json").write_text(json.dumps({"status": "failed", "query": "target"}))
            (root / "target_context.json").write_text(json.dumps({"query": "target", "target_profile": {}}))
            (root / "agent_state" / "transcript.json").write_text(json.dumps({
                "iteration": 17, "messages": [],
                "research_progress": {"idle_turns": 3, "signature": "saved-evidence"},
            }))
            agent = self._agent(root)
            agent._load_resume_payload()
            self.assertEqual(agent._consecutive_no_progress_turns, 0)
            self.assertEqual(agent._last_research_signature, "saved-evidence")

    def test_resume_keeps_valid_events_before_a_truncated_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_state").mkdir()
            (root / "manifest.json").write_text(json.dumps({
                "status": "running", "query": "find enzyme",
            }))
            (root / "target_context.json").write_text(json.dumps({
                "query": "find enzyme", "target_profile": {},
            }))
            (root / "agent_state" / "transcript.json").write_text(json.dumps({
                "iteration": 2, "messages": [],
            }))
            (root / "events.jsonl").write_text(
                json.dumps({"type": "gate", "gate": "literature_harvest"}) + "\n{"
            )
            agent = self._agent(root)
            payload = agent._load_resume_payload()
            self.assertEqual(payload["iteration"], 2)
            self.assertTrue(agent._gate_state.literature_harvest_done)

    def test_gate_interception_checkpoints_injected_guidance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_state").mkdir()
            agent = self._agent(root)
            agent._log_gate = lambda *args: None
            agent.log = lambda *args, **kwargs: None
            messages = [{"role": "assistant", "content": "discard me"}]
            iteration, intercepted = agent._apply_gate_result(
                "test",
                GateResult(
                    intercept=True,
                    new_messages=[{"role": "user", "content": "gate guidance"}],
                    state_updates={},
                ),
                4,
                messages,
            )
            saved = json.loads(
                (root / "agent_state" / "transcript.json").read_text()
            )
            self.assertTrue(intercepted)
            self.assertEqual(iteration, 3)
            self.assertEqual(saved["iteration"], 3)
            self.assertEqual(saved["messages"][-1]["content"], "gate guidance")
            self.assertEqual(saved["research_progress"]["automatic_evidence_batches"], 1)

    def test_resume_preserves_reflection_due_after_automated_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_state").mkdir()
            (root / "manifest.json").write_text(json.dumps({"status": "budget_exhausted"}))
            (root / "target_context.json").write_text(json.dumps({"target_profile": {}}))
            transcript = {"iteration": 7, "messages": [],
                          "research_progress": {"automatic_evidence_batches": 10}}
            path = root / "agent_state/transcript.json"
            path.write_text(json.dumps(transcript))
            agent = self._agent(root)
            agent._load_resume_payload()
            self.assertFalse(agent._automatic_reflection_due())
            agent._consecutive_no_progress_turns = 2
            self.assertTrue(agent._automatic_reflection_due())
            # Legacy checkpoints recover the same debt from the event stream.
            transcript.pop("research_progress")
            path.write_text(json.dumps(transcript))
            events = [{"type": "gate", "intercept": True}, {"type": "assistant_decision"},
                      *[{"type": "gate", "intercept": True} for _ in range(3)]]
            (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            agent._load_resume_payload()
            self.assertEqual(agent._automatic_evidence_batches, 3)


if __name__ == "__main__":
    unittest.main()


def test_resume_recovers_only_proven_empty_uniprot_searches(tmp_path):
    from enzyme_discovery_agent.bootstrap import register_builtin_tools
    from enzyme_discovery_agent.tool_registry import check_search_duplicate
    register_builtin_tools()
    (tmp_path / 'agent_state').mkdir()
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'running'}))
    (tmp_path / 'target_context.json').write_text(json.dumps({'target_profile': {}}))
    (tmp_path / 'agent_state/transcript.json').write_text(json.dumps({'iteration': 1, 'messages': []}))
    folder = tmp_path / 'artifacts/database'; folder.mkdir(parents=True)
    args = {'gene_name': 'exampleQ', 'function_filter': 'synthetic enzyme', 'max_results': 5}
    (folder / 'search_uniprot_001.json').write_text(json.dumps({'tool_name': 'search_uniprot', 'args': args,
        'result': {'status': 'partial', 'summary': 'Found 0 UniProt entries for query: gene:exampleQ', 'data': [], 'count': 0}}))
    other = {**args, 'gene_name': 'anotherQ'}
    (folder / 'search_uniprot_002.json').write_text(json.dumps({'tool_name': 'search_uniprot', 'args': other,
        'result': {'status': 'error', 'summary': 'Request failed', 'data': [], 'count': 0}}))
    agent = TestResumeCheckpoint()._agent(tmp_path)
    agent._load_resume_payload()
    assert check_search_duplicate('search_uniprot', {**args, 'function_filter': 'synthetic enzyme extra'}, history=agent._search_history)
    assert check_search_duplicate('search_uniprot', other, history=agent._search_history) is None
