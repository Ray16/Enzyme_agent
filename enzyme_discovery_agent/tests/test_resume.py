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


if __name__ == "__main__":
    unittest.main()
