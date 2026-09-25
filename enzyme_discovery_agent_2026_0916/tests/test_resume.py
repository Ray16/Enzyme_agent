"""Offline tests for interrupted-run checkpoint restoration."""

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.gates import AgentState


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


if __name__ == "__main__":
    unittest.main()
