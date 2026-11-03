"""User control survives compaction, pause and an explicit controller restart."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from enzyme_discovery_agent import steering
from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.context import build_model_context


class SteeringTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "manifest.json").write_text('{}')

    def controller(self):
        agent = object.__new__(EnzymeAgent)
        agent.run_dir = self.root
        agent.llm = Mock()
        agent.log = Mock()
        agent._log_event = Mock()
        agent._checkpoint_transcript = Mock()
        return agent

    def test_concurrent_writers_do_not_lose_commands_and_consumption_is_idempotent(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            submitted = list(pool.map(lambda i: steering.submit(self.root, "instruction", f"priority {i}"), range(20)))
        state, accepted = steering.consume(self.root)
        self.assertEqual(len(accepted), 20)
        self.assertEqual(len(state["instructions"]), 20)
        self.assertFalse(steering.has_pending(self.root))
        self.assertEqual(steering.consume(self.root)[1], [])
        ids = [x["id"] for x in state["instructions"]]
        steering.mark_delivered(self.root, ids, 12)
        self.assertEqual(steering.status(self.root)["receipts"][submitted[0]["id"]]["status"], "delivered_to_model")

    def test_active_instructions_survive_model_context_compaction_and_can_be_cleared(self):
        steering.submit(self.root, "instruction", "Prioritize missing experimental evidence before another expansion.")
        steering.consume(self.root)
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "target"}]
        messages += [{"role": "assistant", "content": "old narrative" * 2000}] * 30
        with patch("enzyme_discovery_agent.context.build_research_state", return_value={}):
            projected = build_model_context(messages, self.root)
            self.assertIn("Prioritize missing experimental", json.dumps(projected))
            steering.submit(self.root, "clear_instructions")
            steering.consume(self.root)
            self.assertNotIn("Prioritize missing experimental", json.dumps(build_model_context(messages, self.root)))

    def test_pause_checkpoints_then_waits_without_a_provider_call_and_resume_preserves_guidance(self):
        instruction = steering.submit(self.root, "instruction", "Inspect the unresolved family evidence.")
        steering.submit(self.root, "pause")
        agent = self.controller()
        def resume(seconds):
            self.assertTrue(steering.read_state(self.root)["paused"])
            self.assertEqual(steering.read_state(self.root)["controller_status"], "paused")
            agent.llm.call.assert_not_called()
            steering.submit(self.root, "resume")
        with patch("enzyme_discovery_agent.agent.RunFinalizer") as finalizer, \
             patch("enzyme_discovery_agent.agent._time.sleep", side_effect=resume):
            self.assertTrue(agent._service_steering([], 7))
            finalizer.return_value.snapshot.assert_called_once()
        self.assertFalse(steering.read_state(self.root)["paused"])
        self.assertEqual(steering.read_state(self.root)["snapshot_completed"], 1)
        self.assertEqual(steering.read_state(self.root)["receipts"][instruction["id"]]["status"], "accepted")
        # A restart before the model has seen the instruction must schedule reflection again.
        self.assertTrue(self.controller()._service_steering([], 7))

    def test_checkpoint_does_not_finish_campaign_or_drop_pending_evaluations(self):
        evidence = self.root / "agent_state" / "pending.json"
        evidence.parent.mkdir()
        evidence.write_text('{"candidate": "screened"}')
        steering.submit(self.root, "checkpoint")
        agent = self.controller()
        with patch("enzyme_discovery_agent.agent.RunFinalizer") as finalizer:
            self.assertFalse(agent._service_steering([], 3))
            self.assertFalse(agent._service_steering([], 3))
            finalizer.return_value.snapshot.assert_called_once()
        self.assertEqual(evidence.read_text(), '{"candidate": "screened"}')
        self.assertEqual((self.root / "manifest.json").read_text(), '{}')

    def test_new_control_defers_an_action_before_tool_dispatch(self):
        steering.submit(self.root, "pause")
        with patch("enzyme_discovery_agent.agent.execute_tool") as tool:
            result = self.controller()._execute_single_tool("example", {}, 1)
        self.assertTrue(result["deferred_for_user_steering"])
        tool.assert_not_called()

    def test_invalid_queue_command_is_rejected_without_losing_valid_commands(self):
        commands, _ = steering._paths(self.root)
        commands.mkdir(parents=True)
        (commands / "invalid.json").write_text('{"id": "different", "action": "pause"}')
        good = steering.submit(self.root, "instruction", "Follow the newly supplied evidence.")
        state, accepted = steering.consume(self.root)
        self.assertEqual(state["receipts"]["invalid"]["status"], "rejected")
        self.assertEqual(accepted[0]["id"], good["id"])

    def test_live_loop_sends_steering_before_an_automatic_gate_and_records_delivery(self):
        class Message:
            content = "continue"
            tool_calls = None

            def model_dump(self):
                return {"role": "assistant", "content": self.content}

        provider = Mock()
        provider.with_options.return_value = provider
        provider.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=Message())],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        with patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                 {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]), \
             patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}):
            agent = EnzymeAgent(model="test", output_dir=self.root, token_budget=10)
            # The run writes its manifest during setup, before its first boundary.
            (agent.run_dir / "manifest.json").write_text('{}')
            command = steering.submit(agent.run_dir, "instruction", "Resolve uncertainty before more expansion.")
            agent._literature_harvest_gate.should_fire = Mock(return_value=True)
            agent._literature_harvest_gate.execute = Mock(side_effect=AssertionError("Gate preempted user steering"))
            agent.run("target", substrate_smiles="CCO")
        sent = provider.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("Resolve uncertainty before more expansion.", json.dumps(sent))
        self.assertEqual(steering.status(agent.run_dir)["receipts"][command["id"]]["status"], "delivered_to_model")
        agent._literature_harvest_gate.execute.assert_not_called()
