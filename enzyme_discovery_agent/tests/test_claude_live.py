"""Explicitly opt-in first-party integration test; consumes subscription usage."""
import json
import os
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_CLAUDE_SUBSCRIPTION_LIVE") != "1",
                                reason="requires explicit subscription usage")


def test_controller_gate_tool_checkpoint_and_resume_model(tmp_path):
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.gates import GateResult
    from enzyme_discovery_agent.run_layout import run_file
    from enzyme_discovery_agent.tool_registry import get_tool_info
    from enzyme_discovery_agent.tools import substrate  # Registers the genuine tool.

    schema = get_tool_info("analyze_substrate")["schema"]
    with patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
         patch("enzyme_discovery_agent.agent.get_all_tools", return_value=[schema]), \
         patch("enzyme_discovery_agent.agent.build_system_prompt", return_value="Use the available tool exactly as requested."), \
         patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}), \
         patch("enzyme_discovery_agent.models.recovery.next_recovery_action", return_value=None):
        agent = EnzymeAgent(model="claude-code:claude-sonnet-5", output_dir=tmp_path,
                            token_budget=30000, llm_timeout=90)
        gate = Mock()
        gate.should_fire.return_value = True
        gate.execute.side_effect = [GateResult(intercept=True), GateResult(intercept=False)]
        agent._early_gates = [("test_gate", gate)]
        for item in (agent._literature_harvest_gate, agent._mechanism_family_gate,
                     agent._family_survey_gate, agent._gate_pipeline.gates[2]):
            item.should_fire = Mock(return_value=False)
        actual_execute = agent._execute_single_tool
        def execute(name, arguments, iteration):
            result = actual_execute(name, arguments, iteration)
            assert result["status"] == "success"
            agent.token_budget = agent._run_token_usage.total_tokens - agent._budget_start_tokens
            return result
        with patch.object(agent, "_execute_single_tool", side_effect=execute) as performed:
            agent.run("Call analyze_substrate with smiles CCO and name sample.", substrate_smiles="CCO")
        assert gate.execute.call_count == 2
        performed.assert_called_once_with("analyze_substrate", {"smiles": "CCO", "name": "sample"}, 1)
        checkpoint = json.loads((agent.run_dir / "agent_state" / "budget_checkpoint.json").read_text())
        assert "allowance" in checkpoint["reason"]
        usage = json.loads(run_file(agent.run_dir, "token_usage.json").read_text())
        assert usage["accounting_complete"] and len(usage["calls"]) == 2
        assert all(c["backend"] == "claude-code" for c in usage["calls"])
        events = [json.loads(line) for line in run_file(agent.run_dir, "events.jsonl").read_text().splitlines()]
        assert any(e.get("event") == "tool_call" or e.get("type") == "tool_call" for e in events)
        resumed = EnzymeAgent(resume_run_dir=agent.run_dir)
        assert resumed.model_name == "claude-code:claude-sonnet-5"
        resumed._close_logging()
