"""Subscription transport and controller invariants using synthetic tools only."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from enzyme_discovery_agent.claude_subscription import (
    ClaudeTransportError, Response, Choice, Message, Usage, _parse_output, call_claude,
)
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage


def stream(*blocks, subtype="error_max_turns", provider="firstParty", usage=True):
    result = {"type": "result", "subtype": subtype, "is_error": subtype != "success",
              "modelUsage": {"claude-sonnet-5": {"provider": provider}}}
    if usage:
        result["usage"] = {"input_tokens": 3, "cache_creation_input_tokens": 4,
                           "cache_read_input_tokens": 5, "output_tokens": 6}
    return "\n".join(map(json.dumps, [
        {"type": "assistant", "message": {"content": list(blocks)}}, result]))


def tool(name="mcp__discovery__probe", ident="call1"):
    return {"type": "tool_use", "name": name, "id": ident, "input": {"query": "x"}}


def test_native_tool_selection_and_all_input_tokens_counted():
    response = _parse_output(stream(tool()), {"probe"}, "claude-sonnet-5")
    call = response.choices[0].message.tool_calls[0]
    assert call["function"]["name"] == "probe"
    assert json.loads(call["function"]["arguments"]) == {"query": "x"}
    # The prompt breakdown is retained: a cache read does not cost what fresh input costs.
    assert response.usage == Usage(12, 6, input_tokens=3, cache_creation_tokens=4, cache_read_tokens=5)


def test_unqualified_tool_name_requires_exact_controller_allowlist():
    response = _parse_output(stream(tool("probe")), {"probe"}, "claude-sonnet-5")
    assert response.choices[0].message.tool_calls[0]["function"]["name"] == "probe"
    with pytest.raises(ClaudeTransportError, match="unauthorized_name"):
        _parse_output(stream(tool("other")), {"probe"}, "claude-sonnet-5")


def test_guarded_controller_can_receive_an_unoffered_proposal_without_executing_it():
    response=_parse_output(stream(tool('get_decision_context')),{'conclude_research_review'},
                           'claude-sonnet-5',defer_tool_rejections=True)
    assert response.choices[0].message.tool_calls[0]['function']['name']=='get_decision_context'
    assert response.usage==Usage(12, 6, input_tokens=3, cache_creation_tokens=4, cache_read_tokens=5)


def test_text_response():
    response = _parse_output(stream({"type": "text", "text": "done"}, subtype="success"), set(), "claude-sonnet-5")
    assert response.choices[0].message.content == "done"
    assert response.choices[0].message.tool_calls is None


@pytest.mark.parametrize("payload", [
    stream(tool("mcp__discovery__forbidden")),
    stream(tool(None)),
    stream(tool(), tool()),
    stream(tool(), subtype="success"),
    stream(tool(), provider="other"),
    stream(tool(), usage=False),
])
def test_fail_closed_on_invalid_or_unaccountable_response(payload):
    with pytest.raises(ClaudeTransportError):
        _parse_output(payload, {"probe"}, "claude-sonnet-5")


def test_provider_failure_stops_without_retry_and_records_known_usage(tmp_path):
    client = LLMClient("claude-code:claude-sonnet-5")
    usage = TokenUsage()
    client.attach_usage(usage, tmp_path)
    with patch("enzyme_discovery_agent.claude_subscription.call_claude", side_effect=ClaudeTransportError("failed", Usage(10, 3))) as call:
        assert client.call([{"role": "user", "content": "hello"}]) is None
    assert call.call_count == 1
    assert usage.total_tokens == 13
    assert client.budget_exhausted
    assert client.resource_stop_reason
    assert json.loads((tmp_path / "token_usage.json").read_text())["accounting_complete"]


def test_unknown_usage_is_marked_incomplete(tmp_path):
    client = LLMClient("claude-code:claude-sonnet-5")
    usage = TokenUsage()
    client.attach_usage(usage, tmp_path)
    with patch("enzyme_discovery_agent.claude_subscription.call_claude", side_effect=ClaudeTransportError("timeout")):
        assert client.call([]) is None
    assert usage.unmetered_calls == 1
    assert not usage.to_dict()["accounting_complete"]
    assert client.resource_stop_reason


def test_rejects_invalid_image_inputs_without_silently_ignoring_them():
    with pytest.raises(ClaudeTransportError, match="Image must"):
        call_claude([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}], [], "claude-sonnet-5", 10, None)


def test_controller_gate_and_budget_checkpoint(tmp_path):
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.gates import GateResult
    from enzyme_discovery_agent.run_layout import run_file
    schema = {"type": "function", "function": {"name": "search_pubmed", "description": "synthetic",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    def model(*args, **kwargs):
        return Response([Choice(Message(None, [{"id": "synthetic", "type": "function", "function": {
            "name": "search_pubmed", "arguments": '{"query": "synthetic"}'}}]), "tool_calls")], Usage(7, 3))
    with patch("enzyme_discovery_agent.agent.register_builtin_tools"), \
         patch("enzyme_discovery_agent.agent.get_all_tools", return_value=[schema]), \
         patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}), \
         patch("enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status", return_value={'required':False}), \
         patch("enzyme_discovery_agent.models.recovery.next_recovery_action", return_value=None), \
         patch("enzyme_discovery_agent.claude_subscription.call_claude", side_effect=model):
        agent = EnzymeAgent(model="claude-code:claude-sonnet-5", output_dir=tmp_path, token_budget=100)
        gate = Mock()
        gate.should_fire.return_value = True
        gate.execute.side_effect = [GateResult(intercept=True), GateResult(intercept=False)]
        agent._early_gates = [("test_gate", gate)]
        def execute(*args):
            agent.token_budget = 20
            return {"status": "success", "data": []}
        with patch.object(agent, "_execute_single_tool", side_effect=execute) as called:
            agent.run("synthetic task", substrate_smiles="CCO")
        assert gate.execute.call_count == 2
        called.assert_called_once_with("search_pubmed", {"query": "synthetic"}, 1)
        checkpoint = json.loads((agent.run_dir / "agent_state" / "budget_checkpoint.json").read_text())
        assert "allowance" in checkpoint["reason"]
        assert checkpoint["stop_cause"] == "token_budget"
        assert checkpoint["token_overrun"] == 0
        assert checkpoint["token_accounting_complete"] is True
        ledger = json.loads(run_file(agent.run_dir, "token_usage.json").read_text())
        assert ledger["total_tokens"] == 20
        assert ledger["accounting_complete"]
        assert all(item["backend"] == "claude-code" for item in ledger["calls"])
        resumed = EnzymeAgent(resume_run_dir=agent.run_dir, token_budget=10)
        assert resumed.model_name == "claude-code:claude-sonnet-5"
        resumed._early_gates = []
        for foundation in (resumed._literature_harvest_gate, resumed._mechanism_family_gate,
                           resumed._family_survey_gate, resumed._gate_pipeline.gates[2]):
            foundation.should_fire = Mock(return_value=False)
        with patch.object(resumed, "_execute_single_tool") as unexpected:
            resumed.run("ignored on resume")
        unexpected.assert_not_called()  # The response consumed the resumed allowance.
        ledger = json.loads(run_file(agent.run_dir, "token_usage.json").read_text())
        assert ledger["total_tokens"] == 30
        assert ledger["accounting_complete"]


def test_image_is_transmitted_as_native_content_and_environment_is_isolated(tmp_path, monkeypatch):
    import base64
    from types import SimpleNamespace
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    monkeypatch.setenv("ENZYME_CLAUDE_SETTINGS", str(settings))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-not-to-forward")
    image = base64.b64encode(b"test-image-bytes").decode()
    outputs = stream({"type": "text", "text": "ok"}, subtype="success")
    auth = SimpleNamespace(stdout=json.dumps({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}), returncode=0)
    with patch("enzyme_discovery_agent.claude_subscription.subprocess.run", side_effect=[auth, SimpleNamespace(stdout=outputs, returncode=0)]) as run:
        response = call_claude([{"role": "user", "content": [
            {"type": "text", "text": "inspect"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]}], None, "claude-sonnet-5", 30, 800)
    assert response.choices[0].message.content == "ok"
    cmd = run.call_args.args[0]
    assert "--input-format" in cmd and "--restricted" in cmd
    assert "ANTHROPIC_API_KEY" not in run.call_args.kwargs["env"]
    assert run.call_args.kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "25376"
    content = json.loads(run.call_args.kwargs["input"])["message"]["content"]
    assert content[-1]["source"]["data"] == image
    assert image not in content[0]["text"]


def test_rejects_settings_that_would_route_to_proxy(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://example.invalid"}}))
    monkeypatch.setenv("ENZYME_CLAUDE_SETTINGS", str(settings))
    with patch("enzyme_discovery_agent.claude_subscription.subprocess.run") as run:
        with pytest.raises(ClaudeTransportError, match="authentication override") as exc:
            call_claude([], [], "claude-sonnet-5", 20, None)
    assert not exc.value.attempted
    run.assert_not_called()


def test_rejects_api_key_login_before_any_model_request(tmp_path, monkeypatch):
    from types import SimpleNamespace
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    monkeypatch.setenv("ENZYME_CLAUDE_SETTINGS", str(settings))
    auth = SimpleNamespace(stdout=json.dumps({"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty"}), returncode=0)
    with patch("enzyme_discovery_agent.claude_subscription.subprocess.run", return_value=auth) as run:
        with pytest.raises(ClaudeTransportError, match="authentication") as exc:
            call_claude([], [], "claude-sonnet-5", 20, None)
    assert not exc.value.attempted
    assert run.call_count == 1


def test_provider_stop_is_distinguished_from_exhausted_token_budget(tmp_path):
    from enzyme_discovery_agent.agent import EnzymeAgent
    with patch('enzyme_discovery_agent.agent.register_builtin_tools'), \
         patch('enzyme_discovery_agent.agent.get_all_tools', return_value=[]), \
         patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning', return_value={}), \
         patch('enzyme_discovery_agent.models.recovery.next_recovery_action', return_value=None), \
         patch('enzyme_discovery_agent.claude_subscription.call_claude', side_effect=ClaudeTransportError('provider error', Usage(4, 2))):
        agent = EnzymeAgent(model='claude-code:claude-sonnet-5', output_dir=tmp_path, token_budget=100)
        for gate in (agent._literature_harvest_gate, agent._mechanism_family_gate,
                     agent._family_survey_gate, agent._gate_pipeline.gates[2]):
            gate.should_fire = Mock(return_value=False)
        agent.run('synthetic request', substrate_smiles='CCO')
        checkpoint = json.loads((agent.run_dir / 'agent_state/budget_checkpoint.json').read_text())
        assert checkpoint['stop_cause'] == 'provider_unavailable'
        assert checkpoint['tokens_this_invocation'] == 6
        assert checkpoint['token_overrun'] == 0
        assert checkpoint['token_accounting_complete'] is True


def test_token_ledger_separates_cache_reads_from_fresh_input():
    """A cache read bills at a fraction of fresh input; collapsing them hides real spend."""
    from types import SimpleNamespace
    from enzyme_discovery_agent.llm_client import TokenUsage
    usage = TokenUsage()
    for _ in range(3):
        usage.record("main", SimpleNamespace(usage=Usage(
            12, 6, input_tokens=3, cache_creation_tokens=4, cache_read_tokens=5)))
    payload = usage.to_dict()
    assert payload["total_prompt_tokens"] == 36
    assert payload["total_input_tokens"] == 9
    assert payload["total_cache_creation_tokens"] == 12
    assert payload["total_cache_read_tokens"] == 15
    assert payload["cache_read_fraction_of_prompt"] == round(15 / 36, 4)
    assert payload["calls"][0]["cache_read_tokens"] == 5


def test_a_backend_without_a_prompt_breakdown_is_unchanged():
    from types import SimpleNamespace
    from enzyme_discovery_agent.llm_client import TokenUsage
    usage = TokenUsage()
    usage.record("main", SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2)))
    payload = usage.to_dict()
    assert payload["total_tokens"] == 12
    assert "total_cache_read_tokens" not in payload
    assert "cache_read_fraction_of_prompt" not in payload
