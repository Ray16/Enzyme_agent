import json

import pytest

from enzyme_discovery_agent.claude_tool_bridge import ClaudeToolBridge


DEFINITION = {"type": "function", "function": {
    "name": "example", "description": "Example tool",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
}}


def test_lists_detached_schemas_and_dispatches():
    seen = []
    bridge = ClaudeToolBridge([DEFINITION], lambda name, args: seen.append((name, args)) or {"status": "success", "data": 3})
    listed = bridge.list_tools()
    listed[0]["inputSchema"]["properties"].clear()
    assert "query" in bridge.list_tools()[0]["inputSchema"]["properties"]
    output = bridge.call_tool("example", {"query": "test"})
    assert seen == [("example", {"query": "test"})]
    assert json.loads(output["content"][0]["text"])["data"] == 3
    assert not output["isError"]


def test_disallowed_calls_and_invalid_arguments_do_not_dispatch():
    calls = []
    bridge = ClaudeToolBridge([DEFINITION], lambda *args: calls.append(args))
    assert bridge.call_tool("other", {})["isError"]
    assert bridge.call_tool("example", [])["isError"]
    assert calls == []


def test_dispatch_errors_are_not_disclosed():
    def fail(*_):
        raise RuntimeError("sensitive diagnostic")
    output = ClaudeToolBridge([DEFINITION], fail).call_tool("example", {})
    assert output["isError"]
    assert "sensitive" not in json.dumps(output)


def test_tool_error_is_propagated():
    output = ClaudeToolBridge([DEFINITION], lambda *_: {"status": "error"}).call_tool("example", {})
    assert output["isError"]


def test_duplicate_tools_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        ClaudeToolBridge([DEFINITION, DEFINITION], lambda *_: None)
