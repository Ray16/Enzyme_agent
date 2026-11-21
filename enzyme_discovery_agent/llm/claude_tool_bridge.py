"""Opt-in building blocks for exposing controller-approved tools over MCP.

This module is not a discovery backend. The controller must supply the allowed
tool definitions and a dispatcher that applies its gates and records results.
It deliberately does not execute registry functions directly.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class ClaudeToolBridge:
    """Translate controller-provided tool definitions and dispatch calls.

    A fresh instance must be made whenever the controller changes its allowed
    tools. Injected arguments are controller-owned and are never exposed here.
    """

    def __init__(self, definitions: Sequence[Mapping[str, Any]],
                 dispatch: Callable[[str, dict], Any]):
        self._dispatch = dispatch
        self._tools: dict[str, dict] = {}
        for definition in definitions:
            function = definition.get("function")
            if definition.get("type") != "function" or not isinstance(function, Mapping):
                raise ValueError("Expected an OpenAI function tool definition")
            name = function.get("name")
            params = function.get("parameters")
            if not isinstance(name, str) or not name or not isinstance(params, Mapping):
                raise ValueError("Tool name and parameters are required")
            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name}")
            self._tools[name] = {"name": name,
                                 "description": function.get("description", ""),
                                 "inputSchema": copy.deepcopy(dict(params))}

    def list_tools(self) -> list[dict]:
        """Return detached MCP tool descriptions for this controller turn."""
        return copy.deepcopy(list(self._tools.values()))

    def call_tool(self, name: str, arguments: Any) -> dict:
        """Invoke only an allowed tool through the supplied controller dispatcher."""
        if name not in self._tools:
            return {"content": [{"type": "text", "text": "Tool is not allowed in this turn."}],
                    "isError": True}
        if not isinstance(arguments, dict):
            return {"content": [{"type": "text", "text": "Tool arguments must be an object."}],
                    "isError": True}
        try:
            result = self._dispatch(name, copy.deepcopy(arguments))
            text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        except Exception:
            # Details can contain credentials or private tool inputs; the
            # dispatcher should record diagnostic details in its own logs.
            return {"content": [{"type": "text", "text": "Tool dispatch failed."}],
                    "isError": True}
        is_error = isinstance(result, dict) and result.get("status") == "error"
        return {"content": [{"type": "text", "text": text}], "isError": is_error}
