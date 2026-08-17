"""LLM client: call with retry, unified token tracking, and response normalization.

This module is the single boundary at which provider-specific quirks are
absorbed.  In particular, some Argo-served models serialize ``tool_use`` dicts
into the assistant's *text content* instead of the structured ``tool_calls``
field; :meth:`LLMClient.extract_tool_calls` recovers those so the agent loop
only ever sees a uniform tool-call shape.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field

try:
    from openai import OpenAI
except ImportError:  # Keep CLI help and offline tools usable without the optional client.
    OpenAI = None  # type: ignore[assignment,misc]

from enzyme_discovery_agent import config

logger = logging.getLogger("enzyme_agent")

# Marker that identifies an assistant message whose tool calls leaked into text.
_TOOL_USE_TEXT_MARKER = "'type': 'tool_use'"
_TOOLU_SPLIT_RE = re.compile(r"\{['\"]id['\"]:\s*['\"]toolu_")


def recover_tool_calls_from_text(text: str) -> list[dict]:
    """Parse ``tool_use`` dicts that a provider serialized into text content.

    Returns a list of OpenAI-style tool-call dicts
    (``{"id", "function": {"name", "arguments"}}``); empty if none are found.
    """
    recovered: list[dict] = []
    parts = re.split(r"(?=\{['\"]id['\"]:\s*['\"]toolu_)", text)
    for part in parts:
        part = part.strip()
        if not part.startswith("{"):
            continue
        try:
            d = ast.literal_eval(part)
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict) and d.get("type") == "tool_use" and "name" in d:
            args = d.get("input", {})
            recovered.append({
                "id": d.get("id", f"recovered_{len(recovered)}"),
                "function": {
                    "name": d["name"],
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })
    return recovered


@dataclass
class TokenUsage:
    """Accumulator for LLM token usage across a run.

    Thread-safe: finalization may record from concurrent LLM calls (e.g. the two
    report prose passes run in parallel), so the accumulation is guarded by a lock.
    """
    model: str = ""
    calls: list[dict] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, label: str | int, response) -> None:
        """Record token usage from an LLM response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        with self._lock:
            self.calls.append({
                "iteration": label,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            })
            self.total_prompt_tokens += pt
            self.total_completion_tokens += ct
            self.total_tokens += pt + ct

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }


class LLMClient:
    """Wrapper around the OpenAI-compatible client with retry logic."""

    def __init__(self, model: str = config.DEFAULT_MODEL):
        if OpenAI is None:
            raise RuntimeError(
                "The OpenAI client is required to run the agent. "
                "Install the project dependencies with `pip install -e .`."
            )
        self.client = OpenAI(
            base_url=config.ARGO_BASE_URL,
            api_key=config.ARGO_API_KEY,
        )
        self.model = model

    @staticmethod
    def extract_tool_calls(assistant_msg) -> tuple[list, str, bool]:
        """Normalize an assistant message into (tool_calls, content, recovered).

        When the provider returns structured ``tool_calls`` this is a passthrough.
        When it instead leaks ``tool_use`` dicts into the text content, those are
        recovered, the leaked text is stripped from ``content``, and
        ``recovered`` is True.  The agent loop is thereby insulated from the
        provider-specific serialization quirk.
        """
        tool_calls = assistant_msg.tool_calls or []
        content = assistant_msg.content or ""
        if tool_calls or _TOOL_USE_TEXT_MARKER not in content:
            return tool_calls, content, False
        recovered = recover_tool_calls_from_text(content)
        if not recovered:
            return tool_calls, content, False
        cleaned = _TOOLU_SPLIT_RE.split(content)[0].strip()
        cleaned = cleaned or f"[Recovered {len(recovered)} tool calls]"
        return recovered, cleaned, True

    def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = config.LLM_TEMPERATURE,
        timeout: int = config.LLM_TIMEOUT,
        max_retries: int = 3,
        max_tokens: int | None = None,
    ):
        """Call the LLM with retry logic. Returns the response or None on failure."""
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if tools:
                    kwargs["tools"] = tools
                # Whole-list ranking calls emit one JSON row per candidate; the
                # provider's default completion cap (~4k tokens) silently
                # truncates them past ~100 candidates, which then fails to parse
                # and drops the ranker to a worse ungated path.  Let callers lift
                # the ceiling for those calls.
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                # Apply the timeout to the HTTP client itself.  A
                # ThreadPoolExecutor timeout is not sufficient here: leaving
                # its context waits for the still-running request thread, so a
                # stalled proxy call can block the whole campaign indefinitely.
                return self.client.with_options(timeout=timeout).chat.completions.create(
                    **kwargs
                )
            except TimeoutError:
                logger.warning("LLM timeout (%ds), attempt %d/%d", timeout, attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                logger.warning("LLM error: %s, attempt %d/%d", e, attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None
