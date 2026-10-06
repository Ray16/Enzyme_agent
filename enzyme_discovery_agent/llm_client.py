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
import threading
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

# Some providers (e.g. argo past its quota) return a block/limit NOTICE as a
# normal HTTP 200 whose text IS the notice, not model output.  Detect it so the
# client fails over to the next backend instead of treating the notice as an answer.
_PROVIDER_BLOCK_RE = re.compile(
    r"ACCESS REVOKED|USAGE NOTICE FROM ARGO|usage limit has been exceeded", re.I)


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


@dataclass
class _Backend:
    """One LLM endpoint in the failover chain."""
    name: str
    client: object
    model: str


class LLMClient:
    """OpenAI-compatible client with transient retries AND availability failover.

    Backends are tried in priority order (``config.llm_backends()``); the client is
    STICKY per run -- once a backend serves a call it stays active, and the client
    only advances to the next backend when the active one is unavailable (exhausted
    transient retries, an auth failure, or a provider block/limit notice returned as
    a 200).  That keeps one model per run while surviving a primary-backend outage.
    """

    def __init__(self, model: str = config.DEFAULT_MODEL):
        if OpenAI is None:
            raise RuntimeError(
                "The OpenAI client is required to run the agent. "
                "Install the project dependencies with `pip install -e .`."
            )
        # Ordered backends for availability failover (first = primary).  The explicit
        # `model` arg overrides the PRIMARY backend's model so `--model` still works;
        # fallback backends keep their own configured model.
        backend_cfgs = config.llm_backends()
        if backend_cfgs:
            backend_cfgs[0] = {**backend_cfgs[0], "model": model}
        self.backends = [
            _Backend(name=b["name"],
                     client=OpenAI(base_url=b["base_url"], api_key=b["api_key"]),
                     model=b["model"])
            for b in backend_cfgs
        ]
        self._active_idx = 0
        # Back-compat: callers read `.model`; it tracks the ACTIVE backend's model.
        self.model = self.backends[0].model if self.backends else model
        self.last_backend = self.backends[0].name if self.backends else ""
        # Accurate per-backend request accounting: every LLM request in the whole run
        # funnels through this client, so this is the single honest count (the agent's
        # llm_call events only cover ~5 of the call-sites and undercount the rest).
        self.request_counts: dict[str, int] = {}

    @property
    def active_label(self) -> str:
        """'<backend>:<model>' that actually served the run (failover-aware).

        Honest provenance: after a failover this reports the fallback that produced
        output, not the configured primary.  Falls back to the primary's model when
        no call has succeeded yet.
        """
        if self.last_backend:
            backend, model = self.last_backend, self.model
        elif self.backends:
            backend, model = self.backends[0].name, self.backends[0].model
        else:
            return self.model
        # Avoid a redundant prefix when the model id already encodes its provider
        # (e.g. argo's "argo:claude-5-sonnet", or an "nvidia/..." slug).
        if model.startswith(f"{backend}:"):
            return model
        return f"{backend}:{model}"

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
        """Call the LLM, failing over across backends. Returns the response or None.

        Sticky per run (see the class docstring): only advances to the next backend
        when the active one is unavailable.
        """
        while self._active_idx < len(self.backends):
            backend = self.backends[self._active_idx]
            response = self._call_backend(
                backend, messages, tools, temperature, timeout, max_retries, max_tokens)
            if response is not None:
                self.model = backend.model          # track the active model for callers
                self.last_backend = backend.name
                return response
            nxt = self._active_idx + 1
            if nxt < len(self.backends):
                logger.warning("LLM backend '%s' unavailable; failing over to '%s'",
                               backend.name, self.backends[nxt].name)
            self._active_idx = nxt
        logger.error("All %d LLM backend(s) exhausted; returning no response",
                     len(self.backends))
        return None

    def _is_provider_block(self, response) -> bool:
        """True if a 200 response is actually a provider block/limit NOTICE, not output."""
        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return False
        return bool(_PROVIDER_BLOCK_RE.search(content))

    def _call_backend(
        self, backend, messages, tools, temperature, timeout, max_retries, max_tokens,
    ):
        """Attempt ONE backend with transient retries.  Returns the response, or None
        if the backend is unavailable (caller then fails over)."""
        for attempt in range(max_retries):
            try:
                kwargs = {"model": backend.model, "messages": messages}
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if tools:
                    kwargs["tools"] = tools
                # Whole-list ranking calls emit one JSON row per candidate; the
                # provider's default completion cap (~4k tokens) silently truncates
                # them past ~100 candidates, which then fails to parse and drops the
                # ranker to a worse ungated path.  Let callers lift the ceiling.
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                # Apply the timeout to the HTTP client itself.  A ThreadPoolExecutor
                # timeout is not sufficient: leaving its context waits for the
                # still-running request thread, so a stalled proxy call can block the
                # whole campaign indefinitely.
                self.request_counts[backend.name] = self.request_counts.get(backend.name, 0) + 1
                response = backend.client.with_options(timeout=timeout).chat.completions.create(
                    **kwargs
                )
                if self._is_provider_block(response):
                    logger.warning("Backend '%s' returned a block/limit notice (not model "
                                   "output); treating as unavailable", backend.name)
                    return None     # permanent for this backend -> fail over
                return response
            except TimeoutError:
                logger.warning("LLM timeout (%ds) on '%s', attempt %d/%d",
                               timeout, backend.name, attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    return None
            except Exception as e:  # noqa: BLE001 - normalize provider errors to failover
                status = getattr(e, "status_code", None)
                if status in (401, 403):   # auth/quota -> no point retrying this backend
                    logger.warning("Backend '%s' auth failed (%s); treating as unavailable",
                                   backend.name, status)
                    return None
                logger.warning("LLM error on '%s': %s, attempt %d/%d",
                               backend.name, e, attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None
