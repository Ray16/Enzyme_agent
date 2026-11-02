"""LLM client: call with retry, unified token tracking, and response normalization.

This module is the single boundary at which provider-specific quirks are
absorbed.  In particular, some Argo-served models serialize ``tool_use`` dicts
into the assistant's *text content* instead of the structured ``tool_calls``
field; :meth:`LLMClient.extract_tool_calls` recovers those so the agent loop
only ever sees a uniform tool-call shape.
"""

from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import uuid

from enzyme_discovery_agent.state_store import atomic_write_json

try:
    from openai import OpenAI
except ImportError:  # Keep CLI help and offline tools usable without the optional client.
    OpenAI = None  # type: ignore[assignment,misc]

from enzyme_discovery_agent import config
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")

# Marker that identifies an assistant message whose tool calls leaked into text.
_TOOL_USE_TEXT_MARKER = "'type': 'tool_use'"
_TOOLU_SPLIT_RE = re.compile(r"\{['\"]id['\"]:\s*['\"]toolu_")

# Some providers (e.g. argo past its quota) return a block/limit NOTICE as a
# normal HTTP 200 whose text IS the notice, not model output.  Detect it so the
# client fails over to the next backend instead of treating the notice as an answer.
_PROVIDER_BLOCK_RE = re.compile(
    r"ACCESS REVOKED|USAGE NOTICE FROM ARGO|usage limit has been exceeded", re.I)


# A provider can deliver a tool call whose argument JSON it could not itself
# parse, substituting a placeholder that holds only a truncated copy of the raw
# text.  None of the declared fields survive, so such a call records no decision
# the model actually made and must never be scored as a model-side failure.
UNPARSED_TOOL_INPUT_KEY = "__unparsedToolInput"


class UnparsedToolInput(Exception):
    """A tool call arrived with provider-mangled arguments and no usable fields."""


def unparsed_tool_input(arguments) -> bool:
    """True when ``arguments`` is a provider placeholder rather than real tool input."""
    return isinstance(arguments, dict) and UNPARSED_TOOL_INPUT_KEY in arguments


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
    unmetered_calls: int = 0
    legacy_accounting: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _responses: dict[int, object] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, payload: dict) -> "TokenUsage":
        """Restore known usage without declaring an older partial ledger complete."""
        usage = cls(model=payload.get("model", ""))
        usage.calls = [dict(call) for call in payload.get("calls", [])]
        usage.total_prompt_tokens = int(payload.get("total_prompt_tokens", sum(
            c.get("prompt_tokens", 0) for c in usage.calls)))
        usage.total_completion_tokens = int(payload.get("total_completion_tokens", sum(
            c.get("completion_tokens", 0) for c in usage.calls)))
        usage.total_tokens = int(payload.get("total_tokens",
            usage.total_prompt_tokens + usage.total_completion_tokens))
        usage.unmetered_calls = int(payload.get("unmetered_calls", 0))
        usage.legacy_accounting = payload.get("legacy_accounting", "accounting_complete" not in payload)
        if usage.total_tokens != sum(c.get("total_tokens", 0) for c in usage.calls):
            usage.legacy_accounting = True
        return usage

    def record(self, label: str | int, response) -> dict | None:
        """Record token usage from an LLM response."""
        usage = getattr(response, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        with self._lock:
            if id(response) in self._responses:
                return None
            self._responses[id(response)] = response
            if not usage:
                self.unmetered_calls += 1
                return None
            call = {
                "iteration": label,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            }
            self.calls.append(call)
            self.total_prompt_tokens += pt
            self.total_completion_tokens += ct
            self.total_tokens += pt + ct
            return call

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "model": self.model,
                "calls": [dict(call) for call in self.calls],
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "unmetered_calls": self.unmetered_calls,
                "legacy_accounting": self.legacy_accounting,
                "accounting_complete": self.unmetered_calls == 0 and not self.legacy_accounting,
            }


@dataclass
class _Backend:
    """One LLM endpoint in the failover chain."""
    name: str
    client: object
    model: str


_CONTROLLER_VALIDATES_TOOLS = ContextVar('controller_validates_tool_selection',default=False)


class LLMClient:
    """OpenAI-compatible client with transient retries AND availability failover.

    Backends are tried in priority order (``config.llm_backends()``); the client is
    STICKY per run -- once a backend serves a call it stays active, and the client
    only advances to the next backend when the active one is unavailable (exhausted
    transient retries, an auth failure, or a provider block/limit notice returned as
    a 200). Google cascades also advance on model-specific daily quota exhaustion.
    Serving models and transitions are recorded separately from the requested ladder.
    """

    @contextmanager
    def controller_tool_validation(self):
        """Only a controller with its own per-turn execution guard may opt in."""
        token=_CONTROLLER_VALIDATES_TOOLS.set(True)
        try:yield
        finally:_CONTROLLER_VALIDATES_TOOLS.reset(token)

    def __init__(self, model: str = config.DEFAULT_MODEL):
        if OpenAI is None:
            raise RuntimeError(
                "The OpenAI client is required to run the agent. "
                "Install the project dependencies with `pip install -e .`."
            )
        # Ordered backends for availability failover (first = primary).  The explicit
        # `model` arg overrides the PRIMARY backend's model so `--model` still works;
        # fallback backends keep their own configured model.
        self.requested_model = model
        backend_cfgs = config.llm_backends()
        # A provider-qualified model selects Google directly instead of sending
        # a Gemini model identifier to the primary Argo endpoint.
        if model.startswith("claude-code:"):
            model = model.split(":", 1)[1]
            if not model:
                raise ValueError("Use claude-code:<model-id> to select a Claude subscription model")
            backend_cfgs = [{"name": "claude-code", "model": model}]
        elif model.startswith("gemini:"):
            if not config.GEMINI_API_KEY:
                raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use a Gemini model")
            model = model.split(":", 1)[1]
            if not model:
                raise RuntimeError("Use gemini:<model-id> to select a Google model")
            selected = config.GEMINI_CASCADE_MODELS if model in {"auto", "cascade"} else (model,)
            if not selected:
                raise ValueError("GEMINI_CASCADE_MODELS must contain at least one model")
            backend_cfgs = (config.free_model_backends() if model in {"auto", "cascade"} else
                            [{"name": "gemini", "base_url": config.GEMINI_BASE_URL,
                              "api_key": config.GEMINI_API_KEY, "model": item} for item in selected])
            model = backend_cfgs[0]["model"]
        elif model.startswith("groq:"):
            if not config.GROQ_API_KEY:
                raise RuntimeError("Set GROQ_API_KEY in the environment to use Groq")
            model = model.split(":", 1)[1]
            if model != config.GROQ_MODEL:
                raise ValueError(f"The free Groq integration supports only {config.GROQ_MODEL}")
            backend_cfgs = [{"name": "groq", "base_url": config.GROQ_BASE_URL,
                             "api_key": config.GROQ_API_KEY, "model": model}]
        if backend_cfgs:
            backend_cfgs[0] = {**backend_cfgs[0], "model": model}
        self.backends = [
            _Backend(name=b["name"],
                     client=(None if b["name"] == "claude-code" else
                             OpenAI(base_url=b["base_url"], api_key=b["api_key"])),
                     model=b["model"])
            for b in backend_cfgs
        ]
        self._active_idx = 0
        self._backend_state_lock = threading.RLock()
        self._cascade_enabled = (self.requested_model in {"gemini:auto", "gemini:cascade"}
                                 or sum(b.name == "gemini" for b in self.backends) > 1
                                 or any(b.name == "groq" for b in self.backends))
        self._daily_exhausted = {}
        self._backend_failures = {}
        self._gemini_model_states = {}
        self.model_profiles = {}
        self.model_request_counts = {}
        self._groq_daily_usage = []
        for backend in self.backends:
            if backend.name == "gemini":
                self._configure_gemini_profile(backend.model)
            elif backend.name == "groq":
                self._configure_groq_profile(backend.model)
        self._history_default_model = self.requested_model
        # Back-compat: callers read `.model`; it tracks the ACTIVE backend's model.
        self.model = self.backends[0].model if self.backends else model
        self.last_backend = self.backends[0].name if self.backends else ""
        # Accurate per-backend request accounting: every LLM request in the whole run
        # funnels through this client, so this is the single honest count (the agent's
        # llm_call events only cover ~5 of the call-sites and undercount the rest).
        self.request_counts: dict[str, int] = {}
        self._accounting_lock = threading.Lock()
        self._request_count_lock = threading.Lock()
        self._gemini_rate_lock = threading.Lock()
        self._gemini_rate_condition = threading.Condition(self._gemini_rate_lock)
        self._gemini_next_request = 0.0
        self._gemini_input_limit = max(0, config.GEMINI_INPUT_TOKENS_PER_MINUTE)
        self._gemini_request_window: list[dict] = []
        self._usage: TokenUsage | None = None
        self._usage_run_dir: Path | None = None
        self.resource_stop_reason: str | None = None
        self.budget_exhausted = False

    def attach_usage(self, usage: TokenUsage, run_dir: str | Path) -> None:
        """Account for every internal and tool request at the shared boundary."""
        self._usage = usage
        self._usage_run_dir = Path(run_dir)
        try:
            previous = json.loads((run_file(self._usage_run_dir, "token_usage.json")).read_text())
            self.request_counts = previous.get("request_counts", {})
            self.model_request_counts = previous.get("model_request_counts", {})
            self._history_default_model = previous.get("model") or self.requested_model
        except (OSError, ValueError):
            pass
        if not self._gemini_input_limit:
            try:
                limits = json.loads((self._usage_run_dir / "agent_state" / "provider_limits.json").read_text())
                saved = limits.get("gemini") or {}
                if not self._cascade_enabled and any(b.name == "gemini" and b.model == saved.get("model") for b in self.backends):
                    self._gemini_input_limit = max(0, int(saved.get("input_tokens_per_minute", 0)))
            except (OSError, ValueError, TypeError):
                pass
        if self._cascade_enabled:
            self._restore_cascade()

    def _configure_gemini_profile(self, model, metadata=None):
        """Apply project limits without replacing provider-observed quotas."""
        profile = self.model_profiles.setdefault(model, {})
        profile.setdefault("requests_per_minute", config.GEMINI_CASCADE_REQUESTS_PER_MINUTE)
        profile.setdefault("rate_source", "conservative pacing; account quota not yet observed")
        sources = profile.setdefault("quota_sources", {})
        sources.setdefault("requests_per_minute", "conservative pacing")
        profile.update(metadata or {})
        if model in config.GEMINI_FREE_MODEL_PRIORITY:
            profile.setdefault("input_token_limit", 1048576)
            profile.setdefault("output_token_limit", 65536)
        for field, value in config.GEMINI_MODEL_LIMITS.get(model, {}).items():
            quota = field in {"requests_per_minute", "input_tokens_per_minute", "requests_per_day"}
            if quota and sources.get(field) == "Google QuotaFailure":
                continue
            profile[field] = value
            if quota:
                sources[field] = "configured project quota"
        if model in config.GEMINI_FREE_MODEL_PRIORITY:
            profile["reasoning_effort"] = "high"

    def _configure_groq_profile(self, model):
        profile = self.model_profiles.setdefault(model, {})
        profile.update({"requests_per_minute": config.GROQ_REQUESTS_PER_MINUTE,
                        "tokens_per_minute": config.GROQ_TOKENS_PER_MINUTE,
                        "input_tokens_per_minute": config.GROQ_TOKENS_PER_MINUTE,
                        "requests_per_day": config.GROQ_REQUESTS_PER_DAY,
                        "tokens_per_day": config.GROQ_TOKENS_PER_DAY,
                        "input_token_limit": 131072, "output_token_limit": 65536,
                        "reasoning_effort": "high",
                        "rate_source": "configured Groq free-plan limits"})

    def _restore_cascade(self):
        from enzyme_discovery_agent.models.llm_cascade import fetch_model_metadata, next_quota_reset
        root = self._usage_run_dir / "agent_state"
        try:
            saved = json.loads((root / "llm_cascade.json").read_text())
        except (OSError, ValueError):
            saved = {}
        self._daily_exhausted = {k: v for k, v in saved.get("daily_exhausted", {}).items()
                                if v.get("reset_at", 0) > time.time()}
        self.model_profiles = saved.get("model_profiles", {})
        self._groq_daily_usage = [row for row in saved.get("groq_daily_usage", [])
                                 if row.get("time", 0) + 86400 > time.time()]
        try:
            previous_limits = json.loads((root / "provider_limits.json").read_text()).get("gemini", {})
            if previous_limits.get("model"):
                self.model_profiles.setdefault(previous_limits["model"], {}).setdefault(
                    "input_tokens_per_minute", previous_limits.get("input_tokens_per_minute", 0))
        except (OSError, ValueError):
            pass
        # Migrate an earlier single-model daily-quota checkpoint without a retry.
        try:
            checkpoint = root / "budget_checkpoint.json"
            reason = json.loads(checkpoint.read_text()).get("reason", "")
            match = re.search(r"Google AI Studio daily quota exhausted for ([\w.-]+)", reason)
            reset = next_quota_reset(checkpoint.stat().st_mtime)
            if match and reset > time.time():
                self._daily_exhausted.setdefault(f"gemini:{match[1]}", {"reset_at": reset})
        except (OSError, ValueError):
            pass
        metadata = {}
        if any(backend.name == "gemini" for backend in self.backends):
            try:
                metadata = fetch_model_metadata()
            except Exception as exc:
                logger.warning("Google model capability preflight unavailable: %s", type(exc).__name__)
        for backend in self.backends:
            if backend.name == "gemini":
                self._configure_gemini_profile(backend.model, metadata.get(backend.model))
            elif backend.name == "groq":
                self._configure_groq_profile(backend.model)
        self._persist_cascade()

    def _persist_cascade(self):
        root = getattr(self, "_usage_run_dir", None)
        if root is not None and self._cascade_enabled:
            atomic_write_json(root / "agent_state" / "llm_cascade.json", {
                "requested_model": self.requested_model,
                "ladder": [self._backend_label(b) for b in self.backends],
                "active_index": self._active_idx, "daily_exhausted": self._daily_exhausted,
                "model_profiles": copy.deepcopy(self.model_profiles),
                "groq_daily_usage": copy.deepcopy(self._groq_daily_usage),
                "model_request_counts": dict(self.model_request_counts)})

    def _reserve_groq_daily(self, backend, tokens):
        """Reserve both daily allowances before concurrent requests are sent."""
        with self._backend_state_lock:
            now = time.time()
            rows = [row for row in self._groq_daily_usage if row["time"] + 86400 > now]
            self._groq_daily_usage = rows
            profile = self.model_profiles[backend.model]
            request_limit, token_limit = profile["requests_per_day"], profile["tokens_per_day"]
            used = sum(row["tokens"] for row in rows)
            if len(rows) >= request_limit or used + tokens > token_limit:
                remaining_count = len(rows)
                reset = now + 86400
                for row in sorted(rows, key=lambda row: row["time"]):
                    used -= row["tokens"]
                    remaining_count -= 1
                    reset = row["time"] + 86400
                    if remaining_count < request_limit and used + tokens <= token_limit:
                        break
                key = self._backend_label(backend)
                self._daily_exhausted[key] = {"reset_at": reset}
                self._backend_failures[key] = "local Groq daily allowance reserved"
                self._persist_cascade()
                return None
            reservation = {"id": uuid.uuid4().hex, "time": now, "tokens": tokens}
            rows.append(reservation)
            self._groq_daily_usage = rows
            self._persist_cascade()
            return reservation["id"]

    def _settle_groq_daily(self, reservation, actual):
        if not isinstance(actual, int) or actual < 0:
            return
        with self._backend_state_lock:
            for row in self._groq_daily_usage:
                if row["id"] == reservation:
                    row["tokens"] = actual
                    break
            self._persist_cascade()

    def _backend_exhausted(self, backend):
        return self._daily_exhausted.get(self._backend_label(backend), {}).get("reset_at", 0) > time.time()

    @staticmethod
    def _backend_label(backend):
        return backend.model if backend.model.startswith(f"{backend.name}:") else f"{backend.name}:{backend.model}"

    def _model_event(self, event, **fields):
        root = getattr(self, "_usage_run_dir", None)
        if root is not None:
            from enzyme_discovery_agent.models.event_log import append_event
            append_event(root, event, **fields)

    def _advance_backend(self, failed_index):
        # A delayed failure from another caller must not skip the working model.
        with self._backend_state_lock:
            if self._active_idx != failed_index:
                return
            failed = self.backends[failed_index]
            self._active_idx += 1
            if self._active_idx < len(self.backends):
                following = self.backends[self._active_idx]
                reason = self._backend_failures.get(self._backend_label(failed), "unavailable")
                logger.warning("LLM model %s:%s -> %s:%s (%s)",
                               failed.name, failed.model, following.name, following.model, reason)
                self._model_event("llm_model_switch", from_model=self._backend_label(failed),
                                  to_model=self._backend_label(following), reason=reason)
            self._persist_cascade()

    def _account_response(self, response, backend, label) -> None:
        usage = getattr(self, "_usage", None)
        if usage is None:
            return
        with self._accounting_lock:
            missing_before = usage.unmetered_calls
            call = usage.record(label, response)
            from enzyme_discovery_agent.models.event_log import append_event
            usage.model = self.active_label
            if call is not None:
                call.update({"request_id": uuid.uuid4().hex,
                             "backend": backend.name, "model": backend.model})
                if (backend.name == "groq" or (backend.name == "gemini"
                        and backend.model in config.GEMINI_FREE_MODEL_PRIORITY)
                        or (backend.name == "argo" and backend.model == config.ARGO_MODEL)):
                    call["reasoning_effort"] = "high"
                append_event(self._usage_run_dir, "llm_call", **call,
                             cumulative_tokens=usage.total_tokens,
                             accounting="client_boundary")
            elif usage.unmetered_calls > missing_before:
                append_event(self._usage_run_dir, "llm_usage_unavailable", iteration=label,
                             backend=backend.name, model=backend.model)
            try:
                atomic_write_json(run_file(self._usage_run_dir, "token_usage.json"), {
                    **usage.to_dict(), "request_counts": dict(self.request_counts),
                    "model_request_counts": dict(self.model_request_counts),
                })
            except OSError as exc:
                logger.warning("Could not checkpoint LLM usage: %s", exc)

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
        usage_label: str | int = "internal",
    ):
        """Call the LLM, failing over across backends. Returns the response or None.

        Sticky per run (see the class docstring): only advances to the next backend
        when the active one is unavailable.
        """
        if not hasattr(self, "_backend_state_lock"):  # lightweight offline client stubs
            self._backend_state_lock = threading.RLock()
            self._daily_exhausted = {}
            self._backend_failures = {}
            self._cascade_enabled = False
        # An oversized request cannot use Groq, but a later shorter extraction
        # may fit. Do not demote it permanently because of one request's size.
        with self._backend_state_lock:
            for index, backend in enumerate(self.backends[:self._active_idx]):
                reason = self._backend_failures.get(self._backend_label(backend), "")
                if (backend.name == "groq" and not self._backend_exhausted(backend)
                        and reason in {"request exceeds Groq input/output token allowance",
                                       "Groq model requires text-only input"}):
                    self._active_idx = index
                    self._model_event("llm_model_reconsider", model=self._backend_label(backend),
                                      reason="previous request-specific skip")
                    break
        while self._active_idx < len(self.backends):
            if getattr(self, "resource_stop_reason", None):
                return None
            check = getattr(self, "budget_check", None)
            if check and check():
                self.budget_exhausted = True
                return None
            with self._backend_state_lock:
                index = self._active_idx
                if index >= len(self.backends):
                    break
                backend = self.backends[index]
            if self._backend_exhausted(backend):
                self._backend_failures[self._backend_label(backend)] = "daily quota already exhausted"
                self._advance_backend(index)
                continue
            response = self._call_backend(
                backend, messages, tools, temperature, timeout, max_retries, max_tokens)
            if response is not None:
                self.model = backend.model          # track the active model for callers
                self.last_backend = backend.name
                # Carry response provenance into the durable tool-call history.
                if hasattr(response, "choices"):
                    setattr(response, "_enzyme_serving_model", self._backend_label(backend))
                self._account_response(response, backend, usage_label)
                return response
            if getattr(self, "budget_exhausted", False):
                return None
            self._advance_backend(index)
        if self._cascade_enabled:
            self.resource_stop_reason = "All configured LLM models are quota-exhausted or unavailable; resume when capacity returns"
            self.budget_exhausted = True
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
        if backend.name == "claude-code":
            from enzyme_discovery_agent.claude_subscription import call_claude, ClaudeTransportError, Response
            with self._request_count_lock:
                self.request_counts[backend.name] = self.request_counts.get(backend.name, 0) + 1
                key = self._backend_label(backend)
                self.model_request_counts[key] = self.model_request_counts.get(key, 0) + 1
            try:
                if _CONTROLLER_VALIDATES_TOOLS.get():
                    return call_claude(messages, tools, backend.model, max(timeout, 300), max_tokens,
                                       defer_tool_rejections=True)
                return call_claude(messages, tools, backend.model, max(timeout, 300), max_tokens)
            except ClaudeTransportError as exc:
                # A request may have consumed subscription tokens even when it
                # fails. Account any reported usage; never retry an unknown charge.
                if exc.attempted or exc.usage is not None:
                    self._account_response(Response([], exc.usage), backend, "claude_code_failed")
                self.resource_stop_reason = f"Claude Code request stopped: {exc}"
                self.budget_exhausted = True
                logger.warning("%s", self.resource_stop_reason)
                return None
        attempt = 0
        while attempt < max_retries:
            try:
                groq_reservation = None
                if self._backend_exhausted(backend):
                    return None
                kwargs = {"model": backend.model, "messages": [
                    {k: v for k, v in m.items() if not k.startswith("_llm_")} for m in messages]}
                if backend.name != "gemini":
                    for message in kwargs["messages"]:
                        if isinstance(message.get("tool_calls"), list):
                            message["tool_calls"] = [{k: v for k, v in call.items() if k != "extra_content"}
                                                     for call in message["tool_calls"]]
                if backend.name == "gemini":
                    # SDK message dumps include absent struct-valued fields as
                    # null (audio/function_call). Google's protobuf adapter
                    # rejects these on the next tool turn. Preserve signatures
                    # and tool arguments, omitting only absent envelope fields.
                    kwargs["messages"] = self._gemini_messages(messages, backend.model,
                        getattr(self, "_history_default_model", None))
                    if self._cascade_enabled:
                        profile = self.model_profiles.get(backend.model, {})
                        context_limit = profile.get("input_token_limit") or 32768
                        estimated_input = self._estimate_input_tokens(messages, tools)
                        if estimated_input > context_limit:
                            self._backend_failures[f"gemini:{backend.model}"] = "request exceeds model input context"
                            return None
                        minute_limit = profile.get("input_tokens_per_minute", 0)
                        if minute_limit > 0 and estimated_input > minute_limit:
                            self._backend_failures[f"gemini:{backend.model}"] = "request exceeds model input TPM"
                            return None
                        kwargs["max_tokens"] = min(max_tokens or 8192, profile.get("output_token_limit") or 8192)
                    if backend.model in config.GEMINI_FREE_MODEL_PRIORITY:
                        profile = self.model_profiles.get(backend.model, {})
                        answer_allowance = max_tokens if max_tokens is not None else 8192
                        total_allowance = config.GEMINI_HIGH_REASONING_OUTPUT_RESERVE + answer_allowance
                        if answer_allowance <= 0 or total_allowance > (profile.get("output_token_limit") or 65536):
                            self._backend_failures[self._backend_label(backend)] = "insufficient output capacity for high reasoning and answer"
                            return None
                        kwargs.update(reasoning_effort="high", max_tokens=total_allowance)
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if tools:
                    kwargs["tools"] = tools
                # Whole-list ranking calls emit one JSON row per candidate; the
                # provider's default completion cap (~4k tokens) silently truncates
                # them past ~100 candidates, which then fails to parse and drops the
                # ranker to a worse ungated path.  Let callers lift the ceiling.
                if max_tokens is not None and backend.name != "groq" and not (
                        backend.name == "gemini" and (self._cascade_enabled
                        or backend.model in config.GEMINI_FREE_MODEL_PRIORITY)):
                    kwargs["max_tokens"] = max_tokens
                if backend.name == "groq":
                    kwargs["messages"] = self._groq_messages(messages)
                    profile = self.model_profiles[backend.model]
                    estimated_input = self._estimate_input_tokens(messages, tools)
                    output = max_tokens if max_tokens is not None else config.GROQ_MAX_COMPLETION_TOKENS
                    key = self._backend_label(backend)
                    if any(isinstance(m.get("content"), list) and any(
                            part.get("type") != "text" for part in m["content"] if isinstance(part, dict))
                           for m in messages):
                        self._backend_failures[key] = "Groq model requires text-only input"
                        return None
                    if (output <= 0 or output > profile["output_token_limit"]
                            or estimated_input > profile["input_token_limit"]
                            or estimated_input + output > profile["tokens_per_minute"]):
                        self._backend_failures[key] = "request exceeds Groq input/output token allowance"
                        return None
                    kwargs.update(max_completion_tokens=output, reasoning_effort="high",
                                  extra_body={"reasoning_format": "hidden"})
                    if tools:
                        kwargs["parallel_tool_calls"] = False
                if backend.name == "argo" and backend.model == config.ARGO_MODEL:
                    # The installed Argo proxy translates this to adaptive
                    # thinking plus Anthropic output_config.effort=high.
                    answer_allowance = max_tokens if max_tokens is not None else 8192
                    total_allowance = config.ARGO_HIGH_REASONING_OUTPUT_RESERVE + answer_allowance
                    if answer_allowance <= 0 or total_allowance > 128000:
                        self._backend_failures[self._backend_label(backend)] = "insufficient output capacity for high reasoning and answer"
                        return None
                    kwargs.update(reasoning_effort="high", max_tokens=total_allowance)
                reasoning_budget = config.OPENROUTER_STRUCTURED_REASONING_TOKENS
                if (backend.name == "openrouter" and not tools and max_tokens is not None
                        and max_tokens <= 4096 and reasoning_budget > 0):
                    if reasoning_budget >= max_tokens:
                        raise ValueError("structured reasoning budget must leave room for visible output")
                    kwargs["extra_body"] = {"reasoning": {"max_tokens": reasoning_budget}}

                # Apply the timeout to the HTTP client itself.  A ThreadPoolExecutor
                # timeout is not sufficient: leaving its context waits for the
                # still-running request thread, so a stalled proxy call can block the
                # whole campaign indefinitely.
                count_lock = getattr(self, "_request_count_lock", None)
                if count_lock is None:  # compatibility for lightweight client stubs
                    self._request_count_lock = count_lock = threading.Lock()
                client_options = {"timeout": timeout, "max_retries": 0}
                if backend.name == "gemini" and backend.model in config.GEMINI_FREE_MODEL_PRIORITY:
                    client_options["timeout"] = max(timeout, config.GEMINI_HIGH_REASONING_MIN_TIMEOUT)
                elif backend.name == "argo" and backend.model == config.ARGO_MODEL:
                    client_options["timeout"] = max(timeout, config.ARGO_HIGH_REASONING_MIN_TIMEOUT)
                reservation = None
                if backend.name == "gemini":
                    reservation = self._pace_gemini(self._estimate_input_tokens(messages, tools), backend)
                    # SDK retries bypass the pacer and hide actual HTTP attempts.
                    client_options["max_retries"] = 0
                elif backend.name == "groq":
                    reservation = self._pace_gemini(estimated_input + output, backend)
                    if reservation is None:
                        return None
                if self._backend_exhausted(backend):
                    return None
                check = getattr(self, "budget_check", None)
                if getattr(self, "resource_stop_reason", None) or getattr(self, "budget_exhausted", False) or (check and check()):
                    self.budget_exhausted = True
                    return None
                if backend.name == "groq":
                    groq_reservation = self._reserve_groq_daily(backend, estimated_input + output)
                    if groq_reservation is None:
                        return None
                with count_lock:
                    self.request_counts[backend.name] = self.request_counts.get(backend.name, 0) + 1
                    if hasattr(self, "model_request_counts"):
                        key = self._backend_label(backend)
                        self.model_request_counts[key] = self.model_request_counts.get(key, 0) + 1
                response = backend.client.with_options(**client_options).chat.completions.create(
                    **kwargs
                )
                usage = getattr(response, "usage", None)
                if reservation is not None:
                    self._settle_gemini_input(reservation, getattr(usage,
                        "total_tokens" if backend.name == "groq" else "prompt_tokens", None))
                if backend.name == "groq":
                    self._settle_groq_daily(groq_reservation, getattr(usage, "total_tokens", None))
                if (backend.name == "groq" or (backend.name == "gemini"
                        and backend.model in config.GEMINI_FREE_MODEL_PRIORITY)
                        or (backend.name == "argo" and backend.model == config.ARGO_MODEL)):
                    choices = getattr(response, "choices", [])
                    message = getattr(choices[0], "message", None) if choices else None
                    if (not choices or getattr(choices[0], "finish_reason", None) == "length"
                            or not (getattr(message, "content", None) or getattr(message, "tool_calls", None))):
                        # Account for consumed tokens, but never use truncated evidence.
                        self._account_response(response, backend, "discarded_truncated_response")
                        self._backend_failures[self._backend_label(backend)] = f"{backend.name} response truncated or empty"
                        return None
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
                if backend.name == "groq" and status == 429:
                    from enzyme_discovery_agent.models.groq_limits import quota_failure
                    failure = quota_failure(e)
                    key = self._backend_label(backend)
                    if failure["daily"]:
                        with self._backend_state_lock:
                            self._daily_exhausted[key] = {"reset_at": time.time() + failure["retry_after"]}
                            self._backend_failures[key] = "Groq daily quota exhausted"
                            self._model_event("llm_model_quota", model=key, **self._daily_exhausted[key])
                            self._persist_cascade()
                        return None
                    if attempt == max_retries - 1 or failure["retry_after"] > 300:
                        self._backend_failures[key] = "Groq minute quota unavailable after retries"
                        return None
                    self._wait_for_quota(failure["retry_after"] + .1, backend)
                    attempt += 1
                    continue
                if backend.name == "groq" and status == 413:
                    self._backend_failures[self._backend_label(backend)] = "Groq rejected oversized request"
                    return None
                if backend.name == "gemini" and self._google_daily_quota(e):
                    if self._cascade_enabled:
                        from enzyme_discovery_agent.models.llm_cascade import next_quota_reset, quota_values
                        with self._backend_state_lock:
                            key = f"gemini:{backend.model}"
                            self._daily_exhausted[key] = {"reset_at": next_quota_reset()}
                            self._backend_failures[key] = "daily quota exhausted"
                            observed = quota_values(e)
                            profile = self.model_profiles.setdefault(backend.model, {})
                            profile.update(observed)
                            profile.setdefault("quota_sources", {}).update({k: "Google QuotaFailure" for k in observed})
                            self._model_event("llm_model_quota", model=key, **self._daily_exhausted[key])
                            self._persist_cascade()
                        with self._gemini_rate_condition:
                            self._gemini_rate_condition.notify_all()
                        return None
                    self.resource_stop_reason = (
                        f"Google AI Studio daily quota exhausted for {backend.model}; "
                        "resume when quota is available again"
                    )
                    self.budget_exhausted = True
                    logger.warning(self.resource_stop_reason)
                    with self._gemini_rate_condition:
                        self._gemini_rate_condition.notify_all()
                    return None
                if backend.name == "gemini":
                    limit = self._google_input_token_limit(e)
                    if self._cascade_enabled:
                        from enzyme_discovery_agent.models.llm_cascade import quota_values
                        observed = quota_values(e)
                        with self._backend_state_lock:
                            self.model_profiles.setdefault(backend.model, {}).update(observed)
                            self.model_profiles[backend.model].setdefault("quota_sources", {}).update(
                                {k: "Google QuotaFailure" for k in observed})
                            if observed:
                                self.model_profiles[backend.model]["rate_source"] = "Google QuotaFailure"
                            self._persist_cascade()
                        if any(value == 0 for value in observed.values()):
                            self._backend_failures[f"gemini:{backend.model}"] = "model quota is zero for this project"
                            return None
                    elif limit:
                        self._gemini_input_limit = limit
                        run_dir = getattr(self, "_usage_run_dir", None)
                        if run_dir:
                            atomic_write_json(run_dir / "agent_state" / "provider_limits.json", {
                                "gemini": {"model": backend.model, "input_tokens_per_minute": limit,
                                           "source": "provider_quota_failure"}})
                        logger.warning("Google input-token pacing updated to %s tokens/minute", limit)
                    if status == 429 and self._google_minute_quota(e):
                        if limit and self._estimate_input_tokens(messages, tools) > limit:
                            if self._cascade_enabled:
                                self._backend_failures[f"gemini:{backend.model}"] = "request exceeds model input TPM"
                                return None
                            self.resource_stop_reason = (
                                "Request context exceeds Google's input-token allowance per minute; "
                                "reduce the request context before resuming"
                            )
                            self.budget_exhausted = True
                            return None
                        delay = max(1.0, self._retry_delay(e) or 60.1)
                        logger.warning("Google minute quota reached; waiting %.1fs before retrying", delay)
                        self._wait_for_quota(delay, backend)
                        if getattr(self, "budget_exhausted", False):
                            return None
                        # A minute quota is pacing, not backend unavailability.
                        # Keep the model and do not consume transient retries.
                        continue
                if status in (400, 401, 403, 404):
                    self._backend_failures[self._backend_label(backend)] = f"provider rejection HTTP {status}"
                    logger.warning("Backend '%s' rejected the request (%s); treating as unavailable",
                                   backend.name, status)
                    return None
                logger.warning("LLM error on '%s': %s, attempt %d/%d",
                               backend.name, e, attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    return None
                delay = self._retry_delay(e)
                if delay is None:
                    delay = 2 ** attempt
                # Avoid an accidental day-long wait on a daily quota exhaustion.
                if delay > 300:
                    logger.warning("Provider retry delay exceeds five minutes; returning unavailable")
                    return None
                time.sleep(delay)
            attempt += 1
        return None

    def _wait_for_quota(self, seconds, backend=None):
        """Wait in short intervals so a campaign budget can interrupt pacing."""
        remaining = seconds
        while remaining > 0:
            if backend is not None and self._backend_exhausted(backend):
                return
            check = getattr(self, "budget_check", None)
            if getattr(self, "resource_stop_reason", None) or (check and check()):
                self.budget_exhausted = True
                return
            step = min(remaining, 10.0)
            time.sleep(step)
            remaining -= step

    @staticmethod
    def _google_minute_quota(error) -> bool:
        """Only explicit per-minute QuotaFailure identifiers are indefinitely retryable."""
        body = getattr(error, "body", None)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                return False
        def find(value):
            if isinstance(value, dict):
                if str(value.get("@type", "")).endswith("google.rpc.QuotaFailure"):
                    return any("perminute" in re.sub(r"[^a-z]", "", str(row.get("quotaId", "")).lower())
                               for row in value.get("violations") or [] if isinstance(row, dict))
                return any(find(child) for child in value.values())
            return isinstance(value, list) and any(find(child) for child in value)
        return find(body)

    @staticmethod
    def _google_daily_quota(error) -> bool:
        """Distinguish a daily QuotaFailure from a transient request-rate limit."""
        body = getattr(error, "body", None)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                pass

        def daily_violation(value):
            if isinstance(value, dict):
                if str(value.get("@type", "")).endswith("google.rpc.QuotaFailure"):
                    for violation in value.get("violations") or []:
                        if not isinstance(violation, dict):
                            continue
                        identifier = " ".join(str(violation.get(key, "")) for key in (
                            "quotaId", "quotaMetric", "description", "subject"))
                        normalized = re.sub(r"[^a-z0-9]", "", identifier.lower())
                        if "perday" in normalized or "daily" in normalized:
                            return True
                return any(daily_violation(v) for v in value.values())
            if isinstance(value, list):
                return any(daily_violation(v) for v in value)
            return False

        if daily_violation(body):
            return True
        if getattr(error, "status_code", None) != 429:
            return False
        text = str(error) + " " + (body if isinstance(body, str) else json.dumps(body, default=str))
        return bool(re.search(
            r"(?:quota|limit)[^\n]{0,200}(?:per[_ -]?day|daily|\bRPD\b)|"
            r"(?:per[_ -]?day|daily|\bRPD\b)[^\n]{0,100}(?:quota|limit)", text, re.I))

    @staticmethod
    def _groq_messages(messages):
        """Translate provider envelopes while retaining text, tool calls and results."""
        normalized = []
        for message in messages:
            clean = {key: value for key, value in message.items()
                     if key in {"role", "content", "tool_call_id", "tool_calls"} and value is not None}
            if isinstance(clean.get("tool_calls"), list):
                clean["tool_calls"] = [
                    {key: value for key, value in call.items() if key in {"id", "type", "function"}}
                    for call in clean["tool_calls"]]
            normalized.append(clean)
        return normalized

    @staticmethod
    def _gemini_messages(messages: list[dict], model=None, history_default_model=None) -> list[dict]:
        """Normalize optional SDK envelopes without changing conversation state."""
        normalized = []
        for message in messages:
            clean = {key: value for key, value in message.items()
                     if value is not None and not key.startswith("_llm_")}
            source = message.get("_llm_model") or history_default_model
            migrated = model is not None and source is not None and source != f"gemini:{model}"
            if isinstance(clean.get("tool_calls"), list):
                calls = []
                for call in clean["tool_calls"]:
                    kept = {key: value for key, value in call.items() if value is not None}
                    if isinstance(kept.get("function"), dict):
                        kept["function"] = {key: value for key, value in kept["function"].items()
                                            if value is not None}
                    if migrated:
                        # Google documents this sentinel for importing another
                        # model's tool trace. Leave the original audit untouched.
                        extra = copy.deepcopy(kept.get("extra_content") or {})
                        extra.setdefault("google", {})["thought_signature"] = "skip_thought_signature_validator"
                        kept["extra_content"] = extra
                    calls.append(kept)
                clean["tool_calls"] = calls
            normalized.append(clean)
        return normalized

    @staticmethod
    def _estimate_input_tokens(messages, tools) -> int:
        """Conservative text estimate; provider-reported usage settles reservations."""
        images = 0
        def without_media(value):
            nonlocal images
            if isinstance(value, dict):
                if value.get("type") in {"image_url", "input_image"}:
                    images += 1
                    return {"type": value["type"]}
                return {k: without_media(v) for k, v in value.items()}
            if isinstance(value, list):
                return [without_media(v) for v in value]
            return value
        serialized = json.dumps(without_media({"messages": messages, "tools": tools}), ensure_ascii=False, default=str)
        return max(1, int(len(serialized) / 3 * 1.1) + 1 + images * 4096)

    def _pace_gemini(self, input_tokens: int = 0, backend=None):
        """Reserve request and input-token capacity across every internal caller."""
        separate = self._cascade_enabled and backend is not None
        state = self._gemini_model_states.setdefault(backend.model, {"window": [], "next_request": 0.0}) if separate else None
        with self._gemini_rate_condition:
            while True:
                if backend is not None and self._backend_exhausted(backend):
                    return None
                check = getattr(self, "budget_check", None)
                if self.resource_stop_reason or self.budget_exhausted or (check and check()):
                    return None
                now = time.monotonic()
                profile = self.model_profiles.get(backend.model, {}) if separate else {}
                rpm = profile.get("requests_per_minute", config.GEMINI_CASCADE_REQUESTS_PER_MINUTE) if separate else config.GEMINI_REQUESTS_PER_MINUTE
                window = state["window"] if separate else self._gemini_request_window
                window = [r for r in window if now - r["time"] < 60]
                if separate:
                    state["window"] = window
                else:
                    self._gemini_request_window = window
                next_request = state["next_request"] if separate else self._gemini_next_request
                wait = max(0.0, next_request - now) if rpm > 0 else 0.0
                limit = profile.get("input_tokens_per_minute", 0) if separate else self._gemini_input_limit
                if limit:
                    reserved = min(max(0, input_tokens), limit)
                    used = sum(r["tokens"] for r in window)
                    for row in window:
                        if used + reserved <= limit:
                            break
                        wait = max(wait, row["time"] + 60.1 - now)
                        used -= row["tokens"]
                if wait <= 0:
                    break
                provider = "Groq" if backend is not None and backend.name == "groq" else "Google"
                logger.info("%s pacing wait %.1fs (estimated tokens: %s)", provider, wait, input_tokens)
                # Release the lock so completed requests can settle their actual
                # usage and wake waiting callers without a full-window delay.
                self._gemini_rate_condition.wait(wait)
            started = time.monotonic()
            window = [r for r in window if started - r["time"] < 60]
            reservation = {"id": uuid.uuid4().hex, "time": started, "tokens": max(0, input_tokens)}
            window.append(reservation)
            following = started + 60.0 / rpm + 0.1 if rpm > 0 else started
            if separate:
                state.update(window=window, next_request=following)
            else:
                self._gemini_request_window = window
                self._gemini_next_request = following
            return reservation["id"]

    def _settle_gemini_input(self, reservation, actual_tokens):
        if not isinstance(actual_tokens, int) or actual_tokens < 0:
            return
        with self._gemini_rate_condition:
            windows = [self._gemini_request_window] + [state["window"] for state in self._gemini_model_states.values()]
            for row in [row for window in windows for row in window]:
                if row["id"] == reservation:
                    row["tokens"] = actual_tokens
                    self._gemini_rate_condition.notify_all()
                    break

    @staticmethod
    def _google_input_token_limit(error) -> int | None:
        body = getattr(error, "body", None)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                return None
        def find(value):
            if isinstance(value, dict):
                if str(value.get("@type", "")).endswith("google.rpc.QuotaFailure"):
                    for row in value.get("violations") or []:
                        if not isinstance(row, dict):
                            continue
                        identifier = re.sub(r"[^a-z]", "", str(row.get("quotaId", "")).lower())
                        if "inputtokens" in identifier and "perminute" in identifier:
                            try:
                                limit = int(row.get("quotaValue", 0))
                                if limit > 0:
                                    return limit
                            except (ValueError, TypeError):
                                pass
                for child in value.values():
                    result = find(child)
                    if result:
                        return result
            elif isinstance(value, list):
                for child in value:
                    result = find(child)
                    if result:
                        return result
            return None
        return find(body)

    @staticmethod
    def _retry_delay(error) -> float | None:
        """Read Retry-After or Google's structured RetryInfo without guessing."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        try:
            return max(0.0, float(headers["retry-after"])) + 0.1
        except (KeyError, TypeError, ValueError):
            pass
        body = getattr(error, "body", None)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                body = None
        def retry_info(value):
            if isinstance(value, dict):
                if str(value.get("@type", "")).endswith("google.rpc.RetryInfo"):
                    match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(value.get("retryDelay", "")))
                    if match:
                        return float(match.group(1)) + 0.1
                for child in value.values():
                    delay = retry_info(child)
                    if delay is not None:
                        return delay
            elif isinstance(value, list):
                for child in value:
                    delay = retry_info(child)
                    if delay is not None:
                        return delay
            return None
        return retry_info(body)
