"""LLM client: call with retry and unified token tracking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

try:
    from openai import OpenAI
except ImportError:  # Keep CLI help and offline tools usable without the optional client.
    OpenAI = None  # type: ignore[assignment,misc]

from enzyme_discovery_agent import config

logger = logging.getLogger("enzyme_agent")


@dataclass
class TokenUsage:
    """Accumulator for LLM token usage across a run."""
    model: str = ""
    calls: list[dict] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    def record(self, label: str | int, response) -> None:
        """Record token usage from an LLM response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
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

    def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = config.LLM_TEMPERATURE,
        timeout: int = config.LLM_TIMEOUT,
        max_retries: int = 3,
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
