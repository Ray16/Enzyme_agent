"""The enzyme design agent.

A loop that decides HOW to design, not just one that runs a design tool. The
decision it exists to make is the strategy choice -- generate a scaffold around
a catalytic motif, or saturate the pocket of one that already binds -- plus the
inputs that choice implies: which motif, which positions, which conditions.
Hard-coding that choice would make the tools a pipeline and this file
unnecessary, so it is left to the model and recorded explicitly.

Deliberately much smaller than the discovery agent's loop. That one carries
coverage gates, family-promotion nudges and a multi-stage finalizer, all of
which exist to keep a *search* honest. A design run has no search space to
cover; its risk is committing GPU-weeks to a badly specified motif, which is
addressed by the prompt and by the readiness tools, not by gates.

Shared infrastructure -- the LLM client, config, the tool registry, run-dir
conventions, the event log -- is reused from ``enzyme_discovery_agent`` rather
than forked, so both agents stay on one backend chain and one run layout.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.capabilities import create_run_dir, init_capabilities
from enzyme_discovery_agent.context import compact_for_llm
from enzyme_discovery_agent.llm.client import LLMClient, TokenUsage
from enzyme_core.state_store import atomic_write_text as _atomic_write_text
from enzyme_discovery_agent.tool_registry import execute_tool, get_tool_info

from .bootstrap import design_tool_names, design_tool_schemas, register_design_tools
from .prompts import (
    FINALIZE_PROMPT, REFLECTION_PROMPT, build_state_header, build_system_prompt,
)

logger = logging.getLogger("enzyme_agent")

#: Turns reserved at the end so the agent always gets to write its report
#: rather than being cut off mid-campaign with nothing recorded.
FINALIZATION_RESERVE = 3

#: Default character allowance for the transient LLM context window.
DESIGN_CONTEXT_MAX_CHARS = 64000


def _build_design_context(
    messages: list[dict],
    state_header: str,
    *,
    max_chars: int = DESIGN_CONTEXT_MAX_CHARS,
) -> list[dict]:
    """Assemble a bounded, prompt-cache-friendly message list for one LLM turn.

    Keeps the durable ``messages`` prefix immutable across turns (never
    inserting/removing ``[STATE: ...]`` in the middle of history) and appends
    the volatile state header only at the tail. When history exceeds
    ``max_chars``, drops oldest whole assistant/tool groups without splitting
    a tool call from its tool responses.
    """
    state_msg = {"role": "user", "content": state_header}
    if len(messages) <= 2:
        return [*messages, state_msg]

    prefix = messages[:2]
    groups: list[list[dict]] = []
    for msg in messages[2:]:
        if msg.get("role") == "assistant" or not groups:
            groups.append([])
        groups[-1].append(msg)

    prefix_chars = len(json.dumps(prefix, default=str))
    tail_chars = len(json.dumps(state_msg, default=str))
    group_sizes = [len(json.dumps(g, default=str)) for g in groups]
    total = prefix_chars + tail_chars + sum(group_sizes)

    start = 0
    while start < len(groups) - 1 and total > max_chars:
        total -= group_sizes[start]
        start += 1

    kept = [m for g in groups[start:] for m in g]
    return [*prefix, *kept, state_msg]

#: Tools whose execution pins the run's strategy. Readiness probes are
#: intentionally excluded so checking hardware availability does not commit a
#: strategy before the agent actually chooses one.
_STRATEGY_BY_TOOL = {
    "design_de_novo_enzyme": "generative",
    "summarize_de_novo_campaign": "generative",
    "design_enzyme_with_disco": "generative",
    "summarize_disco_campaign": "generative",
    "design_enzyme_with_genzyme": "generative",
    "get_genzyme_design_status": "generative",
    "design_enzyme_with_rfd4": "generative",
    "get_rfd4_design_status": "generative",
    "scan_design_pocket": "redesign",
    "build_saturation_library": "redesign",
    "prepare_saturation_campaign": "redesign",
    "summarize_saturation_campaign": "redesign",
}


def _load_design_handoff(handoff_path: str | Path) -> tuple[dict, dict]:
    """Load a structured structural-modeling design handoff JSON and its provenance."""
    resolved = Path(handoff_path).expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Design handoff must be a JSON object: {resolved}")
    provenance = {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "kind": payload.get("kind"),
        "source_accession": payload.get("source_accession"),
        "source_task_sha256": payload.get("source_task_sha256"),
    }
    return payload, provenance


def _format_handoff_context(handoff: dict) -> str:
    """Render a compact, structured summary of a modeling-to-design handoff."""
    summary = {
        k: handoff.get(k)
        for k in (
            "kind",
            "purpose",
            "requested_tool",
            "source_accession",
            "target",
            "catalytic_profile_summary",
            "soft_defect_design_targets",
            "failure_modes",
            "question",
            "cofold_evidence",
        )
        if handoff.get(k) is not None
    }
    return (
        "Structured structural-modeling handoff (evidence and diagnostic targets; "
        "verify before committing compute, and do not treat as proof of catalysis):\n"
        + json.dumps(summary, indent=2, default=str)
    )


def _count_designs(name: str, result: dict, counted_jobs: set[str]) -> int:
    """Count newly produced designs/variants without double-counting polled jobs."""
    if result.get("status") != "success":
        return 0
    data = result.get("data") or {}
    if not isinstance(data, dict):
        return 0
    if name == "design_de_novo_enzyme":
        counts = data.get("artifact_counts") or {}
        return int(counts.get("generated") or data.get("num_designs") or 0)
    if name == "design_enzyme_with_disco":
        return int(data.get("n_designs") or 0)
    if name in ("get_genzyme_design_status", "get_rfd4_design_status"):
        job_key = f"{name}:{data.get('manifest') or data.get('job_dir') or ''}"
        if job_key in counted_jobs:
            return 0
        designs = data.get("designs") or []
        if designs:
            counted_jobs.add(job_key)
            return len(designs)
        return 0
    if name == "build_saturation_library":
        return int(data.get("n_variants") or 0)
    if name == "prepare_saturation_campaign":
        cands = data.get("candidates") or []
        return sum(int(c.get("n_variants") or 0) for c in cands if isinstance(c, dict))
    return 0


class EnzymeDesignAgent:
    """Autonomous enzyme design: choose a strategy, specify it, run it, report."""

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        output_dir: str | Path = "runs",
        max_iterations: int = 40,
        llm_timeout: int = config.LLM_TIMEOUT,
        query: str = "",
        token_budget: int | None = None,
        runtime_budget_seconds: float | None = None,
    ):
        register_design_tools()
        self.model_name = model
        self.llm = LLMClient(model=model)
        self.max_iterations = max_iterations
        self.llm_timeout = llm_timeout
        self.token_budget = token_budget
        self.runtime_budget_seconds = runtime_budget_seconds

        self.run_dir = create_run_dir(Path(output_dir), query=query)
        self.capabilities = init_capabilities()
        self.tools = design_tool_schemas()
        self.tool_names = design_tool_names()
        self._allowed_tools = frozenset(self.tool_names)

        self.strategy = "undecided"
        self.tools_used: list[str] = []
        self.designs = 0
        self._counted_jobs: set[str] = set()
        self._source_handoff: dict | None = None
        self._strategy_decision: dict | None = None
        self._artifact_paths: list[str] = []
        self._log_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._event_path = self.run_dir / "events.jsonl"
        self._console_path = self.run_dir / "console_log.txt"

    # -- logging -----------------------------------------------------------
    def log(self, msg: str, level: str = "info") -> None:
        from datetime import datetime

        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        with self._log_lock:
            with self._console_path.open("a") as fh:
                fh.write(line + "\n")
        getattr(logger, level, logger.info)(msg)

    def _event(self, event_type: str, **data) -> None:
        try:
            from enzyme_discovery_agent.models.event_log import append_event

            append_event(self.run_dir, event_type, **data)
        except Exception:                                   # pragma: no cover
            pass          # the event log is best-effort; it never breaks a run

    # -- tools -------------------------------------------------------------
    def _execute(self, name: str, args: dict, iteration: int) -> dict:
        self.log(f"  -> {name}({json.dumps(args, default=str)[:160]})")
        self._event("tool_call", tool=name, args=args, iteration=iteration)
        try:
            result = execute_tool(
                name, args,
                injected_args={"run_dir": str(self.run_dir), "llm_client": self.llm},
            )
        except Exception as exc:                            # tool bugs must not kill the run
            logger.warning("tool %s raised: %s", name, exc)
            return {"status": "error", "summary": f"{name} raised {type(exc).__name__}: {exc}",
                    "data": None, "suggestions": []}
        result = json.loads(json.dumps(result, default=str))

        with self._state_lock:
            if name == "declare_design_strategy" and result.get("status") == "success":
                decision_data = result.get("data") or {}
                declared = decision_data.get("strategy")
                if declared in {"generative", "redesign", "both"}:
                    self.strategy = declared
                    self._strategy_decision = decision_data
                    self.log(f"  strategy declared: {declared} ({decision_data.get('backend', '')})")
                    self._event("strategy_selected", strategy=declared, via=name)
            elif name in _STRATEGY_BY_TOOL and result.get("status") in ("success", "running"):
                picked = _STRATEGY_BY_TOOL[name]
                if self.strategy == "undecided":
                    self.strategy = picked
                    self.log(f"  strategy: {picked}")
                    self._event("strategy_selected", strategy=picked, via=name)
                elif self.strategy != picked and not self.strategy.startswith("both"):
                    self.strategy = "both"
                    self._event("strategy_selected", strategy="both", via=name)
            self.designs += _count_designs(name, result, self._counted_jobs)

        self._save_artifact(name, args, result, iteration)
        return result

    def _save_artifact(self, name: str, args: dict, result: dict, iteration: int) -> None:
        d = self.run_dir / "artifacts" / "design"
        with self._log_lock:
            d.mkdir(parents=True, exist_ok=True)
            n = len(list(d.glob(f"{name}_*.json"))) + 1
            dest = d / f"{name}_{n:03d}.json"
            _atomic_write_text(
                dest,
                json.dumps({"tool_name": name, "args": args, "iteration": iteration,
                            "result": result}, indent=2, default=str),
            )
            self._artifact_paths.append(str(dest.relative_to(self.run_dir)))

    # -- the loop ----------------------------------------------------------
    def run(self, query: str, motif_hint: str | None = None,
            scaffold_path: str | None = None,
            handoff_path: str | Path | None = None) -> str:
        """Design for ``query``; returns the final report text.

        Args:
            query: the target chemistry, in the user's words.
            motif_hint: optional notes on the catalytic motif. A hint, never a
                substitute for the agent deriving and justifying one.
            scaffold_path: optional holo structure. Supplying one makes
                redesign *possible*; it does not make it the right choice, and
                the agent is still expected to argue the strategy.
            handoff_path: optional path to a structured ``design_handoff.json``
                produced by ``structural_modeling_agent``.
        """
        handoff_data: dict | None = None
        if handoff_path is not None:
            handoff_data, self._source_handoff = _load_design_handoff(handoff_path)
        self.log(f"query: {query}")
        self.log(f"run_dir: {self.run_dir}")
        self._event("run_start", query=query, model=self.model_name,
                    tools=list(self.tool_names), scaffold=scaffold_path,
                    source_handoff=self._source_handoff)

        system = build_system_prompt(available_tools=list(self.tool_names),
                                     capabilities=self.capabilities)
        user = [f"Target chemistry:\n{query}"]
        if handoff_data is not None:
            user.append(_format_handoff_context(handoff_data))
        if motif_hint:
            user.append(f"Notes on the catalytic motif (a starting point, to be "
                        f"justified or revised):\n{motif_hint}")
        if scaffold_path:
            user.append(f"A candidate holo scaffold is available at: {scaffold_path}\n"
                        f"Its presence does not decide the strategy.")
        else:
            user.append("No scaffold has been supplied.")
        user.append(f"Output directory: {self.run_dir}")

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": "\n\n".join(user)}]
        usage = TokenUsage(model=self.model_name)
        started_at = time.monotonic()
        finalizing = False
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            self._event("iteration_start", iteration=iteration, strategy=self.strategy)

            budget_reason = None
            if self.token_budget is not None and usage.total_tokens >= self.token_budget:
                budget_reason = f"token budget reached ({usage.total_tokens}/{self.token_budget})"
            elif (self.runtime_budget_seconds is not None
                  and (time.monotonic() - started_at) >= self.runtime_budget_seconds):
                budget_reason = f"runtime budget reached ({self.runtime_budget_seconds:.0f}s)"

            if not finalizing and (
                iteration >= self.max_iterations - FINALIZATION_RESERVE
                or budget_reason is not None
            ):
                self.log(f"entering finalization{f' ({budget_reason})' if budget_reason else ''}")
                messages.append({"role": "user", "content": FINALIZE_PROMPT})
                finalizing = True

            if iteration > 1 and iteration % 6 == 0 and not finalizing:
                messages.append({"role": "user", "content": REFLECTION_PROMPT})

            state_header = build_state_header(
                iteration, self.max_iterations, self.strategy, self.designs,
                self.tools_used,
            )
            # Keep durable history append-only for provider prompt caching;
            # attach the volatile [STATE: ...] header only at the tail of the
            # transient request payload.
            model_messages = _build_design_context(messages, state_header)

            response = self.llm.call(messages=model_messages, tools=self.tools,
                                     timeout=self.llm_timeout)
            if response is None:
                return self._finish(messages, usage, "LLM unavailable; no report produced.",
                                    status="failed", iterations=iteration)
            usage.record(iteration, response)
            assistant = response.choices[0].message
            messages.append(assistant.model_dump())
            tool_calls, text, _recovered = self.llm.extract_tool_calls(assistant)
            self._event("assistant_decision", iteration=iteration,
                        n_tool_calls=len(tool_calls or []), has_text=bool(text))

            if not tool_calls:
                if finalizing or iteration > 1:
                    return self._finish(messages, usage, text or "",
                                        status="completed", iterations=iteration)
                # An empty first turn is usually the model asking to be told to
                # begin, not a finished campaign.
                messages.append({"role": "user", "content":
                                 "Begin: state your strategy choice and its reason, "
                                 "then make your first tool call."})
                continue

            prepared = []
            for call in tool_calls:
                name = call.function.name
                raw_args = call.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    prepared.append((call, name, None, {
                        "status": "error",
                        "summary": f"Malformed JSON arguments for {name}: {exc}",
                        "data": None,
                        "suggestions": ["Re-call the tool with a valid JSON object of arguments."],
                    }))
                    continue
                if name not in self._allowed_tools or get_tool_info(name) is None:
                    prepared.append((call, name, None, {
                        "status": "error",
                        "summary": f"{name} is not a design tool. Available: "
                                   f"{', '.join(self.tool_names)}",
                        "data": None,
                        "suggestions": [],
                    }))
                    continue
                prepared.append((call, name, args, None))

            # Execute parallel_safe calls concurrently when multiple calls arrive.
            runnable = [(i, name, args) for i, (_c, name, args, err) in enumerate(prepared)
                        if err is None]
            results_by_idx: dict[int, dict] = {
                i: err for i, (_c, _n, _a, err) in enumerate(prepared) if err is not None
            }
            if (len(runnable) > 1
                    and all((get_tool_info(n) or {}).get("parallel_safe", False) for _i, n, _a in runnable)):
                with ThreadPoolExecutor(max_workers=min(4, len(runnable))) as pool:
                    futures = {
                        i: (name, pool.submit(self._execute, name, args, iteration))
                        for i, name, args in runnable
                    }
                    for i, (name, fut) in futures.items():
                        results_by_idx[i] = fut.result()
                        self.tools_used.append(name)
            else:
                for i, name, args in runnable:
                    results_by_idx[i] = self._execute(name, args, iteration)
                    self.tools_used.append(name)

            for i, (call, _name, _args, _err) in enumerate(prepared):
                out = results_by_idx[i]
                body = json.dumps(compact_for_llm(out), default=str)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": body[:config.MAX_RESULT_CHARS]})

            _atomic_write_text(
                self.run_dir / "agent_state" / "transcript.json",
                json.dumps(
                    {
                        "iteration": iteration,
                        "messages": [*messages, {"role": "user", "content": state_header}],
                    },
                    indent=2,
                    default=str,
                ),
            )

        return self._finish(messages, usage, "Max iterations reached.",
                            status="max_iterations", iterations=iteration)

    # -- finish ------------------------------------------------------------
    def _finish(self, messages: list, usage: TokenUsage, final_text: str,
                status: str, iterations: int) -> str:
        report = final_text or "(no report text produced)"
        _atomic_write_text(self.run_dir / "DESIGN_REPORT.md", report + "\n")
        _atomic_write_text(
            self.run_dir / "token_usage.json",
            json.dumps(usage.to_dict(), indent=2, default=str),
        )
        manifest = {
            "agent": "enzyme_design_agent",
            "status": status,
            "model": self.model_name,
            "strategy": self.strategy,
            "iterations_used": iterations,
            "designs": self.designs,
            "tools_used": self.tools_used,
            "source_handoff": self._source_handoff,
            "strategy_decision": self._strategy_decision,
            "artifacts": list(self._artifact_paths),
            "tokens": usage.to_dict(),
        }
        _atomic_write_text(self.run_dir / "manifest.json", json.dumps(manifest, indent=2, default=str))
        if self._source_handoff is not None or self.designs > 0 or self._strategy_decision is not None:
            _atomic_write_text(
                self.run_dir / "design_output_handoff.json",
                json.dumps({
                    "kind": "design_output_handoff",
                    "agent": "enzyme_design_agent",
                    "status": status,
                    "strategy": self.strategy,
                    "strategy_decision": self._strategy_decision,
                    "designs": self.designs,
                    "source_handoff": self._source_handoff,
                    "artifacts": list(self._artifact_paths),
                    "report_path": "DESIGN_REPORT.md",
                    "note": "Design agent outputs; refolding and catalytic validation remain separate.",
                }, indent=2, default=str),
            )
        self._event("run_complete", status=status, strategy=self.strategy,
                    iterations=iterations, designs=self.designs)
        self.log(f"done: status={status} strategy={self.strategy} "
                 f"iterations={iterations}")
        return report
