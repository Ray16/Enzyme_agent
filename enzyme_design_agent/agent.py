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

import json
import logging
import threading
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.capabilities import create_run_dir, init_capabilities
from enzyme_discovery_agent.context import compact_for_llm
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.tool_registry import execute_tool, get_tool_info

from .bootstrap import design_tool_names, design_tool_schemas, register_design_tools
from .prompts import (
    FINALIZE_PROMPT, REFLECTION_PROMPT, build_state_header, build_system_prompt,
)

logger = logging.getLogger("enzyme_agent")

#: Turns reserved at the end so the agent always gets to write its report
#: rather than being cut off mid-campaign with nothing recorded.
FINALIZATION_RESERVE = 3

#: Tools whose names, if called, pin the run's strategy. Recorded for the state
#: header and the manifest; it is an observation of what the agent did, never a
#: constraint on what it may do next.
_STRATEGY_BY_TOOL = {
    "design_de_novo_enzyme": "generative",
    "check_de_novo_design_readiness": "generative",
    "summarize_de_novo_campaign": "generative",
    "scan_design_pocket": "redesign",
    "build_saturation_library": "redesign",
    "prepare_saturation_campaign": "redesign",
}


class EnzymeDesignAgent:
    """Autonomous enzyme design: choose a strategy, specify it, run it, report."""

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        output_dir: str | Path = "runs",
        max_iterations: int = 40,
        llm_timeout: int = config.LLM_TIMEOUT,
        query: str = "",
    ):
        register_design_tools()
        self.model_name = model
        self.llm = LLMClient(model=model)
        self.max_iterations = max_iterations
        self.llm_timeout = llm_timeout

        self.run_dir = create_run_dir(Path(output_dir), query=query)
        self.capabilities = init_capabilities()
        self.tools = design_tool_schemas()
        self.tool_names = design_tool_names()

        self.strategy = "undecided"
        self.tools_used: list[str] = []
        self.designs = 0
        self._log_lock = threading.Lock()
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

        if name in _STRATEGY_BY_TOOL and result.get("status") == "success":
            picked = _STRATEGY_BY_TOOL[name]
            if self.strategy == "undecided":
                self.strategy = picked
                self.log(f"  strategy: {picked}")
                self._event("strategy_selected", strategy=picked, via=name)
            elif self.strategy != picked and not self.strategy.startswith("both"):
                self.strategy = "both"
                self._event("strategy_selected", strategy="both", via=name)
        if name in ("design_de_novo_enzyme", "build_saturation_library"):
            data = result.get("data") or {}
            self.designs += int(data.get("n_variants") or data.get("num_designs") or 0)

        self._save_artifact(name, args, result, iteration)
        return result

    def _save_artifact(self, name: str, args: dict, result: dict, iteration: int) -> None:
        d = self.run_dir / "artifacts" / "design"
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob(f"{name}_*.json"))) + 1
        (d / f"{name}_{n:03d}.json").write_text(
            json.dumps({"tool_name": name, "args": args, "iteration": iteration,
                        "result": result}, indent=2, default=str)
        )

    # -- the loop ----------------------------------------------------------
    def run(self, query: str, motif_hint: str | None = None,
            scaffold_path: str | None = None) -> str:
        """Design for ``query``; returns the final report text.

        Args:
            query: the target chemistry, in the user's words.
            motif_hint: optional notes on the catalytic motif. A hint, never a
                substitute for the agent deriving and justifying one.
            scaffold_path: optional holo structure. Supplying one makes
                redesign *possible*; it does not make it the right choice, and
                the agent is still expected to argue the strategy.
        """
        self.log(f"query: {query}")
        self.log(f"run_dir: {self.run_dir}")
        self._event("run_start", query=query, model=self.model_name,
                    tools=list(self.tool_names), scaffold=scaffold_path)

        system = build_system_prompt(available_tools=list(self.tool_names),
                                     capabilities=self.capabilities)
        user = [f"Target chemistry:\n{query}"]
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
        finalizing = False
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            self._event("iteration_start", iteration=iteration, strategy=self.strategy)

            if not finalizing and iteration >= self.max_iterations - FINALIZATION_RESERVE:
                self.log("entering finalization")
                messages.append({"role": "user", "content": FINALIZE_PROMPT})
                finalizing = True

            messages.append({"role": "user", "content": build_state_header(
                iteration, self.max_iterations, self.strategy, self.designs,
                self.tools_used)})
            if iteration > 1 and iteration % 6 == 0 and not finalizing:
                messages.append({"role": "user", "content": REFLECTION_PROMPT})

            response = self.llm.call(messages=messages, tools=self.tools,
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

            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if get_tool_info(name) is None:
                    out = {"status": "error",
                           "summary": f"{name} is not a design tool. Available: "
                                      f"{', '.join(self.tool_names)}"}
                else:
                    out = self._execute(name, args, iteration)
                    self.tools_used.append(name)
                body = json.dumps(compact_for_llm(out), default=str)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": body[:config.MAX_RESULT_CHARS]})

            (self.run_dir / "agent_state").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "agent_state" / "transcript.json").write_text(
                json.dumps({"iteration": iteration, "messages": messages},
                           indent=2, default=str))

        return self._finish(messages, usage, "Max iterations reached.",
                            status="max_iterations", iterations=iteration)

    # -- finish ------------------------------------------------------------
    def _finish(self, messages: list, usage: TokenUsage, final_text: str,
                status: str, iterations: int) -> str:
        report = final_text or "(no report text produced)"
        (self.run_dir / "DESIGN_REPORT.md").write_text(report + "\n")
        (self.run_dir / "token_usage.json").write_text(
            json.dumps(usage.to_dict(), indent=2, default=str))
        (self.run_dir / "manifest.json").write_text(json.dumps({
            "agent": "enzyme_design_agent",
            "status": status,
            "model": self.model_name,
            "strategy": self.strategy,
            "iterations_used": iterations,
            "designs": self.designs,
            "tools_used": self.tools_used,
            "tokens": usage.to_dict(),
        }, indent=2, default=str))
        self._event("run_complete", status=status, strategy=self.strategy,
                    iterations=iterations, designs=self.designs)
        self.log(f"done: status={status} strategy={self.strategy} "
                 f"iterations={iterations}")
        return report
