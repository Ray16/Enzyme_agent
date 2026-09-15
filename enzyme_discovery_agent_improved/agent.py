"""Core agent loop: thin orchestrator that delegates to specialized modules.

Reduced from ~1915 lines to ~450 lines by extracting:
- Gate logic -> gates/
- Data models -> models/
- Report/CSV/structure/image generation -> output/
- LLM call + token tracking -> llm_client.py
- Context compaction -> context.py
- Prompts -> prompts/
"""

from __future__ import annotations

import json
import logging
import threading
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from enzyme_discovery_agent_improved import config, provenance
from enzyme_discovery_agent_improved.bootstrap import register_builtin_tools
from enzyme_discovery_agent_improved.capabilities import create_run_dir, init_capabilities
from enzyme_discovery_agent_improved.context import ARTIFACT_CATEGORIES, compact_for_llm
from enzyme_discovery_agent_improved.gates import AgentState, GatePipeline
from enzyme_discovery_agent_improved.gates.ec_triage import ECTriageGate, auto_screen_candidates
from enzyme_discovery_agent_improved.gates.discovery_coverage import DiscoveryCoverageGate
from enzyme_discovery_agent_improved.gates.evidence_selection import EvidenceSelectionGate
from enzyme_discovery_agent_improved.gates.entity_extraction import EntityExtractionGate
from enzyme_discovery_agent_improved.gates.literature_harvest import LiteratureHarvestGate
from enzyme_discovery_agent_improved.gates.mechanism_family_coverage import MechanismFamilyGate
from enzyme_discovery_agent_improved.gates.family_survey import FamilySurveyGate
from enzyme_discovery_agent_improved.gates.homolog_expansion import HomologExpansionGate
from enzyme_discovery_agent_improved.gates.systematic_coverage import SystematicCoverageGate
from enzyme_discovery_agent_improved.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent_improved.models.candidate_registry import (
    current_candidate_status,
    extract_and_register_from_search,
    family_label,
    load_candidate_registry,
    register_structural_homologs,
)
from enzyme_discovery_agent_improved.models.target_profile import normalize_target_profile
from enzyme_discovery_agent_improved.output.finalizer import RunFinalizer
from enzyme_discovery_agent_improved.prompts import (
    EXPLORATIVE_RERANK_PROMPT,
    build_reflection_prompt,
    build_state_header,
    build_system_prompt,
)
from enzyme_discovery_agent_improved.tool_registry import (
    check_search_duplicate,
    clear_search_history,
    execute_tool,
    get_all_tools,
    get_parallel_safe_tools,
    get_tools_by_category,
    record_search,
)

warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")
warnings.filterwarnings("ignore", message=".*not removing hydrogen.*")

logger = logging.getLogger("enzyme_agent")


def _extract_reasoning(assistant_msg, text_content: str) -> str:
    """Best rationale available for a turn, for the ``assistant_decision`` event.

    On a tool-calling turn the ordinary ``content`` is usually empty -- the model
    puts no prose alongside the calls -- which is why the reasoning field read as
    blank in early runs.  Some providers surface the model's chain-of-thought in a
    separate ``reasoning_content`` field (or as ``thinking``/``reasoning`` blocks
    inside a structured content list); recover it so the "why" is captured when
    the provider offers it.  Falls back to the plain text content, then "".
    Purely defensive: any odd shape yields "".
    """
    text = (text_content or "").strip()
    if text:
        return text
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(assistant_msg, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Structured content list (e.g. [{"type": "thinking", "thinking": "..."}]).
    content = getattr(assistant_msg, "content", None)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            for key in ("thinking", "reasoning", "text"):
                val = block.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""


class EnzymeAgent:
    """LLM-driven enzyme discovery agent."""

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        output_dir: str | Path = "runs",
        max_iterations: int = config.MAX_ITERATIONS,
        llm_timeout: int = config.LLM_TIMEOUT,
        query: str = "",
        resume_run_dir: str | Path | None = None,
    ):
        register_builtin_tools()
        self.llm = LLMClient(model=model)
        self.model_name = model
        self.max_iterations = max_iterations
        self.llm_timeout = llm_timeout

        self.run_dir = (
            Path(resume_run_dir).resolve()
            if resume_run_dir else create_run_dir(Path(output_dir), query=query)
        )
        self._resume_requested = resume_run_dir is not None
        if resume_run_dir and not self.run_dir.is_dir():
            raise RuntimeError(f"Resume run directory not found: {self.run_dir}")
        self.artifacts_dir = self.run_dir / "artifacts"

        self.tools = get_all_tools()
        self.parallel_safe = get_parallel_safe_tools()
        self.capabilities = init_capabilities()

        self._artifact_counter = 0
        self._artifact_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._search_log: list[str] = []
        self._candidate_count = 0
        self._iteration_start: float = 0.0
        self._accumulated_papers: list[dict] = []
        self._literature_search_iterations: set[int] = set()
        self._confidence_report_text: str | None = None
        self._consecutive_no_tool_turns = 0
        self._notebook_tool_cooldown = 0
        self._iterations_used = 0
        # Family-coverage guard state: every search query issued (so we can tell
        # which derived families were actually drilled) and how many coverage
        # nudges we have spent (bounded so coverage is encouraged, never a hard
        # block that could stall the run).
        self._search_queries: list[str] = []
        self._coverage_nudges_used = 0

        self._gate_state = AgentState(run_dir=self.run_dir)
        self._systematic_gate = SystematicCoverageGate()
        self._literature_harvest_gate = LiteratureHarvestGate(self.llm, self.log)
        self._mechanism_family_gate = MechanismFamilyGate(self.llm, self.log)
        self._family_survey_gate = FamilySurveyGate(self.log)
        self._evidence_selection_gate = EvidenceSelectionGate()
        self._gate_pipeline = GatePipeline([
            EntityExtractionGate(),
            DiscoveryCoverageGate(),
            ECTriageGate(),
            HomologExpansionGate(),
        ])
        # Gates that fire (in this order) on every discovery turn, before the
        # pre-evaluation pipeline.  Each conforms to the Gate protocol, so they run
        # through one shared intercept ritual (_apply_gate_result) instead of the
        # copy-pasted pop/extend/iteration-=1 block that used to sit inline per gate.
        self._early_gates = [
            ("literature_harvest", self._literature_harvest_gate),
            ("mechanism_family", self._mechanism_family_gate),
            ("family_survey", self._family_survey_gate),
            ("evidence_selection", self._evidence_selection_gate),
            ("systematic_coverage", self._systematic_gate),
        ]

        self._event_log_path = self.run_dir / "events.jsonl"
        self._setup_logging()

    def _load_resume_payload(self) -> dict | None:
        """Restore durable loop state for an explicitly resumed run."""
        manifest_path = self.run_dir / "manifest.json"
        transcript_path = self.run_dir / "agent_state" / "transcript.json"
        if not getattr(self, "_resume_requested", False):
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
            transcript = json.loads(transcript_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Run cannot be resumed: missing or invalid checkpoint ({exc})") from exc
        if manifest.get("status") != "running":
            raise RuntimeError(
                f"Run cannot be resumed because manifest status is {manifest.get('status')!r}"
            )
        messages = transcript.get("messages")
        if not isinstance(messages, list):
            raise RuntimeError("Run cannot be resumed: checkpoint has no message list")
        try:
            context = json.loads((self.run_dir / "target_context.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Run cannot be resumed: missing or invalid target context ({exc})") from exc
        if not isinstance(context.get("target_profile"), dict):
            raise RuntimeError("Run cannot be resumed: target context has no target profile")

        events = []
        try:
            events = [json.loads(line) for line in (self.run_dir / "events.jsonl").read_text().splitlines()
                      if line.strip()]
        except (OSError, json.JSONDecodeError):
            pass
        gates = {e.get("gate") for e in events if e.get("type") == "gate"}
        has_candidates = bool(load_candidate_registry(self.run_dir).get("candidates"))
        state_flags = {
            "entity_extraction_done": has_candidates,
            "literature_harvest_done": "literature_harvest" in gates,
            "mechanism_families_done": "mechanism_family" in gates,
            "family_survey_done": False,
            "evidence_selection_prompted": "evidence_selection" in gates,
            "eval_gate_fired": "pre_evaluation" in gates,
            "critique_injected": "pre_evaluation" in gates,
            "homolog_expansion_done": "homolog_expansion" in gates,
        }
        for key, value in state_flags.items():
            setattr(self._gate_state, key, value)
        self._search_queries = [
            str((e.get("args") or {}).get("query") or "")
            for e in events if e.get("type") == "tool_call"
        ]
        self._search_log = [
            f"{e.get('tool')}:{str((e.get('args') or {}).get('query') or '')[:30]}"
            for e in events if e.get("type") == "tool_call"
        ]
        clear_search_history()
        for event in events:
            if event.get("type") == "tool_call" and event.get("tool"):
                record_search(event["tool"], event.get("args") or {}, {})
        calls = [e for e in events if e.get("type") == "llm_call"]
        usage = TokenUsage(model=self.model_name)
        usage.calls = [{
            "iteration": e.get("iteration"),
            "prompt_tokens": e.get("prompt_tokens", 0),
            "completion_tokens": e.get("completion_tokens", 0),
            "total_tokens": e.get("total_tokens", 0),
        } for e in calls]
        usage.total_prompt_tokens = sum(c["prompt_tokens"] for c in usage.calls)
        usage.total_completion_tokens = sum(c["completion_tokens"] for c in usage.calls)
        usage.total_tokens = sum(c["total_tokens"] for c in usage.calls)
        return {
            "manifest": manifest,
            "target_context": context,
            "messages": messages,
            "iteration": int(transcript.get("iteration") or 0),
            "token_usage": usage,
        }

    def _setup_logging(self):
        root_logger = logging.getLogger("enzyme_agent")
        if not root_logger.handlers:
            root_logger.setLevel(logging.DEBUG)
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter(
                "%(asctime)s | %(message)s", datefmt="%H:%M:%S"
            ))
            root_logger.addHandler(console)
        file_handler = logging.FileHandler(self.run_dir / "agent_state" / "warnings.log")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        root_logger.addHandler(file_handler)

    def log(self, msg: str, level: str = "info"):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        with self._log_lock:
            with open(self.run_dir / "console_log.txt", "a") as f:
                f.write(entry + "\n")
        if level == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)

    def _log_event(self, event_type: str, data: dict | None = None):
        # Delegate to the canonical, module-level event stream so the agent,
        # gates, and the registry all write one source-of-truth JSONL with a
        # shared lock and a monotonic per-run ``seq``.  Best-effort inside.
        from enzyme_discovery_agent_improved.models.event_log import append_event
        append_event(self.run_dir, event_type, **(data or {}))

    def _log_gate(self, gate: str, result, iteration: int):
        """Log a gate that fired: to the human console (as before) and to the
        canonical event stream, so a redirect that reshaped the run is visible
        in the replayable record instead of only in console prose."""
        msg = getattr(result, "log_message", None)
        if msg:
            self.log(f"  Gate: {msg}")
        self._log_event("gate", {
            "gate": gate,
            "message": msg,
            "intercept": bool(getattr(result, "intercept", False)),
            "iteration": iteration,
        })

    def _apply_gate_result(self, name, result, iteration, messages):
        """Apply a gate/pipeline result and run the intercept ritual once.

        Always applies ``state_updates`` (a non-intercepting gate can still report
        a state transition).  If the result intercepts, it POPS the assistant turn,
        injects the gate's messages, and refunds the iteration credit so the gate's
        guidance is answered without consuming a discovery step.  Returns
        ``(iteration, intercepted)``; the caller ``continue``s the loop when
        ``intercepted`` is True.  ``result`` may be None (gate declined).
        """
        if result is None:
            return iteration, False
        for k, v in result.state_updates.items():
            setattr(self._gate_state, k, v)
        if not result.intercept:
            return iteration, False
        self._log_gate(name, result, iteration)
        messages.pop()
        messages.extend(result.new_messages)
        self.log("  Gate transition: iteration credit restored")
        return iteration - 1, True

    def _checkpoint_transcript(self, messages: list, iteration: int) -> None:
        """Persist the full conversation each iteration (crash-safe provenance).

        The finalizer writes ``messages.json`` only at the end of a run, so an
        interrupted campaign would otherwise lose its entire reasoning trace.
        This rolling checkpoint is both a provenance record and the foundation
        for future resume support.  Best-effort: never breaks the loop.
        """
        try:
            payload = {"iteration": iteration, "messages": messages}
            (self.run_dir / "agent_state" / "transcript.json").write_text(
                json.dumps(payload, default=str)
            )
        except (OSError, TypeError) as exc:
            logger.warning("Failed to checkpoint transcript at iteration %d: %s",
                           iteration, exc)

    def _ensure_artifacts_dir(self) -> None:
        for subdir in ("literature", "database", "candidates", "fulltext", "evidence"):
            (self.artifacts_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _save_artifact(self, category: str, tool_name: str, result: dict,
                       tool_args: dict, iteration: int):
        with self._artifact_lock:
            self._artifact_counter += 1
            counter = self._artifact_counter
        filename = f"{tool_name}_{counter:03d}.json"
        out_path = self.artifacts_dir / category / filename
        artifact = {
            "tool_name": tool_name, "args": tool_args,
            "timestamp": datetime.now().astimezone().isoformat(),
            "iteration": iteration, "result": result,
        }
        try:
            with open(out_path, "w") as f:
                json.dump(artifact, f, indent=2, default=str)
        except FileNotFoundError:
            self._ensure_artifacts_dir()
            try:
                with open(out_path, "w") as f:
                    json.dump(artifact, f, indent=2, default=str)
            except Exception as e:
                logger.warning("Failed to save artifact %s after retry: %s", filename, e)
        except Exception as e:
            logger.warning("Failed to save artifact %s: %s", filename, e)

    def _execute_single_tool(self, tool_name: str, tool_args: dict, iteration: int) -> dict:
        t0 = _time.time()

        dup = check_search_duplicate(tool_name, tool_args)
        if dup:
            self.log(f"    ~ {tool_name}: deduplicated (similar search already done)")
            self._log_event("tool_dedup", {"tool": tool_name, "args": tool_args})
            return dup

        query_preview = tool_args.get("query", tool_args.get("ec_number",
                        tool_args.get("accession", tool_args.get("smiles", ""))))
        self.log(f"    -> {tool_name}({str(query_preview)[:80]})")

        injected = {"run_dir": str(self.run_dir), "llm_client": self.llm}
        result = execute_tool(tool_name, tool_args, injected_args=injected)
        result = json.loads(json.dumps(result, default=str))

        elapsed = _time.time() - t0
        record_search(tool_name, tool_args, result)

        # Accumulate every search's query text so the family-coverage guard can
        # tell which derived families were actually drilled vs only named.
        _q = tool_args.get("query") or tool_args.get("term") or tool_args.get("terms")
        if isinstance(_q, (list, tuple)):
            _q = " ".join(str(x) for x in _q)
        if _q:
            self._search_queries.append(str(_q))

        if tool_name in ARTIFACT_CATEGORIES:
            query_desc = tool_args.get("query", tool_args.get("ec_number", tool_args.get("accession", "")))
            self._search_log.append(f"{tool_name}:{query_desc[:30]}")
        if tool_name == "evaluate_candidate":
            self._candidate_count += 1
        if tool_name in ("search_pubmed", "search_openalex", "search_europepmc",
                         "search_semantic_scholar", "search_crossref"):
            papers = result.get("data", [])
            if isinstance(papers, list):
                self._accumulated_papers.extend(papers)
                self._literature_search_iterations.add(iteration)
        if tool_name == "deep_literature_search":
            refs = result.get("data", {})
            if isinstance(refs, dict):
                for paper_list in refs.values():
                    if isinstance(paper_list, list):
                        self._accumulated_papers.extend(paper_list)
                self._literature_search_iterations.add(iteration)

        category = ARTIFACT_CATEGORIES.get(tool_name)
        if category:
            self._save_artifact(category, tool_name, result, tool_args, iteration)

        # Broad-search then expand-within-family: register every distinct enzyme
        # a search surfaces as a *discovered* lead so none is lost to the LLM's
        # ad-hoc downselection.  These are leads, not final candidates — the
        # auto-screen step consolidates orthologs to one gene representative and
        # auto-evaluate triages each on mechanism feasibility, so the registry
        # never becomes an unreviewed candidate pool.  (Structural homologs carry
        # their own provenance and are registered separately.)
        if tool_name == "search_uniprot" and result.get("status") in ("success", "partial"):
            extract_and_register_from_search(self.run_dir, tool_name, result)
        if tool_name == "search_structural_homologs" and result.get("status") == "success":
            register_structural_homologs(self.run_dir, result)

        status = result.get("status", "?")
        summary = result.get("summary", "")
        count = result.get("count", "")
        status_icon = "+" if status == "success" else "~" if status == "partial" else "x"
        count_str = f" [{count} results]" if count else ""
        self.log(f"    {status_icon} {summary[:100]}{count_str} ({elapsed:.1f}s)")

        tool_event = {
            "tool": tool_name, "args": tool_args, "status": status,
            "count": count or 0, "elapsed_s": round(elapsed, 2), "iteration": iteration,
        }
        # Carry search novelty onto the tool_call itself so every search records
        # how much was actually new, not just how many hits it returned.
        if isinstance(result.get("novelty"), dict):
            tool_event["novelty"] = result["novelty"]
        self._log_event("tool_call", tool_event)
        return result

    def _uncovered_promising_families(self) -> list[dict]:
        """Promising derived families that no search has drilled yet.

        Reads the run's own ``mechanism_families.json`` and compares each
        promising family's distinctive terms against every search query issued.
        Returns the families still un-searched, so the loop can steer the agent to
        DIFFERENT families before it finalizes.  Best-effort: any read/parse
        failure yields an empty list (the guard simply does not fire)."""
        try:
            families = json.loads(
                (self.run_dir / "mechanism_families.json").read_text()
            )
        except (OSError, ValueError, TypeError):
            return []
        try:
            from enzyme_discovery_agent_improved.tools.mechanism_families import (
                unexplored_promising_families,
            )
            return unexplored_promising_families(
                families, getattr(self, "_search_queries", [])
            )
        except Exception as exc:  # noqa: BLE001 - guard must never break the loop
            logger.warning("coverage guard failed: %s", exc)
            return []

    def _infer_phase(self, iteration: int) -> str:
        if iteration <= 3:
            return "reaction_analysis"
        if self._candidate_count == 0:
            return "discovery"
        if self._candidate_count < 5:
            return "candidate_evaluation"
        return "ranking_and_report"

    def _should_continue(self, iteration: int) -> bool:
        # Keep exploring for at least ~half the iteration budget before allowing
        # an early finalization.  A literature-first pass reaches a handful of
        # coarse families quickly, but the budget is meant for BROADER and DEEPER
        # coverage — searching each derived mechanism family and its specific
        # members (the target-substrate-matched one).  Stopping at iter ~9 of 30
        # leaves the run shallow.  A genuinely broad pool may still finalize.
        if iteration < self.max_iterations // 2 and self._candidate_count < 20:
            return True
        if self._candidate_count == 0:
            return iteration < self.max_iterations - 2
        if self._candidate_count < 3:
            return iteration < self.max_iterations - 2
        if self._candidate_count < 10:
            return iteration < self.max_iterations - 1
        return False

    def _sync_gate_state(self):
        """Sync mutable agent state into the gate state dataclass."""
        # Candidates are registered by database and structural-homolog tools,
        # not only by explicit evaluate_candidate calls.  Use the registry as
        # the source of truth so phase transitions cannot remain in discovery
        # after candidates have already been found.
        registry = load_candidate_registry(self.run_dir).get("candidates", {})
        # Raw registry size can include many close homologs. Phase decisions
        # should reflect distinct coarse families, not an unbounded hit count.
        active_candidates = {
            acc: info for acc, info in registry.items()
            if current_candidate_status(info) in {"screened", "evaluated", "ranked"}
        }
        self._candidate_count = len({
            family_label(info.get("metadata", {}), acc)
            for acc, info in active_candidates.items()
        })
        self._gate_state.candidate_count = self._candidate_count
        # Count DISTINCT papers for gate sufficiency: now that several sources
        # feed the pool, the same paper arrives multiple times and must not
        # inflate the "enough literature" signal (which would stop search early).
        from enzyme_discovery_agent_improved.tools.literature import dedupe_and_merge
        unique_papers, _ = dedupe_and_merge(self._accumulated_papers)
        self._gate_state.accumulated_papers = unique_papers
        self._gate_state.lit_search_count = len(
            getattr(self, "_literature_search_iterations", set())
        )

    def run(
        self,
        query: str,
        substrate_smiles: str | None = None,
        substrate_name: str | None = None,
        target_profile: dict | None = None,
    ) -> str:
        resume = self._load_resume_payload()
        if resume:
            query = str(resume["manifest"].get("query") or query)
            saved_context = resume["target_context"]
            substrate_smiles = saved_context.get("substrate_smiles")
            substrate_name = saved_context.get("substrate_name")
            target_profile = saved_context.get("target_profile")
        self.log(f"  Run ID:    {self.run_dir.name}")
        self.log(f"  Model:     {self.model_name}")
        self.log(f"  Query:     {query}")
        if substrate_smiles:
            self.log(f"  Substrate: {substrate_smiles}")
        self._log_event("run_resume" if resume else "run_start", {
            "query": query, "substrate": substrate_smiles, "model": self.model_name,
            **({"from_iteration": resume["iteration"]} if resume else {}),
        })

        # Record run identity before substrate resolution or any model call so a
        # setup failure still leaves an auditable, resumable run record.
        if not resume:
            provenance.write_manifest_start(
                self.run_dir,
                query=query,
                model=self.model_name,
                config_module=config,
                args={"max_iterations": self.max_iterations, "llm_timeout": self.llm_timeout},
                substrate={"smiles": substrate_smiles, "name": substrate_name},
                capabilities=self.capabilities,
            )

        self._query = query
        # Derive target-substrate SMILES when the caller did not supply them:
        # without a target structure the substrate-accommodation axis is inert
        # and ranking collapses to annotation depth (fame), unable to tell an
        # enzyme whose substrate fits the target from one built for a bulky or
        # tethered substrate.  The substrate names come from the user's own
        # query, so resolving them is task setup, not answer leakage.
        if not substrate_smiles and not resume:
            try:
                from enzyme_discovery_agent_improved.chemistry_logic import (
                    resolve_substrate_names_to_smiles,
                    reason_target_substrate_smiles,
                )
                # The agent reasons about the reaction + substrates at the outset,
                # so REASON the target-substrate SMILES from the query (a whole
                # sentence does not resolve to a structure).  Fall back to a
                # name->PubChem lookup on the supplied name or raw text.
                derived = None
                if not substrate_name:
                    derived = reason_target_substrate_smiles(query, self.llm)
                if not derived:
                    derived = resolve_substrate_names_to_smiles(substrate_name or query)
                if derived:
                    substrate_smiles = derived
                    self.log(f"  Derived substrate SMILES: {substrate_smiles}")
            except Exception as exc:  # never block a run on resolution
                logger.warning("Substrate SMILES derivation failed: %s", exc)
        self._target_context = saved_context if resume else {
            "query": query,
            "substrate_smiles": substrate_smiles,
            "substrate_name": substrate_name,
            "target_profile": normalize_target_profile(target_profile),
        }
        # Token accounting starts here so the contract-derivation LLM call below
        # is counted like every other call (it is not part of the main loop).
        token_usage = resume["token_usage"] if resume else TokenUsage(model=self.model_name)
        # Derive the shared chemistry contract ONCE, at run start, from the target
        # only.  Every downstream stage reads its required-chemistry facts from
        # this single contract instead of re-deriving them independently, so no
        # stage can invent a different account of what chemistry is required.
        try:
            from enzyme_discovery_agent_improved.models.target_reasoning import derive_target_reasoning
            target_reasoning = (json.loads((self.run_dir / "target_reasoning.json").read_text())
                                if resume else derive_target_reasoning(
                query=query,
                substrate_smiles=substrate_smiles,
                substrate_name=substrate_name,
                llm_client=self.llm,
                target_profile=self._target_context["target_profile"],
                # Account for the derivation's tokens in the same ledger as the
                # main loop, so cost is complete and auditable.
                on_response=lambda resp: token_usage.record("contract", resp),
            ))
            self._target_context["target_reasoning"] = target_reasoning
            (self.run_dir / "target_reasoning.json").write_text(
                json.dumps(target_reasoning, indent=2, default=str) + "\n"
            )
            self.log(
                "  Target chemistry: "
                f"direction={target_reasoning.get('reaction_direction')}, "
                f"bonds={target_reasoning.get('bond_types')}, "
                f"topology={target_reasoning.get('substrate_topology')}"
            )
            # Make the contract visible in the canonical event stream: a
            # dedicated event for the derived facts, plus a standard llm_call
            # event so existing cost tooling counts its tokens.
            contract_call = (token_usage.calls[-1]
                             if token_usage.calls and token_usage.calls[-1].get("iteration") == "contract"
                             else {})
            if not resume:
                self._log_event("contract_derived", {
                "reaction_direction": target_reasoning.get("reaction_direction"),
                "bond_types": target_reasoning.get("bond_types"),
                "substrate_topology": target_reasoning.get("substrate_topology"),
                "reactive_groups": target_reasoning.get("reactive_groups"),
                "n_claims": len(target_reasoning.get("claims") or []),
                "n_inferred": sum(1 for c in target_reasoning.get("claims") or []
                                  if c.get("status") == "inferred"),
                **contract_call,
                })
            if contract_call and not resume:
                self._log_event("llm_call", {
                    "iteration": 0,
                    "phase": "contract_derivation",
                    "prompt_tokens": contract_call.get("prompt_tokens", 0),
                    "completion_tokens": contract_call.get("completion_tokens", 0),
                    "total_tokens": contract_call.get("total_tokens", 0),
                    "cumulative_tokens": token_usage.total_tokens,
                })
        except Exception as exc:  # never block a run on contract derivation
            logger.warning("Target-reasoning contract derivation failed: %s", exc)
        # Gates run independently of the live agent instance. Persist the
        # structured target once so automatic evaluation never has to infer it
        # from a truncated notebook paragraph.
        (self.run_dir / "target_context.json").write_text(
            json.dumps(self._target_context, indent=2, default=str) + "\n"
        )
        if not resume:
            provenance.refresh_manifest_target_reasoning(self.run_dir)
        if not resume:
            clear_search_history()
        self._ensure_artifacts_dir()

        system_prompt = build_system_prompt(
            available_tools=list(get_tools_by_category().keys()),
            capabilities=self.capabilities,
        )

        user_msg = query
        profile = self._target_context["target_profile"]
        if target_profile is not None:
            user_msg += (
                "\n\nTarget profile (constraints, not a scoring rubric):\n"
                + json.dumps({
                    "reaction": profile["reaction"],
                    "substrate": profile["substrate"],
                    "enzyme_requirements": profile["enzyme_requirements"],
                    "required_evidence": profile["required_evidence"],
                    "review_requirements": profile["review_requirements"],
                    "discovery_coverage": profile["discovery_coverage"],
                }, indent=2)
            )
        if substrate_smiles:
            user_msg += f"\n\nSubstrate SMILES: {substrate_smiles}"
        if substrate_name:
            user_msg += f"\nSubstrate name: {substrate_name}"
        user_msg += f"\n\nOutput directory: {self.run_dir}"

        messages = resume["messages"] if resume else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        # ``iteration`` is a budget for substantive model/tool work.  Gate
        # interceptions only redirect orchestration and explicitly restore
        # their credit below, so they cannot silently shorten a campaign.
        iteration = resume["iteration"] if resume else 0
        while iteration < self.max_iterations:
            iteration += 1
            self._iterations_used = iteration
            self._iteration_start = _time.time()
            # Tool calls in the preceding iteration may have promoted a
            # candidate.  Synchronize before announcing state or asking the
            # LLM for another turn, so the evaluation gate cannot miss a
            # screen that occurred near the iteration limit.
            self._sync_gate_state()
            self._log_event("iteration_start", {
                "iteration": iteration,
                "phase": self._infer_phase(iteration),
                "candidate_count": self._candidate_count,
            })

            completion_iteration = max(
                1, self.max_iterations - config.FINALIZATION_RESERVE_ITERATIONS + 1
            )
            if iteration >= completion_iteration:
                self.log(
                    "  Completion reserve reached; closing discovery and generating evaluated reports."
                )
                return self._finalize(
                    messages, token_usage,
                    "Completion reserve reached",
                    force_completion=True,
                )

            state_header = build_state_header(
                iteration=iteration, max_iterations=self.max_iterations,
                candidate_count=self._candidate_count,
                searches_done=self._search_log,
                phase=self._infer_phase(iteration),
            )

            reflection_msg = None
            if iteration > 1 and iteration % config.REFLECTION_INTERVAL == 0:
                reflection_msg = build_reflection_prompt(iteration, self._candidate_count)

            if reflection_msg:
                messages.append({"role": "user", "content": f"{state_header}\n\n{reflection_msg}"})
            elif iteration > 1:
                messages.append({"role": "user", "content": state_header})

            available_tools = self.tools
            # Early turns must establish chemistry and retrieve evidence.
            # Delaying optional notebook writes avoids provider-generated
            # section-only calls before there is any evidence to summarize.
            if iteration <= 3 or self._notebook_tool_cooldown:
                available_tools = [
                    tool for tool in self.tools
                    if tool["function"]["name"] != "update_notebook"
                ]
                self._notebook_tool_cooldown -= 1
            if self._gate_state.discovery_coverage_incomplete:
                # The coverage gate is a hard evaluation boundary, but it
                # must not discard a whole model turn.  Keep all discovery
                # tools available so the model can close evidence gaps.
                available_tools = [
                    tool for tool in available_tools
                    if tool["function"]["name"] != "evaluate_candidate"
                ]
            if self._gate_state.evidence_selection_prompted and self._candidate_count == 0:
                # Once enough literature is gathered, do not spend the
                # remaining campaign rediscovering the same landscape. Resolve
                # identifiers and screen the bounded evidence-backed list.
                selection_tools = {
                    "screen_candidates", "search_uniprot", "lookup_genbank_to_uniprot",
                    "get_notebook",
                }
                available_tools = [
                    tool for tool in available_tools
                    if tool["function"]["name"] in selection_tools
                ]
            # Cap prompt growth: the re-sent history is 97% of token spend and
            # grows unboundedly as raw tool dumps accumulate.  Everything a later
            # turn needs is already persisted (candidate registry, notebook,
            # artifacts), so stub the CONTENT of tool results older than the most
            # recent few turns.  Messages are kept (never removed) so tool_call /
            # tool-result pairing stays intact.
            self._prune_history(messages)
            response = self.llm.call(
                messages=messages, tools=available_tools,
                timeout=self.llm_timeout,
            )
            if response is None:
                return self._finalize(messages, token_usage,
                                      f"LLM failed at iteration {iteration}",
                                      status="failed")

            token_usage.record(iteration, response)
            # Per-call token cost into the canonical stream: a 100-iteration
            # campaign's spend is otherwise only a single total at finalize.
            if token_usage.calls:
                last = token_usage.calls[-1]
                self._log_event("llm_call", {
                    "iteration": iteration,
                    "prompt_tokens": last.get("prompt_tokens", 0),
                    "completion_tokens": last.get("completion_tokens", 0),
                    "total_tokens": last.get("total_tokens", 0),
                    "cumulative_tokens": token_usage.total_tokens,
                })

            choice = response.choices[0]
            assistant_msg = choice.message
            messages.append(assistant_msg.model_dump())

            # Normalize provider quirks at the client boundary: recovered tool
            # calls (leaked into text by some Argo models) come back in a uniform
            # shape so the loop below never has to know how they were serialized.
            tool_calls, text_content, recovered = self.llm.extract_tool_calls(assistant_msg)
            if recovered:
                self.log(f"Recovered {len(tool_calls)} tool calls from text")
                messages[-1]["tool_calls"] = [
                    {"id": r["id"], "type": "function",
                     "function": {"name": r["function"]["name"],
                                  "arguments": r["function"]["arguments"]}}
                    for r in tool_calls
                ]
                messages[-1]["content"] = text_content

            self.log(f"\n  ITERATION {iteration}/{self.max_iterations}  |  "
                     f"{len(tool_calls)} tool calls  |  "
                     f"{self._candidate_count} candidates")

            # Record the model's decision for THIS turn -- the reasoning text and
            # the batch of tools it chose -- so the "why" behind the actions is a
            # first-class event, not something buried in the raw transcript.
            def _tc_name(tc):
                return tc.function.name if hasattr(tc, "function") else tc["function"]["name"]
            reasoning = _extract_reasoning(assistant_msg, text_content)
            self._log_event("assistant_decision", {
                "iteration": iteration,
                "n_tool_calls": len(tool_calls),
                "tools": [_tc_name(tc) for tc in tool_calls],
                "reasoning": reasoning[:800],
                # Distinguish "the model gave no rationale" from "we failed to
                # capture one": most tool-calling turns carry empty content, so an
                # empty reasoning field is expected, not a logging bug.
                "reasoning_source": (
                    "text" if (text_content or "").strip()
                    else "thinking" if reasoning
                    else "none"
                ),
            })

            # No tool calls: check if done
            if not tool_calls:
                self._consecutive_no_tool_turns += 1
                # No-progress guard (let the agent's own decision stand): making no
                # tool call is how the model says it is finished.  Allow ONE recovery
                # nudge in case it paused mid-phase, but once it already has results
                # and still makes no tool call, stop forcing continuation — otherwise
                # `_should_continue` keeps returning True (candidate count below its
                # exploration threshold) and the loop re-invites an idle model every
                # iteration until the mid-run boundary, burning most of a run's
                # tokens.  The bounded coverage checkpoint below still runs.
                stalled = (self._consecutive_no_tool_turns >= config.MAX_NO_PROGRESS_TURNS
                           and self._candidate_count > 0)
                if stalled:
                    self.log(
                        f"  No-progress: model made {self._consecutive_no_tool_turns} "
                        f"consecutive turns with no tool call and "
                        f"{self._candidate_count} candidates found; proceeding to "
                        "coverage check / finalization."
                    )
                if self._should_continue(iteration) and not stalled:
                    if self._candidate_count == 0:
                        recovery_instruction = (
                            "You have not begun evidence-backed discovery. Take a concrete tool action now: "
                            "analyze the substrate or search primary literature for direct target/substrate evidence. "
                            "Do not return a plan or prose-only answer. Call at least one tool."
                        )
                    else:
                        recovery_instruction = "You stopped before completing all phases. Continue with concrete tool calls."
                    messages.append({
                        "role": "user",
                        "content": recovery_instruction,
                    })
                    continue
                # Coverage guard: the agent thinks it is done, but do not finalize
                # while a family it ITSELF derived as promising was never searched.
                # Nudge a bounded number of times with that family's own terms,
                # steering budget to DIFFERENT families instead of more orthologs.
                uncovered = self._uncovered_promising_families()
                if uncovered and self._coverage_nudges_used < config.MAX_FAMILY_COVERAGE_NUDGES:
                    self._coverage_nudges_used += 1
                    lines = []
                    for fam in uncovered[:4]:
                        terms = "; ".join(
                            fam.get("literature_terms") or fam.get("descriptors") or []
                        )
                        lines.append(f"- {fam.get('name', 'family')}: {terms}")
                    self.log(
                        f"  Coverage checkpoint: {len(uncovered)} promising family(ies) "
                        f"never searched; nudging targeted search "
                        f"({self._coverage_nudges_used}/{config.MAX_FAMILY_COVERAGE_NUDGES})"
                    )
                    self._log_event("coverage_checkpoint", {
                        "iteration": iteration,
                        "uncovered_families": [f.get("name") for f in uncovered],
                        "nudge": self._coverage_nudges_used,
                    })
                    messages.append({"role": "user", "content": (
                        "COVERAGE CHECKPOINT — do not finalize yet. You derived these "
                        "mechanism families as promising but never searched them. Run a "
                        "TARGETED search for EACH (search_uniprot and/or primary "
                        "literature with the terms below), then screen only the grounded "
                        "hits. Do NOT expand more orthologs of families you already have "
                        "— reach these DIFFERENT families:\n" + "\n".join(lines)
                    )})
                    continue
                self._consecutive_no_tool_turns = 0
                if self._confidence_report_text is None:
                    self._confidence_report_text = text_content
                    self.log("  Confidence report captured, requesting explorative re-ranking...")
                    messages.append({"role": "user", "content": EXPLORATIVE_RERANK_PROMPT})
                    continue
                return self._finalize(messages, token_usage, text_content)

            self._consecutive_no_tool_turns = 0

            # Systematic coverage gate (fires during discovery, independent of eval calls)
            has_eval_calls = any(
                (tc.function.name if hasattr(tc, "function") else tc["function"]["name"])
                == "evaluate_candidate"
                for tc in tool_calls
            )
            # A set_family_focus call is the agent acting on the family landscape.
            # The pre-evaluation gates below intercept by POPPING the assistant
            # message, which would discard this call before it executes — so the
            # agent's expand/stop decision would silently never land.  When the
            # turn carries a focus call, let NO gate intercept it: the tool runs,
            # the decision persists to family_decisions.json, and the eval gates
            # fire on the NEXT turn with the stops already in place.
            has_focus_call = any(_tc_name(tc) == "set_family_focus" for tc in tool_calls)

            self._sync_gate_state()
            # Deterministic pre-evaluation gates, all skipped on a focus-call turn
            # (see has_focus_call above) so the agent's set_family_focus tool runs
            # and its stop/expand decisions persist before the eval gates spend
            # budget — they fire on the NEXT turn with the stops in place.  The early
            # gates (literature-first harvest → mechanism-family derivation → family
            # survey → evidence selection → systematic coverage) fire every discovery
            # turn; the pre-evaluation pipeline fires only once there is something to
            # evaluate.  Every gate runs through the one shared intercept ritual.
            if not has_focus_call:
                intercepted = False
                for name, gate in self._early_gates:
                    if not gate.should_fire(self._gate_state, has_eval_calls, messages):
                        continue
                    result = gate.execute(self._gate_state, messages)
                    iteration, intercepted = self._apply_gate_result(
                        name, result, iteration, messages)
                    if intercepted:
                        break
                if intercepted:
                    continue
                if (self._gate_state.candidate_count > 0
                        or has_eval_calls
                        or self._gate_state.critique_injected):
                    result = self._gate_pipeline.check(
                        self._gate_state, has_eval_calls, messages)
                    iteration, intercepted = self._apply_gate_result(
                        "pre_evaluation", result, iteration, messages)
                    if intercepted:
                        continue

            # Parse and execute tool calls
            parsed = []
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    name, args_raw, tc_id = tc.function.name, tc.function.arguments, tc.id
                else:
                    name = tc["function"]["name"]
                    args_raw, tc_id = tc["function"]["arguments"], tc["id"]
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, TypeError) as e:
                    args = {"_parse_error": str(e), "_raw": str(args_raw)[:200]}
                parsed.append((tc_id, name, args))

            parallel = [(tc_id, n, a) for tc_id, n, a in parsed if n in self.parallel_safe]
            sequential = [(tc_id, n, a) for tc_id, n, a in parsed if n not in self.parallel_safe]

            results: dict[str, str] = {}
            max_result_chars = config.MAX_RESULT_CHARS

            def _exec(tc_id: str, name: str, args: dict):
                result = self._execute_single_tool(name, args, iteration)
                result_for_llm = compact_for_llm(result)
                result_str = json.dumps(result_for_llm, default=str)
                if len(result_str) > max_result_chars:
                    result_str = (
                        f"[NOTE: Result truncated to {max_result_chars} chars. "
                        f"Full data saved to artifacts.]\n"
                        + result_str[:max_result_chars]
                    )
                results[tc_id] = result_str

            if len(parallel) > 1:
                with ThreadPoolExecutor(max_workers=config.TOOL_THREAD_POOL_WORKERS) as pool:
                    futures = [pool.submit(_exec, *p) for p in parallel]
                    for f in futures:
                        f.result()
            else:
                for p in parallel:
                    _exec(*p)
            for s in sequential:
                _exec(*s)

            for tc_id, name, args in parsed:
                messages.append({
                    "role": "tool", "tool_call_id": tc_id,
                    "content": results[tc_id],
                })

            if any(name == "update_notebook" and not str(args.get("content", "")).strip()
                   for _, name, args in parsed):
                # Prevent a malformed notebook-only response from burning all
                # remaining iterations.  The next two turns cannot select the
                # write tool and must take a substantive discovery action.
                self._notebook_tool_cooldown = max(self._notebook_tool_cooldown, 2)
                messages.append({
                    "role": "user",
                    "content": (
                        "Your notebook call had no content and was safely skipped. Do not retry it now; "
                        "continue with a substantive discovery action. Only write a notebook note after "
                        "you have concise, evidence-backed content to preserve."
                    ),
                })

            # Make newly screened candidates visible immediately.  The next
            # iteration will deterministically enter the evaluation gate even
            # when screening happened in the final permitted tool turn.
            self._sync_gate_state()
            self._checkpoint_transcript(messages, iteration)

        self.log("Max iterations reached.")
        return self._finalize(messages, token_usage, "Max iterations reached")

    def _prune_history(
        self,
        messages: list,
        keep_recent_tools: int = config.HISTORY_KEEP_RECENT_TOOL_MSGS,
        keep_recent_assistant: int = config.HISTORY_KEEP_RECENT_ASSISTANT_MSGS,
        stub_over: int = config.HISTORY_STUB_OVER_CHARS,
    ) -> None:
        """Bound the re-sent prompt by stubbing SUPERSEDED history in place.

        The prompt re-sent each iteration is the dominant cost (~97% of loop
        spend), growing as the transcript accumulates.  Three categories carry no
        forward-looking information once superseded, because what the agent needs
        next is preserved elsewhere:

          * tool results  — candidates are in the registry, full data under
            artifacts/, so an old raw dump is never re-read;
          * assistant TEXT — the ACTION is recorded by its ``tool_calls`` (kept
            intact for pairing); only the verbose narration is dropped, and the
            running state lives in the state header;
          * state headers — each iteration injects a fresh cumulative snapshot,
            so every earlier one is strictly superseded — keeping only the latest
            is lossless.

        Recent turns of each kind stay full so the model reasons over fresh
        context.  In place and idempotent (recomputed each turn); the complete,
        un-stubbed data for every tool call remains on disk under artifacts/ for
        audit, so nothing is truly lost.
        """
        def _stub_old(positions: list[int], stub: str, keep: int) -> None:
            if len(positions) <= keep:
                return
            for i in positions[:-keep]:
                c = messages[i].get("content")
                if isinstance(c, str) and len(c) > stub_over and c != stub:
                    messages[i]["content"] = stub

        # 1. Tool results.
        _stub_old(
            [i for i, m in enumerate(messages) if m.get("role") == "tool"],
            "[elided older tool result — candidates are in the registry, "
            "reasoning is in the notebook, full data is under artifacts/]",
            keep_recent_tools,
        )
        # 2. Assistant narration (tool_calls are preserved; only text is stubbed).
        _stub_old(
            [i for i, m in enumerate(messages) if m.get("role") == "assistant"],
            "[earlier reasoning elided to save context — see the notebook]",
            keep_recent_assistant,
        )
        # 3. Superseded state headers — keep only the most recent, drop the rest.
        state_positions = [
            i for i, m in enumerate(messages)
            if m.get("role") == "user" and isinstance(m.get("content"), str)
            and m["content"].lstrip().startswith("[STATE:")
        ]
        for i in state_positions[:-1]:
            messages[i]["content"] = "[state superseded]"

    def _finalize(
        self,
        messages: list,
        token_usage: TokenUsage,
        final_text: str,
        force_completion: bool = False,
        status: str = "completed",
    ) -> str:
        report, n_evals = RunFinalizer(self.run_dir, self.llm, self.log).finalize(
            messages=messages,
            token_usage=token_usage,
            query=getattr(self, "_query", "enzyme discovery"),
            final_text=final_text,
            target_context=getattr(self, "_target_context", None),
            force_completion=force_completion,
        )
        self.log(f"  Candidates:     {n_evals} evaluated")
        self.log(f"  Total tokens:   {token_usage.total_tokens:,}")
        self._log_event("run_complete", {
            "candidates": n_evals,
            "total_tokens": token_usage.total_tokens,
            "prompt_tokens": token_usage.total_prompt_tokens,
            "completion_tokens": token_usage.total_completion_tokens,
            "iterations_used": self._iterations_used,
            "status": status,
        })
        # Close out run-level provenance: outcome, timing, tokens, and the
        # human-readable RUN_SUMMARY.md.  Best-effort; never masks the report.
        report_path = next(
            (str(p) for name in ("report_confidence.md", "report_explorative.md")
             if (p := self.run_dir / name).exists()),
            None,
        )
        provenance.finalize_manifest(
            self.run_dir,
            status=status,
            iterations_used=self._iterations_used,
            candidates_evaluated=n_evals,
            tokens=token_usage.to_dict(),
            outcome=final_text[:500] if final_text else None,
            report_path=report_path,
        )
        return report
