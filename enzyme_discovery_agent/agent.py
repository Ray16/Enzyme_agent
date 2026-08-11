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

from enzyme_discovery_agent import config, provenance
from enzyme_discovery_agent.bootstrap import register_builtin_tools
from enzyme_discovery_agent.capabilities import create_run_dir, init_capabilities
from enzyme_discovery_agent.context import ARTIFACT_CATEGORIES, compact_for_llm
from enzyme_discovery_agent.gates import AgentState, GatePipeline
from enzyme_discovery_agent.gates.ec_triage import ECTriageGate, auto_screen_candidates
from enzyme_discovery_agent.gates.discovery_coverage import DiscoveryCoverageGate
from enzyme_discovery_agent.gates.evidence_selection import EvidenceSelectionGate
from enzyme_discovery_agent.gates.entity_extraction import EntityExtractionGate
from enzyme_discovery_agent.gates.literature_harvest import LiteratureHarvestGate
from enzyme_discovery_agent.gates.mechanism_family_coverage import MechanismFamilyGate
from enzyme_discovery_agent.gates.family_survey import FamilySurveyGate
from enzyme_discovery_agent.gates.homolog_expansion import HomologExpansionGate
from enzyme_discovery_agent.gates.systematic_coverage import SystematicCoverageGate
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.candidate_registry import (
    extract_and_register_from_search,
    load_candidate_registry,
    register_structural_homologs,
)
from enzyme_discovery_agent.models.target_profile import normalize_target_profile
from enzyme_discovery_agent.output.finalizer import RunFinalizer
from enzyme_discovery_agent.prompts import (
    EXPLORATIVE_RERANK_PROMPT,
    build_reflection_prompt,
    build_state_header,
    build_system_prompt,
)
from enzyme_discovery_agent.tool_registry import (
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

class EnzymeAgent:
    """LLM-driven enzyme discovery agent."""

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        output_dir: str | Path = "runs",
        max_iterations: int = config.MAX_ITERATIONS,
        llm_timeout: int = config.LLM_TIMEOUT,
        query: str = "",
    ):
        register_builtin_tools()
        self.llm = LLMClient(model=model)
        self.model_name = model
        self.max_iterations = max_iterations
        self.llm_timeout = llm_timeout

        self.run_dir = create_run_dir(Path(output_dir), query=query)
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

        self._event_log_path = self.run_dir / "events.jsonl"
        self._setup_logging()

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
        from enzyme_discovery_agent.models.event_log import append_event
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
        for subdir in ("literature", "database", "candidates", "fulltext"):
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
            "timestamp": datetime.now().isoformat(),
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
        from enzyme_discovery_agent.gates.ec_triage import _family_key
        active_candidates = {
            acc: info for acc, info in registry.items()
            if any(s.get("status") in {"screened", "evaluated", "ranked"}
                   for s in info.get("statuses", []))
        }
        self._candidate_count = len({
            _family_key(info.get("metadata", {}), acc)
            for acc, info in active_candidates.items()
        })
        self._gate_state.candidate_count = self._candidate_count
        # Count DISTINCT papers for gate sufficiency: now that several sources
        # feed the pool, the same paper arrives multiple times and must not
        # inflate the "enough literature" signal (which would stop search early).
        from enzyme_discovery_agent.tools.literature import dedupe_and_merge
        unique_papers, _ = dedupe_and_merge(self._accumulated_papers)
        self._gate_state.accumulated_papers = unique_papers
        self._gate_state.lit_search_count = len(
            getattr(self, "_literature_search_iterations", set())
        )
        self._gate_state.search_log = self._search_log

    def run(
        self,
        query: str,
        substrate_smiles: str | None = None,
        substrate_name: str | None = None,
        target_profile: dict | None = None,
    ) -> str:
        self.log(f"  Run ID:    {self.run_dir.name}")
        self.log(f"  Model:     {self.model_name}")
        self.log(f"  Query:     {query}")
        if substrate_smiles:
            self.log(f"  Substrate: {substrate_smiles}")
        self._log_event("run_start", {
            "query": query, "substrate": substrate_smiles, "model": self.model_name,
        })

        self._query = query
        # Derive target-substrate SMILES when the caller did not supply them:
        # without a target structure the substrate-accommodation axis is inert
        # and ranking collapses to annotation depth (fame), unable to tell an
        # enzyme whose substrate fits the target from one built for a bulky or
        # tethered substrate.  The substrate names come from the user's own
        # query, so resolving them is task setup, not answer leakage.
        if not substrate_smiles:
            try:
                from enzyme_discovery_agent.chemistry_logic import (
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
        self._target_context = {
            "query": query,
            "substrate_smiles": substrate_smiles,
            "substrate_name": substrate_name,
            "target_profile": normalize_target_profile(target_profile),
        }
        # Gates run independently of the live agent instance. Persist the
        # structured target once so automatic evaluation never has to infer it
        # from a truncated notebook paragraph.
        (self.run_dir / "target_context.json").write_text(
            json.dumps(self._target_context, indent=2, default=str) + "\n"
        )
        # Run-level provenance: stamp the code revision, resolved config, and
        # environment before any model work, so an interrupted run is still
        # identifiable and a completed run is reproducible.
        provenance.write_manifest_start(
            self.run_dir,
            query=query,
            model=self.model_name,
            config_module=config,
            args={"max_iterations": self.max_iterations, "llm_timeout": self.llm_timeout},
            substrate={"smiles": substrate_smiles, "name": substrate_name},
            capabilities=self.capabilities,
        )
        clear_search_history()
        self._ensure_artifacts_dir()
        token_usage = TokenUsage(model=self.model_name)

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        # ``iteration`` is a budget for substantive model/tool work.  Gate
        # interceptions only redirect orchestration and explicitly restore
        # their credit below, so they cannot silently shorten a campaign.
        iteration = 0
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
            self._prune_tool_history(messages)
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
            self._log_event("assistant_decision", {
                "iteration": iteration,
                "n_tool_calls": len(tool_calls),
                "tools": [_tc_name(tc) for tc in tool_calls],
                "reasoning": (text_content or "").strip()[:800],
            })

            # No tool calls: check if done
            if not tool_calls:
                self._consecutive_no_tool_turns += 1
                if self._should_continue(iteration):
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
            # Literature-first: harvest full-text-mined enzymes as discovery
            # seeds BEFORE the agent registers/expands candidates, so tested
            # enzymes lead and database search expands around them.
            if not has_focus_call and self._literature_harvest_gate.should_fire(self._gate_state):
                harvest_result = self._literature_harvest_gate.execute(self._gate_state)
                self._log_gate("literature_harvest", harvest_result, iteration)
                for k, v in harvest_result.state_updates.items():
                    setattr(self._gate_state, k, v)
                messages.pop()
                messages.extend(harvest_result.new_messages)
                iteration -= 1
                self.log("  Gate transition: iteration credit restored")
                continue
            # Derive the reaction's distinct mechanism families and make each a
            # coverage requirement, searched from literature before evaluation.
            if not has_focus_call and self._mechanism_family_gate.should_fire(self._gate_state):
                mf_result = self._mechanism_family_gate.execute(self._gate_state)
                for k, v in mf_result.state_updates.items():
                    setattr(self._gate_state, k, v)
                if mf_result.intercept:
                    self._log_gate("mechanism_family", mf_result, iteration)
                    messages.pop()
                    messages.extend(mf_result.new_messages)
                    iteration -= 1
                    self.log("  Gate transition: iteration credit restored")
                    continue
            # Auto-run the family-reaction survey once families + seeds exist, and
            # inject the grounded family landscape so the agent navigates by it.
            if not has_focus_call and self._family_survey_gate.should_fire(self._gate_state):
                fs_result = self._family_survey_gate.execute(self._gate_state)
                for k, v in fs_result.state_updates.items():
                    setattr(self._gate_state, k, v)
                if fs_result.intercept:
                    self._log_gate("family_survey", fs_result, iteration)
                    messages.pop()
                    messages.extend(fs_result.new_messages)
                    iteration -= 1
                    self.log("  Gate transition: iteration credit restored")
                    continue
            if not has_focus_call and self._evidence_selection_gate.should_fire(self._gate_state):
                selection_result = self._evidence_selection_gate.execute(self._gate_state)
                self._log_gate("evidence_selection", selection_result, iteration)
                for k, v in selection_result.state_updates.items():
                    setattr(self._gate_state, k, v)
                messages.pop()
                messages.extend(selection_result.new_messages)
                iteration -= 1
                self.log("  Gate transition: iteration credit restored")
                continue
            if not has_focus_call and self._systematic_gate.should_fire(self._gate_state, has_eval_calls, messages):
                sys_result = self._systematic_gate.execute(self._gate_state, messages)
                if sys_result and sys_result.intercept:
                    self._log_gate("systematic_coverage", sys_result, iteration)
                    for k, v in sys_result.state_updates.items():
                        setattr(self._gate_state, k, v)
                    messages.pop()
                    messages.extend(sys_result.new_messages)
                    iteration -= 1
                    self.log("  Gate transition: iteration credit restored")
                    continue

            # Pre-evaluation gates.  Skipped on a focus-call turn so the agent's
            # set_family_focus executes and its stop/expand decisions persist
            # before auto-screen/eval spends the budget (they fire next turn).
            if not has_focus_call and (
                self._gate_state.candidate_count > 0
                or has_eval_calls
                or self._gate_state.critique_injected
            ):
                gate_result = self._gate_pipeline.check(
                    self._gate_state, has_eval_calls, messages
                )
                if gate_result and gate_result.intercept:
                    self._log_gate("pre_evaluation", gate_result, iteration)
                    for k, v in gate_result.state_updates.items():
                        setattr(self._gate_state, k, v)
                    messages.pop()
                    messages.extend(gate_result.new_messages)
                    iteration -= 1
                    self.log("  Gate transition: iteration credit restored")
                    continue
                for k, v in (gate_result.state_updates if gate_result else {}).items():
                    setattr(self._gate_state, k, v)

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

    def _prune_tool_history(
        self,
        messages: list,
        keep_recent: int = config.HISTORY_KEEP_RECENT_TOOL_MSGS,
        stub_over: int = config.HISTORY_STUB_OVER_CHARS,
    ) -> None:
        """Stub the content of tool results older than the most recent
        ``keep_recent`` tool messages, in place.

        The prompt re-sent each iteration is the dominant cost (in one run: 2.1M
        of 2.2M loop tokens were prompt, growing 19k -> 164k as raw tool dumps
        piled up).  A later turn never needs the raw dump of an old search: the
        candidates it surfaced are already in the registry, the reasoning is in
        the notebook, and the full result is on disk under artifacts/.  So the
        old tool CONTENT is replaced with a short stub while the message itself
        is kept, preserving the assistant tool_call / tool-result pairing the
        provider requires.  Recent results stay full so the model still reasons
        over fresh evidence.
        """
        tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_positions) <= keep_recent:
            return
        stub = ("[elided older tool result — candidates are in the registry, "
                "reasoning is in the notebook, full data is under artifacts/]")
        for i in tool_positions[:-keep_recent]:
            content = messages[i].get("content")
            if isinstance(content, str) and len(content) > stub_over and content != stub:
                messages[i]["content"] = stub

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
