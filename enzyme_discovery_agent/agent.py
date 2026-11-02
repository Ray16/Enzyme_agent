"""Core discovery orchestration, with gates, evidence and output modules.

Owns the conversation, tool dispatch and durable recovery state. Chemistry,
candidate validation, provider accounting and report delivery have separate
implementations shared with the downstream modeling interface.
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
from enzyme_discovery_agent.context import ARTIFACT_CATEGORIES, compact_for_llm, build_model_context, compact_tool_definitions
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
    current_candidate_status,
    discovery_status,
    extract_and_register_from_search,
    family_label,
    load_candidate_registry,
    register_atlas_homologs,
    register_structural_homologs,
)
from enzyme_discovery_agent.models.target_profile import normalize_target_profile
from enzyme_discovery_agent.state_store import atomic_write_json, write_tool_artifact, controller_lock
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
from enzyme_discovery_agent.run_layout import run_file

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
        model: str | None = None,
        output_dir: str | Path = "runs",
        max_iterations: int = config.MAX_ITERATIONS,
        llm_timeout: int = config.LLM_TIMEOUT,
        query: str = "",
        resume_run_dir: str | Path | None = None,
        token_budget: int | None = None,
        runtime_budget_seconds: float | None = None,
    ):
        if model is None and resume_run_dir:
            try:
                cascade = json.loads((Path(resume_run_dir) / "agent_state/llm_cascade.json").read_text())
                model = cascade.get("requested_model")
            except (OSError, ValueError):
                pass
        if model is None and resume_run_dir:
            try:
                saved_model = json.loads(run_file(Path(resume_run_dir), "token_usage.json").read_text()).get("model", "")
                if saved_model.startswith(("gemini:", "groq:", "claude-code:")):
                    model = saved_model
                else:
                    model = json.loads((Path(resume_run_dir) / "manifest.json").read_text()).get("model", "").split(" -> ")[0]
            except (OSError, ValueError, AttributeError):
                pass
        model = model or config.DEFAULT_MODEL
        register_builtin_tools()
        self.llm = LLMClient(model=model)
        self.model_name = model
        self.max_iterations = max_iterations
        self.token_budget = token_budget
        self.runtime_budget_seconds = runtime_budget_seconds
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if runtime_budget_seconds is not None and runtime_budget_seconds <= 0:
            raise ValueError("runtime_budget_seconds must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
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

        self._log_lock = threading.Lock()
        self._search_log: list[str] = []
        self._search_history: list[dict] = []
        self._candidate_count = 0
        self._iteration_start: float = 0.0
        self._accumulated_papers: list[dict] = []
        self._literature_search_iterations: set[int] = set()
        self._confidence_report_text: str | None = None
        self._consecutive_no_tool_turns = 0
        self._consecutive_no_progress_turns = 0
        self._automatic_evidence_batches = 0
        self._model_turns_since_recovery = 0
        self._last_automatic_action = None
        self._last_research_signature = None
        self._notebook_tool_cooldown = 0
        self._iterations_used = 0
        # Family-coverage guard state: every search query issued (so we can tell
        # which derived families were actually drilled) and how many coverage
        # nudges we have spent (bounded so coverage is encouraged, never a hard
        # block that could stall the run).
        self._search_queries: list[str] = []
        self._coverage_nudges_used = 0

        self._gate_state = AgentState(run_dir=self.run_dir)
        self._systematic_gate = SystematicCoverageGate(self.llm)
        self._literature_harvest_gate = LiteratureHarvestGate(self.llm, self.log)
        self._mechanism_family_gate = MechanismFamilyGate(self.llm, self.log)
        self._family_survey_gate = FamilySurveyGate(self.log)
        self._evidence_selection_gate = EvidenceSelectionGate()
        self._gate_pipeline = GatePipeline([
            EntityExtractionGate(),
            DiscoveryCoverageGate(),
            ECTriageGate(self.llm),
            HomologExpansionGate(self.llm),
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

        self._event_log_path = run_file(self.run_dir, 'events.jsonl')
        self._warning_file_handler = None
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
        if manifest.get("status") not in {"running", "budget_exhausted", "failed", "discovery_incomplete"}:
            raise RuntimeError(
                f"Run cannot be resumed because manifest status is {manifest.get('status')!r}"
            )
        messages = transcript.get("messages")
        if not isinstance(messages, list):
            raise RuntimeError("Run cannot be resumed: checkpoint has no message list")
        try:
            context = json.loads((run_file(self.run_dir, 'target_context.json')).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Run cannot be resumed: missing or invalid target context ({exc})") from exc
        if not isinstance(context.get("target_profile"), dict):
            raise RuntimeError("Run cannot be resumed: target context has no target profile")

        from enzyme_discovery_agent.models.event_log import read_events
        events = read_events(self.run_dir)
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
        saved_state = transcript.get("gate_state") or {}
        for key, value in saved_state.items():
            if hasattr(self._gate_state, key) and key != "run_dir":
                setattr(self._gate_state, key, value)
        progress = transcript.get("research_progress") or {}
        if "model_turns_since_recovery" in progress:
            self._model_turns_since_recovery = max(0, int(progress["model_turns_since_recovery"]))
        else:
            self._model_turns_since_recovery = 0
            for event in reversed(events):
                if event.get("type") in {"discovery_recovery", "run_start"}:
                    break
                if event.get("type") == "assistant_decision":
                    self._model_turns_since_recovery += 1
        # Keep the evidence signature, but grant the resumed controller a fresh
        # opportunity to act. Restoring a terminal idle streak immediately stops
        # the next run, even after its controller or provider settings are fixed.
        self._consecutive_no_progress_turns = 0
        self._last_research_signature = progress.get("signature")
        self._last_automatic_action = progress.get("last_automatic_action")
        if "automatic_evidence_batches" in progress:
            self._automatic_evidence_batches = max(0, int(progress["automatic_evidence_batches"]))
        else:
            # Older checkpoints still record gate work in the event stream.
            # Preserve its scheduling debt so resume cannot postpone reflection again.
            self._automatic_evidence_batches = 0
            for event in reversed(events):
                if event.get("type") in {"assistant_decision", "run_start"}:
                    break
                if (event.get("type") == "discovery_recovery"
                        or event.get("type") == "gate" and event.get("intercept")):
                    self._automatic_evidence_batches += 1
        from enzyme_discovery_agent.models.recovery import resume_failed_routes
        resume_failed_routes(self.run_dir)
        survey = transcript.get("family_survey_state") or {}
        gate = getattr(self, "_family_survey_gate", None)
        if gate is not None and survey:
            for key in ("_attempts", "_last_attempt_count", "_all_folds", "_claim_validations"):
                if key in survey:
                    setattr(gate, key, survey[key])
            gate._surveyed_folds = set(survey.get("_surveyed_folds", []))
            gate._promoted = set(survey.get("_promoted", []))
        self._search_queries = [
            str((e.get("args") or {}).get("query") or "")
            for e in events if e.get("type") == "tool_call"
        ]
        self._search_log = [
            f"{e.get('tool')}:{str((e.get('args') or {}).get('query') or '')[:30]}"
            for e in events if e.get("type") == "tool_call"
        ]
        if not hasattr(self, "_search_history"):
            self._search_history = []
        clear_search_history(history=self._search_history)
        # Restore the literature inputs used by gate sufficiency, not just the
        # conversation whose older tool results may already have been elided.
        self._accumulated_papers = []
        self._literature_search_iterations = set()
        for path in sorted((self.run_dir / "artifacts" / "literature").glob("*.json")):
            try:
                artifact = json.loads(path.read_text())
                data = artifact.get("result", artifact).get("data", [])
                if isinstance(data, list):
                    self._accumulated_papers.extend(p for p in data if isinstance(p, dict))
                elif artifact.get("tool_name") == "deep_literature_search" and isinstance(data, dict):
                    for papers in data.values():
                        if isinstance(papers, list):
                            self._accumulated_papers.extend(p for p in papers if isinstance(p, dict))
                if artifact.get("iteration") is not None:
                    self._literature_search_iterations.add(artifact["iteration"])
            except (OSError, ValueError, AttributeError):
                continue
        for event in events:
            if event.get("type") == "tool_call" and event.get("tool"):
                record_search(event["tool"], event.get("args") or {},
                              {"status": event.get("status"), "count": event.get("count"),
                               "search_completed": event.get("search_completed")}, history=self._search_history)
                if event.get("tool") in {
                    "search_pubmed", "search_openalex", "search_europepmc",
                    "search_semantic_scholar", "search_crossref", "deep_literature_search",
                }:
                    self._literature_search_iterations.add(event.get("iteration"))
        # Older UniProt events did not distinguish a completed empty result
        # from other partial results. Recover only artifacts whose saved output
        # explicitly records a completed, empty search in this same run.
        for path in (self.run_dir / "artifacts" / "database").glob("search_uniprot_*.json"):
            try:
                artifact = json.loads(path.read_text())
                result = artifact.get("result") or {}
                args = artifact.get("args") or {}
                if (artifact.get("tool_name") == "search_uniprot" and args.get("gene_name")
                        and result.get("status") == "partial" and result.get("count") == 0
                        and result.get("data") == [] and
                        str(result.get("summary") or "").startswith("Found 0 UniProt entries for query:")):
                    record_search("search_uniprot", args, {**result, "search_completed": True},
                                  history=self._search_history)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        calls = [e for e in events if e.get("type") == "llm_call"]
        usage = TokenUsage(model=self.model_name)
        usage.calls = [{
            "iteration": e.get("iteration"),
            "prompt_tokens": e.get("prompt_tokens", 0),
            "completion_tokens": e.get("completion_tokens", 0),
            "total_tokens": e.get("total_tokens", 0),
        } for e in calls]
        usage.legacy_accounting = any(e.get("accounting") != "client_boundary" for e in calls)
        usage.total_prompt_tokens = sum(c["prompt_tokens"] for c in usage.calls)
        usage.total_completion_tokens = sum(c["completion_tokens"] for c in usage.calls)
        usage.total_tokens = sum(c["total_tokens"] for c in usage.calls)
        # New runs checkpoint the complete client-boundary ledger after each
        # response. Legacy events remain the fallback for older checkpoints.
        try:
            saved_usage = json.loads((run_file(self.run_dir, 'token_usage.json')).read_text())
            saved_calls = saved_usage["calls"]
            if (isinstance(saved_calls, list)
                    and saved_usage.get("total_tokens", 0) >= usage.total_tokens):
                usage.calls = saved_calls
                usage.total_prompt_tokens = sum(c.get("prompt_tokens", 0) for c in saved_calls)
                usage.total_completion_tokens = sum(c.get("completion_tokens", 0) for c in saved_calls)
                usage.total_tokens = usage.total_prompt_tokens + usage.total_completion_tokens
                usage.unmetered_calls = saved_usage.get("unmetered_calls", 0)
                usage.legacy_accounting = saved_usage.get("legacy_accounting", "accounting_complete" not in saved_usage)
        except (OSError, ValueError, KeyError, TypeError):
            pass
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
        self._warning_file_handler = file_handler

    def _close_logging(self) -> None:
        handler = getattr(self, "_warning_file_handler", None)
        if handler is None:
            return
        root_logger = logging.getLogger("enzyme_agent")
        root_logger.removeHandler(handler)
        handler.close()
        self._warning_file_handler = None

    def log(self, msg: str, level: str = "info"):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        with self._log_lock:
            with open(run_file(self.run_dir, 'console_log.txt'), "a") as f:
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

    def _apply_gate_result(self, name, result, iteration, messages, *, discard_assistant=True):
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
        self._last_automatic_action = name
        if not result.intercept:
            return iteration, False
        self._automatic_evidence_batches = getattr(self, "_automatic_evidence_batches", 0) + 1
        self._log_gate(name, result, iteration)
        if discard_assistant:
            messages.pop()
        messages.extend(result.new_messages)
        self.log("  Gate transition: iteration credit restored")
        refunded_iteration = iteration - 1
        self._record_research_progress(count_turn=True)
        self._checkpoint_transcript(messages, refunded_iteration)
        return refunded_iteration, True

    def _checkpoint_transcript(self, messages: list, iteration: int) -> None:
        """Persist the full conversation each iteration (crash-safe provenance).

        The finalizer writes ``messages.json`` only at the end of a run, so an
        interrupted campaign would otherwise lose its entire reasoning trace.
        This rolling checkpoint is both a provenance record and the foundation
        for future resume support.  Best-effort: never breaks the loop.
        """
        try:
            payload = {"iteration": iteration, "messages": messages}
            payload["research_progress"] = {
                "idle_turns": getattr(self, "_consecutive_no_progress_turns", 0),
                "signature": getattr(self, "_last_research_signature", None),
                "automatic_evidence_batches": getattr(self, "_automatic_evidence_batches", 0),
                "model_turns_since_recovery": getattr(self, "_model_turns_since_recovery", 0),
                "last_automatic_action": getattr(self, "_last_automatic_action", None),
            }
            state = getattr(self, "_gate_state", None)
            if state is not None:
                payload["gate_state"] = {
                    key: value for key, value in vars(state).items()
                    if key not in {"run_dir", "accumulated_papers"}
                }
            survey = getattr(self, "_family_survey_gate", None)
            if survey is not None:
                payload["family_survey_state"] = {
                    key: list(getattr(survey, key)) if key in {"_surveyed_folds", "_promoted"}
                    else getattr(survey, key)
                    for key in ("_attempts", "_last_attempt_count", "_all_folds",
                                "_claim_validations", "_surveyed_folds", "_promoted")
                }
            atomic_write_json(self.run_dir / "agent_state" / "transcript.json", payload)
        except (OSError, TypeError) as exc:
            logger.warning("Failed to checkpoint transcript at iteration %d: %s",
                           iteration, exc)

    def _ensure_artifacts_dir(self) -> None:
        for subdir in ("literature", "database", "candidates", "fulltext", "evidence"):
            (self.artifacts_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _save_artifact(self, category: str, tool_name: str, result: dict,
                       tool_args: dict, iteration: int):
        artifact = {
            "tool_name": tool_name, "args": tool_args,
            "timestamp": datetime.now().astimezone().isoformat(),
            "iteration": iteration, "result": result,
        }
        try:
            write_tool_artifact(self.run_dir, category, tool_name, artifact)
        except Exception as e:
            logger.warning("Failed to save artifact for %s: %s", tool_name, e)

    def _execute_model_tool(self,tool_name,tool_args,iteration):
        previous=getattr(self,'_executing_model_selection',False)
        self._executing_model_selection=True
        try:return self._execute_single_tool(tool_name,tool_args,iteration)
        finally:self._executing_model_selection=previous

    def _execute_single_tool(self, tool_name: str, tool_args: dict, iteration: int, *, _planned_decision=None) -> dict:
        from enzyme_discovery_agent.steering import has_pending
        if has_pending(self.run_dir):
            return {"status": "skipped", "deferred_for_user_steering": True,
                    "summary": "New user control is queued; reconsider this action at the next boundary."}
        offered=getattr(self,'_offered_direct_tools',None)
        if (getattr(self,'_executing_model_selection',False) and _planned_decision is None
                and offered is not None and tool_name not in offered):
            self._log_event('unoffered_tool_rejected',{'tool':tool_name,'executed':False})
            return {'status':'error','decision_executed':False,
                    'summary':'This direct tool is unavailable for the current turn. Choose an offered control or use the planner for a permitted evidence action.',
                    'offered_tools':sorted(offered)}
        if getattr(self,'_require_research_decision',False) and _planned_decision is None and tool_name!='plan_research_step':
            from enzyme_discovery_agent.models.research_decisions import review_action_allowed
            if not review_action_allowed(self.run_dir,tool_name):
                return {'status':'error','summary':'Review work allowance reached; conclude_research_review before further evaluation or research.'}
        if tool_name == 'plan_research_step' and _planned_decision is None:
            from enzyme_discovery_agent.models.research_decisions import start_decision, finish_decision, validate_plan
            try:
                allowed_actions = getattr(self, '_decision_action_tools', {t['function']['name'] for t in self.tools})
                proposal = validate_plan(self.run_dir, tool_args, allowed_actions)
                usage=getattr(self,'_run_token_usage',None)
                calls=getattr(usage,'calls',[]) if usage else []
                provenance={'serving_label':getattr(self.llm,'active_label',None),
                            'request':{key:calls[-1].get(key) for key in ('request_id','model','backend','iteration')} if calls else None}
                identifier = start_decision(self.run_dir, tool_args,
                    allowed_actions,model_provenance=provenance)
            except (ValueError, TypeError) as exc:
                self._invalid_research_decisions = getattr(self, '_invalid_research_decisions', 0) + 1
                self._log_event('research_proposal_rejected',{'reason':str(exc),'executed':False})
                from enzyme_discovery_agent.models.research_decisions import proposal_repair_context
                try:
                    repair=proposal_repair_context(self.run_dir,proposal=tool_args)
                except (ValueError,TypeError,OSError):
                    repair={'status':'unavailable; inspect the preserved source state'}
                return {'status': 'error', 'summary': str(exc), 'decision_executed': False,
                        'repair_context':repair}
            self._invalid_research_decisions = 0
            try:
                result = self._execute_single_tool(proposal['tool_name'], proposal['arguments'], iteration,
                                                   _planned_decision=identifier)
            except Exception as exc:
                result = {'status': 'error', 'summary': f'Research action failed: {type(exc).__name__}: {exc}'}
            evidence_id = finish_decision(self.run_dir, identifier, result)
            return {**result, 'research_decision': identifier, 'outcome_evidence': evidence_id,
                    'interpretation_required': 'Compare this observation with the registered competing predictions before choosing the next test.'}
        if (getattr(self, '_require_research_decision', False) and _planned_decision is None
                and tool_name not in {'get_decision_context', 'get_exploration_state',
                                      'get_candidate_evidence', 'get_research_memory', 'inspect_discovery_source',
                                      'conclude_research_review','interpret_research_outcome'}):
            self._invalid_research_decisions = getattr(self, '_invalid_research_decisions', 0) + 1
            return {'status': 'error', 'summary': 'New consequential evidence requires a grounded plan_research_step before further work.',
                    'decision_executed': False}
        t0 = _time.time()

        if not isinstance(tool_args, dict):
            return execute_tool(tool_name, tool_args)
        history = getattr(self, "_search_history", None)
        dup = check_search_duplicate(tool_name, tool_args, history=history)
        if dup:
            self.log(f"    ~ {tool_name}: deduplicated (similar search already done)")
            self._log_event("tool_dedup", {"tool": tool_name, "args": tool_args})
            return dup

        query_preview = tool_args.get("query", tool_args.get("ec_number",
                        tool_args.get("accession", tool_args.get("smiles", ""))))
        self.log(f"    -> {tool_name}({str(query_preview)[:80]})")

        injected = {"run_dir": str(self.run_dir), "llm_client": self.llm,
                    "research_decision_id":_planned_decision or ''}
        result = execute_tool(tool_name, tool_args, injected_args=injected)
        result = json.loads(json.dumps(result, default=str))
        if tool_name in {'interpret_research_outcome','conclude_research_review'}:
            if result.get('status')=='success':
                self._invalid_research_decisions=0
                self._research_repair_only=False
                self._research_repair_attempted=False
            else:
                self._invalid_research_decisions=getattr(self,'_invalid_research_decisions',0)+1
                if 'repair_context' not in result:
                    from enzyme_discovery_agent.models.research_decisions import proposal_repair_context
                    result['repair_context']=proposal_repair_context(self.run_dir,proposal=tool_args)
                from enzyme_discovery_agent.tool_registry import get_tool_info
                result['repair_context']['required_fields']=get_tool_info(tool_name)['schema']['function']['parameters'].get('required',[])
                self._log_event('research_control_rejected',{'tool':tool_name,'reason':result.get('summary'),'executed':False})
        if (getattr(self, '_require_research_decision', False) and _planned_decision is None
                and tool_name in {'get_decision_context','get_exploration_state','get_candidate_evidence','get_research_memory','inspect_discovery_source'}
                and result.get('status') in {'success','ok','partial'}):
            from enzyme_discovery_agent.models.research_decisions import record_inspection
            record_inspection(self.run_dir)

        elapsed = _time.time() - t0
        record_search(tool_name, tool_args, result, history=history)

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
            from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates
            refresh_ranked_candidates(self.run_dir)
        if tool_name in ("search_pubmed", "search_openalex", "search_europepmc",
                         "search_semantic_scholar", "search_crossref", "fetch_paper_metadata"):
            papers = result.get("data", [])
            if isinstance(papers, list):
                self._accumulated_papers.extend(papers)
                self._literature_search_iterations.add(iteration)
        if tool_name in {"deep_literature_search", "follow_paper_citations"}:
            refs = result.get("data", {})
            if isinstance(refs, dict):
                for key in ("seed_papers", "reference_papers"):
                    paper_list = refs.get(key)
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
        if tool_name == "search_esm_atlas_homologs" and result.get("status") == "success":
            register_atlas_homologs(self.run_dir, result)

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
        if result.get("search_completed") is True:
            tool_event["search_completed"] = True
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
                (run_file(self.run_dir, 'mechanism_families.json')).read_text()
            )
        except (OSError, ValueError, TypeError):
            return []
        try:
            from enzyme_discovery_agent.tools.mechanism_families import (
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
        return "rolling_discovery"

    def _should_continue(self, iteration: int) -> bool:
        from enzyme_discovery_agent.models.rolling import new_evaluation_work
        if new_evaluation_work(self.run_dir):
            return True
        from enzyme_discovery_agent.models.recovery import next_recovery_action
        if next_recovery_action(self.run_dir, {t["function"]["name"] for t in getattr(self, "tools", [])}):
            return True
        return getattr(self, "_consecutive_no_progress_turns", 0) < 3

    def _recover_discovery(self, messages, iteration) -> bool:
        """Execute one grounded queued search rather than pay for another nudge."""
        from enzyme_discovery_agent.models.recovery import next_recovery_action, record_recovery_attempt
        action = next_recovery_action(self.run_dir, {t["function"]["name"] for t in self.tools})
        if action is None or self._budget_reason():
            return False
        record_recovery_attempt(self.run_dir, action, "started")
        self.log(f"  Discovery recovery: {action['reason']}")
        result = self._execute_single_tool(action["tool"], action["args"], iteration)
        status = "deduplicated" if result.get("deduplicated") else result.get("status", "error")
        record_recovery_attempt(self.run_dir, action, status, result)
        self._log_event("discovery_recovery", {"iteration": iteration, **action, "status": status})
        self._model_turns_since_recovery = 0
        messages.append({"role": "user", "content": (
            f"The controller gathered evidence for an unresolved task: {action['reason']}. "
            f"Tool: {action['tool']}; arguments: {json.dumps(action['args'])}. "
            f"Retrieved result: {json.dumps(result, default=str)[:12000]}. "
            "Use the retrieved evidence to resolve identifiers, screen candidates or reassess the hypothesis. "
            "Do not treat the hypothesis itself as evidence."
        )})
        self._sync_gate_state()
        self._record_research_progress()
        self._automatic_evidence_batches = getattr(self, "_automatic_evidence_batches", 0) + 1
        self._checkpoint_transcript(messages, iteration)
        return True

    def _automatic_reflection_due(self) -> bool:
        return (getattr(self, "_consecutive_no_progress_turns", 0) >= 2 and
                getattr(self, "_automatic_evidence_batches", 0) >= config.MAX_AUTOMATED_BATCHES_BEFORE_REFLECTION)

    def _service_steering(self, messages, iteration):
        """Honor user control before scheduling further discovery work."""
        from enzyme_discovery_agent import steering
        changed = False
        while True:
            state, commands = steering.consume(self.run_dir)
            for command in commands:
                self._log_event("user_steering_accepted", {
                    "command_id": command["id"], "action": command["action"],
                    "iteration": iteration, "message": command.get("message")})
            revision = state["revision"]
            pending_instruction = any(state["receipts"].get(item["id"], {}).get("status") == "accepted"
                                      for item in state["instructions"])
            if revision != getattr(self, "_steering_revision", revision) or pending_instruction:
                changed = True
            self._steering_revision = revision
            if state["snapshot_requested"] > state["snapshot_completed"]:
                self._checkpoint_transcript(messages, iteration)
                RunFinalizer(self.run_dir, self.llm, self.log).snapshot(
                    query=getattr(self, "_query", "enzyme discovery"), reason="User requested checkpoint")
                state = steering.update_state(self.run_dir, snapshot_completed=state["snapshot_requested"])
                self._log_event("user_checkpoint_saved", {"iteration": iteration})
            phase = "paused" if state["paused"] else "running"
            if state.get("controller_status") != phase or commands:
                steering.update_state(self.run_dir, controller_status=phase, iteration=iteration)
                self._log_event("user_control_boundary", {"status": phase, "iteration": iteration})
            if commands:
                steering.mark_controls_applied(self.run_dir, iteration)
            if not state["paused"]:
                return changed
            # No discovery or provider calls while paused. The queue remains live.
            _time.sleep(1)

    def _record_research_progress(self, *, count_turn=False) -> bool:
        from enzyme_discovery_agent.models.progress import research_signature
        signature = research_signature(self.run_dir, getattr(self, "_accumulated_papers", []))
        previous = getattr(self, "_last_research_signature", None)
        changed = previous is not None and signature != previous
        self._last_research_signature = signature
        if changed:
            self._consecutive_no_progress_turns = 0
        elif count_turn:
            self._consecutive_no_progress_turns = getattr(self, "_consecutive_no_progress_turns", 0) + 1
        return changed

    def _budget_reason(self) -> str | None:
        provider_reason = getattr(getattr(self, "llm", None), "resource_stop_reason", None)
        if isinstance(provider_reason, str) and provider_reason:
            return provider_reason
        usage = getattr(self, "_run_token_usage", None)
        token_budget = getattr(self, "token_budget", None)
        if usage is not None and token_budget is not None:
            spent = usage.total_tokens - self._budget_start_tokens
            if spent >= token_budget:
                return f"LLM token allowance reached ({spent}/{token_budget})"
        runtime = getattr(self, "runtime_budget_seconds", None)
        if runtime is not None and _time.monotonic() - self._budget_started >= runtime:
            return f"Elapsed-time allowance reached ({runtime:g} seconds)"
        return None

    def _finish_budget_checkpoint(self, messages, iteration, token_usage, reason, *, status="budget_exhausted"):
        """Publish existing evidence without starting more discovery or LLM work."""
        from enzyme_discovery_agent.models.rolling import pending_evaluations
        from enzyme_discovery_agent.models.literature_strategy import coverage_summary
        from enzyme_discovery_agent.models.recovery import _ledger
        if isinstance(self.llm, LLMClient):
            token_usage.model = self.llm.active_label
        self._checkpoint_transcript(messages, iteration)
        checkpoint_name = "budget_checkpoint.json" if status == "budget_exhausted" else "progress_checkpoint.json"
        spent = token_usage.total_tokens - self._budget_start_tokens
        provider_reason = getattr(self.llm, "resource_stop_reason", None)
        stop_cause = ("provider_unavailable" if provider_reason and reason == provider_reason else
                      "token_budget" if self.token_budget is not None and spent >= self.token_budget else
                      "runtime_budget" if self.runtime_budget_seconds is not None and
                      _time.monotonic() - self._budget_started >= self.runtime_budget_seconds else "other")
        atomic_write_json(self.run_dir / "agent_state" / checkpoint_name, {
            "reason": reason, "iteration": iteration,
            "stop_cause": stop_cause,
            "pending_candidates": sorted(pending_evaluations(self.run_dir)),
            "token_budget": self.token_budget,
            "tokens_this_invocation": spent,
            "token_overrun": max(0, spent - self.token_budget) if self.token_budget is not None else None,
            "token_accounting_complete": token_usage.to_dict().get("accounting_complete"),
            "runtime_budget_seconds": self.runtime_budget_seconds,
            "elapsed_seconds": _time.monotonic() - self._budget_started,
            "literature_coverage": coverage_summary(self.run_dir, _ledger(self.run_dir)),
        })
        report, n_evals = RunFinalizer(self.run_dir, self.llm, self.log).snapshot(
            query=getattr(self, "_query", "enzyme discovery"), reason=reason)
        atomic_write_json(run_file(self.run_dir, 'token_usage.json'), {
            **token_usage.to_dict(), "request_counts": getattr(self.llm, "request_counts", {}),
            "model_request_counts": getattr(self.llm, "model_request_counts", {})
            if isinstance(self.llm, LLMClient) else {},
        })
        provenance.finalize_manifest(
            self.run_dir, status=status, iterations_used=iteration,
            candidates_evaluated=n_evals, tokens=token_usage.to_dict(), outcome=reason,
            report_path=str(run_file(self.run_dir, 'report_interim.md')),
        )
        self._log_event("run_budget_checkpoint" if status == "budget_exhausted" else "run_progress_checkpoint",
                        {"reason": reason, "iteration": iteration, "resumable": True})
        self.log(f"  {reason}; interim report saved. Resume this run for further discovery.")
        return report

    def _sync_gate_state(self):
        """Sync mutable agent state into the gate state dataclass."""
        # Candidates are registered by database and structural-homolog tools,
        # not only by explicit evaluate_candidate calls.  Use the registry as
        # the source of truth so phase transitions cannot remain in discovery
        # after candidates have already been found.
        registry = load_candidate_registry(self.run_dir).get("candidates", {})
        from enzyme_discovery_agent.models.family_learning import refresh_family_learning
        refresh_family_learning(self.run_dir, registry)
        # Raw registry size can include many close homologs. Phase decisions
        # should reflect distinct coarse families, not an unbounded hit count.
        active_candidates = {
            acc: info for acc, info in registry.items()
            if discovery_status(info) in {"screened", "evaluated", "ranked"}
        }
        self._candidate_count = len({
            family_label(info.get("metadata", {}), acc)
            for acc, info in active_candidates.items()
        })
        self._gate_state.candidate_count = self._candidate_count
        self._gate_state.discovery_coverage_incomplete = False
        from enzyme_discovery_agent.models.rolling import pending_evaluations
        self._gate_state.pending_evaluation_count = len(pending_evaluations(self.run_dir))
        # Count DISTINCT papers for gate sufficiency: now that several sources
        # feed the pool, the same paper arrives multiple times and must not
        # inflate the "enough literature" signal (which would stop search early).
        from enzyme_discovery_agent.tools.literature import dedupe_and_merge
        unique_papers, _ = dedupe_and_merge(self._accumulated_papers)
        from enzyme_discovery_agent.tools.literature import _recover_literature_papers
        unique_papers, _ = dedupe_and_merge(unique_papers + _recover_literature_papers(str(self.run_dir)))
        self._accumulated_papers = unique_papers
        self._gate_state.accumulated_papers = unique_papers
        self._gate_state.lit_search_count = len(
            getattr(self, "_literature_search_iterations", set())
        )
        from enzyme_discovery_agent.output.evolving_map import update_candidate_map
        try:
            update_candidate_map(self.run_dir, iteration=getattr(self, "_iterations_used", None))
        except (OSError, ValueError) as exc:
            logger.warning("Run map update deferred: %s", exc)

    def run(
        self,
        query: str,
        substrate_smiles: str | None = None,
        substrate_name: str | None = None,
        target_profile: dict | None = None,
    ) -> str:
        try:
            with controller_lock(self.run_dir):
                return self._run(
                    query=query,
                    substrate_smiles=substrate_smiles,
                    substrate_name=substrate_name,
                    target_profile=target_profile,
                )
        finally:
            self._close_logging()

    def _run(
        self,
        query: str,
        substrate_smiles: str | None = None,
        substrate_name: str | None = None,
        target_profile: dict | None = None,
    ) -> str:
        self._budget_started = _time.monotonic()
        resume = self._load_resume_payload()
        token_usage = resume["token_usage"] if resume else TokenUsage(model=self.model_name)
        self._run_token_usage = token_usage
        self._budget_start_tokens = token_usage.total_tokens
        from datetime import timezone
        atomic_write_json(self.run_dir / "agent_state" / "invocation_resources.json", {
            "kind": "resume" if resume else "new", "started_at": datetime.now(timezone.utc).isoformat(),
            "starting_tokens": self._budget_start_tokens, "token_budget": self.token_budget,
            "runtime_budget_seconds": self.runtime_budget_seconds,
        })
        if isinstance(self.llm, LLMClient):
            self.llm.attach_usage(token_usage, self.run_dir)
            self.llm.budget_check = self._budget_reason
            self.llm.budget_exhausted = False
        if resume:
            query = str(resume["manifest"].get("query") or query)
            saved_context = resume["target_context"]
            substrate_smiles = saved_context.get("substrate_smiles")
            substrate_name = saved_context.get("substrate_name")
            target_profile = saved_context.get("target_profile")
            manifest = dict(resume["manifest"], status="running")
            atomic_write_json(self.run_dir / "manifest.json", manifest)
        # Show the full backend chain (primary -> fallbacks) so the banner is honest
        # about failover: the run serves on the primary when it is up, else a fallback.
        _fallback_models = [b.model for b in getattr(self.llm, "backends", [])[1:]]
        model_chain = self.model_name + (
            " -> " + " -> ".join(_fallback_models) if _fallback_models else "")
        self.log(f"  Run ID:    {self.run_dir.name}")
        self.log(f"  Model:     {model_chain}")
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
                model=model_chain,
                config_module=config,
                args={"max_iterations": self.max_iterations, "llm_timeout": self.llm_timeout,
                      "token_budget": self.token_budget,
                      "runtime_budget_seconds": self.runtime_budget_seconds},
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
                    validations = []
                    derived = reason_target_substrate_smiles(query, self.llm, on_validation=validations.append)
                    if validations:
                        (run_file(self.run_dir, 'substrate_identity_verification.json')).write_text(
                            json.dumps({"query": query, "substrates": validations}, indent=2) + "\n")
                        for validation in validations:
                            self._log_event("substrate_identity_verification", validation)
                if not derived and not self._budget_reason():
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
        # The ledger was attached before substrate resolution; the contract
        # callback remains compatible with standalone and stub clients.
        # Derive the shared chemistry contract ONCE, at run start, from the target
        # only.  Every downstream stage reads its required-chemistry facts from
        # this single contract instead of re-deriving them independently, so no
        # stage can invent a different account of what chemistry is required.
        try:
            from enzyme_discovery_agent.models.target_reasoning import derive_target_reasoning
            target_reasoning = (json.loads((run_file(self.run_dir, 'target_reasoning.json')).read_text())
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
            atomic_write_json(run_file(self.run_dir, 'target_reasoning.json'), target_reasoning)
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
            if contract_call and not resume and not isinstance(self.llm, LLMClient):
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
        atomic_write_json(run_file(self.run_dir, 'target_context.json'), self._target_context)
        if not resume:
            provenance.refresh_manifest_target_reasoning(self.run_dir)
        if not resume:
            clear_search_history(history=self._search_history)
        self._ensure_artifacts_dir()

        system_prompt = build_system_prompt(
            available_tools=list(get_tools_by_category().keys()),
            capabilities=self.capabilities,
        )

        user_msg = query
        profile = self._target_context["target_profile"]
        if target_profile is not None and not resume:
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
        if resume:
            for message in messages:
                if message.get("role") == "system":
                    message["content"] = system_prompt
                    break

        # ``iteration`` is a budget for substantive model/tool work.  Gate
        # interceptions only redirect orchestration and explicitly restore
        # their credit below, so they cannot silently shorten a campaign.
        iteration = resume["iteration"] if resume else 0
        while True:
            user_reflection = self._service_steering(messages, iteration)
            budget_reason = self._budget_reason()
            if budget_reason:
                return self._finish_budget_checkpoint(messages, iteration, token_usage, budget_reason)
            iteration += 1
            self._iterations_used = iteration
            self._iteration_start = _time.time()
            # Tool calls in the preceding iteration may have promoted a
            # candidate.  Synchronize before announcing state or asking the
            # LLM for another turn, so the evaluation gate cannot miss a
            # screen that occurred near the iteration limit.
            self._sync_gate_state()
            if iteration % 2 == 0:
                # Servicing a lead costs no LLM call, so the cadence should track the backlog
                # rather than a fixed trickle; leads the run paid to discover were otherwise
                # still unresolved when the token budget ended.
                from enzyme_discovery_agent.models.lead_followup import service_leads
                followup=service_leads(self.run_dir)
                if followup['resolved']:
                    from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates
                    refresh_ranked_candidates(self.run_dir)
                    messages.append({'role':'user','content':
                        'Source-grounded literature follow-up linked and assessed these sequence records: '+json.dumps(followup['resolved'])+
                        '. Inspect their current evidence and record target-specific priorities when justified. Identity/native-function resolution does not establish target activity.'})
            self._record_research_progress()
            from enzyme_discovery_agent.models.target_review import review_queue,run_review
            from enzyme_discovery_agent.models.research_decisions import _root as decision_root,_read as read_decision_state
            pending_interpretation=read_decision_state(decision_root(self.run_dir)/'state.json',{}).get('awaiting_interpretation')
            if not pending_interpretation and review_queue(self.run_dir,limit=4):
                review=run_review(self.run_dir,self.llm,limit=4)
                if review['assessed']:
                    from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates
                    refresh_ranked_candidates(self.run_dir)
                self._checkpoint_transcript(messages,iteration)
                if self._budget_reason():
                    return self._finish_budget_checkpoint(messages,iteration,token_usage,self._budget_reason())
                if review['errors'] or review['unassessed']:
                    self.log('Target review checkpoint: '+json.dumps(review))
                else:
                    continue
            from enzyme_discovery_agent.models.literature_bootstrap import bootstrap_status
            literature_bootstrap=bootstrap_status(self.run_dir,{tool['function']['name'] for tool in self.tools},
                mechanisms_derived=self._gate_state.mechanism_families_done)
            reconnaissance=literature_bootstrap['required']
            if not self._gate_state.mechanism_families_done:
                atomic_write_json(self.run_dir/'agent_state/literature_bootstrap.json',literature_bootstrap)
            from enzyme_discovery_agent.models.rolling import fresh_evaluation_backlog
            backlog = fresh_evaluation_backlog(self.run_dir)
            from enzyme_discovery_agent.models.research_decisions import review_due
            adaptive_review = (config.ADAPTIVE_REASONING and review_due(
                self.run_dir, minimum_assessed=config.ADAPTIVE_REVIEW_MIN_ASSESSED))
            if getattr(self, '_invalid_research_decisions', 0) >= 2:
                if backlog:
                    from enzyme_discovery_agent.models.research_decisions import defer_ungrounded_review
                    defer_ungrounded_review(self.run_dir,len(backlog))
                    self._invalid_research_decisions=0
                    adaptive_review=False
                    reconnaissance=False
                    messages.append({'role':'user','content':
                        'The controller deferred an ungrounded review and is resuming ready candidate evaluation. '
                        'Uncertainty and existing scientific priorities remain unchanged; new evidence can reopen review.'})
                elif not getattr(self,'_research_repair_attempted',False):
                    from enzyme_discovery_agent.models.research_decisions import proposal_repair_context
                    repair=proposal_repair_context(self.run_dir)
                    self._research_repair_attempted=True
                    self._research_repair_only=True
                    self._invalid_research_decisions=0
                    messages.append({'role':'user','content':
                        'BOUNDED PROTOCOL REPAIR: interpret_research_outcome can record your interpretation '
                        'of the pending observation without proposing another test, or conclude_research_review '
                        'can close the review provisionally. No scientific interpretation has been supplied by '
                        'the controller. Use the exact saved references: '+json.dumps(repair)})
                else:
                    return self._finish_budget_checkpoint(messages, iteration - 1, token_usage,
                        'Adaptive decision could not be grounded after two attempts; review the preserved evidence and proposals.', status='discovery_incomplete')
            if backlog and not adaptive_review and not reconnaissance:
                ready_gate = self._gate_pipeline.gates[2]
                result = ready_gate.execute(self._gate_state, messages)
                iteration, intercepted = self._apply_gate_result(
                    "evaluation_backlog", result, iteration, messages, discard_assistant=False)
                if intercepted:
                    continue
            forced_reflection = user_reflection or self._automatic_reflection_due()
            # A productive automatic queue still yields periodically to a
            # normal model decision, without injecting a reflection request.
            automation_yield = self._automatic_evidence_batches >= config.MAX_AUTOMATED_BATCHES_BEFORE_REFLECTION
            skip_automation = forced_reflection or automation_yield or adaptive_review or reconnaissance
            from enzyme_discovery_agent.models.recovery import next_recovery_action
            initial_action = next_recovery_action(self.run_dir, {t["function"]["name"] for t in self.tools})
            if (reconnaissance and self._consecutive_no_progress_turns >= 2 and initial_action
                    and initial_action['tool'] in literature_bootstrap['available_tools']):
                # A model ignoring the source-only contract cannot indefinitely
                # starve a grounded, permitted literature retrieval.
                if self._recover_discovery(messages, iteration):
                    continue
            research_due = self._model_turns_since_recovery >= config.DISCOVERY_RESEARCH_INTERVAL
            if not skip_automation and initial_action and research_due:
                if self._recover_discovery(messages, iteration):
                    continue
            self._log_event("iteration_start", {
                "iteration": iteration,
                "phase": self._infer_phase(iteration),
                "candidate_count": self._candidate_count,
            })

            intercepted = False
            if not skip_automation and self._last_automatic_action != "rolling_evaluation":
                ready_gate = self._gate_pipeline.gates[2]
                if ready_gate.should_fire(self._gate_state, False, messages):
                    result = ready_gate.execute(self._gate_state, messages)
                    iteration, intercepted = self._apply_gate_result(
                        "rolling_evaluation", result, iteration, messages, discard_assistant=False)
                if intercepted:
                    continue
            foundation_gates = (
                ("literature_harvest", self._literature_harvest_gate),
                ("mechanism_family", self._mechanism_family_gate),
                ("family_survey", self._family_survey_gate),
            )
            for name, gate in (() if skip_automation else foundation_gates):
                from enzyme_discovery_agent.steering import has_pending
                if has_pending(self.run_dir) or self._budget_reason():
                    intercepted = True
                    break
                if gate.should_fire(self._gate_state, False, messages):
                    result = gate.execute(self._gate_state, messages)
                    iteration, intercepted = self._apply_gate_result(
                        name, result, iteration, messages, discard_assistant=False)
                    if intercepted:
                        break
            if intercepted:
                continue

            # Completed phases are revisited when durable work changes. Run these
            # gates before paying for another model turn, including prose-only turns.
            if not skip_automation and (self._gate_state.critique_injected
                    or self._gate_state.pending_evaluation_count > 0):
                intercepted = False
                for name, gate in [
                    ("homolog_expansion", self._gate_pipeline.gates[3]),
                    ("systematic_coverage", self._systematic_gate),
                    ("rolling_evaluation", self._gate_pipeline.gates[2]),
                ]:
                    from enzyme_discovery_agent.steering import has_pending
                    if has_pending(self.run_dir) or self._budget_reason():
                        intercepted = True
                        break
                    if gate.should_fire(self._gate_state, False, messages):
                        result = gate.execute(self._gate_state, messages)
                        iteration, intercepted = self._apply_gate_result(
                            name, result, iteration, messages, discard_assistant=False)
                        if intercepted:
                            break
                if intercepted:
                    continue

            if (not skip_automation and self._consecutive_no_progress_turns >= 2
                    and self._recover_discovery(messages, iteration)):
                continue

            state_header = build_state_header(
                iteration=iteration, max_iterations=self.max_iterations,
                candidate_count=self._candidate_count,
                searches_done=self._search_log,
                phase=self._infer_phase(iteration),
            )

            reflection_msg = None
            if forced_reflection or (self._consecutive_no_progress_turns >= 2 and
                                    iteration > 1 and iteration % config.REFLECTION_INTERVAL == 0):
                reflection_msg = build_reflection_prompt(iteration, self._candidate_count)
                if forced_reflection:
                    reflection_msg += (
                        f"\n\nThe controller completed {self._automatic_evidence_batches} automated evidence batches "
                        "since your last decision. Review the CURRENT RESEARCH STATE and take a substantive action now. "
                        "Use set_family_focus to record evidence-backed changes in family priorities, or gather "
                        "missing primary evidence through a different source, query or citation route. "
                        "Do not claim coverage is complete from a populated candidate list."
                    )
                self._log_event("reflection_scheduled", {"iteration": iteration,
                    "reason": "automated_batch_limit" if forced_reflection else "periodic",
                    "automated_batches": self._automatic_evidence_batches})
                # Surface a plateau signal (a cue, not a gate): if recent searches
                # stopped finding new families, nudge toward breadth before the run
                # settles into mining one neighbourhood.
                from enzyme_discovery_agent.models.event_log import exploration_saturation
                sat = exploration_saturation(self.run_dir)
                if sat["plateaued"]:
                    reflection_msg += (
                        f"\n\nEXPLORATION SIGNAL: the last {sat['events_in_window']} searches added "
                        f"no new families (and {sat['recent_new_accessions']} new sequence(s)). If "
                        "promising families remain unexplored, search a DIFFERENT one; if you are "
                        "deepening an already-strong family, broaden with expand_candidates_esm "
                        "(steer_mode='explore') rather than re-mining the same neighbourhood."
                    )

            if reflection_msg:
                messages.append({"role": "user", "content": f"{state_header}\n\n{reflection_msg}"})
            elif iteration > 1:
                messages.append({"role": "user", "content": state_header})

            available_tools = self.tools
            if not config.ADAPTIVE_REASONING:
                available_tools = [tool for tool in available_tools if tool['function']['name'] not in
                                   {'plan_research_step', 'get_decision_context'}]
            if self._consecutive_no_progress_turns >= 2 or forced_reflection:
                available_tools = [tool for tool in available_tools
                                   if tool["function"]["name"] not in {"get_notebook", "update_notebook"}]
                messages.append({"role": "user", "content": (
                    ("STRATEGY CHECK: automated evidence work must now yield to a substantive decision. "
                     if forced_reflection else "PROGRESS CHECK: recent turns added no new candidate, paper or evidence. ") +
                    "Notebook activity and completion notices do not advance discovery. "
                    "Take an evidence-gathering action in an unresolved family or candidate, "
                    "or explain that the remaining searches are exhausted."
                )})
            # Early turns must establish chemistry and retrieve evidence.
            # Delaying optional notebook writes avoids provider-generated
            # section-only calls before there is any evidence to summarize.
            if iteration <= 3 or self._notebook_tool_cooldown > 0:
                available_tools = [
                    tool for tool in available_tools
                    if tool["function"]["name"] != "update_notebook"
                ]
                self._notebook_tool_cooldown = max(0, self._notebook_tool_cooldown - 1)
            if self._gate_state.discovery_coverage_incomplete:
                # The coverage gate is a hard evaluation boundary, but it
                # must not discard a whole model turn.  Keep all discovery
                # tools available so the model can close evidence gaps.
                available_tools = [
                    tool for tool in available_tools
                    if tool["function"]["name"] != "evaluate_candidate"
                ]
            if (not forced_reflection and self._gate_state.evidence_selection_prompted
                    and self._candidate_count == 0):
                # Once enough literature is gathered, do not spend the
                # remaining campaign rediscovering the same landscape. Resolve
                # identifiers and screen the bounded evidence-backed list.
                selection_tools = {
                    "screen_candidates", "search_uniprot", "lookup_genbank_to_uniprot",
                    "get_notebook", "get_research_memory",
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
            if adaptive_review:
                available_tools = [tool for tool in self.tools if tool['function']['name'] not in
                                   {'get_notebook', 'update_notebook'}]
                from enzyme_discovery_agent.models.research_decisions import review_limits,review_action_allowed
                review_budget=review_limits(self.run_dir)
                available_tools=[tool for tool in available_tools
                                 if tool['function']['name']=='plan_research_step' and not review_budget['conclusion_required']
                                 or tool['function']['name']!='plan_research_step' and
                                 review_action_allowed(self.run_dir,tool['function']['name'])]
                messages.append({'role': 'user', 'content': (
                    'ADAPTIVE RESEARCH DECISION. Use plan_research_step with one available tool and valid arguments. '
                    'Cite evidence IDs from adaptive_reasoning or get_decision_context. Register a testable hypothesis, '
                    'a competing explanation, contrasting predicted observations, why this test distinguishes them, and a revision condition. '
                    'Interpret the previous action outcome, cite its evidence ID, set revises to its decision ID, and state retain, revise or suspend. '
                    'You may instead call interpret_research_outcome to record that interpretation separately before selecting another action. '
                    'A priority update supplies no new functional evidence. Explain any reinterpretation of existing observations, and choose a discriminating evidence action when more information is needed. '
                    'Native mismatch, retrieval failure and a weak structural proxy have different meanings; none alone establishes target inactivity. '
                    'assess_next_batch gathers a bounded native-evidence batch. set_family_focus can defer a reported family or annotation scope reversibly. '
                    'Check actual test coverage: an unsampled mechanism remains unresolved, not refuted. '
                    'Choose an evidence action rather than a narrative-only answer or a repeated completed query. '
                    'Deliberation is the expensive step and retrieval is the cheap one: having reasoned a mechanism '
                    'family in or out, spend the next actions retrieving its sequence-database members rather than '
                    'restating the reasoning. A family judged relevant and never searched is an untested family, and '
                    'a named protein without a retrieved sequence cannot be assessed or modelled.'
                )})
            if reconnaissance:
                available_tools=[tool for tool in self.tools if tool['function']['name'] in literature_bootstrap['available_tools']]
                messages.append({'role':'user','content':
                    'INITIAL SOURCE RECONNAISSANCE: retrieve primary literature or a supplied primary source before the broad annotation survey. '
                    'Construct a discriminating query from the user target and reaction requirements. Prefer direct target/substrate evidence; '
                    'use a complementary source or reformulate if the first lookup fails or is empty. Names from model memory remain hypotheses '
                    'until retrieved. The next mechanism derivation receives the actual retrieved titles and abstracts. This reconnaissance '
                    'is bounded; unavailable sources will be recorded as degraded evidence, never biological absence.'})
            self._require_research_decision = adaptive_review
            self._decision_action_tools = {tool['function']['name'] for tool in available_tools}
            try:
                model_messages = build_model_context(messages, self.run_dir)
            except ValueError as exc:
                if not str(exc).startswith('Mandatory instructions'):
                    raise
                return self._finish_budget_checkpoint(messages, iteration - 1, token_usage,
                    'Context allowance cannot fit mandatory instructions and source-linked evidence. '
                    'The full history and pending work are preserved for resume.', status='discovery_incomplete')
            from enzyme_discovery_agent import steering
            delivered_ids = [item["id"] for item in steering.read_state(self.run_dir)["instructions"]]
            self._log_event("model_context", {"iteration": iteration,
                "history_characters": len(json.dumps(messages, default=str)),
                "sent_characters": len(json.dumps(model_messages, default=str))})
            from enzyme_discovery_agent.models.research_decisions import inspection_completed
            from enzyme_discovery_agent.context import strategic_tool_definitions
            permitted = (strategic_tool_definitions(available_tools,include_inspections=not inspection_completed(self.run_dir))
                         if adaptive_review else compact_tool_definitions(available_tools))
            from enzyme_discovery_agent.models.research_decisions import review_action_allowed
            if not review_action_allowed(self.run_dir,'interpret_research_outcome'):
                permitted=[tool for tool in permitted if tool['function']['name']!='interpret_research_outcome']
            if getattr(self,'_research_repair_only',False):
                permitted=[tool for tool in permitted if tool['function']['name'] in
                           {'interpret_research_outcome','conclude_research_review'}]
            # The allowlist still decides what may run; it no longer decides what schemas
            # are sent. Varying the schemas rewrote the provider's cache prefix every turn.
            self._offered_direct_tools={tool['function']['name'] for tool in permitted}
            from enzyme_discovery_agent.context import stable_tool_definitions
            model_tools = stable_tool_definitions(self.tools,
                                                  include_inspections=not inspection_completed(self.run_dir))
            # Volatile, so it goes last in the payload and is never written to the durable
            # history: anything placed earlier would invalidate the cache behind it.
            model_messages = model_messages + [{'role': 'user', 'content': 'PERMITTED THIS TURN: '
                + ', '.join(sorted(self._offered_direct_tools))
                + '. Other listed tools are unavailable this turn and will be refused; choose from these.'}]
            from contextlib import nullcontext
            validation_context=self.llm.controller_tool_validation() if isinstance(self.llm,LLMClient) else nullcontext()
            with validation_context:
                response = self.llm.call(
                    messages=model_messages, tools=model_tools,
                    timeout=self.llm_timeout,
                    usage_label=iteration,
                )
            if response is None:
                budget_reason = self._budget_reason()
                if budget_reason:
                    return self._finish_budget_checkpoint(messages, iteration - 1,
                                                          token_usage, budget_reason)
                return self._finalize(messages, token_usage,
                                      f"LLM failed at iteration {iteration}",
                                      status="failed")

            self._automatic_evidence_batches = 0
            token_usage.record(iteration, response)
            steering.mark_delivered(self.run_dir, delivered_ids, iteration)
            if steering.has_pending(self.run_dir):
                # A decision made before new steering arrived must not start work
                # under superseded priorities. Keep its response in the event audit.
                self._log_event("decision_deferred_for_user_steering", {"iteration": iteration})
                self._checkpoint_transcript(messages, iteration)
                continue
            # Per-call token cost into the canonical stream: a 100-iteration
            # campaign's spend is otherwise only a single total at finalize.
            if token_usage.calls and not isinstance(self.llm, LLMClient):
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
            dumped = assistant_msg.model_dump()
            served_model = getattr(response, "_enzyme_serving_model", None)
            if isinstance(served_model, str):
                dumped["_llm_model"] = served_model
            messages.append(dumped)

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

            self.log(f"\n  TURN {iteration}  |  "
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
                "user_steering_ids": delivered_ids,
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
            self._model_turns_since_recovery += 1

            # No tool calls: check if done
            if not tool_calls:
                if adaptive_review:
                    self._invalid_research_decisions = getattr(self, '_invalid_research_decisions', 0) + 1
                self._consecutive_no_tool_turns += 1
                self._record_research_progress(count_turn=True)
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
                return self._finish_budget_checkpoint(
                    messages, iteration, token_usage,
                    "Discovery incomplete: complementary search routes currently exhausted; "
                    "literature and candidate coverage remain unproven", status="discovery_incomplete")

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
            has_focus_call = any(_tc_name(tc) in {"set_family_focus", "plan_research_step", "conclude_research_review"} for tc in tool_calls)

            self._sync_gate_state()
            # Deterministic pre-evaluation gates, all skipped on a focus-call turn
            # (see has_focus_call above) so the agent's set_family_focus tool runs
            # and its stop/expand decisions persist before the eval gates spend
            # budget — they fire on the NEXT turn with the stops in place.  The early
            # gates (literature-first harvest → mechanism-family derivation → family
            # survey → evidence selection → systematic coverage) fire every discovery
            # turn; the pre-evaluation pipeline fires only once there is something to
            # evaluate.  Every gate runs through the one shared intercept ritual.
            if not has_focus_call and not skip_automation:
                intercepted = False
                for name, gate in self._early_gates:
                    if steering.has_pending(self.run_dir):
                        break
                    if not gate.should_fire(self._gate_state, has_eval_calls, messages):
                        continue
                    result = gate.execute(self._gate_state, messages)
                    iteration, intercepted = self._apply_gate_result(
                        name, result, iteration, messages)
                    if intercepted:
                        break
                if intercepted:
                    continue
                if not steering.has_pending(self.run_dir) and (self._gate_state.candidate_count > 0
                        or self._gate_state.pending_evaluation_count > 0
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
                if not isinstance(args, dict):
                    args = {"_parse_error": "Tool arguments must be a JSON object", "_raw": str(args)[:200]}
                parsed.append((tc_id, name, args))

            parallel = [(tc_id, n, a) for tc_id, n, a in parsed if n in self.parallel_safe]
            sequential = [(tc_id, n, a) for tc_id, n, a in parsed if n not in self.parallel_safe]

            results: dict[str, str] = {}
            max_result_chars = config.MAX_RESULT_CHARS

            def _exec(tc_id: str, name: str, args: dict):
                budget_reason = self._budget_reason()
                allowed = {tool["function"]["name"] for tool in available_tools}
                result = ({"status": "skipped", "summary": "Tool unavailable this turn; take an evidence-gathering action."}
                          if name not in allowed else
                          {"status": "skipped", "deferred_for_budget": True,
                           "summary": budget_reason + "; retry this tool on continuation."}
                          if budget_reason else self._execute_model_tool(name, args, iteration))
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
            self._record_research_progress(count_turn=True)
            self._checkpoint_transcript(messages, iteration)
            if reconnaissance:
                permitted={tool['function']['name'] for tool in available_tools}
                self._reconnaissance_contract_failures=(
                    getattr(self, '_reconnaissance_contract_failures', 0)+1
                    if parsed and all(name not in permitted for _, name, _ in parsed) else 0)
                if self._reconnaissance_contract_failures >= 2:
                    return self._finish_budget_checkpoint(messages, iteration, token_usage,
                        'Initial literature retrieval could not be executed: repeated tool selections outside the source contract. '
                        'Evidence remains unresolved; the checkpoint is resumable.', status='discovery_incomplete')
            if self._consecutive_no_progress_turns >= 3 and not self._should_continue(iteration):
                return self._finish_budget_checkpoint(
                    messages, iteration, token_usage,
                    "Discovery stalled: no new research evidence and queued recovery searches exhausted",
                    status="discovery_incomplete",
                )

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
        budget_reason = self._budget_reason()
        if budget_reason:
            return self._finish_budget_checkpoint(messages, self._iterations_used,
                                                  token_usage, budget_reason)
        # Record the backend/model that ACTUALLY served the run (failover-aware), so
        # token_usage.json and the manifest reflect reality, not just the requested model.
        try:
            token_usage.model = self.llm.active_label
        except Exception:  # noqa: BLE001 - provenance must never break finalization
            pass
        try:
            _counts = getattr(self.llm, "request_counts", {}) or {}
            if _counts:
                self.log("  LLM requests:   "
                         + ", ".join(f"{k}={v}" for k, v in _counts.items())
                         + f" (total {sum(_counts.values())})")
        except Exception:  # noqa: BLE001
            pass
        try:
            report, n_evals = RunFinalizer(self.run_dir, self.llm, self.log).finalize(
                messages=messages,
                token_usage=token_usage,
                query=getattr(self, "_query", "enzyme discovery"),
                final_text=final_text,
                target_context=getattr(self, "_target_context", None),
                force_completion=force_completion,
            )
            budget_reason = self._budget_reason()
            if budget_reason:
                return self._finish_budget_checkpoint(messages, self._iterations_used,
                                                      token_usage, budget_reason)
            self.log(f"  Candidates:     {n_evals} evaluated")
            self.log(f"  Total tokens:   {token_usage.total_tokens:,}")
            self._log_event("run_complete", {
                "candidates": n_evals,
                "served_model": token_usage.model,
                "llm_requests": getattr(self.llm, "request_counts", {}),
                "total_tokens": token_usage.total_tokens,
                "prompt_tokens": token_usage.total_prompt_tokens,
                "completion_tokens": token_usage.total_completion_tokens,
                "iterations_used": self._iterations_used,
                "status": status,
            })
            report_path = next(
                (str(p) for name in ("report_confidence.md", "report_explorative.md")
                 if (p := run_file(self.run_dir, name)).exists()),
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
        finally:
            self._close_logging()
