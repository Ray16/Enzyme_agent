"""Gate 2: Select, auto-screen, and auto-evaluate candidate representatives.

Fires before the first evaluate_candidate call (after entity extraction).
Groups exact sequences, preserves fold breadth, and allocates deeper evaluation to
the most relevant families while protecting named and high-fit candidates.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.chemistry_logic import classify_reaction_chemistry
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status,
    discovery_status,
    evidence_fingerprint,
    defer_non_uniprot_deposit,
    family_key as _family_key,
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent.models.event_log import append_event
from enzyme_discovery_agent.models.exploration_log import append_exploration_log
from enzyme_discovery_agent.models.notebook import load_notebook
from enzyme_discovery_agent.tools._http import request_with_retry
from enzyme_discovery_agent.state_store import write_tool_artifact
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")

# ``_family_key`` is re-exported from the registry module (the canonical
# definition) so the registry and every gate group candidates identically.
# Kept under this name for the modules that import it from here.


def _evidence_tier(metadata: dict) -> int:
    """Order representatives by explicit evidence, never a composite score."""
    if metadata.get("direct_target_evidence") or metadata.get("native_substrates"):
        return 0
    if metadata.get("catalytic_activities") or metadata.get("structural_homology"):
        return 1
    return 2


def auto_screen_candidates(run_dir: Path, log_fn=None) -> None:
    """Allocate screening to new discoveries in the current rolling batch.

    Existing screened/evaluated candidates do not consume the new batch's
    representative budget or repeat annotation fetches. Deferred discoveries
    remain in the registry and compete again when evidence or decisions change.
    """
    import concurrent.futures

    from enzyme_discovery_agent.gates.family_relevance import build_relevance_and_allocate
    from enzyme_discovery_agent.tools.reasoning import _fetch_uniprot_entry_light

    registry = load_candidate_registry(run_dir)
    candidates = registry.get("candidates", {})
    if not candidates:
        return

    from enzyme_discovery_agent.models.ortholog import ortholog_key
    family_groups: dict[str, list[tuple[str, dict]]] = {}
    for acc, info in candidates.items():
        if discovery_status(info) != "discovered":
            continue
        if (info.get("metadata") or {}).get("is_literature_lead"):
            continue
        if defer_non_uniprot_deposit(run_dir, acc, info):
            continue
        meta = info.get("metadata", {})
        family_groups.setdefault(ortholog_key(meta) or f"accession:{acc}", []).append((acc, info))

    already_screened = set()
    for acc, info in candidates.items():
        if any(s["status"] in ("screened", "evaluated") for s in info.get("statuses", [])):
            already_screened.add(acc)

    # Candidates the reading step surfaced BY NAME.  These are protected end to
    # end: kept as their own representative through the ortholog collapse below,
    # and force-selected for evaluation regardless of any fold-level stop verdict.
    literature_named = {
        acc for acc, info in candidates.items()
        if str(info.get("metadata", {}).get("literature_name") or "").strip()
    }
    # Candidates PROMOTED by native-substrate resemblance within their fold (the
    # family survey selected them because their native substrate resembles the
    # target — the strongest grounded relevance signal we have).  Protect them so a
    # promoted-then-budget-dropped candidate does not defeat the substrate-fit
    # selection — but budget-consciously: protect only the GLOBALLY best-fit
    # promotions across ALL families (relative fit rank, capped at a resource
    # budget), so a low-fit family's members do not force-consume evaluation budget
    # just for being their own fold's best.  The rest stay registered and eligible
    # via the normal relevance-budgeted screen; they simply are not force-evaluated.
    _promoted = [
        (acc, float(info.get("metadata", {}).get("substrate_fit") or 0.0))
        for acc, info in candidates.items()
        if str(info.get("metadata", {}).get("source_tool") or "")
        .startswith("family_substrate_fit")
    ]
    _promoted.sort(key=lambda x: (-x[1], x[0]))   # best fit first, stable tiebreak
    substrate_fit_promoted = {acc for acc, _ in _promoted[:config.FAMILY_PROTECT_BUDGET]}
    protected_named = literature_named | substrate_fit_promoted

    # One representative per exact sequence where known; otherwise keep each
    # accession separate. Fold-relevance budgeting selects evaluation work.
    gene_reps: list[str] = []
    for family, members in family_groups.items():
        # Keep a member if it is known to be full-length (>=100 aa) OR if its
        # length is not known yet.  Full-text-mined accessions are registered
        # WITHOUT a length (only their source); screening is the step that
        # fetches the length, so applying a >=100 cutoff here — before the
        # length exists — silently discards exactly the deterministically
        # recovered enzymes we mined the literature to reach.  Genuine short
        # fragments (a KNOWN length below 100) are still dropped.
        members = [
            (acc, info) for acc, info in members
            if (info.get("metadata", {}).get("seq_length") or 0) >= 100
            or not info.get("metadata", {}).get("seq_length")
        ]
        if not members:
            continue
        members.sort(
            key=lambda x: x[1].get("metadata", {}).get("seq_length", 0),
            reverse=True,
        )
        # The first member represents an exact-sequence group, PLUS any literature-named
        # member of the group: a named hit must never be collapsed away behind a
        # longer ortholog, or the downstream protection can't reach it.
        reps_here = [members[0][0]] + [
            acc for acc, _ in members[1:] if acc in protected_named
        ]
        for acc in reps_here:
            if acc not in gene_reps:
                gene_reps.append(acc)

    if not gene_reps:
        return

    # Fetch light annotations (EC, catalytic activity, InterPro folds, length)
    # for every representative up front — this is the same volume of light
    # fetches as before (we previously fetched one per screened rep), just moved
    # ahead of selection so fold and direction are known when budgeting.
    def _light(acc):
        metadata = candidates.get(acc, {}).get("metadata", {})
        cached = metadata.get("screening_annotation")
        if isinstance(cached, dict) and cached.get("accession") == acc:
            return acc, cached
        if metadata.get("is_metagenomic"):
            sequence = str(metadata.get("sequence") or "")
            return acc, {
                "accession": acc,
                "protein_name": metadata.get("protein_name", ""),
                "ec_numbers": [],
                "catalytic_activities": [],
                "function": "Unannotated metagenomic structural homolog",
                "organism": "",
                "gene_names": [],
                "interpro_ids": metadata.get("interpro_ids", []),
                "seq_length": len(sequence),
                "is_metagenomic": True,
            }
        return acc, _fetch_uniprot_entry_light(acc)

    entry_by_acc: dict[str, dict] = {}
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry_batch
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.EVAL_POOL_WORKERS
    ) as pool:
        for acc, entry in pool.map(_light, gene_reps):
            if entry:
                entry_by_acc[acc] = entry
    update_candidate_registry_batch(run_dir, [
        (acc, None, {"screening_annotation": {**entry, "accession": acc}})
        for acc, entry in entry_by_acc.items()
    ])

    if not entry_by_acc:
        return

    # Regroup by fold and spend the evaluation budget by grounded relevance.
    # Protected candidates — surfaced BY NAME in the literature, OR promoted by
    # native-substrate resemblance within their fold — are evaluated on their own
    # evidence even if their fold is later stopped, so a specific relevant enzyme is
    # never deleted by a fold-level breadth verdict made from generic annotation.
    reps = [(acc, {
        **entry_by_acc[acc],
        "substrate_fit": (candidates[acc].get("metadata") or {}).get("substrate_fit"),
        "survey_functional_group_similarity": (candidates[acc].get("metadata") or {}).get("survey_functional_group_similarity"),
    }) for acc in gene_reps if acc in entry_by_acc]
    selected, landscape = build_relevance_and_allocate(
        run_dir, reps, protected=protected_named)
    family_by_acc = {
        acc: item.get("fold_id")
        for item in landscape
        if item.get("fold_id")
        for acc in item.get("members", [])
    }

    if log_fn:
        log_fn(f"  Auto-screening {len(selected)}/{len(reps)} candidate representatives "
               f"across {len(landscape)} fold families "
               f"({len(already_screened)} already screened)")

    screened_count = 0
    screening_updates = []
    for acc in selected:
        if acc in already_screened:
            continue
        entry = entry_by_acc.get(acc)
        if entry:
            screening_updates.append((acc, "screened", {
                "protein_name": entry.get("protein_name", ""),
                "organism": entry.get("organism", ""),
                "ec_numbers": entry.get("ec_numbers", []),
                "catalytic_activities": entry.get("catalytic_activities", []),
                "function": entry.get("function", ""),
                # Persist the fold ids fetched free with the light screen call so
                # family grouping survives past this transient allocation step.
                "interpro_ids": entry.get("interpro_ids", []),
                "screening_family_id": family_by_acc.get(acc),
                "evaluation_selected": True,
                "is_metagenomic": bool(entry.get("is_metagenomic")),
            }))
            screened_count += 1

    update_candidate_registry_batch(run_dir, screening_updates)

    lower_depth_folds = [f for f in landscape if f["n_representatives"] > f["n_selected_for_evaluation"]]
    append_exploration_log(
        run_dir,
        entry_type="SCREEN",
        title=f"Auto-screened {screened_count} representatives for progressive evaluation",
        details={
            "Candidate representatives": len(reps),
            "Fold families": len(landscape),
            "Selected for evaluation": len(selected),
            "Deferred after light screening": len(reps) - len(selected),
            "Folds given lower ranking depth by relevance": len(lower_depth_folds),
            "Already screened": len(already_screened),
            "Newly screened": screened_count,
        },
    )
    append_event(
        run_dir, "candidates_screened",
        newly_screened=screened_count,
        distinct_genes=len(reps),
        grouping_basis="exact known sequence, otherwise accession",
        fold_families=len(landscape),
        selected_for_evaluation=len(selected),
        folds_held_at_floor=len(lower_depth_folds),
        already_screened=len(already_screened),
    )
    if log_fn:
        log_fn(f"  Auto-screening complete: {screened_count} newly screened")


def _extract_target_params(run_dir: Path) -> dict:
    """Load durable target context, with notebook parsing only as legacy fallback."""
    context_path = run_file(run_dir, 'target_context.json')
    try:
        context = json.loads(context_path.read_text())
        profile = context.get("target_profile", {})
        reaction = profile.get("reaction", {})
        substrate = profile.get("substrate", {})
        reaction_text = " ".join(filter(None, [
            str(reaction.get("class", "")),
            str(reaction.get("direction", "")),
            str(context.get("query", "")),
        ]))
        if reaction_text:
            return {
                "target_ec_class": str(reaction.get("ec_class", "")),
                "target_reaction_type": reaction_text,
                "target_substrate_smiles": str(context.get("substrate_smiles") or ""),
                "target_substrate_description": json.dumps(substrate, sort_keys=True),
            }
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    # Legacy runs written before target_context.json.
    notebook = load_notebook(run_dir)
    reaction_text = notebook.get("sections", {}).get("reaction_analysis", "")

    smiles_match = re.search(r'SMILES[:\s]+([A-Za-z0-9@+\-\[\]\\/()\.\=#]+)', reaction_text)
    smiles_list = []
    for m in re.finditer(r'(?:SMILES|validated)[:\s]+([A-Za-z0-9@+\-\[\]\\/()\.\=#]{3,})', reaction_text):
        s = m.group(1).strip()
        if s and len(s) >= 3:
            smiles_list.append(s)

    ec_match = re.search(r'EC\s*(\d\.\d+(?:\.\d+)?)', reaction_text)

    return {
        "target_ec_class": ec_match.group(1) if ec_match else "",
        "target_reaction_type": reaction_text[:200] if reaction_text else "",
        "target_substrate_smiles": ".".join(smiles_list[:2]) if smiles_list else "",
        "target_substrate_description": "",
    }


def _auto_evaluate_all(run_dir: Path, log_fn=None, llm_client=None, limit=None, native_first=False, stats=None, fresh_only=False, candidate_accessions=None) -> int:
    """Auto-evaluate ALL screened candidates that haven't been evaluated yet."""
    from enzyme_discovery_agent.tools.reasoning import evaluate_candidate

    registry = load_candidate_registry(run_dir)
    if stats is None:
        stats = {}
    stats.update(attempted=0, supported=0, evidence_pending=0, rejected=0, failed=0)
    target_params = _extract_target_params(run_dir)

    to_evaluate = []
    from enzyme_discovery_agent.models.family_decisions import paused_folds
    paused = paused_folds(run_dir)
    from enzyme_discovery_agent.models.rolling import evaluation_ready, evaluation_basis
    for acc, info in registry.get("candidates", {}).items():
        if candidate_accessions is not None and acc not in candidate_accessions:
            continue
        status = current_candidate_status(info)
        if fresh_only and status not in {'screened', 'reopened'}:
            continue
        metadata = info.get("metadata", {})
        new_evidence = bool(metadata.get("literature_evidence")) and (
            evidence_fingerprint(metadata["literature_evidence"])
            != metadata.get("literature_recovery_fingerprint"))
        if evaluation_ready(info, paused=paused) and (status != "evidence_pending" or llm_client is not None):
            to_evaluate.append(acc)

    if not to_evaluate:
        return 0
    if limit is not None:
        # Fresh paper-linked leads enter the feedback loop ahead of an older
        # database tail. Remaining screened representatives stay in the queue.
        to_evaluate.sort(key=lambda acc: (
            current_candidate_status(registry["candidates"][acc]) not in {"screened", "reopened"},
            not bool(registry["candidates"][acc].get("metadata", {}).get("literature_name")),
            _evidence_tier(registry["candidates"][acc].get("metadata", {})), acc))
        from enzyme_discovery_agent.models.rolling import diversify_evaluation_batch
        to_evaluate = diversify_evaluation_batch(registry["candidates"], to_evaluate, limit)

    if log_fn:
        log_fn(f"  Auto-evaluating {len(to_evaluate)} screened candidates")
    stats["attempted"] = len(to_evaluate)
    stats["accessions_attempted"] = list(to_evaluate)

    evaluated = 0
    evidence_pending = 0
    artifacts_dir = run_dir / "artifacts" / "candidates"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime as _dt, timezone as _timezone

    def record_failure(acc, reason):
        import time
        meta = load_candidate_registry(run_dir).get("candidates", {}).get(acc, {}).get("metadata", {})
        previous = meta.get("evaluation_failure") or {}
        basis = evaluation_basis(meta)
        attempts = previous.get("attempts", 0) + 1 if previous.get("basis") == basis else 1
        update_candidate_registry(run_dir, acc, None, {"evaluation_failure": {
            "basis": basis, "attempts": attempts, "retry_after": time.time() + 60, "reason": reason}})

    def _eval_one(acc):
        budget_check = getattr(llm_client, "budget_check", None)
        if budget_check and budget_check():
            llm_client.budget_exhausted = True
            return acc, None
        try:
            use_llm = None if native_first and current_candidate_status(registry["candidates"][acc]) == "screened" else llm_client
            result = evaluate_candidate(
                run_dir=run_dir,
                accession=acc,
                llm_client=use_llm,
                **target_params,
            )
            if use_llm is not None and not getattr(llm_client, "budget_exhausted", False):
                update_candidate_registry(run_dir, acc, None,
                    metadata={"literature_recovery_attempted": True,
                              "literature_recovery_fingerprint": evidence_fingerprint(
                                  load_candidate_registry(run_dir).get("candidates", {}).get(
                                      acc, {}).get("metadata", {}).get("literature_evidence"))})
            if result and result.get("status") == "success":
                update_candidate_registry(run_dir, acc, None, {"evaluation_failure": None})
                write_tool_artifact(run_dir, "candidates", "evaluate_candidate", {
                    "tool_name": "evaluate_candidate",
                    "args": {"accession": acc},
                    "timestamp": _dt.now(_timezone.utc).isoformat(),
                    "iteration": "auto",
                    "result": result,
                })
            elif not getattr(llm_client, "budget_exhausted", False):
                record_failure(acc, "evaluation returned no successful evidence")
            return acc, result
        except Exception as e:
            logger.warning("Auto-evaluate failed for %s: %s", acc, e)
            record_failure(acc, type(e).__name__)
            return acc, None

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.EVAL_POOL_WORKERS
    ) as pool:
        futures = [pool.submit(_eval_one, acc) for acc in to_evaluate]
        for future in concurrent.futures.as_completed(futures):
            acc, result = future.result()
            if result and result.get("status") == "success":
                validation = (result.get("data") or {}).get("claim_validation")
                if validation and validation.get("promotion_eligible") is not True:
                    evidence_pending += 1
                else:
                    evaluated += 1

    if log_fn:
        log_fn(f"  Auto-evaluation complete: {evaluated}/{len(to_evaluate)} supported, "
               f"{evidence_pending} awaiting evidence")

    final_registry = load_candidate_registry(run_dir).get("candidates", {})
    from collections import Counter
    outcomes = Counter(discovery_status(final_registry.get(acc, {})) for acc in to_evaluate)
    stats.update(supported=evaluated, evidence_pending=outcomes["evidence_pending"],
                 rejected=outcomes["rejected"],
                 failed=sum(bool((final_registry.get(acc, {}).get("metadata") or {}).get("evaluation_failure"))
                            for acc in to_evaluate))

    append_exploration_log(
        run_dir,
        entry_type="EVAL",
        title=f"Auto-evaluated {evaluated} candidates",
        details={
            "Total screened": len(to_evaluate),
            "Successfully evaluated": evaluated,
        },
    )
    append_event(
        run_dir, "candidates_evaluated",
        evaluated=evaluated,
        evidence_pending=evidence_pending,
        screened_pending=len(to_evaluate),
    )
    from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates
    refresh_ranked_candidates(run_dir)

    return evaluated


def _check_landscape_coverage(run_dir: Path) -> tuple[bool, list[str]]:
    """Check if the agent has searched for the genes it enumerated in its landscape.

    Returns (is_covered, missing_genes). Coverage is considered sufficient
    when at least 70% of landscape genes have been searched.
    """
    registry = load_candidate_registry(run_dir).get("candidates", {})
    represented_families = {_family_key(info.get("metadata", {}), acc) for acc, info in registry.items()}
    if len(represented_families) >= 3:
        # Focused discovery has sufficient mechanistic breadth; do not force
        # searches for every incidental gene mentioned in a literature note.
        return True, []

    notebook = load_notebook(run_dir)
    landscape = notebook.get("sections", {}).get("enzyme_landscape", "")
    if not landscape:
        return True, []

    gene_re = re.compile(r'\b([A-Z][a-z]{1,3}[A-Z]\d{0,2})\b')
    blacklist = {
        "ATP", "ADP", "AMP", "GTP", "CoA", "NAD", "FAD", "NRPS", "KEGG",
        "PDB", "NCBI", "BLAST", "FPLC", "HPLC", "DMSO", "HEPES",
        "Fig", "Tab", "Ref",
    }
    landscape_genes = set()
    for m in gene_re.finditer(landscape):
        name = m.group(1)
        if name not in blacklist:
            landscape_genes.add(name.lower())

    if not landscape_genes:
        return True, []

    log_path = run_file(run_dir, 'exploration_log.md')
    if not log_path.exists():
        return False, sorted(landscape_genes)

    log_text = log_path.read_text()
    searched = set()
    for m in re.finditer(r'gene:(\w+)', log_text):
        searched.add(m.group(1).lower())
    for m in re.finditer(r'query:.*?(\w{3,})', log_text):
        searched.add(m.group(1).lower())

    missing = landscape_genes - searched
    coverage = 1.0 - (len(missing) / len(landscape_genes)) if landscape_genes else 1.0

    return coverage >= 0.7, sorted(missing)


class ECTriageGate:
    """Auto-screen and evaluate selected representatives, then present results."""

    def __init__(self, llm_client=None):
        self._coverage_redirected = False
        self.llm = llm_client

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        # Screening is an explicit candidate-promotion action.  Once one or
        # more candidates exist, evaluation must not depend on the LLM
        # remembering to issue a second, redundant evaluate_candidate call.
        from enzyme_discovery_agent.models.rolling import new_evaluation_work
        return not state.discovery_coverage_incomplete and new_evaluation_work(state.run_dir)

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        from enzyme_discovery_agent.models.rolling import fresh_evaluation_backlog
        ready_backlog = fresh_evaluation_backlog(state.run_dir)
        if not ready_backlog and not self._coverage_redirected:
            covered, missing = _check_landscape_coverage(state.run_dir)
            if not covered and missing:
                self._coverage_redirected = True
                missing_str = ", ".join(missing[:30])
                logger.info("Redirecting: %d landscape genes not yet searched", len(missing))
                return GateResult(
                    intercept=True,
                    new_messages=[
                        {
                            "role": "user",
                            "content": (
                                f"STOP — you listed {len(missing)} genes in your enzyme "
                                f"landscape that you have NOT searched for yet. Search "
                                f"UniProt for these genes BEFORE evaluating:\n\n"
                                f"{missing_str}\n\n"
                                f"Search each individually using gene_name parameter. "
                                f"Then call evaluate_candidate to trigger evaluation."
                            ),
                        },
                    ],
                    log_message=f"Coverage redirect: {len(missing)} genes unsearched",
                )

        if not ready_backlog:
            logger.info("Selecting and screening discovered candidate representatives")
            auto_screen_candidates(state.run_dir, log_fn=lambda msg: logger.info(msg))

        logger.info("Auto-evaluating selected screened candidates")
        step_stats = {}
        n_evaluated = _auto_evaluate_all(
            state.run_dir, log_fn=lambda msg: logger.info(msg), llm_client=self.llm,
            limit=config.EVALUATIONS_PER_STEP,
            native_first=True,
            stats=step_stats,
        )
        from enzyme_discovery_agent.models.rolling import record_evaluation_batch
        if not getattr(self.llm, "budget_exhausted", False):
            record_evaluation_batch(state.run_dir)

        if not step_stats.get("attempted", 0):
            return GateResult(intercept=False,
                              state_updates={"critique_injected": True, "eval_gate_fired": True})

        registry = load_candidate_registry(state.run_dir)
        eval_lines = []
        for acc, info in registry.get("candidates", {}).items():
            if discovery_status(info) not in {"evaluated", "ranked"}:
                continue
            meta = info.get("metadata", {})
            name = meta.get("protein_name", "unnamed")
            genes = meta.get("gene_names", [])
            gene = genes[0] if genes else ""
            ec_list = meta.get("ec_numbers", [])
            ec_str = ec_list[0] if ec_list else "no EC"
            cat_acts = meta.get("catalytic_activities", [])
            reaction = cat_acts[0][:80] if cat_acts else "no reaction data"
            func = meta.get("function", "")
            func_tag = f" | Function: {func[:60]}" if func else ""

            eval_lines.append(
                f"  - {gene} ({acc}) {name} | EC {ec_str} | {reaction}{func_tag}"
            )

        if not eval_lines:
            return GateResult(
                intercept=step_stats.get("attempted", 0) > 0,
                log_message=f"Evaluation batch: {step_stats.get('attempted', 0)} attempted; no supported records yet.",
                state_updates={"critique_injected": True, "eval_gate_fired": True},
            )

        summary = (f"Evaluation step: {step_stats.get('attempted', 0)} attempted; "
                   f"{n_evaluated} supported, {step_stats.get('evidence_pending', 0)} awaiting evidence, "
                   f"{step_stats.get('rejected', 0)} rejected, {step_stats.get('failed', 0)} failed; "
                   f"{len(eval_lines)} total supported evaluations")
        logger.info(summary)

        new_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Auto-screening and auto-evaluation complete. "
                    f"{len(eval_lines)} candidates have supported evaluation records. "
                    f"Batch outcomes: {summary}. Missing evidence remains explicitly unresolved."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The {len(eval_lines)} selected representatives have been "
                    f"evaluated using currently available evidence. "
                    f"These records do not imply complete literature, structural modeling, "
                    f"or demonstrated target activity. Inspect the evidence and gaps for each candidate.\n\n"
                    f"Evaluated candidates:\n"
                    + "\n".join(sorted(eval_lines))
                    + "\n\nUpdate candidate priorities from this evaluation batch. "
                    "Expand the most promising evaluated leads, then evaluate new "
                    "candidates as discovery continues. Avoid repeating completed "
                    "evaluations unless new evidence changes the candidate."
                ),
            },
        ]

        return GateResult(
            intercept=True,
            new_messages=new_messages,
            log_message=summary,
            state_updates={"critique_injected": True, "eval_gate_fired": True},
        )
