"""Shared selection and durable search history for evaluated expansion seeds."""

from __future__ import annotations

from enzyme_discovery_agent.models.candidate_evidence import candidate_record
import json
import hashlib
import math
import time
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import discovery_status, load_candidate_registry
from enzyme_discovery_agent.models.event_log import append_event
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent import config


def expansion_feedback(candidates, policy, searches, method):
    """Learn method yield from assessed independent groups, never raw hit counts."""
    from enzyme_discovery_agent.models.candidate_registry import family_label
    from enzyme_discovery_agent.models.ortholog import tentative_label_key as ortholog_key
    groups = {}
    for acc, entry in candidates.items():
        data = seed_evidence(acc, entry)
        groups[acc] = ortholog_key(data) or f"accession:{acc}"
    output = {}
    for seed, attempt in searches.get(method, {}).items():
        if seed not in candidates or attempt.get("status") not in {"success", "partial"}:
            continue
        family = family_label(seed_evidence(seed, candidates[seed]), seed)
        row = output.setdefault(family, {"attempts": 0, "pending": set(), "supported": set(),
                                        "contradicted": set(), "duplicates": set(), "seed_groups": set(), "seed_fits": []})
        row["attempts"] += 1
        row["seed_groups"].add(groups[seed])
        fit = policy.get("candidates", {}).get(seed, {}).get("fit")
        if fit is not None:
            row["seed_fits"].append(fit)
        children = set(attempt.get("new_accessions") or [])
        # Older runs recorded provenance on candidates instead of search rows.
        for acc, entry in (candidates.items() if "new_accessions" not in attempt else []):
            meta = entry.get("metadata") or {}
            structural = meta.get("structural_homology") or {}
            sequence = meta.get("sequence_expansion") or {}
            if (method == "foldseek" and structural.get("seed_accession") == seed
                    and meta.get("source_tool") != "search_esm_atlas_homologs"
                    or method.startswith("esm_expansion") and
                    (sequence.get("seed_accession") == seed or meta.get("source_tool") == f"esm_expansion:{seed}")):
                children.add(acc)
        for acc in children:
            if acc not in candidates:
                continue
            group = groups[acc]
            if group == groups[seed]:
                row["duplicates"].add(group)
                continue
            state = policy.get("candidates", {}).get(acc, {}).get("state")
            if state in {"supported", "deprioritized"}:
                row["supported" if state == "supported" else "contradicted"].add(group)
            else:
                row["pending"].add(group)
    for row in output.values():
        row["pending"] -= row["supported"] | row["contradicted"]
        for field in ("pending", "supported", "contradicted", "duplicates", "seed_groups"):
            row[field] = sorted(row[field])
    return output


def expansion_assessments(run_dir, candidates, policy, method):
    """Return evidence-backed decisions about whether more expansion can help."""
    from enzyme_discovery_agent.models.candidate_registry import family_label
    from enzyme_discovery_agent.models.ortholog import tentative_label_key as ortholog_key
    try:
        searches = json.loads((Path(run_dir) / "agent_state/expansion_searches.json").read_text())
    except FileNotFoundError:
        searches = {}
    outcomes = expansion_feedback(candidates, policy, searches, method)
    decisions = {}
    for acc, entry in candidates.items():
        if discovery_status(entry) not in {'evaluated','ranked'}:
            continue
        data = seed_evidence(acc, entry)
        family = family_label(data, acc)
        feedback = policy.get("candidates", {}).get(acc, {})
        family_policy = policy.get("families", {}).get(family, {})
        previous = outcomes.get(family, {})
        eligible=seed_eligible(entry)
        focused=(entry.get('metadata') or {}).get('priority_assessment',{}).get('decision')=='prioritize' and eligible
        decision,reason=("expand","eligible evaluated seed; test for useful independent candidates") if eligible else (
            'gather_evidence','Current claim, role or priority evidence does not support expansion of this seed.')
        if feedback.get('role_review_required'):
            decision, reason = 'gather_evidence', feedback['next_question']
        elif feedback.get("state") == "deprioritized" and not focused or family_policy.get("agent_decision") == "stop":
            decision, reason = "gather_evidence", "recorded chemistry or family decision does not support expansion"
        elif previous.get("pending"):
            decision, reason = "evaluate_returned", "assess independent candidates from the previous expansion first"
        elif previous.get("attempts", 0) >= 2 and not previous.get("supported"):
            fit = feedback.get("fit")
            best = max(previous.get("seed_fits") or [-1])
            if not (data.get("direct_target_evidence") or fit is not None and fit > best):
                decision, reason = "change_method", "repeated completed expansions added no supported independent group"
        decisions[acc] = {"decision": decision, "reason": reason, "family": family,
                          "group": ortholog_key(data) or f"accession:{acc}",
                          "native_support": feedback.get("state", "unresolved"),
                          "previous_method_outcome": previous}
    return decisions


def seed_evidence(accession: str, entry: dict) -> dict:
    metadata = entry.get("metadata") or {}
    return {**candidate_record(metadata), "accession": accession}


def seed_eligible(entry: dict) -> bool:
    """Require current evaluation and respect contradictions in both payloads."""
    if discovery_status(entry) not in {"evaluated", "ranked"}:
        return False
    metadata = entry.get("metadata") or {}
    if (metadata.get('priority_assessment') or {}).get('decision') in {'exclude','deprioritize'}:
        return False
    for data in (metadata, metadata.get("evaluation_data") or {}):
        from enzyme_discovery_agent.models.evaluations import priority_concerns
        if priority_concerns(data):
            return False
        validation = data.get("claim_validation")
        if validation and validation.get("promotion_eligible") is not True:
            return False
        if data.get("is_enzyme") is False or data.get("critical_warnings"):
            return False
        if any(any(term in str(warning) for term in (
            "BOND TYPE MISMATCH", "SEVERE SIZE MISMATCH", "REACTION DIRECTION MISMATCH",
        )) for warning in data.get("weaknesses") or []):
            return False
    return True


def select_expansion_seeds(candidates: dict, n: int, run_dir: Path | str | None = None,
                           *, method: str = "", record_decision: bool = True) -> list[str]:
    """Select evaluated candidates by saved rank or the report's evidence order.

    A saved ranking is used only if it covers the entire eligible set; otherwise
    newly evaluated candidates compete with existing leads on current evidence.
    No discovered/screened fallback and no additional model request is needed.
    """
    if n <= 0:
        return []
    eligible = {a: e for a, e in candidates.items() if seed_eligible(e)}
    if not eligible:
        return []
    ranks = {}
    if run_dir:
        try:
            rows = json.loads((run_file(Path(run_dir), 'pass1_ranking.json')).read_text())
            for row in rows if isinstance(rows, list) else []:
                rank = float(row.get("rank", 0))
                if row.get("accession") in eligible and math.isfinite(rank) and rank > 0:
                    ranks[row["accession"]] = rank
        except (OSError, ValueError, TypeError, AttributeError):
            ranks = {}
    basis = "saved_ranking"
    if set(ranks) == set(eligible):
        ordered = sorted(eligible, key=lambda a: (ranks[a], a))
    else:
        # Reuse the ranking policy rather than invent a separate expansion score.
        from enzyme_discovery_agent.output.report_generator import (
            _deterministic_mechanistic_ranking, _family_relevance_partition, _family_rank_map,
        )
        evaluations = [{"data": seed_evidence(a, e)} for a, e in eligible.items()]
        family_ranks = {}
        if run_dir:
            stopped, on_target, _ = _family_relevance_partition(Path(run_dir), evaluations)
            family_ranks = _family_rank_map(stopped, on_target)
        fits = {}
        for a, e in eligible.items():
            try:
                fit = float(seed_evidence(a, e).get("substrate_fit"))
                if math.isfinite(fit) and 0 <= fit <= 1:
                    fits[a] = fit
            except (TypeError, ValueError):
                pass
        ordered = [row["accession"] for row in _deterministic_mechanistic_ranking(
            evaluations, {}, family_rank_by_acc=family_ranks,
            fallback_substrate_fit_by_acc=fits,
        )]
        basis = "current_evaluation_evidence"
    if run_dir:
        from enzyme_discovery_agent.models.family_learning import refresh_family_learning, prioritize_seeds
        policy = refresh_family_learning(run_dir, candidates)
        ordered = prioritize_seeds(ordered, candidates, policy)
        if any(row.get("state") != "unresolved" for row in policy["candidates"].values()):
            basis = "adaptive_native_chemistry_feedback"
        if config.ENABLE_EXPANSION_FEEDBACK and method and method != "sequence_literature":
            decisions = expansion_assessments(run_dir, candidates, policy, method)
            chosen, families = [], set()
            attempted = searched_seeds(run_dir, method)
            # Count distinct families in the leading window. Several high-ranked
            # members of one family must not consume the breadth allowance and
            # silently exclude another family's unresolved representative.
            leader_window, leading_families = set(), set()
            for acc in ordered:
                family = decisions.get(acc, {}).get('family')
                if family not in leading_families:
                    leader_window.add(acc)
                    leading_families.add(family)
                    if len(leader_window) == n:
                        break
            for acc in ordered:
                decision = decisions.get(acc, {})
                if (acc in attempted or decision.get("decision") != "expand"
                        or decision.get("family") in families):
                    continue
                if decision.get("native_support") != "supported" and acc not in leader_window:
                    continue
                chosen.append(acc)
                families.add(decision.get("family"))
            ordered = chosen
            if record_decision:
                path = Path(run_dir) / "agent_state/expansion_policy.json"
                payload = {"decisions": decisions, "recommended": ordered[:n]}
                signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                with state_lock(path):
                    try:
                        saved = json.loads(path.read_text())
                    except FileNotFoundError:
                        saved = {}
                    previous = saved.get(method, {})
                    if previous.get("signature") != signature:
                        saved[method] = {**payload, "signature": signature}
                        atomic_write_json(path, saved)
                        append_event(Path(run_dir), "expansion_policy", method=method,
                                     recommended=ordered[:n], signature=signature)
    # A saved complete ranking and disabled adaptive feedback must not restore
    # a fit whose participant-role interpretation is unresolved or stale.
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
    ordered = [acc for acc in ordered if not role_transfer_summary(seed_evidence(acc, eligible[acc]))['review_required']]
    selected = ordered[:n]
    if run_dir and method and record_decision:
        append_event(Path(run_dir), "expansion_seed_selection", method=method,
                     basis=basis, eligible_count=len(eligible), selected=selected, limit=n)
    return selected


def require_evaluated_seed(run_dir: Path | str, accession: str, *, method="") -> dict | None:
    """Reject explicit requests before any network or GPU work starts."""
    if run_dir:
        entry = load_candidate_registry(Path(run_dir), strict=True).get("candidates", {}).get(accession)
        if entry and seed_eligible(entry):
            from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
            roles = role_transfer_summary(seed_evidence(accession, entry))
            if roles['review_required']:
                return {'status':'skipped', 'count':0, 'data':roles,
                        'summary':'Expansion deferred pending participant-role evidence: '+roles['next_question']}
            if method and config.ENABLE_EXPANSION_FEEDBACK:
                from enzyme_discovery_agent.models.family_learning import refresh_family_learning
                candidates = load_candidate_registry(Path(run_dir), strict=True).get("candidates", {})
                policy = refresh_family_learning(run_dir, candidates)
                decision = expansion_assessments(run_dir, candidates, policy, method).get(accession, {})
                if decision.get("decision") != "expand":
                    return {"status": "skipped", "count": 0, "data": decision,
                            "summary": f"Expansion deferred: {decision.get('reason', 'evaluate evidence first')}"}
            return None
    return {"status": "skipped", "count": 0, "data": None,
            "summary": f"Expansion deferred: {accession} must be evaluated and eligible in the current run first."}


def expansion_due(run_dir: Path | str, interval: int) -> bool:
    """Has evaluation progressed far enough to owe expansion a reserved turn?

    Deepening (evaluation) would otherwise intercept every turn while a backlog
    exists, so breadth never runs. This reserves one expansion pass per ``interval``
    evaluation batches: a floor that keeps expansion reachable, not a cap on the
    agent, which can still expand on its own initiative through the search tools.
    Measured from the durable event log so it survives a resume.
    """
    if interval <= 0:
        return True
    from enzyme_discovery_agent.models.event_log import read_events
    eval_batches = expansions = 0
    for event in read_events(run_dir):
        if event.get("type") != "gate":
            continue
        gate = event.get("gate")
        if gate in {"evaluation_backlog", "pre_evaluation"}:
            eval_batches += 1
        elif gate == "homolog_expansion":
            expansions += 1
    # Reserve the next pass once evaluation has advanced a full interval beyond
    # what the expansions already run have been paid for.
    return eval_batches >= (expansions + 1) * interval


def searched_seeds(run_dir: Path | str, method: str) -> set[str]:
    try:
        document = json.loads((Path(run_dir) / "agent_state" / "expansion_searches.json").read_text())
    except FileNotFoundError:
        return set()
    return {acc for acc, row in (document.get(method) or {}).items()
            if row.get("status") in {"success", "empty"}
            or row.get("retry_after", 0) > time.time()}


def record_seed_search(run_dir: Path | str, method: str, accession: str, result: dict) -> None:
    """Remember each automatic attempt across turns and process restarts."""
    path = Path(run_dir) / "agent_state" / "expansion_searches.json"
    with state_lock(path):
        try:
            document = json.loads(path.read_text())
        except FileNotFoundError:
            document = {}
        previous = document.get(method, {}).get(accession, {})
        row = {
            "status": result.get("status", "error"), "count": result.get("count", 0),
            "attempts": previous.get("attempts", 0) + 1,
            "retry_after": time.time() + min(3600, 60 * (2 ** min(previous.get("attempts", 0), 6)))
                if result.get("status") not in {"success", "empty"} else 0,
        }
        if "new_accessions" in result or "new_accessions" in previous:
            row["new_accessions"] = result.get("new_accessions", previous.get("new_accessions", []))
        document.setdefault(method, {})[accession] = row
        atomic_write_json(path, document)
    append_event(Path(run_dir), "expansion_seed_searched", method=method, accession=accession,
                 status=result.get("status", "error"), count=result.get("count", 0))
