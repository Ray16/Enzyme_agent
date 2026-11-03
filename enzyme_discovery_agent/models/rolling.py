"""Durable pending-evaluation snapshots for rolling discovery batches."""

import hashlib
import json
import time
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import discovery_status, load_candidate_registry, evidence_fingerprint
from enzyme_discovery_agent.state_store import atomic_write_json
from enzyme_discovery_agent.run_layout import run_file


def diversify_evaluation_batch(candidates, ordered_accessions, limit):
    """Schedule under-assessed families, then annotation groups within a family.

    Resolved family annotations take precedence; unresolved names remain explicit
    scheduling hints, not evidence of orthology or interchangeability. Retain
    the supplied evidence priority when assessment counts tie. Pending and
    rejected assessments also consume effort and must not monopolize the queue.
    """
    from collections import Counter
    from enzyme_discovery_agent.models.ortholog import tentative_label_key

    def group(accession):
        metadata = (candidates.get(accession) or {}).get("metadata") or {}
        return tentative_label_key(metadata) or f"accession:{accession}"

    def family(accession):
        metadata = (candidates.get(accession) or {}).get("metadata") or {}
        resolved = metadata.get("interpro_family_id") or metadata.get("screening_family_id")
        return f"ipr:{resolved}" if resolved else f"unresolved:{group(accession)}"

    assessed = [acc for acc, entry in candidates.items()
                if discovery_status(entry) in {"evaluated", "ranked", "evidence_pending", "rejected"}]
    family_used = Counter(family(acc) for acc in assessed)
    group_used = Counter((family(acc), group(acc)) for acc in assessed)
    remaining = list(ordered_accessions)
    chosen = []
    while remaining and len(chosen) < limit:
        index = min(range(len(remaining)), key=lambda i: (
            family_used[family(remaining[i])],
            group_used[(family(remaining[i]), group(remaining[i]))], i))
        accession = remaining.pop(index)
        chosen.append(accession)
        family_used[family(accession)] += 1
        group_used[(family(accession), group(accession))] += 1
    return chosen


def evaluation_basis(metadata):
    return evidence_fingerprint({k: metadata.get(k) for k in (
        "literature_evidence", "reopen_evidence", "claim_validation", "catalytic_activities", "native_substrates")})


def work_deferred(metadata, paused):
    """Scheduling scopes can pause work without asserting functional equivalence."""
    if not paused:
        return False
    from enzyme_discovery_agent.models.ortholog import tentative_label_key
    view = {**(metadata.get('screening_annotation') or {}), **metadata}
    group = tentative_label_key(view)
    keys = {view.get('interpro_family_id'), view.get('screening_family_id')}
    if group:
        keys.add('annotation:' + group)
    return bool(keys & set(paused))


def evaluation_ready(entry, *, paused=None):
    metadata = entry.get("metadata") or {}
    if work_deferred(metadata, paused):
        return False
    if metadata.get("is_literature_lead"):
        return False
    failure = metadata.get("evaluation_failure") or {}
    if failure.get("basis") == evaluation_basis(metadata):
        if failure.get("attempts", 0) >= 2 or failure.get("retry_after", 0) > time.time():
            return False
    status = discovery_status(entry)
    if status in {"screened", "reopened"}:
        return True
    return status == "evidence_pending" and (not metadata.get("literature_recovery_attempted") or
        bool(metadata.get("literature_evidence")) and evidence_fingerprint(metadata["literature_evidence"])
            != metadata.get("literature_recovery_fingerprint"))


def fresh_evaluation_backlog(run_dir):
    """Ready first-pass work, excluding unresolved identities and retry cooldowns."""
    from enzyme_discovery_agent.models.family_decisions import paused_folds
    paused = paused_folds(run_dir)
    return [acc for acc, entry in load_candidate_registry(Path(run_dir), strict=True).get("candidates", {}).items()
            if discovery_status(entry) in {"screened", "reopened"} and evaluation_ready(entry, paused=paused)]


def pending_evaluations(run_dir: Path) -> dict:
    from enzyme_discovery_agent.models.family_decisions import paused_folds
    paused = paused_folds(run_dir)
    candidates = load_candidate_registry(run_dir, strict=True).get("candidates", {})
    pending = {}
    for acc, entry in candidates.items():
        status = discovery_status(entry)
        meta = entry.get("metadata") or {}
        if status in {"discovered", "screened", "reopened", "evidence_pending"}:
            pending[acc] = {"status": status, "literature_evidence": meta.get("literature_evidence"),
                            "reopen_evidence": meta.get("reopen_evidence")}
            pending[acc]["evaluation_ready"] = evaluation_ready(entry, paused=paused)
    return pending


def evaluation_signature(run_dir: Path) -> str:
    pending = pending_evaluations(run_dir)
    try:
        decisions = json.loads((run_file(run_dir, 'family_decisions.json')).read_text())
    except FileNotFoundError:
        decisions = {}
    decisions = {fid: value.get("decision") for fid, value in (decisions.get("decisions") or {}).items()
                 if isinstance(value, dict)}
    from enzyme_discovery_agent.models.family_learning import refresh_family_learning
    learning = refresh_family_learning(run_dir)
    priorities = {fid: {"state": row["state"], "fit": row["best_supported_fit"]}
                  for fid, row in learning["families"].items()}
    return hashlib.sha256(json.dumps({"pending": pending, "decisions": decisions, "family_priorities": priorities},
                                    sort_keys=True, default=str).encode()).hexdigest()


def new_evaluation_work(run_dir: Path) -> bool:
    pending = pending_evaluations(run_dir)
    if not pending:
        return False
    if any(row["evaluation_ready"] for row in pending.values()):
        return True
    try:
        saved = json.loads((run_dir / "agent_state" / "evaluation_batch.json").read_text())
    except FileNotFoundError:
        return True
    return saved.get("signature") != evaluation_signature(run_dir)


def record_evaluation_batch(run_dir: Path) -> None:
    atomic_write_json(run_dir / "agent_state" / "evaluation_batch.json", {
        "signature": evaluation_signature(run_dir), "deferred": sorted(pending_evaluations(run_dir)),
    })
