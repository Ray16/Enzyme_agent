"""Fold-family relevance budgeting for pre-evaluation screening.

Ortholog collapse alone (one representative per gene) leaves the evaluation set
spread evenly across fold families — so a large family whose native reactions do
NOT resemble the target consumes as much evaluation budget as the on-target
families, and can crowd them out.  This module regroups the gene representatives
by their FOLD family and spends a bounded evaluation budget by GROUNDED relevance:

  - direction vs the target reaction (a member that natively runs the opposite
    direction — formation vs cleavage — is a grounded dead end), and
  - how closely the fold's members' native substrates resemble the target
    (functional-group overlap, from the family-reaction survey).

Every fold keeps a floor of representatives (breadth is never dropped — an
off-target family still appears, as a documented negative control); extra depth
flows to the folds whose members most resemble the target.

Answer-agnostic by construction: folds come from the DISCOVERED candidates' own
InterPro annotations and relevance is measured against the run's OWN target, so
nothing here encodes an answer.  It also PERSISTS the grounded family landscape
(family_landscape.json) so the picture exists for every run and downstream stages.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.discovery.family_mapping import (
    resolve_family_fold_candidates,
    resolve_family_folds,
)
from enzyme_discovery_agent.tools.family_reactions import (
    _LIST_PAGE_MAX,
    _reviewed_member_count,
    _seed_fold_ids,
    _target_profile_for_run,
    classify_member_reaction,
)

logger = logging.getLogger("enzyme_agent")

# Reaction direction relative to the target, ranked from most to least worth
# pursuing.  "opposite" (a cleavage fold for a formation target, or vice versa)
# is the one grounded dead end and is never given depth beyond the floor.
_DIR_RANK = {"same": 0, "unknown": 1, "unrelated": 2, "opposite": 3}
_OPPOSITE_RANK = _DIR_RANK["opposite"]


# ---------------------------------------------------------------------------
# Pure allocation (offline-testable — no network)
# ---------------------------------------------------------------------------

def allocate_evaluation_budget(
    folds: list[dict], budget: int, floor: int, cap: int,
    stopped: set[str] | None = None,
    contradicted: set[str] | None = None,
) -> dict[str, int]:
    """Return {fold_id: n_to_evaluate} spending ``budget`` by relevance.

    ``folds`` items carry direction and best role-complete compatibility vectors.
    Every fold first gets ``min(floor, n_members)``
    (breadth — never dropped).  The remaining budget is then handed out in
    relevance order (complete one-to-one role fit, then Pareto dominance over
    worst-role properties) up to ``min(cap, n_members)`` per fold; a fold whose native
    reactions run OPPOSITE to the target never rises above its floor.  Pure and
    deterministic (ties broken by fold_id) so it can be unit-tested offline.

    ``stopped`` folds retain the audit floor but receive no extra depth. Only
    ``contradicted`` folds, which require explicit contrary evidence, receive zero.
    """
    stopped = stopped or set()
    contradicted = contradicted or set()
    targets: dict[str, int] = {}
    for f in folds:
        # A stop decision prevents depth but does not erase the minimum evidence
        # sample. Sparse annotations and family-level aggregates are not strong
        # enough to remove a plausible route before one representative is assessed.
        targets[f["fold_id"]] = (
            0 if f["fold_id"] in contradicted else min(floor, f["n_members"])
        )

    remaining = budget - sum(targets.values())
    if remaining <= 0:
        return targets

    # Eligible for extra depth: any non-opposite, non-stopped fold with room to
    # grow. Role-incomplete surveyed folds retain only their audit floor.
    eligible = [
        f for f in folds
        if f["fold_id"] not in stopped
        and f["fold_id"] not in contradicted
        and f["dir_rank"] < _OPPOSITE_RANK
        and f.get("role_complete_count") != 0
        and min(cap, f["n_members"]) > targets[f["fold_id"]]
    ]
    eligible = order_folds_by_relevance(eligible)

    for f in eligible:
        if remaining <= 0:
            break
        room = min(cap, f["n_members"]) - targets[f["fold_id"]]
        grant = min(room, remaining)
        targets[f["fold_id"]] += grant
        remaining -= grant
    return targets


_WORST_ROLE_AXES = (
    "weakest_functional_group_similarity", "weakest_mw_similarity",
    "weakest_volume_similarity", "weakest_chain_length_similarity",
    "weakest_rotatable_bonds_similarity", "weakest_tpsa_similarity",
    "weakest_logp_similarity", "weakest_charge_similarity",
)


def _fold_dominates(left: dict, right: dict) -> bool:
    left_complete = bool(left.get("role_complete_count")
                         or left.get("fully_accommodating_same_direction_members"))
    right_complete = bool(right.get("role_complete_count")
                          or right.get("fully_accommodating_same_direction_members"))
    if left_complete != right_complete:
        return left_complete
    if left.get("dir_rank", 1) != right.get("dir_rank", 1):
        return left.get("dir_rank", 1) < right.get("dir_rank", 1)
    lv = left.get("best_role_compatibility") or {}
    rv = right.get("best_role_compatibility") or {}
    pairs = [(lv.get(axis), rv.get(axis)) for axis in _WORST_ROLE_AXES]
    pairs = [(a, b) for a, b in pairs
             if isinstance(a, (int, float)) and isinstance(b, (int, float))]
    return bool(pairs) and all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)


def order_folds_by_relevance(folds: list[dict]) -> list[dict]:
    """Order folds in Pareto layers over their best complete role vector."""
    prepared = []
    for fold in folds:
        item = dict(fold)
        if not item.get("best_role_compatibility"):
            best = (item.get("best_role_complete_members") or [{}])[0]
            item["best_role_compatibility"] = best.get("role_compatibility") or {}
        prepared.append(item)
    remaining = prepared
    ordered = []
    while remaining:
        front = [fold for fold in remaining if not any(
            _fold_dominates(other, fold) for other in remaining if other is not fold
        )]
        front.sort(key=lambda fold: (
            not bool(fold.get("role_complete_count")
                     or fold.get("fully_accommodating_same_direction_members")),
            fold.get("dir_rank", 1),
            -(fold.get("role_complete_count")
              or fold.get("fully_accommodating_same_direction_members") or 0),
            fold.get("fold_id") or "",
        ))
        ordered.extend(front)
        ids = {id(fold) for fold in front}
        remaining = [fold for fold in remaining if id(fold) not in ids]
    return ordered


# ---------------------------------------------------------------------------
# Fold resolution + relevance (network)
# ---------------------------------------------------------------------------

def _interpro_ids(acc: str, meta: dict) -> list[str]:
    """InterPro ids for a representative — from cached metadata if present, else
    a single UniProt lookup.  Never raises; degrades to an empty list."""
    ids = meta.get("interpro_ids")
    if ids:
        return list(ids)
    try:
        return _seed_fold_ids(acc)
    except Exception as e:  # noqa: BLE001 - a failed lookup must not crash screening
        logger.debug("Fold id lookup failed for %s: %s", acc, e)
        return []


def _specific_family_ids(acc: str) -> list[str]:
    """Return a bounded set of specific InterPro family entries for a seed."""
    try:
        from enzyme_discovery_agent.tools.database import get_protein_domains
        result = get_protein_domains(acc)
    except Exception as e:  # noqa: BLE001
        logger.debug("InterPro family lookup failed for %s: %s", acc, e)
        return []
    entries = (result.get("data") if isinstance(result, dict) else result) or []
    ids = []
    for entry in entries:
        fid = str(entry.get("accession") or "")
        if ((entry.get("type") or "").lower() == "family"
                and fid.upper().startswith("IPR") and fid not in ids):
            ids.append(fid)
    return ids[:5]


def _qualified_literature_seed_folds(run_dir: Path) -> list[tuple[str, str]]:
    """Resolve exact families only from curated same-direction literature seeds."""
    from enzyme_discovery_agent.models.retrieval_leads import load_retrieval_leads
    from enzyme_discovery_agent.tools.reasoning import _fetch_uniprot_entry_light

    target_direction, target_bonds = _target_profile_for_run(run_dir)
    leads = load_retrieval_leads(run_dir).get("leads", {})
    accessions = [
        accession for accession, lead in leads.items()
        if isinstance(lead, dict)
        and lead.get("lead_type") == "evidence_seed"
        and (lead.get("metadata") or {}).get("literature_name")
        and lead.get("literature_evidence")
    ]
    if not accessions:
        return []

    def resolve(accession: str) -> tuple[str, list[str]]:
        entry = _fetch_uniprot_entry_light(accession) or {}
        reactions = entry.get("catalytic_activities") or []
        if not reactions:
            return accession, []
        facts = classify_member_reaction(
            entry.get("ec_numbers") or [], reactions[0],
            target_direction, target_bonds,
        )
        if facts.get("direction_vs_target") != "same" or not facts.get("shared_bond_type"):
            return accession, []
        return accession, _specific_family_ids(accession)

    workers = max(1, min(config.FAMILY_MAPPING_WORKERS, len(accessions)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resolved = list(pool.map(resolve, accessions))
    return [
        (accession, fold_id)
        for accession, fold_ids in resolved
        for fold_id in fold_ids
    ]


def _most_relevant_fold(interpro_ids: list[str], count_cache: dict[str, int]) -> str | None:
    """Pick the most RELEVANT surveyable fold for a member: the entry with the
    BROADEST reviewed coverage that is still surveyable.

    A gene carries several InterPro entries at different granularities (a whole
    catalytic family, a single accessory domain, a conserved site).  Choosing the
    FEWEST-member entry — the old behaviour — collapses onto a narrow accessory
    entry and misses the functionally-related members that share the catalytic
    FAMILY (e.g. a family's substrate-diverse members are simply absent from a
    3-member sub-domain).  Instead pick the entry with the LARGEST reviewed count
    that is still below a full page: the coarse whole-superfamily entries saturate
    at ``_LIST_PAGE_MAX`` (``_reviewed_member_count`` returns the sentinel) and are
    excluded, so the winner is the broadest grouping that is still a coherent,
    surveyable family — the one whose members actually cover the target's
    substrate range.  Member counts are cached across candidates."""
    best: str | None = None
    best_count = None
    for fid in interpro_ids:
        c = count_cache.get(fid)
        if c is None:
            try:
                c = _reviewed_member_count(fid)
            except Exception as e:  # noqa: BLE001
                logger.debug("Member count failed for %s: %s", fid, e)
                c = 0
            count_cache[fid] = c
        if 1 < c < _LIST_PAGE_MAX and (best_count is None or c > best_count):
            best, best_count = fid, c
    return best


def _fold_relevance(run_dir: Path, fold_id: str, fg_cache: dict[str, dict]) -> dict:
    """Substrate-first expansion facts from a fold's member survey."""
    if fold_id in fg_cache:
        return fg_cache[fold_id]
    value = {"role_complete_count": None, "best_role_compatibility": {}}
    try:
        landscape = json.loads((run_dir / "family_landscape.json").read_text())
        existing = next(
            (item for item in landscape if item.get("fold_id") == fold_id), None)
        if existing:
            value["role_complete_count"] = existing.get(
                "fully_accommodating_same_direction_members")
            best = (existing.get("best_role_complete_members") or [{}])[0]
            value["best_role_compatibility"] = best.get("role_compatibility") or {}
            fg_cache[fold_id] = value
            return value
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    # Allocation persists the same reduced facts in family_budget.json. Reuse
    # them on later evaluation epochs instead of repeating a network-backed
    # whole-family survey for every incidental candidate fold.
    try:
        budget = json.loads((run_dir / "family_budget.json").read_text())
        existing = next(
            (item for item in budget if item.get("fold_id") == fold_id), None)
        if existing:
            value["role_complete_count"] = existing.get(
                "fully_accommodating_same_direction_members")
            value["best_role_compatibility"] = existing.get(
                "best_role_compatibility") or {}
            legacy = existing.get(
                "best_role_complete_functional_group_similarity")
            if not value["best_role_compatibility"] and isinstance(
                    legacy, (int, float)):
                value["best_role_compatibility"] = {
                    "weakest_functional_group_similarity": legacy,
                    "legacy_partial_vector": True,
                }
            # Runs created before role vectors persisted only the best
            # functional-group bottleneck. Preserve that grounded axis when
            # resuming, without reconstructing the removed blended fit score.
            if not value["best_role_compatibility"]:
                legacy_fg = existing.get(
                    "best_role_complete_functional_group_similarity")
                if isinstance(legacy_fg, (int, float)):
                    value["best_role_compatibility"] = {
                        "all_target_substrates_accommodated": True,
                        "weakest_functional_group_similarity": legacy_fg,
                        "legacy_partial_vector": True,
                    }
            fg_cache[fold_id] = value
            return value
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    try:
        from enzyme_discovery_agent.tools.family_reactions import survey_family_reactions
        res = survey_family_reactions(run_dir=run_dir, interpro_id=fold_id)
        if res.get("status") == "success":
            profile = ((res.get("data") or {}).get("family_reaction_profile") or {})
            value["role_complete_count"] = profile.get(
                "fully_accommodating_same_direction_members", 0)
            best = profile.get("best_role_complete_members") or []
            value["best_role_compatibility"] = (
                (best[0].get("role_compatibility") or {}) if best else {}
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("Fold relevance survey failed for %s: %s", fold_id, e)
    fg_cache[fold_id] = value
    return value


def collect_survey_folds(
    run_dir: Path,
    derived_families: list[dict] | None,
    count_cache: dict[str, int],
    fold_cache: dict[str, str | None] | None = None,
    max_genes: int = 150,
    max_folds: int = 80,
    log_fn=None,
) -> list[dict]:
    """Return exact InterPro families justified for reviewed-reaction survey.

    (A) Ranked mappings of derived mechanism families cover relevant chemistry
        before any candidate has been discovered. Mapping confidence is not a
        family-relevance verdict: one mapping per hypothesis is surveyed, and a
        negative result unlocks the next ranked alternative.
    Literature and database seeds remain auditable retrieval leads, but do not
    open every InterPro annotation on a protein for exhaustive enumeration. A
    family-wide survey requires a selected mapping of a retained mechanism
    hypothesis; this keeps unrelated accessory domains out of the focused pass.
    Returns one currently selected mapping per derived hypothesis."""

    fold_cache = fold_cache if fold_cache is not None else {}
    ordered: list[dict] = []
    seen: set[str] = set()

    from enzyme_discovery_agent.models.family_hypothesis import normalize_family_hypothesis

    def _add(fid, source, hypothesis: dict | None = None):
        if fid and str(fid).upper().startswith("IPR") and fid in seen:
            if hypothesis:
                existing = next((item for item in ordered if item["fold_id"] == fid), None)
                if existing is not None:
                    normalized = normalize_family_hypothesis(hypothesis)
                    hypotheses = existing.setdefault("derived_family_hypotheses", [])
                    if normalized not in hypotheses:
                        hypotheses.append(normalized)
                    existing.setdefault("derived_family_hypothesis", normalized)
            return
        if fid and str(fid).upper().startswith("IPR"):
            seen.add(fid)
            item = {"fold_id": fid, "source": source}
            if hypothesis:
                normalized = normalize_family_hypothesis(hypothesis)
                item["derived_family_hypothesis"] = normalized
                item["derived_family_hypotheses"] = [normalized]
            ordered.append(item)

    # Resolve all retained hypotheses first and reserve every plausible mapping.
    # This prevents an ortholog-rich discovered pool from consuming the fold cap
    # before an as-yet-undiscovered catalytic route gets surveyed.
    families = list(derived_families or [])
    resolutions: list[list[dict]] = [[] for _ in families]
    if families:
        from enzyme_discovery_agent.models.event_log import append_event
        from enzyme_discovery_agent.models.family_ledger import (
            family_hypothesis_id,
            load_family_ledger,
            next_family_survey_fold,
            record_family_resolution,
        )
        from enzyme_discovery_agent.tools.database import search_interpro

        # Resolve each hypothesis to InterPro entries EXACTLY ONCE per run. This
        # gate re-fires every time the survey pass grows the discovered pool, but
        # InterPro text resolution is intentionally ambiguous and not perfectly
        # stable across calls: re-resolving on every firing kept merging freshly
        # returned entries into the ledger, so ``resolved_folds`` grew without
        # bound, every newly appended mapping re-opened a "pending" survey, and
        # the map->survey->map loop never reached a terminal, evaluable state.
        # A hypothesis that already carries stored resolution candidates reuses
        # them; only unresolved hypotheses hit the network. This makes the
        # resolved fold set a stable fixed point so coverage can complete.
        prior_records = load_family_ledger(run_dir).get("families", {})
        stored_by_index: dict[int, list[dict]] = {}
        pending_families: list[tuple[int, dict, str]] = []
        for index, fam in enumerate(families, start=1):
            record = prior_records.get(family_hypothesis_id(fam), {})
            # ``proposed`` is the only pre-resolution state; every other state
            # means a resolution attempt already ran (possibly to an empty set).
            # Reuse that outcome verbatim rather than re-hitting InterPro, so an
            # unstable text search cannot append new folds on later passes.
            already_resolved = bool(record) and record.get("state", "proposed") != "proposed"
            if already_resolved:
                stored_by_index[index] = list(record.get("resolution_candidates") or [])
            else:
                pending_families.append((index, fam, uuid.uuid4().hex))

        total = len(pending_families)
        workers = max(1, min(config.FAMILY_MAPPING_WORKERS, total)) if total else 1
        if log_fn and total:
            reused = len(families) - total
            log_fn(
                f"  Family mapping: resolving {total} hypotheses with "
                f"{workers} worker(s)"
                + (f" ({reused} reused from ledger)" if reused else "")
            )

        def _resolve_family(item):
            index, fam, operation_id = item
            started = time.monotonic()
            family_name = str(fam.get("name") or "")
            append_event(
                run_dir, "family_mapping_started",
                operation_id=operation_id,
                family_index=index,
                family_total=total,
                family_name=family_name,
            )
            if log_fn:
                log_fn(
                    f"  Family mapping {index}/{total} started: "
                    f"{family_name[:100]}"
                )
            try:
                resolved = resolve_family_fold_candidates(
                    fam, count_cache, search_interpro
                )
            except Exception as exc:  # noqa: BLE001 - one hypothesis is best-effort
                append_event(
                    run_dir, "family_mapping_failed",
                    operation_id=operation_id,
                    family_index=index,
                    family_total=total,
                    family_name=family_name,
                    error_type=type(exc).__name__,
                    elapsed_s=round(time.monotonic() - started, 3),
                )
                logger.debug(
                    "derived-family InterPro resolution failed for %s: %s",
                    fam.get("name"), exc,
                )
                if log_fn:
                    log_fn(
                        f"  Family mapping {index}/{total} failed after "
                        f"{time.monotonic() - started:.1f}s: {family_name[:100]}",
                        level="warning",
                    )
                return []
            append_event(
                run_dir, "family_mapping_completed",
                operation_id=operation_id,
                family_index=index,
                family_total=total,
                family_name=family_name,
                mapping_count=len(resolved),
                fold_ids=[mapping.get("fold_id") for mapping in resolved],
                elapsed_s=round(time.monotonic() - started, 3),
            )
            if log_fn:
                fold_ids = [mapping.get("fold_id") for mapping in resolved]
                log_fn(
                    f"  Family mapping {index}/{total} complete in "
                    f"{time.monotonic() - started:.1f}s: "
                    f"{', '.join(filter(None, fold_ids)) or 'no InterPro mapping'}"
                )
            return resolved

        # Placeholder-fill reused hypotheses from their stored candidates so the
        # per-family loop below sees the same shape whether or not the network ran.
        for index, stored in stored_by_index.items():
            resolutions[index - 1] = stored

        if pending_families:
            jobs = [
                (disp_index, fam, operation_id)
                for disp_index, (_, fam, operation_id) in enumerate(
                    pending_families, start=1
                )
            ]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_resolve_family, job) for job in jobs]
                pending = set(futures)
                while pending:
                    _, pending = wait(
                        pending,
                        timeout=config.FAMILY_PROGRESS_INTERVAL_SECONDS,
                        return_when=ALL_COMPLETED,
                    )
                    if pending and log_fn:
                        log_fn(
                            f"  Family mapping progress: {total - len(pending)}/{total} "
                            f"complete, {len(pending)} pending"
                        )
                # Read futures in submission order so fold precedence is reproducible.
                fresh = [future.result() for future in futures]
            for (orig_index, fam, _), resolved in zip(pending_families, fresh):
                resolutions[orig_index - 1] = resolved
                # Persist a freshly resolved hypothesis's mappings once. Reused
                # hypotheses are already recorded; re-recording them is skipped so
                # the stored resolution stays a stable fixed point.
                record_family_resolution(run_dir, fam, resolved)

    for fam, resolved in zip(families, resolutions):
            selected_fold = next_family_survey_fold(run_dir, fam)
            for mapping in resolved:
                if mapping.get("fold_id") != selected_fold:
                    continue
                _add(
                    mapping.get("fold_id"),
                    f"derived:{fam.get('name','')[:40]}",
                    hypothesis=fam,
                )

    # A specifically named literature enzyme with a curated same-direction
    # native reaction is stronger family evidence than a text-only mapping.
    # Follow only exact InterPro FAMILY annotations from those qualified seeds;
    # unqualified names, cited analogues, and accessory domains remain leads.
    for accession, fold_id in _qualified_literature_seed_folds(run_dir):
        _add(fold_id, f"literature_seed:{accession}")

    return ordered[:max_folds]


def _candidate_direction(meta: dict, target_dir: str | None, target_bonds: set[str]) -> str:
    """Grounded reaction-direction of a representative vs the target (offline)."""
    cats = meta.get("catalytic_activities") or []
    native = cats[0] if cats else ""
    facts = classify_member_reaction(
        meta.get("ec_numbers") or [], native, target_dir, target_bonds,
    )
    return facts["direction_vs_target"]


def build_relevance_and_allocate(
    run_dir: Path,
    reps: list[tuple[str, dict]],
    budget: int | None = None,
    floor: int | None = None,
    cap: int | None = None,
    workers: int = 4,
    protected: set[str] | None = None,
) -> tuple[set[str], list[dict]]:
    """Group gene representatives by fold, spend the evaluation budget by
    relevance, and persist the grounded family landscape.

    ``reps`` are ``(accession, metadata)`` for the gene representatives (metadata
    should carry ``ec_numbers``, ``catalytic_activities``, ``interpro_ids`` and
    ``seq_length`` when available).  Returns ``(selected_accessions, landscape)``
    — the accessions to evaluate and the per-fold landscape (also written to
    ``family_landscape.json``).  Degrades gracefully: any representative whose
    fold cannot be resolved becomes its own singleton fold, so it is still
    evaluated (never silently dropped).

    ``protected`` accessions are ALWAYS selected for evaluation regardless of any
    fold-level stop verdict.  A fold ``stop`` is a BREADTH decision (do not expand
    into orthologs of a fold the agent judged off-target); it must not silently
    delete a specific enzyme the reading step surfaced by name.  Without this, a
    named candidate that happens to be the sole representative of a stopped fold is
    dropped before evaluation on the strength of the fold's generic annotation,
    never on its own evidence."""
    protected = protected or set()
    budget = config.FAMILY_EVAL_BUDGET if budget is None else budget
    floor = config.FAMILY_EVAL_FLOOR_PER_FOLD if floor is None else floor
    cap = config.FAMILY_EVAL_MAX_PER_FOLD if cap is None else cap
    if not reps:
        return set(), []

    target_dir, target_bonds = _target_profile_for_run(run_dir)

    # 1. Resolve each representative to its most specific fold (member-count
    #    cache shared across candidates keeps this cheap).
    count_cache: dict[str, int] = {}

    def _resolve(acc_meta):
        acc, meta = acc_meta
        # Native-reaction survey promotions already identify the exact fold that
        # produced their evidence. Reusing it avoids member-count requests for
        # every other InterPro annotation and preserves provenance alignment.
        hinted_fold = str(meta.get("screening_family_id") or "")
        if hinted_fold.upper().startswith("IPR"):
            return acc, hinted_fold
        fold = _most_relevant_fold(_interpro_ids(acc, meta), count_cache)
        return acc, fold or f"unresolved:{acc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        resolved = dict(pool.map(_resolve, reps))

    meta_by_acc = {acc: meta for acc, meta in reps}

    # 2. Group by fold and compute the per-member direction (offline).
    fold_members: dict[str, list[str]] = {}
    dir_by_acc: dict[str, str] = {}
    for acc, meta in reps:
        fold_members.setdefault(resolved[acc], []).append(acc)
        dir_by_acc[acc] = _candidate_direction(meta, target_dir, target_bonds)

    # 3. Per-fold relevance: best (min-rank) direction among members, and the
    #    fold's substrate resemblance from the survey (skipped for opposite
    #    folds — a grounded dead end — and for unresolved singletons).
    fg_cache: dict[str, dict] = {}
    folds: list[dict] = []
    for fold_id, members in fold_members.items():
        dir_rank = min(_DIR_RANK.get(dir_by_acc[a], 1) for a in members)
        relevance = {"role_complete_count": None, "best_role_compatibility": {}}
        if dir_rank < _OPPOSITE_RANK and not fold_id.startswith("unresolved:"):
            relevance = _fold_relevance(run_dir, fold_id, fg_cache)
        folds.append({
            "fold_id": fold_id,
            "members": members,
            "n_members": len(members),
            "dir_rank": dir_rank,
            **relevance,
        })

    # 4. Allocate the budget, then pick the best representatives within each fold.
    #    Honour expand/stop decisions without letting a provisional stop erase
    #    the audit floor. Explicitly contradicted folds alone receive zero.
    from enzyme_discovery_agent.models.family_decisions import (
        contradicted_folds as _contradicted_folds,
        stopped_folds as _stopped_folds,
    )
    stopped = _stopped_folds(run_dir)
    contradicted = _contradicted_folds(run_dir)
    if stopped:
        logger.info("Family decisions: %d fold(s) stopped by the agent", len(stopped))
    targets = allocate_evaluation_budget(
        folds, budget, floor, cap, stopped=stopped, contradicted=contradicted)
    from enzyme_discovery_agent.gates.ec_triage import _evidence_tier

    selected: set[str] = set()
    for f in folds:
        members = sorted(
            f["members"],
            key=lambda a: (
                _evidence_tier(meta_by_acc[a]),
                -(meta_by_acc[a].get("seq_length") or 0),
                a,
            ),
        )
        chosen = members[:targets[f["fold_id"]]]
        # A named candidate the reading step surfaced is evaluated on its own
        # merits even when its fold was stopped: the stop pruned breadth, not
        # this specific enzyme.  Force-add keeps the budget honest (the pure
        # allocation is unchanged) while never deleting a literature-named hit.
        forced = [a for a in members if a in protected and a not in chosen]
        f["selected"] = chosen + forced
        selected.update(f["selected"])

    # 5. Persist the landscape, ordered by relevance (best first) for the report.
    _rank_name = {0: "same", 1: "unknown", 2: "unrelated", 3: "opposite"}
    landscape = order_folds_by_relevance(folds)
    landscape_out = [
        {
            "fold_id": f["fold_id"] if not f["fold_id"].startswith("unresolved:") else None,
            "direction_vs_target": _rank_name.get(f["dir_rank"], "unknown"),
            "fully_accommodating_same_direction_members": f.get("role_complete_count"),
            "best_role_compatibility": f.get("best_role_compatibility") or {},
            "n_representatives": f["n_members"],
            "n_selected_for_evaluation": len(f["selected"]),
            "agent_stopped": f["fold_id"] in stopped,
            "members": f["members"],
        }
        for f in landscape
    ]
    # Written to a SEPARATE file so it does not clobber the survey gate's richer
    # discovery-time family_landscape.json (native reactions + top members).
    try:
        (run_dir / "family_budget.json").write_text(
            json.dumps(landscape_out, indent=2) + "\n"
        )
    except OSError:
        pass

    return selected, landscape_out
