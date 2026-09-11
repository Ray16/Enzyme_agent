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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from enzyme_discovery_agent_improved import config
from enzyme_discovery_agent_improved.tools.family_reactions import (
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
) -> dict[str, int]:
    """Return {fold_id: n_to_evaluate} spending ``budget`` by relevance.

    ``folds`` items: ``{"fold_id": str, "n_members": int, "dir_rank": int,
    "fg_max": float|None}``.  Every fold first gets ``min(floor, n_members)``
    (breadth — never dropped).  The remaining budget is then handed out in
    relevance order (best direction first, then higher functional-group
    resemblance) up to ``min(cap, n_members)`` per fold; a fold whose native
    reactions run OPPOSITE to the target never rises above its floor.  Pure and
    deterministic (ties broken by fold_id) so it can be unit-tested offline.

    ``stopped`` is the set of fold_ids the AGENT has judged not worth pursuing
    (e.g. native substrate unlike the target).  A stopped fold gets ZERO — its
    floor is waived — so the agent's reasoned decision overrides the breadth
    default rather than the framework forcing coverage of an off-target fold.
    The deterministic allocation still governs every fold the agent did not stop,
    so this is an override layer, not a replacement.
    """
    stopped = stopped or set()
    targets: dict[str, int] = {}
    for f in folds:
        # Agent-stopped folds are waived from the breadth floor entirely.
        targets[f["fold_id"]] = 0 if f["fold_id"] in stopped else min(floor, f["n_members"])

    remaining = budget - sum(targets.values())
    if remaining <= 0:
        return targets

    # Eligible for extra depth: any non-opposite, non-stopped fold with room to
    # grow.  Order by direction (same > unknown > unrelated), then substrate
    # resemblance (fg_max desc), then fold_id for determinism.
    eligible = [
        f for f in folds
        if f["fold_id"] not in stopped
        and f["dir_rank"] < _OPPOSITE_RANK
        and min(cap, f["n_members"]) > targets[f["fold_id"]]
    ]
    eligible.sort(key=lambda f: (
        f["dir_rank"],
        -(f["fg_max"] if f.get("fg_max") is not None else -1.0),
        f["fold_id"],
    ))

    for f in eligible:
        if remaining <= 0:
            break
        room = min(cap, f["n_members"]) - targets[f["fold_id"]]
        grant = min(room, remaining)
        targets[f["fold_id"]] += grant
        remaining -= grant
    return targets


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


# InterPro entry types that name a coherent FUNCTIONAL grouping worth surveying.
# A homologous_superfamily is a coarse structural bin (it pools unrelated
# functions); conserved/active/binding sites, repeats and PTMs are sub-features,
# not families.  Keeping only family/domain avoids resolving a mechanism family
# onto a structural fold shared by unrelated proteins.
_SURVEYABLE_ENTRY_TYPES = frozenset({"family", "domain"})


def resolve_family_folds(fam: dict, count_cache: dict[str, int], search_interpro,
                         max_folds: int = 3) -> list[str]:
    """Resolve one derived family hypothesis to surveyable InterPro folds by
    relevance, robustly.

    A single JOINED descriptor query lets a generic structural-fold term dominate
    the keyword match and resolve onto the wrong proteins (an unrelated protein
    sharing that fold, or a whole superfamily) instead of the functional family
    whose members carry the target chemistry.  So resolve by the family NAME
    first — the
    derivation's cleanest FUNCTIONAL label, which matches the on-topic InterPro
    entries — and use the (often structural, noisy) descriptors only to SUPPLEMENT
    when the name resolves too few.  Coarse superfamilies and non-family
    sub-features are dropped; only surveyable (bounded, non-saturated) family/
    domain entries are kept; broadest coverage first.  Several entries are
    surveyed, not one, so a single noisy resolution can no longer lose the family.
    """
    def _surveyable(term: str) -> list[tuple[str, int]]:
        term = str(term or "").strip()
        if not term:
            return []
        try:
            res = search_interpro(term, max_results=8)
        except Exception:  # noqa: BLE001 - one bad term must not sink resolution
            return []
        out: list[tuple[str, int]] = []
        for e in (res.get("data") if isinstance(res, dict) else res) or []:
            acc = str(e.get("accession") or e.get("id") or "")
            if not acc.upper().startswith("IPR"):
                continue
            if (e.get("type") or "").lower() not in _SURVEYABLE_ENTRY_TYPES:
                continue
            c = count_cache.get(acc)
            if c is None:
                try:
                    c = _reviewed_member_count(acc)
                except Exception:  # noqa: BLE001
                    c = 0
                count_cache[acc] = c
            if 1 < c < _LIST_PAGE_MAX:
                out.append((acc, c))
        return sorted(out, key=lambda kv: -kv[1])

    picked: dict[str, int] = {}
    # 1. The family name is authoritative — its on-topic entries come first.
    for acc, c in _surveyable(fam.get("name", "")):
        picked.setdefault(acc, c)
    # 2. Supplement (breadth) from descriptors only for slots the name left open.
    if len(picked) < max_folds:
        for descriptor in fam.get("descriptors") or []:
            for acc, c in _surveyable(descriptor):
                picked.setdefault(acc, c)
    # Name-derived entries first (by coverage), then descriptor supplements.
    return list(picked)[:max_folds]


def _fold_fg_max(run_dir: Path, fold_id: str, fg_cache: dict[str, float | None]) -> float | None:
    """Max functional-group resemblance of a fold's members to the target, via
    the family-reaction survey (cached per fold).  None when unavailable."""
    if fold_id in fg_cache:
        return fg_cache[fold_id]
    val: float | None = None
    try:
        from enzyme_discovery_agent_improved.tools.family_reactions import survey_family_reactions
        res = survey_family_reactions(run_dir=run_dir, interpro_id=fold_id)
        if res.get("status") == "success":
            fg = ((res.get("data") or {}).get("family_reaction_profile") or {}).get(
                "functional_group_similarity")
            if fg:
                val = fg.get("max")
    except Exception as e:  # noqa: BLE001
        logger.debug("Fold fg survey failed for %s: %s", fold_id, e)
    fg_cache[fold_id] = val
    return val


def collect_survey_folds(
    run_dir: Path,
    derived_families: list[dict] | None,
    count_cache: dict[str, int],
    fold_cache: dict[str, str | None] | None = None,
    max_genes: int = 150,
    max_folds: int = 40,
) -> list[dict]:
    """Return the InterPro folds worth surveying, from BOTH sources.

    (A) Each derived mechanism family resolved to a fold via ``search_interpro``
        — covers a relevant family even before any candidate is discovered in it.
    (B) The most-specific fold of every DISTINCT gene in the discovered-candidate
        pool — so a discovered-but-unscreened family (a highly target-like ligase
        that never got screened) is surveyed rather than missed.

    Returns ``[{"fold_id": str, "source": str}]``, de-duplicated, derived-family
    folds first.  ``fold_cache`` (accession -> fold) and ``count_cache`` persist
    across gate firings so re-runs only resolve NEW accessions."""
    from enzyme_discovery_agent_improved.models.candidate_registry import load_candidate_registry

    fold_cache = fold_cache if fold_cache is not None else {}
    ordered: list[dict] = []
    seen: set[str] = set()

    from enzyme_discovery_agent_improved.models.family_hypothesis import normalize_family_hypothesis

    def _add(fid, source, hypothesis: dict | None = None):
        if fid and str(fid).upper().startswith("IPR") and fid in seen:
            if hypothesis:
                existing = next((item for item in ordered if item["fold_id"] == fid), None)
                if existing is not None:
                    existing["derived_family_hypothesis"] = normalize_family_hypothesis(
                        hypothesis)
            return
        if fid and str(fid).upper().startswith("IPR"):
            seen.add(fid)
            item = {"fold_id": fid, "source": source}
            if hypothesis:
                item["derived_family_hypothesis"] = normalize_family_hypothesis(hypothesis)
            ordered.append(item)

    # (B, first) distinct discovered genes -> most-specific folds.  These are
    # GROUNDED in real candidates (so a discovered-but-unscreened family like a
    # target-like ligase is surveyed), and are prioritised over derived folds.
    reg = load_candidate_registry(run_dir).get("candidates", {})
    genes_seen: set[str] = set()
    accs: list[str] = []
    for _key, v in reg.items():
        if not isinstance(v, dict):
            continue
        if "rejected" in {s.get("status") for s in v.get("statuses", [])}:
            continue
        acc = v.get("accession")
        if not acc:
            continue
        meta = v.get("metadata", {})
        gene = (meta.get("gene_names") or [None])[0] or acc
        if gene in genes_seen:
            continue
        genes_seen.add(gene)
        accs.append(acc)
    accs = accs[:max_genes]

    def _fold_of(acc):
        if acc in fold_cache:
            return fold_cache[acc]
        fold = _most_relevant_fold(_interpro_ids(acc, {}), count_cache)
        fold_cache[acc] = fold
        return fold

    with ThreadPoolExecutor(max_workers=8) as pool:
        for acc, fid in zip(accs, pool.map(_fold_of, accs)):
            _add(fid, "discovered")

    # (A, second) derived mechanism families -> InterPro folds, to cover a family
    # with no discovered candidate yet.  Resolve each family to its BROADEST
    # surveyable functional entries (per-descriptor, superfamily-filtered) and
    # survey SEVERAL, so a single noisy fold-keyword resolution cannot lose the
    # family whose members carry the target chemistry.
    try:
        from enzyme_discovery_agent_improved.tools.database import search_interpro
        fams = list(derived_families or [])[:8]
        # Resolve each family's InterPro folds concurrently — every family issues
        # independent, network-bound search_interpro / member-count calls, so
        # serial resolution just stacks their latencies. pool.map preserves input
        # order and _add runs sequentially afterwards, so the resulting fold set
        # is byte-identical to the serial version (fast path == slow path); only
        # the wall-clock differs. The shared count_cache stays correct: repeated
        # member counts are deterministic, so at worst a fold is counted twice.
        if fams:
            with ThreadPoolExecutor(max_workers=min(len(fams), 8)) as pool:
                resolved = list(pool.map(
                    lambda fam: resolve_family_folds(
                        fam, count_cache, search_interpro),
                    fams,
                ))
            for fam, fids in zip(fams, resolved):
                for fid in fids:
                    _add(fid, f"derived:{fam.get('name','')[:40]}", hypothesis=fam)
    except Exception as e:  # noqa: BLE001 - derived-family resolution is best-effort
        logger.debug("derived-family InterPro resolution failed: %s", e)

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
    fg_cache: dict[str, float | None] = {}
    folds: list[dict] = []
    for fold_id, members in fold_members.items():
        dir_rank = min(_DIR_RANK.get(dir_by_acc[a], 1) for a in members)
        fg_max = None
        if dir_rank < _OPPOSITE_RANK and not fold_id.startswith("unresolved:"):
            fg_max = _fold_fg_max(run_dir, fold_id, fg_cache)
        folds.append({
            "fold_id": fold_id,
            "members": members,
            "n_members": len(members),
            "dir_rank": dir_rank,
            "fg_max": fg_max,
        })

    # 4. Allocate the budget, then pick the best representatives within each fold.
    #    Honour the agent's expand/stop decisions: a fold the agent judged not
    #    worth pursuing is waived from the breadth floor (gets zero), so its
    #    reasoning overrides the default coverage instead of the framework
    #    forcing an off-target evaluation.
    from enzyme_discovery_agent_improved.models.family_decisions import stopped_folds as _stopped_folds
    stopped = _stopped_folds(run_dir)
    if stopped:
        logger.info("Family decisions: %d fold(s) stopped by the agent", len(stopped))
    targets = allocate_evaluation_budget(folds, budget, floor, cap, stopped=stopped)
    from enzyme_discovery_agent_improved.gates.ec_triage import _evidence_tier

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
    landscape = sorted(folds, key=lambda f: (
        f["dir_rank"], -(f["fg_max"] if f.get("fg_max") is not None else -1.0), f["fold_id"],
    ))
    landscape_out = [
        {
            "fold_id": f["fold_id"] if not f["fold_id"].startswith("unresolved:") else None,
            "direction_vs_target": _rank_name.get(f["dir_rank"], "unknown"),
            "functional_group_similarity_max": f["fg_max"],
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
