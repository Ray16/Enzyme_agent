"""Family native-reaction survey gate.

The survey is only useful if it actually runs.  Left as an optional tool the LLM
must remember to call, it gets bypassed by the auto-evaluation machinery (observed
in practice).  This gate makes the survey part of the PIPELINE: once the mechanism
families are derived and the agent has candidate seeds, it surveys the folds those
candidates fall into, builds a grounded family landscape (each fold's members'
native reactions vs the target — direction and substrate resemblance, as reported
numbers, no cutoff), persists it, and injects it so the agent reasons over the map
and concentrates expansion on the target-resembling fold.

Answer-agnostic: folds come from DISCOVERED candidates and the comparison is to the
run's own target; nothing here names an answer.  Soft by construction — it reports
numbers and steers; it never blocks evaluation.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.tool_registry import execute_tool

logger = logging.getLogger("enzyme_agent")

_MAX_NEW_FOLDS_PER_FIRING = 16   # folds surveyed per firing (bounds compute)
_MAX_FOLDS_SHOWN = 12            # folds presented back to the agent


def _select_evaluation_admissions(members: list[dict], cap: int) -> list[dict]:
    """Bound evaluation while representing distinct native functions first."""
    from enzyme_discovery_agent.tools.family_reactions import (
        _member_function_key,
        order_by_role_compatibility,
    )

    ordered = order_by_role_compatibility(members)
    selected: list[dict] = []
    deferred: list[dict] = []
    seen_functions: set[tuple] = set()
    for member in ordered:
        function = _member_function_key(member)
        if function in seen_functions:
            deferred.append(member)
            continue
        seen_functions.add(function)
        selected.append(member)
        if len(selected) >= cap:
            return selected
    return [*selected, *deferred[:max(0, cap - len(selected))]]


class FamilySurveyGate:
    """Comprehensively survey the InterPro folds relevant to the run and inject
    the grounded family map.

    Folds come from TWO sources so no relevant family is missed: (1) each derived
    mechanism family resolved to a fold via search_interpro (covers a family with
    no discovered candidate yet), and (2) the most-specific fold of every distinct
    gene in the discovered pool (covers a discovered-but-unscreened family — e.g.
    a highly target-like ligase that never got screened).  Accumulates across
    firings and re-runs as the discovered pool grows, so late-discovered families
    are surveyed too.
    """

    # Cap on firings: the survey accumulates folds, so a few passes cover the pool
    # as it grows without re-surveying folds already done.
    _MAX_ATTEMPTS = 12
    _MAX_ATTEMPTS_PER_FOLD = 2

    def __init__(self, log_fn=None):
        self.log = log_fn or (lambda *a, **k: None)
        self._attempts = 0
        self._last_attempt_count = -1
        self._surveyed_folds: set[str] = set()   # folds already surveyed
        self._all_folds: dict[str, dict] = {}     # fold_id -> landscape entry
        self._count_cache: dict[str, int] = {}    # fold -> reviewed-member count
        self._fold_cache: dict[str, str | None] = {}  # accession -> most-specific fold
        self._promoted: set[str] = set()          # accessions promoted to candidates
        self._claim_validations: dict[str, dict] = {}  # accession -> latest validation
        self._retry_pending = False
        self._pending_targets = False
        self._failed_attempts: dict[str, int] = {}

    def should_fire(self, state: AgentState, has_eval_calls: bool = False, messages: list[dict] | None = None) -> bool:
        # Fire as soon as the mechanism families are DERIVED (i.e. the agent has
        # deemed which families are relevant) — the derived families themselves seed
        # the survey via search_interpro, so a reviewed enzyme whose fold matches the
        # target on native substrate is registered IMMEDIATELY, not gated behind
        # discovering + screening other candidates first (which pushed such hits to a
        # late stage).  Then RE-FIRE as the discovered pool grows so later folds get
        # surveyed too; each firing only surveys folds NOT already done, so re-runs
        # stay cheap.
        if state.family_survey_done or not state.mechanism_families_done:
            return False
        # Yield to candidate screening/evaluation whenever the survey has already
        # promoted candidates that are still unscreened. Each promotion bumps
        # discovery_revision, so without this the survey re-fires on its own
        # output forever and starves the pre-evaluation pipeline — promoted
        # members pile up as "discovered" and never reach evaluation. Once
        # screening drains the pending set the gate re-opens for any folds still
        # awaiting a survey, so interleaving survey and screening stays complete.
        if state.pending_candidate_count > 0:
            return False
        if self._attempts >= self._MAX_ATTEMPTS:
            return False
        if self._attempts == 0:
            # Reconciliation has already grounded the inventory in retrieved
            # literature. Resolve its folds here instead of waiting for the LLM
            # to repeat gene/name searches whose wording may not match generated
            # family labels in the ledger. Mapping validation and native-reaction
            # checks below remain the hard boundary for candidate promotion.
            from enzyme_discovery_agent.persistence import read_json
            reconciliation = read_json(
                state.run_dir / "family_inventory_reconciliation.json", {}
            )
            return (
                isinstance(reconciliation, dict)
                and reconciliation.get("status") == "completed"
            )
        return (self._retry_pending or self._pending_targets
                or state.discovery_revision > self._last_attempt_count)

    def _derived_families(self, run_dir) -> list[dict]:
        # Survey EVERY derived mechanism family, not only the focus subset.
        # focus vs exploration is an evaluation-PRIORITY split, not a relevance
        # verdict: a free-substrate ligase is demoted to exploration for an
        # incidental "intramolecular" mention in its route text, and a genuinely
        # on-target family can miss the focus cut on a noisy model partition.
        # Gating the SURVEY on that split silently drops relevant families before
        # a single member is examined (observed: an acyl-adenylate amide-ligase
        # route left unsurveyed). Surveying is now a converging fixed point and the
        # survey's own direction/role-fit gate decides promotion, so covering the
        # exploration routes broadens family coverage without admitting off-target
        # members; the evaluation budget still prioritizes by grounded relevance.
        # The ledger is the complete derived-family record (focus + exploration);
        # fall back to the active file only before the ledger exists.
        from enzyme_discovery_agent.models.family_ledger import load_family_ledger
        records = load_family_ledger(run_dir).get("families", {})
        ledgered = [
            rec["hypothesis"] for rec in records.values()
            if isinstance(rec, dict) and rec.get("hypothesis")
        ]
        if ledgered:
            return ledgered
        try:
            data = json.loads((run_dir / "mechanism_families.json").read_text())
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _promote_relevant(
        self, run_dir, fold_id: str, relevant: list[dict],
        route_ids: list[str] | None = None,
        family_hypotheses: list[dict] | None = None,
    ) -> int:
        """Register a fold's role-complete members as candidates.

        Each becomes a ``discovered`` candidate tagged with the family it came from
        and its role-wise accommodation facts, so provenance shows WHY it was
        promoted. Dedup is
        across the whole run so a member surfacing in several folds is promoted once.
        """
        from enzyme_discovery_agent.models.candidate_registry import (
            add_discovered_candidates,
        )
        from enzyme_discovery_agent.models.retrieval_leads import load_retrieval_leads

        promoted = 0
        leads = load_retrieval_leads(run_dir).get("leads", {})
        hypotheses = [item for item in (family_hypotheses or []) if item]
        family_names = list(dict.fromkeys(
            str(item.get("name")) for item in hypotheses if item.get("name")
        ))
        family_aliases = list(dict.fromkeys(
            str(value)
            for item in hypotheses
            for value in (item.get("aliases") or []) + (item.get("search_descriptors") or [])
            if value
        ))
        registrations = []
        accepted_accessions = []
        for m in relevant:
            acc = m.get("accession")
            if not acc or acc in self._promoted:
                continue
            accommodation = m.get("accommodation") or {}
            role_compatibility = m.get("role_compatibility") or {}
            validation = m.get("claim_validation") or {}
            lane = m.get("discovery_lane", "supported")
            if lane == "supported":
                if (m.get("direction_vs_target") != "same"
                        or accommodation.get("all_target_substrates_accommodated") is not True
                        or role_compatibility.get("all_target_substrates_accommodated") is not True
                        or ("claim_validation" in m
                            and validation.get("promotion_eligible") is not True)):
                    continue
            elif lane == "family_supported":
                family_validation = m.get("family_support_validation") or {}
                if (family_validation.get("promotion_eligible") is not True
                        or not m.get("family_anchor_accessions")
                        or m.get("evidence_level") != "validated_activity_annotation_match"):
                    continue
            elif lane == "same_direction_analogue":
                claims = {
                    item.get("claim_type"): item
                    for item in validation.get("claims") or []
                }
                if (m.get("direction_vs_target") != "same"
                        or any((claims.get(name) or {}).get("verdict") != "supported"
                               for name in (
                                   "accession_identity", "record_active",
                                   "family_membership", "native_reaction",
                                   "reaction_direction", "bond_type",
                               ))):
                    continue
            else:
                continue
            seed = leads.get(acc) or {}
            meta = dict(seed.get("metadata") or {})
            if seed.get("literature_evidence"):
                meta["literature_evidence"] = seed["literature_evidence"]
            meta.update({
                "source_tool": (
                    f"family_substrate_fit:{fold_id}" if lane == "supported"
                    else f"family_native_analogue:{fold_id}"
                    if lane == "same_direction_analogue"
                    else f"family_supported:{fold_id}"
                ),
                "substrate_accommodation": m.get("accommodation"),
                "role_compatibility": m.get("role_compatibility"),
                "direction_compatible": (
                    True if lane in {"supported", "same_direction_analogue"} else None
                ),
                "substrate_roles_compatible": True if lane == "supported" else None,
                "target_substrate_plausible": True if lane == "supported" else None,
                "screening_family_id": fold_id,
                "route_ids": list(route_ids or []),
                "mechanism_family_names": family_names,
                "mechanism_family_aliases": family_aliases,
                "discovery_lane": lane,
                "evaluation_depth_priority": bool(
                    m.get("evaluation_depth_priority")
                ),
                "evidence_level": m.get("evidence_level") or "direct_native_reaction",
                "family_anchor_accessions": m.get("family_anchor_accessions") or [],
                "reviewed": m.get("reviewed"),
                "gene_names": m.get("gene_names") or [],
                "ec_numbers": m.get("ec") or [],
                "native_reaction": m.get("native_reaction") or "",
                "catalytic_activities": (
                    [m.get("native_reaction")] if m.get("native_reaction") else []
                ),
            })
            if m.get("exploration_reason"):
                meta["exploration_reason"] = m["exploration_reason"]
            if validation:
                meta["claim_validation"] = validation
            if m.get("family_support_validation"):
                meta["family_support_validation"] = m["family_support_validation"]
            # Use protein_name (not literature_name) — these are found by fold +
            # native-substrate resemblance, NOT named in a paper, so conflating them
            # with the literature-named channel would corrupt provenance-by-name.
            if m.get("name"):
                meta["protein_name"] = m["name"]
            registrations.append((acc, meta))
            accepted_accessions.append(acc)
        try:
            promoted = add_discovered_candidates(run_dir, registrations)
            self._promoted.update(accepted_accessions)
        except Exception as exc:  # noqa: BLE001 - promotion must never crash a survey
            logger.debug("promotion batch failed for %s: %s", fold_id, exc)
        return promoted

    def execute(self, state: AgentState, messages: list[dict] | None = None) -> GateResult:
        from enzyme_discovery_agent.gates.family_relevance import collect_survey_folds

        run_dir = state.run_dir
        # Reconstruct durable progress when a run resumes in a new process.
        if not self._all_folds:
            from enzyme_discovery_agent.discovery.family_survey import (
                load_family_survey_state,
            )
            landscapes, validations = load_family_survey_state(run_dir)
            self._all_folds.update(landscapes)
            self._surveyed_folds.update(landscapes)
            self._claim_validations.update(validations)
        # Preserve the audit trail if a run resumes in a new process.  New
        # validations replace older entries for the same accession below.
        if not self._claim_validations:
            from enzyme_discovery_agent.discovery.family_survey import (
                load_family_survey_state,
            )
            _, validations = load_family_survey_state(run_dir)
            self._claim_validations.update(validations)
        self._attempts += 1
        self._last_attempt_count = state.discovery_revision
        # Only give up permanently once retries are exhausted; an empty early
        # attempt should leave the gate open to retry with more seeds.
        give_up = {"family_survey_done": True} if self._attempts >= self._MAX_ATTEMPTS else {}

        # Comprehensive fold set: derived families (via search_interpro) + every
        # distinct discovered gene's fold.  Only survey folds not already done.
        targets = collect_survey_folds(
            run_dir, self._derived_families(run_dir),
            self._count_cache, self._fold_cache, log_fn=self.log,
        )
        target_by_fold = {t["fold_id"]: t for t in targets}
        new_folds = [
            t["fold_id"] for t in targets
            if t["fold_id"] not in self._surveyed_folds
            and self._failed_attempts.get(t["fold_id"], 0) < self._MAX_ATTEMPTS_PER_FOLD
        ][:_MAX_NEW_FOLDS_PER_FIRING]
        self._pending_targets = any(
            t["fold_id"] not in self._surveyed_folds
            and self._failed_attempts.get(t["fold_id"], 0) < self._MAX_ATTEMPTS_PER_FOLD
            and t["fold_id"] not in new_folds
            for t in targets
        )
        if not new_folds and not self._all_folds:
            if give_up:
                self.log("  Gate: family survey resolved no surveyable folds after "
                         f"{self._attempts} attempts; proceeding without a landscape")
            return GateResult(intercept=False, state_updates=give_up)
        if not new_folds:
            # Nothing new this pass; leave the gate open for later growth.
            return GateResult(intercept=False, state_updates=give_up)

        inj = {"run_dir": str(run_dir)}

        from enzyme_discovery_agent.models.event_log import append_event

        survey_operations = {fid: uuid.uuid4().hex for fid in new_folds}
        survey_indexes = {
            fid: index for index, fid in enumerate(new_folds, start=1)
        }
        workers = max(1, min(config.FAMILY_SURVEY_WORKERS, len(new_folds)))
        self.log(
            f"  Family survey: checking {len(new_folds)} InterPro families with "
            f"{workers} worker(s)"
        )

        def _survey(fid):
            operation_id = survey_operations[fid]
            index = survey_indexes[fid]
            started = time.monotonic()
            append_event(
                run_dir, "family_survey_started",
                operation_id=operation_id,
                fold_id=fid,
                family_index=index,
                family_total=len(new_folds),
            )
            self.log(f"  Family survey {index}/{len(new_folds)} started: {fid}")
            try:
                result = execute_tool(
                    "survey_family_reactions",
                    {"interpro_id": fid}, injected_args=inj,
                )
            except Exception as exc:  # noqa: BLE001 - never let a survey crash the loop
                logger.debug("family survey failed for %s: %s", fid, exc)
                append_event(
                    run_dir, "family_survey_failed",
                    operation_id=operation_id,
                    fold_id=fid,
                    family_index=index,
                    family_total=len(new_folds),
                    error_type=type(exc).__name__,
                    elapsed_s=round(time.monotonic() - started, 3),
                )
                self.log(
                    f"  Family survey {index}/{len(new_folds)} failed after "
                    f"{time.monotonic() - started:.1f}s: {fid}",
                    level="warning",
                )
                return None
            data = result.get("data") or {} if isinstance(result, dict) else {}
            status = result.get("status") if isinstance(result, dict) else None
            append_event(
                run_dir,
                "family_survey_completed" if status == "success" else "family_survey_failed",
                operation_id=operation_id,
                fold_id=fid,
                family_index=index,
                family_total=len(new_folds),
                status=status,
                total_reviewed_member_count=data.get("total_reviewed_member_count"),
                native_annotated_reviewed_member_count=data.get(
                    "native_annotated_reviewed_member_count"
                ),
                reviewed_member_count=len(data.get("members") or []),
                direct_candidate_count=len(data.get("relevant_members") or []),
                analogue_count=len(data.get("same_direction_analogues") or []),
                family_supported_count=len(data.get("family_supported_members") or []),
                elapsed_s=round(time.monotonic() - started, 3),
            )
            self.log(
                f"  Family survey {index}/{len(new_folds)} {status or 'failed'} in "
                f"{time.monotonic() - started:.1f}s: {fid}; "
                f"reviewed={len(data.get('members') or [])}, "
                f"direct={len(data.get('relevant_members') or [])}, "
                f"analogues={len(data.get('same_direction_analogues') or [])}, "
                f"annotation-matched={len(data.get('family_supported_members') or [])}"
            )
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_survey, fold_id) for fold_id in new_folds]
            pending = set(futures)
            while pending:
                _, pending = wait(
                    pending,
                    timeout=config.FAMILY_PROGRESS_INTERVAL_SECONDS,
                    return_when=ALL_COMPLETED,
                )
                if pending:
                    self.log(
                        f"  Family survey progress: "
                        f"{len(new_folds) - len(pending)}/{len(new_folds)} complete, "
                        f"{len(pending)} pending"
                    )
            # Reduce in input order, independent of completion order.
            results = [future.result() for future in futures]

        n_promoted = 0
        self._retry_pending = False
        from enzyme_discovery_agent.models.family_ledger import (
            has_pending_family_survey,
            record_survey_attempt,
        )
        for fid, r in zip(new_folds, results):
            fold_target = target_by_fold.get(fid, {})
            hypotheses = fold_target.get("derived_family_hypotheses") or [
                fold_target.get("derived_family_hypothesis")
            ]
            hypotheses = [hypothesis for hypothesis in hypotheses if hypothesis]
            route_ids = list(dict.fromkeys(
                str(hypothesis.get("route_id")) for hypothesis in hypotheses
                if hypothesis.get("route_id")
            ))
            if not r or r.get("status") != "success":
                self._failed_attempts[fid] = self._failed_attempts.get(fid, 0) + 1
                exhausted = self._failed_attempts[fid] >= self._MAX_ATTEMPTS_PER_FOLD
                self._retry_pending = self._retry_pending or not exhausted
                if hypotheses:
                    for hypothesis in hypotheses:
                        record_survey_attempt(
                            run_dir, fid, success=False, family=hypothesis,
                            error=(r or {}).get("summary") or "survey failed",
                            exhausted=exhausted,
                        )
                else:
                    record_survey_attempt(
                        run_dir, fid, success=False,
                        error=(r or {}).get("summary") or "survey failed",
                        exhausted=exhausted,
                    )
                continue
            self._surveyed_folds.add(fid)
            d = r.get("data") or {}
            got = d.get("fold_id") or fid
            # Promote the relevant enzymes in this family (grounded same-direction
            # native reaction + resolved substrate fit — reviewed + gated TrEMBL) to
            # candidates, so the substrate match actually feeds evaluation instead of
            # only being read by the model.  EVERY relevant member is promoted (no
            # top-N cap): dropping relevant members defeats discovery.  The set is
            # bounded by the relevance gate + the fold's surveyed-member ceiling.
            relevant = d.get("relevant_members") or []
            analogues = d.get("same_direction_analogues") or []
            # Every reviewed same-direction reaction-backed member is a candidate.
            # A function-diverse subset receives deep follow-up first, but that
            # resource decision must not truncate the candidate inventory.
            depth_priority = _select_evaluation_admissions(
                [*relevant, *analogues],
                config.FAMILY_REVIEWED_EVALUATION_CAP_PER_FOLD,
            )
            priority_accessions = {
                member.get("accession") for member in depth_priority
                if member.get("accession")
            }
            admitted = [*relevant, *analogues]
            for member in admitted:
                member["discovery_lane"] = (
                    "supported" if member in relevant
                    else "same_direction_analogue"
                )
                member["evaluation_depth_priority"] = (
                    member.get("accession") in priority_accessions
                )
            n_promoted += self._promote_relevant(
                run_dir, got, admitted, route_ids,
                family_hypotheses=hypotheses,
            )
            family_supported = d.get("family_supported_members") or []
            for member in family_supported:
                member["discovery_lane"] = "family_supported"
            n_promoted += self._promote_relevant(
                run_dir, got, family_supported, route_ids,
                family_hypotheses=hypotheses,
            )
            exploration = d.get("exploration_members") or []
            for member in exploration:
                member["discovery_lane"] = "exploration"
            # Uncertain members remain visible in the durable family landscape,
            # but they are not candidates. Unknown direction, unresolved native
            # substrates, or incomplete target-role accommodation is precisely
            # the evidence gap the candidate registry must not conceal.
            if got in self._all_folds:
                continue
            prof = d.get("family_reaction_profile", {})
            members = d.get("members") or []
            reviewed_validations = [
                m.get("claim_validation") for m in members
                if m.get("reviewed") and m.get("claim_validation")
            ]
            for validation in reviewed_validations:
                self._claim_validations[validation.get("subject", "")] = validation
            eligible = sum(v.get("promotion_eligible") is True for v in reviewed_validations)
            # Centralized falsifiable-hypothesis verdict: "no evidence" stays
            # insufficient (an exploration lead), never silently "feasible".
            from enzyme_discovery_agent.models.family_hypothesis import family_verdict
            verdict = family_verdict(reviewed_validations)
            role_complete_count = prof.get(
                "fully_accommodating_same_direction_members", 0)
            native_substrate_evidence = {
                "direction_vs_target_counts":
                    prof.get("direction_vs_target_counts") or {},
                "role_complete_same_direction_members": role_complete_count,
                "best_role_complete_members":
                    prof.get("best_role_complete_members") or [],
                "distinct_native_reactions": prof.get("distinct_reactions") or [],
                "basis": (
                    "native reactions and role-wise native substrates of "
                    "surveyed family members"
                ),
            }
            if (verdict == "supported_family_hypothesis"
                    and role_complete_count == 0):
                verdict = "insufficient_family_evidence"
            hypothesis = fold_target.get("derived_family_hypothesis")
            if hypotheses:
                for mapped_hypothesis in hypotheses:
                    record_survey_attempt(
                        run_dir, got, success=True, family=mapped_hypothesis,
                        verdict=verdict, member_count=len(members),
                        native_substrate_evidence=native_substrate_evidence,
                    )
            else:
                record_survey_attempt(
                    run_dir, got, success=True, verdict=verdict,
                    member_count=len(members),
                    native_substrate_evidence=native_substrate_evidence,
                )
            self._all_folds[got] = {
                "fold_id": got,
                "fold_name": d.get("fold_name"),
                "source": fold_target.get("source", "discovered"),
                "derived_family_hypothesis": hypothesis,
                "n_members": prof.get("n_members"),
                "direction_vs_target_counts": prof.get("direction_vs_target_counts"),
                "functional_group_similarity": prof.get("functional_group_similarity"),
                "size_ratio": prof.get("size_ratio"),
                # Per-axis context is retained for visualization, while selection
                # uses the explicit worst-role vectors below.
                "fully_accommodating_same_direction_members":
                    role_complete_count,
                "direct_candidate_count": len(relevant),
                "same_direction_analogue_count": len(analogues),
                "family_supported_candidate_count": len(family_supported),
                "reviewed_reaction_inventory": d.get("reviewed_reaction_inventory"),
                "comprehensive_inventory": d.get("comprehensive_inventory"),
                "best_role_complete_members": prof.get("best_role_complete_members") or [],
                "axis_medians": prof.get("axis_medians"),
                "native_chain_length_range": prof.get("native_chain_length_range"),
                "native_mw_range": prof.get("native_mw_range"),
                "members_spanning_aromatic_and_aliphatic":
                    prof.get("members_spanning_aromatic_and_aliphatic"),
                "family_claim_validation": {
                    "verdict": verdict,
                    "reviewed_members_checked": len(reviewed_validations),
                    "promotion_eligible_members": eligible,
                    "basis": "reviewed UniProt members of the exact InterPro family",
                    "hypothesis_tested": hypothesis.get("name") if hypothesis else None,
                },
                "top_members": [
                    {"accession": m.get("accession"),
                     "native_reaction": (m.get("native_reaction") or "")[:80],
                     "direction_vs_target": m.get("direction_vs_target"),
                     "substrate": m.get("substrate"),
                     # Role-aware accommodation vs the target substrates — the
                     # decisive, non-diluted fact the agent reasons over (which
                     # target substrate has no genuine native counterpart).
                     "accommodation": m.get("accommodation"),
                     "claim_validation": m.get("claim_validation")}
                    for m in members[:3]
                ],
            }

        # InterPro text searches can resolve ambiguously. A completed negative
        # primary mapping unlocks one alternate for a subsequent gate pass;
        # supported mappings stop that hypothesis from fanning out further.
        self._retry_pending = (
            self._retry_pending
            or has_pending_family_survey(run_dir, focus_only=False)
        )

        if not self._all_folds:
            if give_up:
                self.log("  Gate: family survey resolved no surveyable folds after "
                         f"{self._attempts} attempts; proceeding without a landscape")
            return GateResult(intercept=False, state_updates=give_up)

        # Families with at least one complete one-to-one role mapping lead. Keep
        # property tradeoffs explicit instead of collapsing them into a score.
        from enzyme_discovery_agent.gates.family_relevance import order_folds_by_relevance
        all_ordered = order_folds_by_relevance(list(self._all_folds.values()))
        ordered = all_ordered[:_MAX_FOLDS_SHOWN]

        # Persist the reducer state atomically for crash-safe resume.
        from enzyme_discovery_agent.discovery.family_survey import (
            persist_family_survey_state,
        )
        persist_family_survey_state(
            run_dir, all_ordered, self._claim_validations,
        )

        lines = []
        for f in ordered:
            best = (f.get("best_role_complete_members") or [{}])[0]
            bottleneck = best.get("role_compatibility") or {}
            lines.append(
                f"- fold {f['fold_id']} ({f.get('source', 'discovered')}, {f.get('n_members')} "
                f"reviewed members): direction {f.get('direction_vs_target_counts')}; "
                f"{f.get('fully_accommodating_same_direction_members', 0)} member(s) "
                "accommodate every target substrate role; best complete member "
                f"worst-role FG={bottleneck.get('weakest_functional_group_similarity')}, "
                f"MW={bottleneck.get('weakest_mw_similarity')}, "
                f"chain={bottleneck.get('weakest_chain_length_similarity')}"
            )
            for m in f.get("top_members", [])[:2]:
                sub = m.get("substrate") or {}
                lines.append(
                    f"    e.g. {m.get('accession')} ({m.get('direction_vs_target')}, "
                    f"fg={sub.get('functional_group_similarity')}, "
                    f"size={sub.get('size_ratio')}): {m.get('native_reaction')}"
                )
        landscape_block = "\n".join(lines)

        guidance = (
            "FAMILY LANDSCAPE (auto-surveyed comprehensively: your derived "
            "mechanism families resolved via InterPro, PLUS every distinct fold "
            "your discovered candidates fall into — so no relevant family is "
            "missed). For each fold below are GROUNDED NUMBERS — how its reviewed members' "
            "native reactions compare to the target: direction vs target, and how "
            "closely their native substrates resemble every target substrate role. "
            "Correct bond direction is only an entry condition. CONCENTRATE follow-up "
            "on folds with role-complete members. Compare each worst-role property "
            "vector directly; do not average properties or let one well-matched "
            "substrate compensate for a poorly matched substrate. Set aside "
            "folds whose native reactions run the opposite direction or whose "
            "substrates are far larger / decorated with extra groups. Prefer reading "
            "the resembling members' OWN characterizing papers over broad keyword "
            "search.\n"
            f"{landscape_block}\n\n"
            "THEN ACT ON THIS: call set_family_focus(stop=[...], hold=[...], expand=[...], "
            "reasons={fold_id: why}) using the InterPro fold ids listed above — "
            "STOP only folds with direct contradictory evidence; HOLD folds with "
            "sparse or unresolved evidence; EXPAND folds with positive evidence. "
            "A stop prevents extra depth but retains one representative for audit. "
            "Contradictory folds may include those whose native substrates are unlike the target "
            "(peptide, amino acid, nucleotide, sugar, or cofactor) or run the "
            "opposite direction, and EXPAND the folds whose native reactions "
            "resemble the target. The evaluation budget HONOURS your decision: a "
            "stopped fold receives no extra depth. Give a one-line reason "
            "per fold. You can revise later by calling it again."
        )
        if n_promoted:
            guidance += (
                f"\n\nNOTE: {n_promoted} role-complete member(s) were auto-promoted to candidates (tagged "
                "family_substrate_fit:<fold>) so they enter evaluation — no action "
                "needed; STOP their fold only if you judge the family off-target."
            )
        return GateResult(
            intercept=True,
            new_messages=[
                {"role": "assistant",
                 "content": f"Comprehensively surveyed {len(self._all_folds)} fold "
                            f"{'family' if len(self._all_folds) == 1 else 'families'} "
                            "(derived mechanism families + every distinct discovered "
                            "fold); comparing their native reactions to the target."},
                {"role": "user", "content": guidance},
            ],
            log_message=f"Family survey: {len(self._all_folds)} fold(s) surveyed "
                        f"(+{len(new_folds)} this pass); {n_promoted} role-complete "
                        f"member(s) promoted; landscape injected",
            # Do NOT mark done on success: re-fire as the discovered pool grows so
            # late-appearing families get surveyed too (bounded by _MAX_ATTEMPTS).
            state_updates=give_up,
        )
