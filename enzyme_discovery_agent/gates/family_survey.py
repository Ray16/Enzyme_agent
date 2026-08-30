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
from concurrent.futures import ThreadPoolExecutor

from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.tool_registry import execute_tool

logger = logging.getLogger("enzyme_agent")

_MAX_NEW_FOLDS_PER_FIRING = 16   # folds surveyed per firing (bounds compute)
_MAX_FOLDS_SHOWN = 12            # folds presented back to the agent
_WORKERS = 6


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
    _MAX_ATTEMPTS = 6

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
        if self._attempts >= self._MAX_ATTEMPTS:
            return False
        if self._attempts == 0:
            return True   # first pass: survey the derived families right away
        return state.candidate_count > self._last_attempt_count

    def _derived_families(self, run_dir) -> list[dict]:
        try:
            return json.loads((run_dir / "mechanism_families.json").read_text())
        except (OSError, json.JSONDecodeError):
            return []

    def _promote_relevant(self, run_dir, fold_id: str, relevant: list[dict]) -> int:
        """Register a fold's relevant members (best substrate fit) as candidates.

        Each becomes a ``discovered`` candidate tagged with the family it came from
        and its substrate fit, so provenance shows WHY it was promoted.  Dedup is
        across the whole run so a member surfacing in several folds is promoted once.
        """
        from enzyme_discovery_agent.models.candidate_registry import (
            update_candidate_registry,
        )

        promoted = 0
        for m in relevant:
            acc = m.get("accession")
            if not acc or acc in self._promoted:
                continue
            validation = m.get("claim_validation") or {}
            if "claim_validation" in m and validation.get("promotion_eligible") is not True:
                continue
            self._promoted.add(acc)
            meta = {
                "source_tool": f"family_substrate_fit:{fold_id}",
                "substrate_fit": m.get("substrate_fit"),
            }
            if validation:
                meta["claim_validation"] = validation
            # Use protein_name (not literature_name) — these are found by fold +
            # native-substrate resemblance, NOT named in a paper, so conflating them
            # with the literature-named channel would corrupt provenance-by-name.
            if m.get("name"):
                meta["protein_name"] = m["name"]
            try:
                update_candidate_registry(run_dir, acc, "discovered", metadata=meta)
                promoted += 1
            except Exception as exc:  # noqa: BLE001 - promotion must never crash a survey
                logger.debug("promotion failed for %s: %s", acc, exc)
        return promoted

    def execute(self, state: AgentState, messages: list[dict] | None = None) -> GateResult:
        from enzyme_discovery_agent.gates.family_relevance import collect_survey_folds

        run_dir = state.run_dir
        # Preserve the audit trail if a run resumes in a new process.  New
        # validations replace older entries for the same accession below.
        if not self._claim_validations:
            try:
                persisted = json.loads((run_dir / "claim_validation.json").read_text())
                for validation in persisted.get("validations") or []:
                    subject = validation.get("subject")
                    if subject:
                        self._claim_validations[subject] = validation
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        self._attempts += 1
        self._last_attempt_count = state.candidate_count
        # Only give up permanently once retries are exhausted; an empty early
        # attempt should leave the gate open to retry with more seeds.
        give_up = {"family_survey_done": True} if self._attempts >= self._MAX_ATTEMPTS else {}

        # Comprehensive fold set: derived families (via search_interpro) + every
        # distinct discovered gene's fold.  Only survey folds not already done.
        targets = collect_survey_folds(
            run_dir, self._derived_families(run_dir),
            self._count_cache, self._fold_cache,
        )
        target_by_fold = {t["fold_id"]: t for t in targets}
        new_folds = [t["fold_id"] for t in targets
                     if t["fold_id"] not in self._surveyed_folds][:_MAX_NEW_FOLDS_PER_FIRING]
        if not new_folds and not self._all_folds:
            if give_up:
                self.log("  Gate: family survey resolved no surveyable folds after "
                         f"{self._attempts} attempts; proceeding without a landscape")
            return GateResult(intercept=False, state_updates=give_up)
        if not new_folds:
            # Nothing new this pass; leave the gate open for later growth.
            return GateResult(intercept=False, state_updates=give_up)

        inj = {"run_dir": str(run_dir)}

        def _survey(fid):
            try:
                return execute_tool("survey_family_reactions",
                                    {"interpro_id": fid}, injected_args=inj)
            except Exception as exc:  # noqa: BLE001 - never let a survey crash the loop
                logger.debug("family survey failed for %s: %s", fid, exc)
                return None

        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            results = list(pool.map(_survey, new_folds))

        n_promoted = 0
        for fid, r in zip(new_folds, results):
            self._surveyed_folds.add(fid)
            if not r or r.get("status") != "success":
                continue
            d = r.get("data") or {}
            got = d.get("fold_id") or fid
            # Promote the relevant enzymes in this family (grounded same-direction
            # native reaction + resolved substrate fit — reviewed + gated TrEMBL) to
            # candidates, so the substrate match actually feeds evaluation instead of
            # only being read by the model.  EVERY relevant member is promoted (no
            # top-N cap): dropping relevant members defeats discovery.  The set is
            # bounded by the relevance gate + the fold's surveyed-member ceiling.
            relevant = d.get("relevant_members") or []
            n_promoted += self._promote_relevant(run_dir, got, relevant)
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
            fold_target = target_by_fold.get(fid, {})
            hypothesis = fold_target.get("derived_family_hypothesis")
            self._all_folds[got] = {
                "fold_id": got,
                "fold_name": d.get("fold_name"),
                "source": fold_target.get("source", "discovered"),
                "derived_family_hypothesis": hypothesis,
                "n_members": prof.get("n_members"),
                "direction_vs_target_counts": prof.get("direction_vs_target_counts"),
                "functional_group_similarity": prof.get("functional_group_similarity"),
                "size_ratio": prof.get("size_ratio"),
                # Comprehensive, even-handed native-substrate-fit picture (drives
                # the fit radar/bar for experimentalists): equal-weight holistic
                # fit, per-axis medians, native size range, and aromatic/aliphatic
                # breadth (a permissive pocket that spans both).
                "holistic_fit": prof.get("holistic_fit"),
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
                     "claim_validation": m.get("claim_validation")}
                    for m in members[:3]
                ],
            }

        if not self._all_folds:
            if give_up:
                self.log("  Gate: family survey resolved no surveyable folds after "
                         f"{self._attempts} attempts; proceeding without a landscape")
            return GateResult(intercept=False, state_updates=give_up)

        # Order folds by how closely their members resemble the target (max
        # functional-group similarity) — a presentation order over a grounded
        # number, NOT a cutoff.
        ordered = sorted(
            self._all_folds.values(),
            key=lambda f: -((f.get("functional_group_similarity") or {}).get("max") or -1.0),
        )[:_MAX_FOLDS_SHOWN]

        # Persist the landscape (holistic family picture for the run + the report).
        try:
            (run_dir / "family_landscape.json").write_text(
                json.dumps(ordered, indent=2) + "\n"
            )
            (run_dir / "claim_validation.json").write_text(json.dumps({
                "schema_version": "1.0",
                "scope": "family_survey_members",
                "validations": list(self._claim_validations.values()),
            }, indent=2) + "\n")
        except OSError:
            pass

        lines = []
        for f in ordered:
            fg = f.get("functional_group_similarity") or {}
            sz = f.get("size_ratio") or {}
            lines.append(
                f"- fold {f['fold_id']} ({f.get('source', 'discovered')}, {f.get('n_members')} "
                f"reviewed members): direction {f.get('direction_vs_target_counts')}; "
                f"functional-group similarity max {fg.get('max')} / median {fg.get('median')}; "
                f"size ratio max {sz.get('max')} / median {sz.get('median')}"
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
            "closely their native substrates resemble the target (functional-group "
            "overlap, size ratio). No cutoff or label is applied — YOU reason over "
            "the numbers. CONCENTRATE follow-up expansion and full-text reading on "
            "the fold(s) whose members most resemble the target (higher functional-"
            "group similarity, closer size, same reaction direction), and set aside "
            "folds whose native reactions run the opposite direction or whose "
            "substrates are far larger / decorated with extra groups. Prefer reading "
            "the resembling members' OWN characterizing papers over broad keyword "
            "search.\n"
            f"{landscape_block}\n\n"
            "THEN ACT ON THIS: call set_family_focus(stop=[...], expand=[...], "
            "reasons={fold_id: why}) using the InterPro fold ids listed above — "
            "STOP the folds whose members' native substrates are unlike the target "
            "(peptide, amino acid, nucleotide, sugar, or cofactor) or run the "
            "opposite direction, and EXPAND the folds whose native reactions "
            "resemble the target. The evaluation budget HONOURS your decision: a "
            "stopped fold is waived from the coverage floor, so this is how you "
            "stop wasting evaluation on off-target families. Give a one-line reason "
            "per fold. You can revise later by calling it again."
        )
        if n_promoted:
            guidance += (
                f"\n\nNOTE: {n_promoted} member(s) whose native substrate most "
                "resembles the target were auto-promoted to candidates (tagged "
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
                        f"(+{len(new_folds)} this pass); {n_promoted} substrate-fit "
                        f"member(s) promoted; landscape injected",
            # Do NOT mark done on success: re-fire as the discovered pool grows so
            # late-appearing families get surveyed too (bounded by _MAX_ATTEMPTS).
            state_updates=give_up,
        )
