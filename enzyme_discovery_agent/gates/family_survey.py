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
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.tool_registry import execute_tool

logger = logging.getLogger("enzyme_agent")

_MAX_SEEDS = 8          # candidates whose folds we survey (deduped by fold)
_MAX_FOLDS_SHOWN = 6    # folds presented back to the agent
_WORKERS = 4


class FamilySurveyGate:
    """Auto-run the family-reaction survey and inject the grounded family map."""

    def __init__(self, log_fn=None):
        self.log = log_fn or (lambda *a, **k: None)

    def should_fire(self, state: AgentState) -> bool:
        # After families are derived and enough candidate seeds exist to survey.
        return (
            not state.family_survey_done
            and state.mechanism_families_done
            and state.candidate_count >= 3
        )

    def _seeds(self, run_dir) -> list[str]:
        """Representative candidate accessions to survey, best-characterised first."""
        registry = load_candidate_registry(run_dir).get("candidates", {})
        scored = []
        for acc, info in registry.items():
            statuses = {s.get("status") for s in info.get("statuses", [])}
            if "rejected" in statuses:
                continue
            # Prefer evaluated > screened > discovered as seeds.
            rank = (0 if "evaluated" in statuses else
                    1 if "screened" in statuses else 2)
            scored.append((rank, acc))
        scored.sort()
        return [acc for _, acc in scored[:_MAX_SEEDS]]

    def execute(self, state: AgentState) -> GateResult:
        run_dir = state.run_dir
        seeds = self._seeds(run_dir)
        if not seeds:
            return GateResult(intercept=False, state_updates={"family_survey_done": True})

        inj = {"run_dir": str(run_dir)}

        def _survey(acc):
            try:
                return execute_tool("survey_family_reactions",
                                    {"seed_accession": acc}, injected_args=inj)
            except Exception as exc:  # noqa: BLE001 - never let a survey crash the loop
                logger.debug("family survey failed for %s: %s", acc, exc)
                return None

        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            results = list(pool.map(_survey, seeds))

        # Keep one landscape entry per DISTINCT fold (candidates share folds).
        folds: dict[str, dict] = {}
        for r in results:
            if not r or r.get("status") != "success":
                continue
            d = r.get("data") or {}
            fid = d.get("fold_id")
            if not fid or fid in folds:
                continue
            prof = d.get("family_reaction_profile", {})
            members = d.get("members") or []
            folds[fid] = {
                "fold_id": fid,
                "seed": d.get("seed_accession"),
                "n_members": prof.get("n_members"),
                "direction_vs_target_counts": prof.get("direction_vs_target_counts"),
                "functional_group_similarity": prof.get("functional_group_similarity"),
                "size_ratio": prof.get("size_ratio"),
                "top_members": [
                    {"accession": m.get("accession"),
                     "native_reaction": (m.get("native_reaction") or "")[:80],
                     "direction_vs_target": m.get("direction_vs_target"),
                     "substrate": m.get("substrate")}
                    for m in members[:3]
                ],
            }

        if not folds:
            return GateResult(intercept=False, state_updates={"family_survey_done": True})

        # Order folds by how closely their members resemble the target (max
        # functional-group similarity) — a presentation order over a grounded
        # number, NOT a cutoff.
        ordered = sorted(
            folds.values(),
            key=lambda f: -((f.get("functional_group_similarity") or {}).get("max") or -1.0),
        )[:_MAX_FOLDS_SHOWN]

        # Persist the landscape (holistic family picture for the run + the report).
        try:
            (run_dir / "family_landscape.json").write_text(
                json.dumps(ordered, indent=2) + "\n"
            )
        except OSError:
            pass

        lines = []
        for f in ordered:
            fg = f.get("functional_group_similarity") or {}
            sz = f.get("size_ratio") or {}
            lines.append(
                f"- fold {f['fold_id']} (from {f.get('seed')}, {f.get('n_members')} "
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
            "FAMILY LANDSCAPE (auto-surveyed from your candidates' fold families). "
            "For each fold below are GROUNDED NUMBERS — how its reviewed members' "
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
            f"{landscape_block}"
        )
        return GateResult(
            intercept=True,
            new_messages=[
                {"role": "assistant",
                 "content": f"Auto-surveyed {len(ordered)} fold "
                            f"{'family' if len(ordered) == 1 else 'families'} my "
                            "candidates fall into; comparing their native reactions "
                            "to the target."},
                {"role": "user", "content": guidance},
            ],
            log_message=f"Family survey: {len(ordered)} fold(s) surveyed and "
                        "landscape injected",
            state_updates={"family_survey_done": True},
        )
