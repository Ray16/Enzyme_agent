"""Mechanism-family coverage gate.

Derives — answer-agnostically, from the reaction's bond chemistry — the distinct
enzyme mechanism families that could catalyse it, and writes them as the run's
discovery-coverage evidence buckets so the existing DiscoveryCoverageGate then
requires a screened representative of EACH family before evaluation.  Fires
early (once enough literature exists to know the reaction) and tells the agent
to search each family FROM THE LITERATURE, so literature-first recall covers the
whole mechanism space instead of collapsing onto whichever family the first good
papers named.
"""

from __future__ import annotations

import json
import logging

from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.exploration_log import append_exploration_log

logger = logging.getLogger("enzyme_agent")


class MechanismFamilyGate:
    """Derive the reaction's mechanism families and make each a coverage
    requirement, searched from the literature before evaluation."""

    def __init__(self, llm_client=None, log_fn=None):
        self.llm = llm_client
        self.log = log_fn or (lambda *a, **k: None)

    def should_fire(self, state: AgentState) -> bool:
        return (
            not state.mechanism_families_done
            and self.llm is not None
            and state.lit_search_count >= config.MIN_LITERATURE_SEARCH_ROUNDS
        )

    def execute(self, state: AgentState) -> GateResult:
        from enzyme_discovery_agent.tools.mechanism_families import (
            derive_mechanism_families,
        )

        ctx_path = state.run_dir / "target_context.json"
        try:
            ctx = json.loads(ctx_path.read_text())
        except (OSError, json.JSONDecodeError):
            ctx = {}
        query = ctx.get("query", "") or ""
        substrate = ctx.get("substrate_smiles") or ""

        families = derive_mechanism_families(query, substrate, self.llm)
        if not families:
            # Derivation unavailable — leave the generic coverage contract in
            # place and don't fire again.
            return GateResult(intercept=False,
                              state_updates={"mechanism_families_done": True})

        # Persist the full derivation (with literature terms) for audit + guidance.
        (state.run_dir / "mechanism_families.json").write_text(
            json.dumps(families, indent=2) + "\n"
        )

        # Express each family as a discovery-coverage evidence bucket, so the
        # existing coverage gate requires a screened member of each before
        # evaluation.  Merge into the run's target profile.
        buckets = [
            {"name": f["name"], "match_any": f["descriptors"], "minimum_candidates": 1}
            for f in families
        ]
        profile = ctx.get("target_profile") or {}
        cov = profile.get("discovery_coverage") or {}
        existing = {b.get("name") for b in cov.get("evidence_buckets", [])}
        cov["evidence_buckets"] = cov.get("evidence_buckets", []) + [
            b for b in buckets if b["name"] not in existing
        ]
        # Require at least as many distinct families as we derived.
        cov["minimum_distinct_families"] = max(
            cov.get("minimum_distinct_families", 0), len(families)
        )
        profile["discovery_coverage"] = cov
        ctx["target_profile"] = profile
        try:
            ctx_path.write_text(json.dumps(ctx, indent=2) + "\n")
        except OSError as exc:
            logger.warning("could not persist mechanism-family coverage: %s", exc)

        append_exploration_log(
            state.run_dir,
            entry_type="COVERAGE",
            title="Mechanism families derived for coverage",
            details={"Families": "; ".join(f["name"] for f in families)},
        )

        lines = []
        for f in families:
            terms = "; ".join(f.get("literature_terms") or f["descriptors"])
            lines.append(f"- {f['name']}: {terms}")
        families_block = "\n".join(lines)

        guidance = (
            "MECHANISM-FAMILY COVERAGE. The target bond-forming reaction can be "
            "catalysed by several DISTINCT enzyme families/folds. Before you "
            "evaluate, cover EACH of these families from the LITERATURE — run "
            "targeted literature searches with the terms below, fetch full text, "
            "and register at least one representative per family. Do not converge "
            "on one family or skip a family because the first papers named a "
            "different one:\n"
            f"{families_block}\n\n"
            "Evaluation stays deferred until each family has a screened "
            "representative (or you have searched it and shown it has no viable "
            "member)."
        )
        return GateResult(
            intercept=True,
            new_messages=[
                {"role": "assistant",
                 "content": f"Derived {len(families)} mechanism families to cover: "
                            + ", ".join(f["name"] for f in families)},
                {"role": "user", "content": guidance},
            ],
            log_message=f"Mechanism families derived: {len(families)} to cover from literature",
            state_updates={"mechanism_families_done": True},
        )
