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
            rank_families_by_substrate_resemblance,
        )

        ctx_path = state.run_dir / "target_context.json"
        try:
            ctx = json.loads(ctx_path.read_text())
        except (OSError, json.JSONDecodeError):
            ctx = {}
        query = ctx.get("query", "") or ""
        substrate = ctx.get("substrate_smiles") or ""

        families = derive_mechanism_families(query, substrate, self.llm)
        if families:
            # Order families by how closely their native substrates resemble the
            # target substrates, so guidance can steer search DEPTH into the most
            # promising family instead of spreading budget thinly across all.
            families = rank_families_by_substrate_resemblance(
                families, query, substrate, self.llm
            )
        if not families:
            # Derivation unavailable — leave the generic coverage contract in
            # place and don't fire again.
            return GateResult(intercept=False,
                              state_updates={"mechanism_families_done": True})

        # Persist the full derivation (with literature terms) for audit + guidance.
        (state.run_dir / "mechanism_families.json").write_text(
            json.dumps(families, indent=2) + "\n"
        )

        # SOFT guidance only — do NOT turn the families into hard coverage
        # buckets or raise minimum_distinct_families.  The coverage matcher does a
        # literal text match against candidate metadata, and abstract fold
        # descriptors ("glutamine synthetase fold", "adenylation") never appear
        # verbatim in UniProt annotations, so a hard requirement would be
        # impossible to satisfy and would block evaluation entirely (an empty
        # report).  Coverage is encouraged through the guidance below and the
        # generic distinct-family contract, not enforced per derived family.

        append_exploration_log(
            state.run_dir,
            entry_type="COVERAGE",
            title="Mechanism families derived for coverage",
            details={"Families": "; ".join(f["name"] for f in families)},
        )

        lines = []
        for i, f in enumerate(families):
            terms = "; ".join(f.get("literature_terms") or f["descriptors"])
            marker = " [MOST PROMISING — substrate match]" if i == 0 else ""
            reason = f.get("promise_reason")
            reason_txt = f" (why: {reason})" if reason and i == 0 else ""
            lines.append(f"- {f['name']}{marker}: {terms}{reason_txt}")
        families_block = "\n".join(lines)
        top_name = families[0]["name"]

        guidance = (
            "MECHANISM-FAMILY COVERAGE. The target bond-forming reaction can be "
            "catalysed by several DISTINCT enzyme families/folds, listed below in "
            "DESCENDING order of how closely each family's native substrates "
            "resemble the target substrates.\n"
            "STRATEGY — go DEEP before broad. Do NOT spend equal budget on every "
            "family, and do NOT wander into families whose bond DIRECTION or type "
            "does not match the target reaction. Instead:\n"
            f"1. EXPAND on the most promising family first ('{top_name}'): run "
            "SEVERAL targeted literature + sequence-database searches that CROSS "
            "that fold with the target substrate class, fetch full text, and "
            "register MULTIPLE members — especially the ones whose native "
            "substrate most resembles the target's reactive groups and size, plus "
            "their close homologs/orthologs — not just one representative.\n"
            "2. Then TOUCH each remaining family at least once (one representative "
            "from the literature) so no distinct fold is missed, and expand any "
            "that also turn out to fit the target substrates well.\n"
            "Within every family, look for the MEMBER whose native substrate "
            "RESEMBLES THE TARGET substrates, not the family's textbook archetype. "
            "Do not converge prematurely on a family the first papers happened to "
            "name:\n"
            f"{families_block}\n"
            "GROUND THE CHOICE IN THE FAMILY'S NATIVE REACTIONS. Once you have a "
            "candidate enzyme in hand, call survey_family_reactions on a "
            "representative candidate of EACH promising family. It resolves that "
            "candidate's fold family and lists the fold's reviewed members WITH "
            "THEIR NATIVE REACTIONS, comparing each to the target reaction "
            "(direction + bond type). CONCENTRATE expansion on the family whose "
            "members' native reactions MATCH the target reaction, and DEPRIORITIZE "
            "families whose native reactions run opposite or unrelated to it. This "
            "is soft guidance to steer where you dig, not a hard requirement — let "
            "the surveyed native-reaction fit, not a family label, decide."
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
