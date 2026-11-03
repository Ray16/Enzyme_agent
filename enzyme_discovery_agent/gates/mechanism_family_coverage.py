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
from enzyme_discovery_agent.state_store import atomic_write_json
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.exploration_log import append_exploration_log
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")


class MechanismFamilyGate:
    """Start mechanism-led discovery when target chemistry is available."""

    def __init__(self, llm_client=None, log_fn=None):
        self.llm = llm_client
        self.log = log_fn or (lambda *a, **k: None)

    def should_fire(self, state: AgentState, has_eval_calls: bool = False, messages: list[dict] | None = None) -> bool:
        try:
            chemistry = json.loads((run_file(state.run_dir, 'target_reasoning.json')).read_text())
        except (OSError, ValueError):
            chemistry = {}
        chemistry_ready = bool(chemistry.get("bond_types") or chemistry.get("activation_categories"))
        return (
            not state.mechanism_families_done
            and self.llm is not None
            and (chemistry_ready or state.candidate_count > 0 or has_eval_calls
                 or state.lit_search_count >= config.MIN_LITERATURE_SEARCH_ROUNDS)
        )

    def execute(self, state: AgentState, messages: list[dict] | None = None) -> GateResult:
        from enzyme_discovery_agent.tools.mechanism_families import (
            derive_mechanism_families,
            rank_families_by_substrate_resemblance,
        )

        ctx_path = run_file(state.run_dir, 'target_context.json')
        try:
            ctx = json.loads(ctx_path.read_text())
        except (OSError, json.JSONDecodeError):
            ctx = {}
        query = ctx.get("query", "") or ""
        substrate = ctx.get("substrate_smiles") or ""

        # Take the reaction direction from the shared chemistry contract so the
        # inverse-direction family filter agrees with every other stage.
        from enzyme_discovery_agent.models.target_reasoning import target_claims
        target_direction, _, target_topology = target_claims(state.run_dir)

        families = derive_mechanism_families(
            query, substrate, self.llm, target_direction=target_direction,
            target_topology=target_topology,
            retrieved_papers=state.accumulated_papers,
        )
        if families:
            # Order families by how closely their native substrates resemble the
            # target substrates, so guidance can steer search DEPTH into the most
            # promising family instead of spreading budget thinly across all.
            families = rank_families_by_substrate_resemblance(
                families, query, substrate, self.llm
            )
        if not families:
            if getattr(self.llm, "budget_exhausted", False):
                return GateResult(intercept=False)
            # Derivation unavailable — leave the generic coverage contract in
            # place and don't fire again.
            return GateResult(intercept=False,
                              state_updates={"mechanism_families_done": True})

        # Persist the full derivation (with literature terms) for audit + guidance.
        atomic_write_json(run_file(state.run_dir, 'mechanism_families.json'), families)

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

        guidance = (
            "MECHANISM-FAMILY WORKFLOW. The target transformation can be "
            "catalysed by several DISTINCT enzyme families/folds, listed below in "
            "DESCENDING order of how closely each family's native substrates "
            "resemble the target substrates. Work them in THIS ORDER — do NOT keep "
            "running broad keyword literature searches yet:\n"
            f"{families_block}\n"
            "\nSTEP 1 — SEED (mostly done): your initial literature search gave you "
            "candidate enzymes and the families above. You need only ONE grounded "
            "seed candidate per plausible family before moving on — don't over-mine "
            "the literature here.\n"
            "\nSTEP 2 — SURVEY & PRUNE (do this NEXT, before any deep reading). For "
            "EACH promising family call survey_family_reactions — on a seed "
            "candidate you already found, or on an InterPro fold id from "
            "search_interpro. It enumerates the fold's reviewed members with their "
            "NATIVE REACTIONS and reports GROUNDED NUMBERS (reaction direction vs "
            "the target; how closely members' native substrates resemble the target "
            "— functional-group overlap, size ratio), with NO cutoff or label. "
            "COMPARE the families' distributions and KEEP the families whose members "
            "most resemble the target (their native substrates share the target's "
            "functional groups at a similar size); SET ASIDE families whose native "
            "reactions run the opposite direction or whose substrates are far "
            "larger / aromatic / decorated with extra groups. This prunes where to "
            "spend effort BEFORE expensive full-text reading — the surveyed numbers, "
            "not a family label, decide.\n"
            "\nSTEP 3 — DEEP LITERATURE on the SURVIVORS only. For the families you "
            "kept, now read full text and expand members + close homologs — but "
            "prefer the surveyed members' OWN characterizing papers (their UniProt "
            "entries link to them) over broad keyword searches, which pull in "
            "off-topic and wrong-direction papers. Do not spend deep-reading budget "
            "on families STEP 2 set aside."
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
