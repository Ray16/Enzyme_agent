"""Gate 3: surface a structural/embedding expansion decision for qualified seeds.

The agent decides whether to broaden from supported seeds, using the search
tools; this gate only puts that choice in front of it on a reserved cadence so
a persistent evaluation backlog cannot keep breadth off the table entirely.
"""

from __future__ import annotations

import logging
from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.expansion import (
    select_expansion_seeds, searched_seeds, expansion_due,
)

logger = logging.getLogger("enzyme_agent")

class HomologExpansionGate:
    """Expand qualified seeds structurally; similarity remains candidate generation only."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        if not state.critique_injected:
            return False
        # Never pre-empt the model's own actions on the turn it took them.
        if has_eval_calls:
            return False
        if not self._seeds(state.run_dir):
            return False
        # The evaluation backlog used to intercept every remaining turn, so this
        # gate (placed after it) never ran: 0 expansions across a full run. Fire
        # only on the reserved cadence, so evaluation keeps most turns while
        # breadth is guaranteed a share. Targeting is unchanged: _seeds draws
        # only from already-supported, promotion-eligible candidates.
        return expansion_due(state.run_dir, config.EXPANSION_EVAL_INTERVAL)

    def _seeds(self, run_dir, *, record_selection=False):
        candidates = load_candidate_registry(run_dir).get("candidates", {})
        top = select_expansion_seeds(candidates, config.MAX_STRUCTURAL_EXPANSION_SEEDS,
                                     run_dir, method="foldseek", record_decision=record_selection)
        searched = searched_seeds(run_dir, "foldseek")
        return [(acc, candidates[acc].get("metadata", {}).get("protein_name", acc))
                for acc in top if acc not in searched]

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        # Surface the decision; the agent chooses whether and how to broaden from
        # these supported seeds, rather than the gate executing a search blindly.
        # Ranked first so the strongest, least-redundant seeds lead the list.
        seeds = self._seeds(state.run_dir, record_selection=True)[:config.EXPANSION_SEEDS_PER_STEP]
        if not seeds:
            return GateResult(intercept=False)

        candidates = load_candidate_registry(state.run_dir).get("candidates", {})

        def _describe(acc: str, name: str) -> str:
            meta = candidates.get(acc, {}).get("metadata", {}) or {}
            data = {**(meta.get("evaluation_data") or {}), **meta}
            family = data.get("interpro_family_id") or data.get("mechanism_family") or "family unrecorded"
            fit = data.get("substrate_fit")
            fit_txt = f", substrate_fit {fit}" if fit not in (None, "") else ""
            return f"  - {name} ({acc}) — {family}{fit_txt}"

        listed = "\n".join(_describe(acc, name) for acc, name in seeds)
        # Name only the expansion tools actually registered in this run, so the
        # advice never points at one filtered out of the current process.
        from enzyme_discovery_agent import tool_registry
        blurbs = [
            ("search_structural_homologs", "remote structural homologs of a seed"),
            ("search_esm_atlas_homologs", "sequence neighbours from the metagenomic ESM Atlas, which reach unannotated enzymes"),
            ("expand_candidates_esm", "embedding neighbours scored against the pool"),
        ]
        available = [f"{tool} ({why})" for tool, why in blurbs if tool in tool_registry._TOOLS]
        tools_line = ("broaden from it with " + "; or ".join(available) + ". "
                      if available else
                      "broaden from it with the available homology/embedding search tools. ")
        advisory = (
            "EXPANSION OPPORTUNITY. These candidates have cleared evaluation and are "
            "supported, so they are sound seeds for broadening the search beyond "
            "keyword/annotation retrieval. Decide, from the current evidence, whether "
            "widening the net around any of them is worthwhile now:\n"
            f"{listed}\n"
            "If a seed's family still has unexplored members or its native chemistry is a "
            "strong analogue of the target, " + tools_line +
            "These are candidate-generation only: a homolog is not reaction evidence and must be "
            "evaluated before promotion. If a family instead looks saturated or off-target, record "
            "that with set_family_focus rather than expanding it. Expanding is optional; skip it "
            "when the evidence does not warrant it."
        )
        logger.info("Surfaced structural-expansion opportunity for %d seed(s)", len(seeds))
        return GateResult(
            intercept=True,
            new_messages=[{"role": "user", "content": advisory}],
            log_message=f"Expansion opportunity surfaced: {len(seeds)} seed(s)",
        )
