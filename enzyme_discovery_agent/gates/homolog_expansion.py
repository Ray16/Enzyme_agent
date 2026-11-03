"""Gate 3: Foldseek expansion for a small set of qualified evaluated seeds."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.candidate_registry import register_structural_homologs
from enzyme_discovery_agent.models.expansion import (
    select_expansion_seeds, searched_seeds, record_seed_search,
)
from enzyme_discovery_agent.tools.structural_homolog_search import search_structural_homologs

logger = logging.getLogger("enzyme_agent")

# Each search is a remote queue wait on the shared public Foldseek server (~40 s);
# a small pool overlaps them without hammering it.
_FOLDSEEK_WORKERS = 3

class HomologExpansionGate:
    """Expand qualified seeds structurally; similarity remains candidate generation only."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        if not state.critique_injected or has_eval_calls:
            return False
        # ECTriageGate may auto-evaluate candidates and replace the triggering
        # tool call with a compact summary, so message-history inspection is
        # not reliable.  The gate state is the durable completion signal.
        return bool(self._seeds(state.run_dir)) or not state.homolog_expansion_done

    def _seeds(self, run_dir, *, record_selection=False):
        candidates = load_candidate_registry(run_dir).get("candidates", {})
        top = select_expansion_seeds(candidates, config.MAX_STRUCTURAL_EXPANSION_SEEDS,
                                     run_dir, method="foldseek", record_decision=record_selection)
        searched = searched_seeds(run_dir, "foldseek")
        return [(acc, candidates[acc].get("metadata", {}).get("protein_name", acc))
                for acc in top if acc not in searched]

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        # Rank first, then remove searched seeds. Lower-ranked candidates do not
        # consume searches merely because the current leaders were already tried.
        seeds = self._seeds(state.run_dir, record_selection=True)[:config.EXPANSION_SEEDS_PER_STEP]
        if not seeds:
            return GateResult(
                intercept=False,
                state_updates={"homolog_expansion_done": True},
            )

        def _search(seed: tuple[str, str]) -> dict:
            check = getattr(self.llm, "budget_check", None)
            reason = check() if callable(check) else None
            if isinstance(reason, str):
                return {"status": "skipped", "count": 0, "summary": reason,
                        "deferred_for_budget": True}
            try:
                return search_structural_homologs(seed[0], max_hits=10, min_tm_score=0.5,
                                                  run_dir=str(state.run_dir))
            except Exception as exc:
                logger.warning("Structural expansion failed for %s: %s", seed[0], exc)
                return {"status": "error", "count": 0, "summary": str(exc)}

        with ThreadPoolExecutor(max_workers=_FOLDSEEK_WORKERS) as pool:
            search_results = list(pool.map(_search, seeds))

        # Registry writes stay serial and in seed order.
        results = []
        known = set(load_candidate_registry(state.run_dir).get("candidates", {}))
        for (accession, name), result in zip(seeds, search_results):
            if result.get("status") in {"success", "partial"}:
                returned = register_structural_homologs(state.run_dir, result)
                result["new_accessions"] = [acc for acc in returned if acc not in known]
                known.update(returned)
            if not result.get("deferred_for_budget"):
                record_seed_search(state.run_dir, "foldseek", accession, result)
            results.append((accession, name, result.get("count", 0), result.get("summary", "")))
        lines = "\n".join(f"  - {name} ({acc}): {count} Foldseek homologs — {summary}" for acc, name, count, summary in results)
        logger.info("Structural homolog expansion completed for %d seeds", len(seeds))

        new_messages = [
            {
                "role": "user",
                "content": (
                    "Foldseek structural-homolog expansion completed for qualified evaluated seeds. "
                    "These are candidate-generation results only; do not treat structural similarity as "
                    "reaction evidence. Evaluate any homolog before promotion:\n"
                    f"{lines}"
                ),
            },
        ]

        return GateResult(
            intercept=True,
            new_messages=new_messages,
            log_message=f"Structural homolog expansion: {len(seeds)} seeds",
            state_updates={"homolog_expansion_done": True},
        )
