"""Gate 3: Foldseek expansion for a small set of qualified evaluated seeds."""

from __future__ import annotations

import logging

from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.candidate_registry import register_structural_homologs
from enzyme_discovery_agent.tools.structural_homolog_search import search_structural_homologs

logger = logging.getLogger("enzyme_agent")

_CRITICAL_WARNINGS = (
    "BOND TYPE MISMATCH", "SEVERE SIZE MISMATCH",
    "REACTION DIRECTION MISMATCH",
)


class HomologExpansionGate:
    """Expand qualified seeds structurally; similarity remains candidate generation only."""

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        if state.homolog_expansion_done or not state.critique_injected or has_eval_calls:
            return False
        # ECTriageGate may auto-evaluate candidates and replace the triggering
        # tool call with a compact summary, so message-history inspection is
        # not reliable.  The gate state is the durable completion signal.
        return True

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        registry = load_candidate_registry(state.run_dir)

        qualified: list[tuple[str, str]] = []
        for acc, info in registry.get("candidates", {}).items():
            statuses = [s["status"] for s in info.get("statuses", [])]
            if "evaluated" not in statuses:
                continue
            meta = info.get("metadata", {})
            weaknesses = meta.get("weaknesses", [])
            has_critical = any(
                any(cw in w for cw in _CRITICAL_WARNINGS)
                for w in weaknesses
            )
            if not has_critical:
                qualified.append((acc, meta.get("protein_name", acc)))

        # A bounded cap keeps Foldseek a within-family expansion, not a broad
        # search, while still seeding from more than the top couple of families.
        seeds = qualified[:config.MAX_STRUCTURAL_EXPANSION_SEEDS]
        if not seeds:
            return GateResult(
                intercept=False,
                state_updates={"homolog_expansion_done": True},
            )

        results = []
        for accession, name in seeds:
            result = search_structural_homologs(accession, max_hits=10, min_tm_score=0.5)
            if result.get("status") in {"success", "partial"}:
                register_structural_homologs(state.run_dir, result)
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
