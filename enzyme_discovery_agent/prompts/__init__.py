"""System prompt construction for the enzyme discovery agent."""

from __future__ import annotations

from enzyme_discovery_agent.prompts.report_prompts import (
    EXPLORATIVE_RERANK_PROMPT,
)
from enzyme_discovery_agent.prompts.sections import (
    ANTI_PATTERNS,
    CANDIDATE_EVALUATION_GUIDANCE,
    DEFAULT_WORKFLOW,
    LITERATURE_GUIDANCE,
    MODELSEED_GUIDANCE,
    NOTEBOOK_GUIDANCE,
    PRINCIPLES,
    REPORT_FORMAT,
    ROLE,
    SEARCH_GUIDANCE,
    SEQUENCE_LITERATURE_GUIDANCE,
    SMILES_RULES,
    STRUCTURAL_EVIDENCE_GUIDANCE,
    STRUCTURAL_GUIDANCE,
)

__all__ = [
    "build_system_prompt",
    "build_state_header",
    "build_reflection_prompt",
    "EXPLORATIVE_RERANK_PROMPT",
]


def build_system_prompt(
    available_tools: list[str] | None = None,
    capabilities: dict[str, bool] | None = None,
) -> str:
    """Build the full system prompt for the enzyme discovery agent."""
    caps = capabilities or {}
    sections = [
        ROLE,
        PRINCIPLES,
        DEFAULT_WORKFLOW,
        ANTI_PATTERNS,
        SEARCH_GUIDANCE,
        LITERATURE_GUIDANCE,
        MODELSEED_GUIDANCE,
        SEQUENCE_LITERATURE_GUIDANCE,
        CANDIDATE_EVALUATION_GUIDANCE,
        NOTEBOOK_GUIDANCE,
        SMILES_RULES,
        REPORT_FORMAT,
        STRUCTURAL_EVIDENCE_GUIDANCE,
    ]
    if caps.get("pocket_analysis"):
        sections.append(STRUCTURAL_GUIDANCE)
    return "\n\n".join(sections)


def build_state_header(
    iteration: int,
    max_iterations: int,
    candidate_count: int,
    searches_done: list[str],
    coverage_gaps: list[str] | None = None,
    phase: str = "discovery",
) -> str:
    """Build a compact state header injected at each iteration."""
    gaps = coverage_gaps or []
    searches_str = ", ".join(searches_done[-8:]) if searches_done else "none"
    gaps_str = ", ".join(gaps[:5]) if gaps else "none identified"
    return (
        f"[STATE: iter={iteration}/{max_iterations}, "
        f"candidates={candidate_count}, "
        f"recent_searches=[{searches_str}], "
        f"coverage_gaps=[{gaps_str}], "
        f"phase={phase}]"
    )


def build_reflection_prompt(iteration: int, candidate_count: int = 0) -> str:
    """Build the reflection prompt injected every REFLECTION_INTERVAL iterations."""
    coverage_warning = ""
    if candidate_count < 3:
        coverage_warning = (
            f"\nWARNING: You have only {candidate_count} distinct candidate families. "
            "Find a small number of mechanistically distinct representatives before evaluation.\n"
        )
    return (
        f"REFLECTION CHECKPOINT (iteration {iteration}):\n"
        "You MUST now reflect before continuing. Do the following:\n"
        "1. Call get_notebook() to review your accumulated analysis.\n"
        "2. List the direct-evidence families and mechanistically distinct analogue families you identified "
        "in reaction_analysis and enzyme_landscape.\n"
        "3. Check whether direct evidence is covered and whether any missing family would add a distinct mechanism. "
        "Do not expand a near-duplicate family.\n"
        "4. Assess search efficiency: if recent searches mostly return papers you "
        "already have (low novelty) or 0 results, do NOT keep re-searching the same "
        "way — switch source, reformulate with shorter 3-5 word queries, run "
        "consolidate_literature to work from one deduplicated pool, or chase "
        "citations (deep_literature_search) instead of more keyword search.\n"
        "5. Think about what other biological contexts use the same bond type that "
        "you haven't explored yet (cell wall biosynthesis, biopolymers, storage "
        "compounds, post-translational modifications, secondary metabolites).\n"
        "6. Take the next evidence-gathering action directly; only write a notebook note "
        "when it preserves concise, evidence-backed conclusions.\n"
        f"{coverage_warning}"
        "If direct evidence and a small diverse set of mechanisms are covered, proceed to evaluation and structural expansion."
    )
