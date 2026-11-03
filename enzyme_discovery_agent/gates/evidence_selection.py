"""Gate that converts sufficient literature evidence into a bounded screen set."""

from __future__ import annotations

from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates import AgentState, GateResult


class EvidenceSelectionGate:
    """Prevent endless retrieval when literature evidence is already available.

    Search tools return leads, not candidates.  Once enough evidence has been
    collected, force an explicit, small accession selection and screening step.
    This is generic: it does not assume a particular enzyme family or target.
    """

    def should_fire(self, state: AgentState, has_eval_calls: bool = False, messages: list[dict] | None = None) -> bool:
        return (
            not state.evidence_selection_prompted
            and state.candidate_count == 0
            and len(state.accumulated_papers) >= config.EVIDENCE_SELECTION_SEARCH_THRESHOLD
        )

    def execute(self, state: AgentState, messages: list[dict] | None = None) -> GateResult:
        return GateResult(
            intercept=True,
            new_messages=[
                {
                    "role": "user",
                    "content": (
                        "EVIDENCE COLLECTION CHECKPOINT: sufficient literature has been gathered "
                        "across several mechanistically distinct families. Now REGISTER candidates "
                        "COMPREHENSIVELY — do NOT downselect here. Register EVERY DISTINCT ENZYME "
                        "you have found: across all families AND the distinct members within each "
                        "family (different genes catalysing related-but-distinct reactions are "
                        "SEPARATE candidates — do not collapse them into one 'family representative'). "
                        "Resolve each distinct enzyme to a UniProt accession and call "
                        "screen_candidates. Consolidate ONLY orthologs (the SAME enzyme from "
                        "different organisms → one representative). An enzyme you found but did not "
                        "screen is a lost candidate. RELEVANCE FILTERING HAPPENS LATER, at "
                        "triage/evaluation — NOT here; registration must be inclusive so evaluation "
                        "can judge each enzyme on catalytic-mechanism feasibility. Do not register a "
                        "raw 500-row search-result page wholesale, but every enzyme you actually "
                        "resolved must be registered; do not continue generic literature or database "
                        "searches before screening."
                    ),
                },
            ],
            log_message="Evidence checkpoint: select and screen a bounded accession set",
            state_updates={"evidence_selection_prompted": True},
        )
