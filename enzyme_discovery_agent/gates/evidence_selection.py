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

    def should_fire(self, state: AgentState) -> bool:
        return (
            not state.evidence_selection_prompted
            and state.candidate_count == 0
            and len(state.accumulated_papers) >= config.EVIDENCE_SELECTION_SEARCH_THRESHOLD
            and state.lit_search_count >= config.MIN_LITERATURE_SEARCH_ROUNDS
        )

    def execute(self, state: AgentState) -> GateResult:
        return GateResult(
            intercept=True,
            new_messages=[
                {
                    "role": "user",
                    "content": (
                        "EVIDENCE COLLECTION CHECKPOINT: sufficient literature has been gathered. "
                        "You have completed several literature-search rounds and should now have evidence "
                        "across several mechanistically distinct families. "
                        "From the papers and identifier lookups already available, select at most 20 concrete "
                        "accessions with a direct target/substrate assay, a close substrate analogue, or a "
                        "distinct supported mechanism. Resolve a "
                        "specific literature-derived gene/accession if needed, then call screen_candidates "
                        "on that bounded accession list. Do not promote a search-result page, and do not "
                        "continue generic literature or database searches before screening."
                    ),
                },
            ],
            log_message="Evidence checkpoint: select and screen a bounded accession set",
            state_updates={"evidence_selection_prompted": True},
        )
