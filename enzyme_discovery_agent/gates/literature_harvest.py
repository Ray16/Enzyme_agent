"""Early literature-harvest gate.

Mines enzymes from the primary literature's FULL TEXT and registers them as
candidates BEFORE the agent expands the candidate space by database search, so
that literature LEADS discovery.

Why it exists: abstract-level search + regex entity extraction see only titles
and abstracts, so an enzyme named only in a paper's body or reference list is
invisible to them.  The full-text harvest — fetch the mechanism-relevant papers,
LLM-extract the characterised + related enzymes with their organism, resolve
each to a UniProt accession, and register it — is the only route to those
experimentally-tested enzymes.  Running it here, once enough literature has been
gathered and before candidate registration/expansion, makes the grounded, tested
enzymes the SEEDS that the subsequent database search expands around, instead of
seeding expansion from names the model happened to recall.
"""

from __future__ import annotations

import json
import logging

from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates import AgentState, GateResult

logger = logging.getLogger("enzyme_agent")


class LiteratureHarvestGate:
    """Fire once, after the literature fan-out, to register full-text-mined
    enzymes as the discovery seeds — before database expansion."""

    def __init__(self, llm_client=None, log_fn=None):
        self.llm = llm_client
        self.log = log_fn or (lambda *a, **k: None)

    def should_fire(self, state: AgentState, has_eval_calls: bool = False, messages: list[dict] | None = None) -> bool:
        # Fire once, after enough literature is gathered, before candidate
        # registration/expansion.  Needs an LLM (full-text extraction).
        return (
            not state.literature_harvest_done
            and self.llm is not None
            and len(state.accumulated_papers) >= config.EVIDENCE_SELECTION_SEARCH_THRESHOLD
            and state.lit_search_count >= config.MIN_LITERATURE_SEARCH_ROUNDS
        )

    def execute(self, state: AgentState, messages: list[dict] | None = None) -> GateResult:
        from enzyme_discovery_agent.tools.fulltext import (
            mine_and_register_fulltext_accessions,
            seed_mechanism_literature,
        )

        query = ""
        substrate = ""
        try:
            tc = json.loads((state.run_dir / "target_context.json").read_text())
            query = tc.get("query", "") or ""
            substrate = tc.get("substrate_smiles", "") or ""
        except Exception:  # noqa: BLE001 - query is best-effort context
            pass

        # Ensure the MECHANISM primary literature is in the pool before mining:
        # run the derived families' own search phrases, so a product-anchored
        # keyword search cannot leave the harvest with only application/materials
        # papers and none of the characterised enzymes.
        try:
            seed_mechanism_literature(
                state.run_dir, query, substrate,
                llm_client=self.llm, log_fn=self.log,
            )
        except Exception as exc:  # noqa: BLE001 - seeding must not block the harvest
            logger.warning("Mechanism-literature seeding failed: %s", exc)

        self.log("  Literature harvest: reading full text of the primary "
                 "literature to seed discovery with tested enzymes...")
        registered: list[str] = []
        try:
            registered = mine_and_register_fulltext_accessions(
                state.run_dir, query, log_fn=self.log, llm_client=self.llm,
            )
        except Exception as exc:  # never let the harvest block the loop
            logger.warning("Literature harvest failed: %s", exc)

        guidance = (
            f"LITERATURE HARVEST read the full text of the gathered papers and "
            f"registered {len(registered)} candidate(s) named or deposited in "
            "them — grounded, often experimentally-characterised enzymes. These "
            "are your discovery SEEDS. Before screening, EXPAND the candidate "
            "space by database search AROUND each seed: find its homologs and "
            "same-family members (UniProt family/domain, structural neighbours). "
            "Keep the distinct seeds distinct — do not collapse them into one "
            "family or converge on a single mechanism."
        )
        return GateResult(
            intercept=True,
            new_messages=[
                {"role": "assistant",
                 "content": f"Ran a full-text literature harvest: "
                            f"{len(registered)} candidate(s) registered as seeds."},
                {"role": "user", "content": guidance},
            ],
            log_message=(f"Literature harvest: {len(registered)} candidate(s) "
                         "registered from full text before database expansion"),
            state_updates={"literature_harvest_done": True},
        )
