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
import hashlib
import time

from enzyme_discovery_agent import config
from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.state_store import atomic_write_json
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")


class LiteratureHarvestGate:
    """Harvest initial literature and revisit genuinely new paper batches."""

    def __init__(self, llm_client=None, log_fn=None):
        self.llm = llm_client
        self.log = log_fn or (lambda *a, **k: None)

    def should_fire(self, state: AgentState, has_eval_calls: bool = False, messages: list[dict] | None = None) -> bool:
        if self.llm is None or not state.accumulated_papers:
            return False
        try:
            failure = json.loads((state.run_dir / "agent_state/literature_harvest_failure.json").read_text())
            if failure.get("batch_signature") == self._batch_signature(state) and (
                    failure.get("attempts", 0) >= 3 or failure.get("retry_after", 0) > time.time()):
                return False
        except (OSError, ValueError, TypeError):
            pass
        if not state.literature_harvest_done:
            return True
        try:
            processed = json.loads((state.run_dir / "agent_state" / "literature_harvest_batch.json").read_text())
        except FileNotFoundError:
            # Older checkpoints never recorded the batch. One cache-aware
            # harvest establishes it and catches papers found after the first.
            return True
        from enzyme_discovery_agent.models.literature_reading import pending_reading
        return bool(self._paper_keys(state) - set(processed.get("paper_keys", []))
                    or pending_reading(state.run_dir, state.accumulated_papers))

    @staticmethod
    def _paper_keys(state):
        from enzyme_discovery_agent.tools.literature import _paper_key
        return {str(key) for paper in state.accumulated_papers
                if (key := _paper_key(paper)) is not None}

    @classmethod
    def _batch_signature(cls, state):
        return hashlib.sha256(json.dumps(sorted(cls._paper_keys(state))).encode()).hexdigest()

    def execute(self, state: AgentState, messages: list[dict] | None = None) -> GateResult:
        from enzyme_discovery_agent.tools.fulltext import (
            mine_and_register_fulltext_accessions,
        )

        query = ""
        substrate = ""
        try:
            tc = json.loads((run_file(state.run_dir, 'target_context.json')).read_text())
            query = tc.get("query", "") or ""
            substrate = tc.get("substrate_smiles", "") or ""
        except Exception:  # noqa: BLE001 - query is best-effort context
            pass

        # Ensure the MECHANISM primary literature is in the pool before mining:
        # run the derived families' own search phrases, so a product-anchored
        # keyword search cannot leave the harvest with only application/materials
        # papers and none of the characterised enzymes.
        # Search strategy is now rolling controller work. Harvest only the pool
        # already retrieved, rather than re-running the same eight phrases on
        # every read batch and overwriting their earlier results.

        self.log("  Literature harvest: reading full text of the primary "
                 "literature to seed discovery with tested enzymes...")
        registered: list[str] = []
        try:
            registered = mine_and_register_fulltext_accessions(
                state.run_dir, query, log_fn=self.log, llm_client=self.llm,
            )
        except Exception as exc:  # never let the harvest block the loop
            logger.warning("Literature harvest failed: %s", exc)
            path = state.run_dir / "agent_state/literature_harvest_failure.json"
            signature = self._batch_signature(state)
            try:
                prior = json.loads(path.read_text())
            except (OSError, ValueError):
                prior = {}
            attempts = prior.get("attempts", 0) + 1 if prior.get("batch_signature") == signature else 1
            atomic_write_json(path, {"batch_signature": signature, "attempts": attempts,
                "status": "degraded" if attempts >= 3 else "retry_pending",
                "retry_after": time.time() + min(300, 30 * 2 ** (attempts - 1)),
                "error_type": type(exc).__name__, "batch_processed": False})
            return GateResult(intercept=False,
                              log_message="Literature harvest unavailable; batch remains unprocessed.")

        if getattr(self.llm, "budget_exhausted", False):
            return GateResult(intercept=False)

        failure_path = state.run_dir / "agent_state/literature_harvest_failure.json"
        if failure_path.exists():
            atomic_write_json(failure_path, {"batch_signature": self._batch_signature(state),
                "attempts": 0, "status": "recovered", "retry_after": 0, "batch_processed": True})

        atomic_write_json(state.run_dir / "agent_state" / "literature_harvest_batch.json", {
            "paper_keys": sorted(self._paper_keys(state)),
        })

        try:
            stats = json.loads((state.run_dir / "agent_state" / "literature_harvest_step.json").read_text())
        except (OSError, ValueError):
            stats = {}
        progress = (f"{stats.get('chunks_processed', 0)} text chunks processed; "
                    f"{stats.get('new_candidates', len(registered))} new candidates, "
                    f"{stats.get('existing_candidates_refreshed', 0)} existing candidates refreshed, "
                    f"{stats.get('unresolved_leads', 0)} unresolved names")
        if not any(stats.get(key, 0) for key in ("chunks_processed", "new_downloads", "new_candidates", "existing_candidates_refreshed")):
            return GateResult(intercept=False, log_message="Literature batch added no new evidence.",
                              state_updates={"literature_harvest_done": True})
        guidance = (
            f"LITERATURE HARVEST: {progress}. Names in a paper are discovery leads; "
            "identity resolution and activity evidence require separate validation. Screen "
            "and evaluate these discovery leads first. Expand only the most "
            "promising evaluated candidates with same-family or structural searches. "
            "Keep the distinct seeds distinct — do not collapse them into one "
            "family or converge on a single mechanism."
        )
        if stats.get('reading_outcomes'):
            guidance += ('\nReading objectives and passage-scoped outcomes (verify source entailment before '
                         'using them as candidate evidence):\n' + json.dumps(stats['reading_outcomes']))
        return GateResult(
            intercept=True,
            new_messages=[
                {"role": "assistant",
                 "content": f"Ran a bounded full-text literature harvest: {progress}."},
                {"role": "user", "content": guidance},
            ],
            log_message=f"Literature harvest: {progress}",
            state_updates={"literature_harvest_done": True},
        )
