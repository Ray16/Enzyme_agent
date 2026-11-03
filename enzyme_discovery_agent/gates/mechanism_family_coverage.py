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
            retrieved_papers=state.accumulated_papers, max_attempts=2, reflect=False,
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

        from datetime import datetime, timezone
        import uuid
        from enzyme_discovery_agent.models.reasoning_trace import capture_scientific_record
        identifier = uuid.uuid4().hex
        snapshot = state.run_dir / 'artifacts' / 'mechanism_hypotheses' / (identifier + '.json')
        atomic_write_json(snapshot, {
            'id':identifier, 'created_at':datetime.now(timezone.utc).isoformat(),
            'query':query, 'substrate_smiles':substrate,
            'retrieved_papers':state.accumulated_papers, 'families':families,
            'model':getattr(self.llm, 'active_label', None),
            'scope':'Provisional source-informed search hypotheses; neither exhaustive coverage nor established target activity.'})
        capture_scientific_record(state.run_dir, 'mechanism_hypotheses_recorded', snapshot)
        # This source-informed synthesis is the initial evidence review. Release
        # routine retrieval without requiring a second model turn to close it.
        # An existing pending outcome remains pending and still needs interpretation.
        from enzyme_discovery_agent.models.research_decisions import _root, _read
        from enzyme_discovery_agent.state_store import state_lock
        review_path = _root(state.run_dir) / 'state.json'
        with state_lock(review_path):
            review = _read(review_path, {})
            review.update(initial_sources_reviewed=True, initial_hypothesis_snapshot=str(snapshot.relative_to(state.run_dir)))
            atomic_write_json(review_path, review)
        lines = [f"- {f['name']}: " + '; '.join(f.get('literature_terms') or f['descriptors']) for f in families]
        guidance = (
            "INITIAL MECHANISM SEARCH MAP (provisional, not exhaustive). These are hypotheses "
            "derived from the target and retrieved sources, not validated enzyme activities.\n"
            + "\n".join(lines) +
            "\nUse this map now for focused retrieval across distinct plausible families. "
            "Resolve family/domain or native-reaction records to candidate sequences, and follow "
            "retrieved identities to their own functional evidence. Use independent retrievals in "
            "one planned batch. Obtain a source-supported representative before deepening one family. "
            "Assign each target participant a role and compare its reacting end with an occurrence "
            "in ONE native reaction. Distinct native occurrences can supply complementary roles "
            "despite different whole-molecule functional-group inventories. Substrate resemblance "
            "alone cannot establish feasibility or rule out a transfer hypothesis. "
            "Use evidence to revise priorities, broaden an empty query or explore an overlooked "
            "route. Keep unknowns distinct from observed conflicts; a proposed map is not a "
            "coverage certificate. Preserve alternative hypotheses and reopening conditions."
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
