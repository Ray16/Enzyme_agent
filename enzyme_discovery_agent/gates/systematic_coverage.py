"""Evidence-led post-evaluation expansion.

EC annotations are useful compatibility metadata, but are too coarse and
incomplete to generate a candidate pool for challenging reactions.  This gate
therefore never performs EC-to-UniProt expansion.  It only follows literature
links from already evaluated, evidence-backed families.

The EC/ModelSEED enumeration path that used to live here was removed: it was
unreachable, and re-enabling it would contradict the evidence-first retrieval
contract enforced by the system prompt and the discovery coverage gate.  The
bond-type rejection filter it used survives as
``chemistry_logic.filter_ec_prefixes_by_bond_type``.
"""

from __future__ import annotations

import logging

from enzyme_discovery_agent.gates import AgentState, GateResult

logger = logging.getLogger("enzyme_agent")


class SystematicCoverageGate:
    """Follow literature links out of evaluated families, once per run."""

    def __init__(self):
        self._expansion_done = False

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        if self._expansion_done:
            return False
        return state.homolog_expansion_done

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        return self._execute_expansion(state)

    def _execute_expansion(self, state: AgentState) -> GateResult:
        """Run sequence_literature_search for evaluated families."""
        from enzyme_discovery_agent.tools.literature import sequence_literature_search
        from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry

        registry = load_candidate_registry(state.run_dir)
        candidates = registry.get("candidates", {})

        family_reps: dict[str, str] = {}
        for acc, info in candidates.items():
            statuses = [s["status"] for s in info.get("statuses", [])]
            if "evaluated" not in statuses or (statuses and statuses[-1] == "rejected"):
                continue
            meta = info.get("metadata", {})
            validation = meta.get("claim_validation")
            if validation and validation.get("promotion_eligible") is not True:
                continue
            fam_id = meta.get("interpro_family_id", "")
            if fam_id and fam_id not in family_reps:
                family_reps[fam_id] = acc

        if not family_reps:
            self._expansion_done = True
            return GateResult(intercept=False)

        logger.info("Sequence literature expansion: %d families", len(family_reps))

        all_new_genes = []
        family_results = []

        for fam_id, rep_acc in list(family_reps.items())[:8]:
            try:
                result = sequence_literature_search(
                    accession=rep_acc,
                    max_homologs=10,
                )
                if result.get("status") == "success":
                    data = result.get("data", {})
                    entities = data.get("extracted_entities", {})
                    genes = entities.get("gene_names", [])
                    fam_name = data.get("family_name", fam_id)
                    n_pubs = result.get("count", 0)
                    all_new_genes.extend(genes)
                    family_results.append(
                        f"  - {fam_name}: {n_pubs} papers, genes: "
                        f"{', '.join(g['name'] for g in genes[:5])}"
                    )
            except Exception as e:
                logger.warning("Sequence literature search failed for %s: %s", rep_acc, e)

        self._expansion_done = True

        if not all_new_genes:
            return GateResult(intercept=False)

        seen = set()
        unique_genes = []
        for g in all_new_genes:
            name = g["name"] if isinstance(g, dict) else str(g)
            if name not in seen:
                seen.add(name)
                unique_genes.append(name)

        results_text = "\n".join(family_results)
        genes_text = ", ".join(unique_genes[:30])

        msg_content = (
            f"SEQUENCE-BASED LITERATURE EXPANSION:\n"
            f"Searched for homologs across {len(family_results)} evaluated families:\n"
            f"{results_text}\n\n"
            f"New gene names discovered: {genes_text}\n\n"
            f"Search UniProt for any of these gene names that you have NOT "
            f"already searched for, especially from families underrepresented "
            f"in your current candidate list."
        )

        return GateResult(
            intercept=True,
            new_messages=[
                {"role": "user", "content": msg_content},
            ],
            log_message=f"Sequence literature expansion: {len(unique_genes)} new genes from {len(family_results)} families",
        )
