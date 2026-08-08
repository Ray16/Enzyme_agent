"""Gate 1: Auto entity extraction + deep citation search.

Fires before the first evaluate_candidate call when there are accumulated
papers that haven't been processed for entity extraction.
"""

from __future__ import annotations

import logging

from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.exploration_log import append_exploration_log

logger = logging.getLogger("enzyme_agent")


class EntityExtractionGate:
    """Extract gene names from accumulated literature before evaluation."""

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        return (
            has_eval_calls
            and not state.critique_injected
            and not state.entity_extraction_done
            and bool(state.accumulated_papers)
        )

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        from enzyme_discovery_agent.tools.literature import extract_entities, deep_literature_search

        logger.info("Auto-extracting entities from %d papers", len(state.accumulated_papers))

        entity_result = extract_entities(state.accumulated_papers)
        entity_data = entity_result.get("data", {})
        new_genes = entity_data.get("gene_names", [])
        new_ecs = entity_data.get("ec_numbers", [])
        new_families = entity_data.get("enzyme_families", [])
        new_genbank = entity_data.get("genbank_accessions", [])

        logger.info("Extracted %d genes, %d ECs, %d families, %d GenBank accessions",
                     len(new_genes), len(new_ecs), len(new_families), len(new_genbank))

        top_papers = [
            p for p in state.accumulated_papers
            if p.get("title") and p.get("doi")
        ][:3]
        deep_genes = []
        if top_papers:
            logger.info("Deep citation search on %d top papers", len(top_papers))
            for paper in top_papers:
                title = paper.get("title", "")[:80]
                try:
                    deep_result = deep_literature_search(
                        query=title,
                        max_seed_papers=3,
                        max_refs_per_seed=10,
                    )
                    deep_data = deep_result.get("data", {})
                    deep_entities = deep_data.get("extracted_entities", {})
                    dg = deep_entities.get("gene_names", [])
                    if dg:
                        deep_genes.extend(dg)
                except Exception as e:
                    logger.warning("Deep search failed: %s", e)

        all_new_genes = new_genes + deep_genes

        if not all_new_genes and not new_genbank:
            return GateResult(intercept=False)

        gene_names_str = ", ".join(
            g["name"] if isinstance(g, dict) else str(g)
            for g in all_new_genes[:30]
        )

        genbank_str = ", ".join(
            g["accession"] if isinstance(g, dict) else str(g)
            for g in new_genbank[:20]
        )

        append_exploration_log(
            state.run_dir,
            entry_type="SEARCH",
            title="Auto entity extraction + deep citation search",
            details={
                "Papers processed": len(state.accumulated_papers),
                "Deep searches": len(top_papers),
                "Gene names found": gene_names_str,
                "EC numbers": len(new_ecs),
                "Families": len(new_families),
                "GenBank accessions": genbank_str,
            },
        )

        genbank_instruction = ""
        if new_genbank:
            genbank_instruction = (
                f"\n\nGenBank/EMBL accessions found in papers:\n"
                f"{genbank_str}\n"
                "Use lookup_genbank_to_uniprot for each accession to "
                "resolve to a UniProt entry. GenBank accessions are "
                "unambiguous and avoid gene-name collisions."
            )

        new_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Auto entity extraction + deep citation search found "
                    f"{len(all_new_genes)} gene names"
                    + (f" and {len(new_genbank)} GenBank accessions" if new_genbank else "")
                    + f" from {len(state.accumulated_papers)} papers."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Entity extraction from ALL literature results + "
                    "citation network search found these gene names:\n"
                    f"{gene_names_str}\n\n"
                    "Search UniProt for any gene names that you have NOT "
                    "already searched for. Then proceed to evaluation."
                    + genbank_instruction
                ),
            },
        ]

        return GateResult(
            intercept=True,
            new_messages=new_messages,
            log_message=f"Entity extraction: {len(all_new_genes)} genes found",
            state_updates={"entity_extraction_done": True},
        )
