"""Evidence-led post-evaluation expansion.

This gate follows literature links from evaluated family representatives.
Reaction-to-enzyme discovery is available through the reaction database tools
and recorded scientific decisions, where native chemistry can justify a small
mapping set. Broad EC membership alone does not authorize automatic expansion.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.expansion import select_expansion_seeds, searched_seeds, record_seed_search

logger = logging.getLogger("enzyme_agent")


class SystematicCoverageGate:
    """Follow literature links as promising evaluated family representatives change."""

    def __init__(self, llm_client=None):
        self._expansion_done = False
        self.llm = llm_client

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        return state.homolog_expansion_done and bool(self._families(state.run_dir))

    def _families(self, run_dir):
        candidates = load_candidate_registry(run_dir).get("candidates", {})
        ranked = select_expansion_seeds(candidates, len(candidates), run_dir)
        representatives = {}
        for acc in ranked:
            fid = candidates[acc].get("metadata", {}).get("interpro_family_id")
            if fid:
                representatives.setdefault(fid, acc)
        searched = searched_seeds(run_dir, "sequence_literature")
        return [(fid, acc) for fid, acc in list(representatives.items())[:8] if acc not in searched]

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        return self._execute_expansion(state)

    def _execute_expansion(self, state: AgentState) -> GateResult:
        """Run sequence_literature_search for evaluated families."""
        from enzyme_discovery_agent.tools.literature import sequence_literature_search
        families = self._families(state.run_dir)
        if not families:
            self._expansion_done = True
            return GateResult(intercept=False)

        logger.info("Sequence literature expansion: %d families", len(families))

        all_new_genes = []
        family_results = []

        def _search(rep_acc: str) -> dict | None:
            check = getattr(self.llm, "budget_check", None)
            reason = check() if callable(check) else None
            if isinstance(reason, str):
                return {"status": "skipped", "count": 0, "deferred_for_budget": True}
            try:
                return sequence_literature_search(accession=rep_acc, max_homologs=10)
            except Exception as e:
                logger.warning("Sequence literature search failed for %s: %s", rep_acc, e)
                return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_search, [rep for _, rep in families]))

        for (fam_id, accession), result in zip(families, results):
            if not (result or {}).get("deferred_for_budget"):
                record_seed_search(state.run_dir, "sequence_literature", accession, result or {"status": "error"})
            if result and result.get("status") == "success":
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
