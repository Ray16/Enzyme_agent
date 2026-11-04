"""Generate ranked_candidates.csv from evaluation data."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

from enzyme_discovery_agent.tools.mechanism_family_assignment import (
    assign_mechanism_family,
    load_mechanism_families,
)
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")


def refresh_ranked_candidates(run_dir):
    """Refresh the decision list from saved evidence without provider calls."""
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry,discovery_status
    from enzyme_discovery_agent.models.expansion import seed_evidence
    from enzyme_discovery_agent.models.evaluations import current_evaluations
    from enzyme_discovery_agent.output.report_generator import _deterministic_mechanistic_ranking,_family_relevance_partition,_family_rank_map,_collapse_orthologs
    from enzyme_discovery_agent.state_store import atomic_write_json
    root=Path(run_dir)
    from enzyme_discovery_agent.state_store import state_lock
    with state_lock(root/'agent_state/live_ranking'):
        registry=load_candidate_registry(root,strict=True)['candidates']
        from enzyme_discovery_agent.models.evaluations import registered_hypothesis
        evaluations=current_evaluations(root,[{'data':seed_evidence(acc,entry)} for acc,entry in registry.items()
                                            if discovery_status(entry) in {'evaluated','ranked'} or
                                            registered_hypothesis(root,acc,entry.get('metadata') or {})])
        # Fill missing comparisons from source-resolved occurrences in one native
        # reaction. Reporting does not initiate compound lookups or pool activities.
        from enzyme_discovery_agent.models.native_substrate_fit import add_property_comparisons
        add_property_comparisons(root,evaluations)
        from enzyme_discovery_agent.models.target_review import admitted_evaluations
        rankable=admitted_evaluations(root,evaluations)
        stopped,supported,_=_family_relevance_partition(root,evaluations)
        ranking=_deterministic_mechanistic_ranking([ev for ev in rankable if ev['data']['accession'] not in stopped],{},
            family_rank_by_acc=_family_rank_map(stopped,supported))
        ranking=_collapse_orthologs(ranking,{ev['data']['accession']:ev for ev in rankable})
        atomic_write_json(run_file(root,'pass1_ranking.json'),ranking)
        generate_candidates_csv(root,evaluations,ranking_entries=ranking)
        return ranking


def _load_ranking(run_dir: Path) -> dict[str, dict]:
    """Map each representative accession to its final ranking metadata.

    Ortholog alternatives remain nested under their representative in
    ``pass1_ranking.json`` for provenance, but are intentionally omitted here:
    the final CSV is a decision list, not an expanded ortholog inventory.
    """
    ranking_path = run_file(run_dir, 'pass1_ranking.json')
    if not ranking_path.exists():
        return {}
    try:
        entries = json.loads(ranking_path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Could not read pass1_ranking.json: %s", exc)
        return {}

    lookup: dict[str, dict] = {}
    for entry in entries or []:
        acc = entry.get("accession")
        if not acc:
            continue
        # Retroactively stop-fold-excluded members are kept in pass1_ranking.json
        # for provenance but omitted from the CSV decision list: a fold the agent
        # stopped (after these were already evaluated) is off-target, so it does not
        # belong in the actionable shortlist an experimentalist reads.
        if entry.get("excluded_stopped_fold"):
            continue
        info = {
            "rank": entry.get("rank"),
            "tier": entry.get("tier", ""),
            "tier_label": entry.get("tier_label", ""),
            "substrate_fit": entry.get("substrate_fit", ""),
            "evidence_level": entry.get("evidence_level", ""),
            "family_relevance": entry.get("family_relevance", ""),
            "direction": entry.get("direction", ""),
            "rationale":entry.get('rationale',''),
        }
        lookup[acc] = info
    return lookup


def generate_candidates_csv(run_dir: Path, all_evals: list[dict], log_fn=None, *, ranking_entries=None) -> None:
    """Generate ranked_candidates.csv: evaluated candidates ordered by rank."""
    from enzyme_discovery_agent.models.evaluations import current_evaluations
    from enzyme_discovery_agent.output.research_portfolio import write_research_portfolio
    write_research_portfolio(run_dir)
    all_evals = current_evaluations(run_dir, all_evals or [])
    # Drop failed evaluations (data is None): not candidates, and the row
    # builder below assumes dict data.
    all_evals = [e for e in (all_evals or [])
                 if isinstance(e, dict) and isinstance(e.get("data", e), dict)]
    from enzyme_discovery_agent.models.target_review import admitted_evaluations
    accepted=admitted_evaluations(run_dir,all_evals)
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
    current_registry=load_candidate_registry(run_dir,strict=True)['candidates']
    accepted_ids={ev.get('data',ev).get('accession') for ev in accepted}
    queue=io.StringIO();work=csv.DictWriter(queue,fieldnames=['accession','protein_name','native_reaction','assessment_status']);work.writeheader()
    for ev in all_evals:
        data=ev.get('data',ev)
        if data.get('accession') not in accepted_ids:
            work.writerow({'accession':data.get('accession'),'protein_name':data.get('protein_name'),
                'native_reaction':' | '.join(data.get('catalytic_activities') or []),'assessment_status':'Not admitted to target shortlist; see target review or obtain one.'})
    from enzyme_discovery_agent.state_store import atomic_write_text
    atomic_write_text(run_file(run_dir,'candidate_assessment_queue.csv'),queue.getvalue())
    all_evals=accepted

    ranking = _load_ranking(run_dir) if ranking_entries is None else {row["accession"]:row for row in ranking_entries}
    mech_families = load_mechanism_families(run_dir)
    # A persisted ranking is authoritative and already contains one selected
    # representative per ortholog group. Do not expand its nested alternatives
    # back into separate rows.
    all_evals = [ev for ev in all_evals if ev.get("data", ev).get("accession", "") in ranking]

    def sort_key(ev):
        acc = ev.get("data", ev).get("accession", "")
        info = ranking.get(acc)
        if not info or info.get("rank") is None:
            return (float("inf"), acc)
        return (info["rank"], acc)

    all_evals = sorted(all_evals, key=sort_key)

    # De-duplicate by accession: an accession evaluated across multiple iterations
    # produces multiple evaluation artifacts, and the ranking is one row per
    # representative accession -- so without this the same candidate lands on
    # several identical CSV rows.  Keep the first occurrence (best rank after the
    # sort above); the decision list is one row per candidate.
    seen: set[str] = set()
    deduped = []
    for ev in all_evals:
        acc = ev.get("data", ev).get("accession", "")
        if acc and acc in seen:
            continue
        if acc:
            seen.add(acc)
        deduped.append(ev)
    all_evals = deduped

    csv_path = run_dir / "ranked_candidates.csv"
    fieldnames = [
        "rank", "tier",
        "substrate_fit", "evidence_level", "family_relevance", "direction",
        "gene", "accession", "uniprot_url", "protein_name", "organism",
        "ec_numbers", "interpro_family", "mechanism_family", "is_swissprot", "sequence_length",
        "pdb_ids", "has_alphafold", "alphafold_url",
        "characterization_papers", "engineering_papers", "n_mutations",
        "tanimoto_score", "native_substrates", "native_substrate_smiles",
        "catalytic_activities", "function_comments",
        "strengths", "weaknesses", "requires_polymer_substrate",
        "tier_label", "priority_rationale",
        "native_annotation_status", "ranking_basis", "native_activity_evidence",
        "native_activity_sources", "target_assessment_decision", "target_activity_evidence",
        "substrate_comparison_complete", "substrate_comparison_required", "substrate_comparison_resolved",
        "substrate_partner_fits", "substrate_fit_scope",
        "target_role_assessments", "substrate_role_mapping_status",
    ]

    try:
        with io.StringIO(newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ev in all_evals:
                d = ev.get("data", ev)
                acc = d.get("accession", "")
                from enzyme_discovery_agent.models.native_annotation import native_annotation_for_report
                annotation = native_annotation_for_report(run_dir, acc, d)
                assessment = ((current_registry.get(acc) or {}).get('metadata') or {}).get('target_assessment') or d.get('target_assessment') or {}
                genes = d.get("gene_names", [])
                ec_list = d.get("ec_numbers", [])
                lit = d.get("literature", {})
                struct = d.get("structural_data", {})
                sub_sim = d.get("substrate_similarity", {})
                from enzyme_discovery_agent.substrate_fit import comparison_coverage
                from enzyme_discovery_agent.chemistry_logic import whole_package_pair_fit
                coverage=comparison_coverage(sub_sim)
                mut = d.get("mutagenesis", {})
                pdb_ids = struct.get("pdb_ids", [])
                has_af = struct.get("has_alphafold", False)
                cat_acts = d.get("catalytic_activities", [])
                func_comments = d.get("function_comments", [])

                native_subs = d.get("native_substrates", [])
                native_names = "; ".join(
                    ns["name"] for ns in native_subs if ns.get("name")
                )
                native_smiles = "; ".join(
                    ns["smiles"] for ns in native_subs if ns.get("smiles")
                )

                rank_info = ranking.get(acc, {})

                writer.writerow({
                    "rank": rank_info.get("rank", ""),
                    "tier": rank_info.get("tier", ""),
                    "substrate_fit": rank_info.get("substrate_fit", ""),
                    "evidence_level": rank_info.get("evidence_level", ""),
                    "family_relevance": rank_info.get("family_relevance", ""),
                    "direction": rank_info.get("direction", ""),
                    "accession": acc,
                    "uniprot_url": f"https://www.uniprot.org/uniprot/{acc}" if acc else "",
                    "protein_name": d.get("protein_name", ""),
                    "gene": genes[0] if genes else "",
                    "organism": d.get("organism", ""),
                    "ec_numbers": "; ".join(ec_list),
                    "interpro_family": d.get("interpro_family", ""),
                    "mechanism_family": (
                        (assign_mechanism_family(d, mech_families) or {}).get("name", "")
                    ),
                    "is_swissprot": d.get("is_swissprot", False),
                    "sequence_length": d.get("sequence_length", ""),
                    "pdb_ids": "; ".join(pdb_ids),
                    "has_alphafold": has_af,
                    "alphafold_url": (
                        f"https://alphafold.ebi.ac.uk/entry/{acc}"
                        if has_af and acc else ""
                    ),
                    "characterization_papers": lit.get("characterization_papers", 0),
                    "engineering_papers": lit.get("engineering_papers", 0),
                    "n_mutations": mut.get("n_mutations", 0),
                    "tanimoto_score": sub_sim.get("tanimoto_score", ""),
                    "native_substrates": native_names,
                    "native_substrate_smiles": native_smiles,
                    "catalytic_activities": " | ".join(cat_acts),
                    "function_comments": " | ".join(func_comments),
                    "strengths": "; ".join(d.get("strengths", [])),
                    "weaknesses": "; ".join(d.get("weaknesses", [])),
                    "requires_polymer_substrate": d.get("requires_polymer_substrate", False),
                    "tier_label":rank_info.get("tier_label", ""),
                    "priority_rationale":rank_info.get("rationale", ""),
                    "native_annotation_status":"annotated_reaction" if cat_acts else "native_reaction_unknown",
                    "ranking_basis":"target_assessed_hypothesis" if assessment else "source_cited_hypothesis" if d.get('priority_assessment') else "provisional_native_analogue",
                    "native_activity_evidence":annotation['native_activity_evidence'],
                    "native_activity_sources":"; ".join(sorted({
                        str(item.get('source',''))+':'+str(item.get('id',''))
                        for reaction in annotation['reactions'] for item in reaction['evidences']
                        if item.get('source') and item.get('id')})),
                    "target_assessment_decision":assessment.get('decision',''),
                    "target_activity_evidence":assessment.get('evidence_level','unresolved'),
                    "substrate_comparison_complete":coverage['complete'],
                    "substrate_comparison_required":coverage['required'],
                    "substrate_comparison_resolved":coverage['resolved'],
                    "substrate_partner_fits":json.dumps([{'target_smiles':p.get('target_smiles'),
                        'native_smiles':p.get('native_smiles'),'analogue_fit':whole_package_pair_fit(p)}
                        for p in sub_sim.get('property_comparisons') or []]),
                    "substrate_fit_scope":"Weakest required-partner analogue resemblance; incomplete comparisons are unknown. Not measured binding or activity.",
                    "target_role_assessments":json.dumps(assessment.get('target_roles') or []),
                    "substrate_role_mapping_status":(sub_sim.get('accommodation') or {}).get('role_transfer_status','not_recorded'),
                })

            content=f.getvalue()
        from enzyme_discovery_agent.state_store import atomic_write_text
        atomic_write_text(csv_path,content)

        if log_fn:
            log_fn(f"  Ranked candidates CSV:  {csv_path}")

    except Exception as e:
        logger.warning("Failed to generate candidates CSV: %s", e)
        if log_fn:
            log_fn(f"  CSV generation failed: {e}")
