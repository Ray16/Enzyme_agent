"""Generate ranked_candidates.csv from evaluation data."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger("enzyme_agent")


def _load_ranking(run_dir: Path) -> dict[str, dict]:
    """Map each representative accession to its final ranking metadata.

    Ortholog alternatives remain nested under their representative in
    ``pass1_ranking.json`` for provenance, but are intentionally omitted here:
    the final CSV is a decision list, not an expanded ortholog inventory.
    """
    ranking_path = run_dir / "pass1_ranking.json"
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
        # for provenance but omitted from the CSV decision list (below the
        # reporting line): a fold the agent stopped late is off-target, so its
        # already-evaluated members do not belong in the ranked shortlist.
        if entry.get("excluded_stopped_fold"):
            continue
        info = {
            "rank": entry.get("rank"),
            "tier": entry.get("tier", ""),
            "tier_label": entry.get("tier_label", ""),
            # Decomposed substrate-fit axes (no single collapsed score): how many of
            # the target's substrates are hosted, and the weakest per-substrate
            # functional-group overlap.
            "substrates_hosted": entry.get("substrates_hosted", ""),
            "shape_mismatches": entry.get("shape_mismatches", ""),
            "weakest_fg_match": entry.get("weakest_fg_match", ""),
            "evidence_level": entry.get("evidence_level", ""),
            "family_relevance": entry.get("family_relevance", ""),
            "direction": entry.get("direction", ""),
        }
        lookup[acc] = info
    return lookup


def generate_candidates_csv(run_dir: Path, all_evals: list[dict], log_fn=None) -> None:
    """Generate ranked_candidates.csv: evaluated candidates ordered by rank."""
    # Drop failed evaluations (data is None): not candidates, and the row
    # builder below assumes dict data.
    all_evals = [e for e in (all_evals or [])
                 if isinstance(e, dict) and isinstance(e.get("data", e), dict)]
    if not all_evals:
        return

    ranking = _load_ranking(run_dir)
    # A persisted ranking is authoritative and already contains one selected
    # representative per ortholog group. Do not expand its nested alternatives
    # back into separate rows.
    if ranking:
        all_evals = [
            ev for ev in all_evals
            if ev.get("data", ev).get("accession", "") in ranking
        ]

    def sort_key(ev):
        acc = ev.get("data", ev).get("accession", "")
        info = ranking.get(acc)
        if not info or info.get("rank") is None:
            return (float("inf"), acc)
        return (info["rank"], acc)

    all_evals = sorted(all_evals, key=sort_key)

    csv_path = run_dir / "ranked_candidates.csv"
    fieldnames = [
        "rank", "tier", "tier_label",
        "substrates_hosted", "shape_mismatches", "weakest_fg_match",
        "evidence_level", "family_relevance", "direction",
        "gene", "accession", "uniprot_url", "protein_name", "organism",
        "ec_numbers", "interpro_family", "is_swissprot", "sequence_length",
        "pdb_ids", "has_alphafold", "alphafold_url",
        "characterization_papers", "engineering_papers", "n_mutations",
        "tanimoto_score", "native_substrates", "native_substrate_smiles",
        "catalytic_activities", "function_comments",
        "strengths", "weaknesses", "requires_polymer_substrate",
    ]

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ev in all_evals:
                d = ev.get("data", ev)
                acc = d.get("accession", "")
                genes = d.get("gene_names", [])
                ec_list = d.get("ec_numbers", [])
                lit = d.get("literature", {})
                struct = d.get("structural_data", {})
                sub_sim = d.get("substrate_similarity", {})
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
                    "tier_label": rank_info.get("tier_label", ""),
                    "substrates_hosted": rank_info.get("substrates_hosted", ""),
                    "weakest_fg_match": rank_info.get("weakest_fg_match", ""),
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
                })

        if log_fn:
            log_fn(f"  Ranked candidates CSV:  {csv_path}")

    except Exception as e:
        logger.warning("Failed to generate candidates CSV: %s", e)
        if log_fn:
            log_fn(f"  CSV generation failed: {e}")
