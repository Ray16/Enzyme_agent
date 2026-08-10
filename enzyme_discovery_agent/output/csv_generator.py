"""Generate candidates.csv from evaluation data."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger("enzyme_agent")


def generate_candidates_csv(run_dir: Path, all_evals: list[dict], log_fn=None) -> None:
    """Generate a CSV file with all evaluated candidates."""
    # Drop failed evaluations (data is None): not candidates, and the row
    # builder below assumes dict data.
    all_evals = [e for e in (all_evals or [])
                 if isinstance(e, dict) and isinstance(e.get("data", e), dict)]
    if not all_evals:
        return

    csv_path = run_dir / "candidates.csv"
    fieldnames = [
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

                writer.writerow({
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
            log_fn(f"  Candidates CSV:  {csv_path}")

    except Exception as e:
        logger.warning("Failed to generate candidates CSV: %s", e)
        if log_fn:
            log_fn(f"  CSV generation failed: {e}")
