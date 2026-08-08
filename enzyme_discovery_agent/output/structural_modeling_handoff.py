"""Create an explicit, reviewable handoff to the structural-modeling pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry


def write_structural_modeling_handoff(
    run_dir: Path,
    evaluations: list[dict],
    target: dict,
    max_candidates: int = 20,
) -> Path | None:
    """Write a shortlist manifest; structural modeling remains a deliberate next step."""
    if not evaluations:
        return None

    registry = load_candidate_registry(run_dir).get("candidates", {})
    candidates = []
    seen = set()
    for evaluation in evaluations:
        data = evaluation.get("data", evaluation)
        accession = data.get("accession", "")
        if not accession or accession in seen:
            continue
        seen.add(accession)
        metadata = registry.get(accession, {}).get("metadata", {})
        candidates.append({
            "accession": accession,
            "protein_name": data.get("protein_name") or metadata.get("protein_name", ""),
            "organism": data.get("organism") or metadata.get("organism", ""),
            "gene_names": data.get("gene_names") or metadata.get("gene_names", []),
            "ec_numbers": data.get("ec_numbers") or metadata.get("ec_numbers", []),
            "structure_sources": data.get("structural_data", {}),
            "native_substrate_evidence": data.get("native_substrates", []),
            "reaction_chemistry": data.get("reaction_chemistry", {}),
            "structural_homology": metadata.get("structural_homology"),
            "selection_reason": "Evaluated discovery candidate; structural modeling is required before geometry-based promotion.",
        })
        if len(candidates) == max_candidates:
            break

    if not candidates:
        return None
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "target": target,
        "max_candidates": max_candidates,
        "candidates": candidates,
        "next_step": (
            "Review the shortlist, then pass selected accessions and verified substrate SMILES "
            "to structural_modeling_pipeline/run_structural_modeling.py."
        ),
    }
    path = run_dir / "structural_modeling_handoff.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path
