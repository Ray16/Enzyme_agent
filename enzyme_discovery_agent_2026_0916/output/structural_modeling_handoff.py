"""Create an explicit, reviewable handoff to the structural-modeling pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.modeling_handoff import (
    SCHEMA_VERSION,
    build_modeling_input,
)


def _first(data: dict, metadata: dict, key: str, default=None):
    """Prefer populated evaluation evidence, then later registry enrichment."""
    if data.get(key):
        return data[key]
    if metadata.get(key):
        return metadata[key]
    if key in data:
        return data[key]
    if key in metadata:
        return metadata[key]
    return default


def _evidence_layer(accession: str, data: dict, metadata: dict) -> dict:
    mechanism = metadata.get("mechanism_evidence") or data.get("mechanism_evidence")
    return {
        "identity": {
            "accession": accession,
            "protein_name": _first(data, metadata, "protein_name", ""),
            "organism": _first(data, metadata, "organism", ""),
            "gene_names": _first(data, metadata, "gene_names", []),
            "ec_numbers": _first(data, metadata, "ec_numbers", []),
        },
        "native_reactions": _first(data, metadata, "catalytic_activities", []),
        "native_substrates": _first(data, metadata, "native_substrates", []),
        "family_assignment": {
            "interpro_family": _first(data, metadata, "interpro_family"),
            "interpro_family_id": _first(data, metadata, "interpro_family_id"),
        },
        "domain_architecture": _first(data, metadata, "domain_architecture", []),
        "structures": _first(data, metadata, "structural_data", {}),
        "cofactor_annotations": {
            "names": _first(data, metadata, "required_cofactors", []),
            "notes": _first(data, metadata, "cofactor_notes", []),
        },
        "reaction_chemistry": _first(data, metadata, "reaction_chemistry", {}),
        "mechanism": {
            "status": (
                mechanism.get("status", "not_attempted")
                if isinstance(mechanism, dict)
                else "not_attempted"
            ),
            "evidence": mechanism,
        },
        "structural_homology": metadata.get("structural_homology"),
    }


def _modeling_source(evidence: dict, metadata: dict) -> dict:
    annotations = evidence["cofactor_annotations"]
    mechanism = evidence["mechanism"]["evidence"]
    return {
        "catalytic_activities": evidence["native_reactions"],
        "required_cofactors": annotations["names"],
        "cofactor_notes": annotations["notes"],
        "mechanism_evidence": mechanism,
        "modeling_specification": metadata.get("modeling_specification"),
    }


def write_structural_modeling_handoff(
    run_dir: Path,
    evaluations: list[dict],
    target: dict,
    max_candidates: int = 20,
    neighbor_map: dict | None = None,
    output_path: Path | None = None,
) -> Path | None:
    """Write a shortlist manifest; structural modeling remains a deliberate next step.

    When ``neighbor_map`` (the persisted ``candidate_map.json``) is supplied, each
    shortlisted candidate carries its ESM neighbors, and neighbor accessions not
    already shortlisted are appended as extra cofold-panel entries -- so "the
    neighbors of the promising candidates" become part of the structural panel.
    """
    if not evaluations:
        return None

    registry = load_candidate_registry(run_dir).get("candidates", {})
    neighbors_by_promising: dict[str, list[dict]] = {}
    for promising in (neighbor_map or {}).get("promising", []):
        acc = promising.get("accession")
        if acc:
            neighbors_by_promising[acc] = promising.get("neighbors", []) or []
    evaluation_by_accession = {
        evaluation.get("data", evaluation).get("accession", ""): evaluation
        for evaluation in evaluations
        if isinstance(evaluation.get("data", evaluation), dict)
    }
    try:
        ranking = json.loads((run_dir / "pass1_ranking.json").read_text())
    except (OSError, json.JSONDecodeError):
        ranking = []
    if ranking:
        # The ranking has already collapsed each ortholog group to its selected
        # representative. Preserve that exact order and do not expand aliases.
        evaluations = [
            evaluation_by_accession[entry.get("accession")]
            for entry in ranking
            if entry.get("accession") in evaluation_by_accession
        ]
    candidates = []
    seen = set()
    for evaluation in evaluations:
        data = evaluation.get("data", evaluation)
        accession = data.get("accession", "")
        if not accession or accession in seen:
            continue
        seen.add(accession)
        metadata = registry.get(accession, {}).get("metadata", {})
        evidence = _evidence_layer(accession, data, metadata)
        projection_source = _modeling_source(evidence, {
            **metadata,
            "modeling_specification": (
                data.get("modeling_specification")
                or metadata.get("modeling_specification")
            ),
        })
        modeling_input = build_modeling_input(projection_source, target)
        candidates.append({
            "accession": accession,
            "evidence": evidence,
            "modeling_input": modeling_input,
            "selection": {
                "source": "evaluated_discovery_candidate",
                "reason": (
                    "Structural modeling is required before geometry-based promotion."
                ),
                "ortholog_alternatives": next((
                    entry.get("orthologs", []) for entry in ranking
                    if entry.get("accession") == accession
                ), []),
                "esm_neighbors": neighbors_by_promising.get(accession, []),
            },
        })
        if len(candidates) == max_candidates:
            break

    # Append ESM neighbors of the shortlisted promising candidates as extra
    # cofold-panel entries (the neighbors themselves become structural targets),
    # respecting the overall cap and skipping anything already shortlisted.
    for promising_acc, nbrs in neighbors_by_promising.items():
        for nbr in nbrs:
            if len(candidates) >= max_candidates:
                break
            nbr_acc = nbr.get("accession")
            if not nbr_acc or nbr_acc in seen:
                continue
            seen.add(nbr_acc)
            metadata = registry.get(nbr_acc, {}).get("metadata", {})
            evidence = _evidence_layer(nbr_acc, {}, metadata)
            projection_source = _modeling_source(evidence, metadata)
            candidates.append({
                "accession": nbr_acc,
                "evidence": evidence,
                "modeling_input": build_modeling_input(projection_source, target),
                "selection": {
                    "source": "esm_neighbor",
                    "reason": f"ESM neighbor of promising candidate {promising_acc}.",
                    "ortholog_alternatives": [],
                    "esm_neighbors": [],
                    "esm_neighbor_of": promising_acc,
                    "esm_distance": nbr.get("distance"),
                },
            })

    if not candidates:
        return None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "target": target,
        "max_candidates": max_candidates,
        "candidates": candidates,
        "next_step": (
            "Review every needs_review modeling_input and record approved substrate "
            "states, stereochemistry, cofactors, and stoichiometry. Then use "
            "structural_modeling/scripts/build_step1_inputs.py with explicit FASTA "
            "and MSA-index paths; it accepts only status=ready candidates unless an "
            "exploratory override is explicit."
        ),
    }
    path = output_path or (run_dir / "structural_modeling_handoff.json")
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path
