"""Import structural-modeling evidence into a discovery run for explicit review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.models.target_profile import normalize_target_profile
from enzyme_discovery_agent.state_store import atomic_write_json


def _evidence_value(value: object) -> float | None:
    """Accept finite normalized evidence values only; malformed values stay unobserved."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def structural_evidence_matrix(evidence: dict, target_profile: dict | None = None) -> tuple[dict, str]:
    """Return raw, per-check evidence without collapsing it into a composite score."""
    catalytic = evidence.get("catalytic_site", {})
    cofold = evidence.get("cofold", {})
    profile = normalize_target_profile(target_profile)
    raw_components = {
        "catalytic_center": catalytic.get("catalytic_center_score"),
        "pocket_engagement": catalytic.get("pocket_engagement_score"),
        "cofold_confidence": cofold.get("confidence_score"),
        "ligand_burial": cofold.get("substrate_sasa_burial"),
    }
    matrix = {
        name: {
            "required": name in profile["required_evidence"],
            "observed": (value := _evidence_value(raw_value)) is not None,
            "value": value,
        }
        for name, raw_value in raw_components.items()
    }
    missing_required = [name for name, check in matrix.items() if check["required"] and not check["observed"]]
    return matrix, "ready_for_review" if not missing_required else "incomplete_structural_evidence"


def import_structural_evidence(run_dir: Path, manifest_path: Path) -> Path:
    """Attach evidence to matching candidates and write a non-scoring review matrix."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") == "modeling-agent-1.0":
        from structural_modeling_agent.evidence import modeling_evidence
        manifest = modeling_evidence(manifest_path)
    if manifest.get("schema_version") not in ("1.0", "2.0", "3.0"):
        raise ValueError("Unsupported structural-evidence schema; expected 1.0, 2.0, 3.0 or a modeling agent report.")
    if manifest.get("schema_version") == "3.0" and manifest.get("kind") != "modeling_validation":
        raise ValueError("schema 3.0 must identify modeling validation evidence")
    if manifest.get("schema_version") == "3.0":
        from structural_modeling_agent.validation import file_digest
        from structural_modeling_agent.evidence import modeling_evidence
        source = manifest["source_report"]
        if file_digest(source["path"]) != source["sha256"]:
            raise ValueError("modeling source report changed after export")
        manifest = modeling_evidence(source["path"])
        source_run = manifest.get("source_discovery_run")
        if source_run and Path(source_run).resolve() != run_dir.resolve():
            raise ValueError("modeling evidence belongs to a different discovery run; review target transfer explicitly")
    identifiers = [c.get("accession") for c in manifest.get("candidates", [])]
    if any(not a for a in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("structural evidence requires unique candidate identifiers")
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = load_candidate_registry(run_dir).get("candidates", {})
    target_profile = normalize_target_profile(manifest.get("target_profile"))
    candidates = []
    for evidence in manifest.get("candidates", []):
        accession = evidence.get("accession")
        if accession not in registry:
            continue
        from enzyme_discovery_agent.models.candidate_registry import discovery_status
        transition = None if discovery_status(registry[accession]) == "rejected" else "structurally_evaluated"
        if manifest["schema_version"] == "3.0":
            review_status = "ready_for_modeling_review" if (evidence.get("status") == "validated"
                or evidence.get('execution_status')=='complete') else "incomplete_modeling_evidence"
            update_candidate_registry(run_dir, accession, transition, metadata={
                "modeling_validation_evidence": evidence,
                "modeling_validation_source": manifest.get("source_report"),
                "modeling_review_status": review_status,
            })
            candidates.append({"accession": accession, "review_status": review_status, "evidence": evidence})
            continue
        if manifest["schema_version"] == "2.0":
            # Frozen proxy applications keep assay units, calibration scope and
            # provenance. They must not be coerced into normalized legacy checks.
            review_status = "ready_for_proxy_review" if evidence.get("status") == "scored" else "inconclusive_proxy_evidence"
            update_candidate_registry(run_dir, accession, transition, metadata={
                "structural_proxy_evidence": evidence,
                "structural_proxy_bundle": manifest.get("bundle"),
                "structural_proxy_review_status": review_status,
            })
            candidates.append({"accession": accession, "review_status": review_status,
                               "evidence": evidence})
            continue
        matrix, review_status = structural_evidence_matrix(evidence, target_profile)
        update_candidate_registry(run_dir, accession, transition, metadata={
            "structural_evidence": evidence,
            "structural_evidence_matrix": matrix,
            "structural_review_status": review_status,
        })
        candidates.append({"accession": accession, "review_status": review_status, "evidence": matrix})
    output = run_dir / ("structural_modeling_review.json" if manifest["schema_version"] == "3.0" else
                        "structural_proxy_review.json" if manifest["schema_version"] == "2.0" else "structural_evidence_review.json")
    atomic_write_json(output, {
        "source_manifest": str(manifest_path),
        "target_profile": target_profile,
        "review_requirements": target_profile["review_requirements"],
        "warning": "No composite score or automatic rank is produced. Review raw structural evidence with functional and literature evidence before selection.",
        "candidates": candidates,
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Import structural evidence into an enzyme-discovery run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(import_structural_evidence(args.run_dir, args.manifest))


if __name__ == "__main__":
    main()
