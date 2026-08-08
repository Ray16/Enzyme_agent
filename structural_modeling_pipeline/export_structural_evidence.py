#!/usr/bin/env python3
"""Export project-local structural results in the discovery-agent evidence schema."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _read_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row.get("uid") or row.get("protein_id"): row for row in csv.DictReader(handle) if row.get("uid") or row.get("protein_id")}


def _number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def export_structural_evidence(project_dir: Path, cofold_metrics: Path | None = None) -> Path:
    """Merge pocket, catalytic-site, and optional co-fold metrics by accession."""
    pocket_rows = _read_csv(project_dir / "pocket_analysis_summary.csv")
    catalytic_path = project_dir / "catalytic_site_validation.json"
    catalytic = json.loads(catalytic_path.read_text()) if catalytic_path.exists() else {}
    cofold_rows = _read_csv(cofold_metrics) if cofold_metrics else {}
    accessions = sorted(set(pocket_rows) | set(catalytic) | set(cofold_rows))
    candidates = []
    for accession in accessions:
        pocket, catalytic_entry, cofold = pocket_rows.get(accession, {}), catalytic.get(accession, {}), cofold_rows.get(accession, {})
        candidates.append({
            "accession": accession,
            "pocket": {
                "evidence_type": pocket.get("evidence_type"),
                "evidence_tm_score": _number(pocket.get("evidence_tmscore")),
                "active_site_overlap": _number(pocket.get("active_site_overlap_in_pocket")),
                "pocket_volume_A3": _number(pocket.get("pocket_volume_A3")),
                "warning": pocket.get("warning") or None,
            },
            "catalytic_site": {
                "status": catalytic_entry.get("status"),
                "evidence_tier": catalytic_entry.get("evidence_tier"),
                "catalytic_center_score": catalytic_entry.get("catalytic_center_score"),
                "pocket_engagement_score": catalytic_entry.get("pocket_engagement_score"),
            },
            "cofold": {
                "confidence_score": _number(cofold.get("confidence_score")),
                "iptm": _number(cofold.get("iptm")),
                "substrate_sasa_burial": _number(cofold.get("substrate_sasa_burial")),
                "substrate_contact_residues": _number(cofold.get("substrate_n_contact_residues")),
                "substrate_min_distance_A": _number(cofold.get("substrate_min_distance")),
            },
        })
    handoff_path = project_dir / "discovery_handoff.json"
    handoff = json.loads(handoff_path.read_text()) if handoff_path.exists() else {}
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(project_dir.resolve()),
        "target_profile": handoff.get("target", {}).get("target_profile"),
        "candidates": candidates,
        "interpretation": "Structural evidence supports or weakens a candidate; it does not prove catalytic turnover.",
    }
    output = project_dir / "structural_evidence_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export structural evidence for discovery-agent ranking.")
    parser.add_argument("project", type=Path)
    parser.add_argument("--cofold-metrics", type=Path, default=None, help="Optional cofolding_metrics.csv")
    args = parser.parse_args()
    print(export_structural_evidence(args.project, args.cofold_metrics))


if __name__ == "__main__":
    main()
