"""Read-only exposure audit of saved modeling exports; never rewrites results.

This checks the review's concrete acceptance/provenance risks, not historical
scientific performance or all possible modeling errors. Missing evidence is
reported as unverified, not proof that a prior calculation was invalid.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from .validation import file_digest, fingerprint, validate_ensemble, validate_pose_plan
from .integration import relocated_source
from .cofold.queue_worker import verify_completion

NAMES = {"report.json", "structural_modeling_evidence.json", "candidate_proxy_evidence.json", "ranking.json", "bundle.json"}
SKIP = {".git", ".venv", "__pycache__", "archive", "source_snapshots", "source-snapshots", "code_snapshots", "mlmm", "node_modules", "artifacts"}


def inspect_file(path):
    row = {"path": str(path.resolve()), "sha256": None, "findings": []}
    try:
        row["sha256"] = file_digest(path)
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") == "modeling-agent-1.0":
            row["kind"] = "controller_report"
            for candidate in data.get("candidates", []):
                acc = candidate.get("accession")
                record = candidate.get("modeling_input") or {}
                if not candidate.get("pose_validation"):
                    row["findings"].append({"accession": acc, "status": "no_pose_validation_claim", "detail": "input preparation or incomplete modeling; not evidence of false structural validation"})
                    continue
                execution_path = Path(candidate.get("input_yaml", "")).with_name("execution.json")
                try:
                    execution = json.loads(execution_path.read_text())
                    if fingerprint(execution) != candidate.get("execution_sha256"):
                        raise ValueError("saved execution identity is inconsistent")
                    plan = validate_pose_plan(record, execution.get("entity_map"))
                    poses = [Path(p["path"]) for p in candidate["pose_validation"].get("poses", [])]
                    checked = validate_ensemble(poses, execution)
                    failed = [c for p in checked["poses"] for c in p["checks"]
                              if c["status"] != "pass" and c["check"].startswith(("protein_completeness", "pose_plan", "structure"))]
                    row["findings"].append({"accession": acc, "status": "requires_review" if plan["status"] != "pass" or checked["status"] != "pass" else "current_physical_checks_pass",
                                            "current_pose_status": checked["status"], "review_checks": failed})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["findings"].append({"accession": acc, "status": "unverified", "detail": str(exc)})
        elif data.get("kind") == "modeling_validation":
            row["kind"] = "modeling_evidence_export"
            from .evidence import modeling_evidence
            source = data.get("source_report") or {}
            try:
                if file_digest(source["path"]) != source["sha256"]:
                    raise ValueError("source report no longer matches the saved export")
                current = modeling_evidence(source["path"])
                for candidate in current["candidates"]:
                    row["findings"].append({"accession": candidate["accession"], "status": candidate["status"], "review_errors": candidate["review_errors"]})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["findings"].append({"status": "unverified", "detail": str(exc)})
        elif path.parent.name == "completed":
            row["kind"] = "queue_completion"
            manifests = []
            for source in path.parent.parent.glob("*manifest*.json"):
                document = json.loads(source.read_text())
                for item in document.get("items", []):
                    if item.get("id") == path.stem:
                        manifests.append((document, item))
            if len(manifests) != 1:
                raise ValueError("current queue manifest association is missing or ambiguous")
            try:
                verify_completion(*manifests[0], data)
                row["findings"].append({"status": "current_execution_and_files_verified"})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                missing, changed, hashes = [], [], []
                sources = data.get("geometry", []) + data.get("confidence_files", [])
                for source in sources:
                    file = relocated_source(Path(source["path"]))
                    try:
                        observed = file_digest(file)
                        if observed != source.get("sha256"):
                            changed.append(str(file))
                        if file.suffix.lower() == ".pdb":
                            hashes.append(observed)
                    except OSError:
                        missing.append(str(file))
                row["findings"].append({"status": "unverified", "detail": str(exc),
                    "execution_binding": "unverified", "supporting_files_checked": len(sources),
                    "supporting_file_integrity": "verified" if sources and not missing and not changed else "unverified",
                    "missing_files": missing, "changed_files": changed,
                    "duplicate_prediction_contents": len(hashes) - len(set(hashes)),
                    "note": "Current file integrity does not reconstruct the original producer/scoring identity or prove historical invalidity."})
        elif path.name == "candidate_proxy_evidence.json":
            row["kind"] = "frozen_proxy_application"
            for candidate in data.get("candidates", []):
                if candidate.get("status") != "scored":
                    row["findings"].append({"accession": candidate.get("accession"), "status": "not_scored_no_sampling_claim",
                                            "original_status": candidate.get("status")})
                    continue
                prediction_hashes, errors = [], []
                for source in candidate.get("source_files", []):
                    file = relocated_source(Path(source["path"]))
                    try:
                        observed = file_digest(file)
                        if observed != source.get("sha256"):
                            errors.append("supporting file changed: " + str(file))
                        if file.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
                            prediction_hashes.append(observed)
                    except OSError:
                        errors.append("supporting file unavailable: " + str(file))
                if len(prediction_hashes) != len(set(prediction_hashes)):
                    errors.append("duplicate prediction contents in claimed ensemble")
                if not candidate.get("source_files"):
                    errors.append("no saved per-candidate supporting files to revalidate")
                elif not prediction_hashes:
                    errors.append("saved feature evidence has no direct prediction files; sampling coverage remains unverified")
                row["findings"].append({"accession": candidate.get("accession"), "status": "requires_review" if errors else "no_detected_file_or_copy_exposure",
                                        "review_errors": errors, "pose_files_checked": len(prediction_hashes)})
        elif path.name == "bundle.json" and "definition" in data:
            row["kind"] = "frozen_bundle"
            definition = data["definition"]
            changed = "approach_consistency" in definition.get("features", []) and (definition.get("descriptor_versions") or {}).get("approach_consistency") != "2.0"
            row["findings"].append({"status": "requires_descriptor_recalibration" if changed else "no_orientation_descriptor_exposure",
                                    "metric_kind": definition.get("kind"), "note": "Historical metric definitions are retained; no score was replaced or refitted."})
        elif path.name == "ranking.json" and "ranked_candidates" in data:
            row["kind"] = "saved_queue_ranking"
            row["findings"].append({"status": "requires_completion_revalidation", "note": "A saved ranked status is not current execution evidence; corresponding completions are audited separately."})
        else:
            return None
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        row.setdefault("kind", "unverified_saved_artifact")
        row["findings"].append({"status": "unverified", "detail": str(exc)})
    try:
        if file_digest(path) != row["sha256"]:
            row["findings"].append({"status": "unverified", "detail": "saved artifact changed during the audit"})
    except OSError:
        row["findings"].append({"status": "unverified", "detail": "saved artifact became unavailable during the audit"})
    return row


def audit(roots):
    rows, visited, seen = [], 0, set()
    for root in roots:
        for directory, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in SKIP and not name.startswith(".")]
            visited += 1
            for name in files:
                path = Path(directory) / name
                if name not in NAMES and not (Path(directory).name == "completed" and path.suffix == ".json"):
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                row = inspect_file(path)
                if row is not None:
                    rows.append(row)
    return {"schema_version": "structural-exposure-audit-1.0", "roots": [str(Path(r).resolve()) for r in roots],
            "directories_visited": visited, "artifacts_inspected": len(rows), "artifacts": rows,
            "finding_status_counts": dict(Counter(f["status"] for row in rows for f in row["findings"])),
            "limitations": ["Read-only current-state checks, not proof that a campaign was historically affected.",
                            "ML/MM test artifacts, archives, code snapshots and unrelated formats are excluded.",
                            "No GPU/provider run, refitting, source result rewrite, or exhaustive historical scientific reanalysis was performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    from enzyme_discovery_agent.state_store import atomic_write_json
    atomic_write_json(args.output, result, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
