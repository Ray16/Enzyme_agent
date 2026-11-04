"""Export decomposed modeling observations for discovery review."""

from pathlib import Path
import copy
import json
import hashlib

from enzyme_discovery_agent.state_store import atomic_write_json
from .validation import file_digest, fingerprint
from .proxies.attack_angle import weighted_attack_angle
from .integrity import proxy_integrity
from .scientific_reasoning import assess_reasoning, execution_reasoning_errors


def modeling_evidence(report_path):
    """Recheck artifact identity and retain failed/incomplete ensembles verbatim."""
    path = Path(report_path).resolve()
    report = json.loads(path.read_text())
    if report.get("schema_version") != "modeling-agent-1.0":
        raise ValueError("expected a modeling-agent-1.0 report")
    rows, seen = [], set()
    for candidate in report.get("candidates", []):
        accession = candidate.get("accession")
        if not accession or accession in seen:
            raise ValueError("modeling report requires unique candidate identifiers")
        seen.add(accession)
        existing=candidate.get('existing_prediction_review')
        if existing:
            from .reuse import checked_evidence
            try:
                row=copy.deepcopy(checked_evidence(existing))
            except (OSError,ValueError,KeyError) as exc:
                row={'status':'incomplete','execution_status':'incomplete','geometry_metrics':{},
                     'scientific_status':'requires_source_review','review_errors':[str(exc)],
                     'experimental_activity':'not_established_by_modeling'}
            row.update(accession=accession,computation_origin='reused_existing_prediction',
                       existing_prediction_review=existing)
            rows.append(row)
            continue
        setup = candidate.get("setup_validation") or {}
        poses = candidate.get("pose_validation") or {}
        errors = []
        reasoning_assessment = assess_reasoning(candidate)
        reasoning_gaps = execution_reasoning_errors(candidate) if "scientific_reasoning" in candidate else ["legacy report has no scientific reasoning record"]
        if "scientific_reasoning" in candidate:
            errors.extend("reasoning: " + gap for gap in reasoning_gaps)
        identity_source = candidate.get("sequence_identity_source") or {}
        if report.get("sequence_identity_requirement") == "exact_uniprot_primary":
            try:
                from enzyme_discovery_agent.models.handoff_contract import read_bundle_source, _unique_object, _reject_constant
                identity_path = Path(identity_source["path"]).resolve(strict=True)
                source_root = (path.parent / "sources").resolve()
                if not identity_path.is_relative_to(source_root):
                    raise ValueError("identity source is outside the modeling run's sources")
                raw = read_bundle_source(path.parent, {"bundle_relative_path": identity_path.relative_to(path.parent).as_posix(),
                                                      "sha256": identity_source["sha256"]})
                entry = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                sequence = (entry.get("sequence") or {}).get("value")
                if (entry.get("primaryAccession") != accession or entry.get("inactiveReason")
                        or str(entry.get("entryType") or "").casefold() == "inactive"
                        or identity_source.get("source_id") != f"uniprot:{accession}"
                        or not isinstance(sequence, str)
                        or hashlib.sha256(sequence.encode("ascii")).hexdigest() != candidate.get("sequence_sha256")):
                    raise ValueError("identity record does not match candidate and sequence")
            except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError) as exc:
                errors.append(f"sequence identity source unavailable, changed, or inconsistent: {exc}")
        if setup.get("record_sha256") != fingerprint(candidate.get("modeling_input")):
            errors.append("setup decision differs from its validated record")
        if setup.get("validator_sha256") != file_digest(Path(__file__).with_name("validation.py")):
            errors.append("setup validator identity is missing or changed")
        inputs = candidate.get("input_yaml")
        if not inputs or not candidate.get("input_sha256"):
            errors.append("rendered input identity is missing")
        else:
            try:
                if file_digest(inputs) != candidate["input_sha256"]:
                    errors.append("rendered input changed")
            except OSError:
                errors.append("rendered input is unavailable")
        records = poses.get("poses") or []
        references = list(records) + list(candidate.get("msa_evidence") or [])
        provenance = poses.get("prediction_provenance") or {}
        if provenance.get("status") != "pass":
            errors.append("prediction provenance is not verified")
        else:
            references.append(provenance)
            references.extend(provenance.get("supporting_files", []))
        for declaration in ((candidate.get("modeling_input") or {}).get("validation_plan") or {}).get("setup", {}).values():
            references.extend(declaration.get("evidence") or [])
        references.extend((((candidate.get("modeling_input") or {}).get("validation_plan") or {}).get("poses") or {}).get("evidence") or [])
        pose_declaration = ((candidate.get("modeling_input") or {}).get("validation_plan") or {}).get("poses") or {}
        atom_policy = pose_declaration.get("protein_atom_policy") or {}
        for rule in atom_policy.get("allowed_missing", []) + atom_policy.get("modified_residues", []):
            references.extend(rule.get("evidence") or [])
        for source in references:
            try:
                if not source.get("sha256") or file_digest(source["path"]) != source["sha256"]:
                    errors.append("supporting artifact changed or lacks a fingerprint")
            except (OSError, KeyError, TypeError):
                errors.append("supporting artifact is unavailable")
        if poses.get("validator_sha256") != file_digest(Path(__file__).with_name("validation.py")):
            errors.append("pose validator identity is missing or changed")
        if candidate.get("execution_sha256") != poses.get("execution_sha256") or not candidate.get("execution_sha256"):
            errors.append("execution identity is missing or inconsistent")
        per_site = {}
        for pose in records:
            for check in pose.get("checks", []):
                if check.get("check", "").startswith("site.") and "distance_A" in check and "angle_deg" in check:
                    per_site.setdefault(check["check"], []).append((check["distance_A"], check["angle_deg"]))
        metrics = {}
        for site, observations in per_site.items():
            distances, angles = zip(*observations)
            metrics[site] = {**weighted_attack_angle(distances, angles),
                             "complete_observed_ensemble": len(observations) == len(records)}
        ready = setup.get("status") == "pass" and poses.get("status") == "pass" and not errors
        expected=(((candidate.get('modeling_input') or {}).get('validation_plan') or {}).get('poses') or {}).get('expected_predictions')
        artifact_errors=[error for error in errors if not error.startswith('reasoning:')]
        execution_complete=(type(expected) is int and expected>0 and len(records)==expected
            and len({record.get('sha256') for record in records if record.get('sha256')})==expected
            and provenance.get('status')=='pass' and not artifact_errors)
        proxy_check = proxy_integrity(candidate.get("proxy_plan"), candidate.get("proxy_evaluation"),
                                      candidate.get("proxy_evaluation_artifact"), records)
        rows.append({"accession": accession, "revision": candidate.get("revision"),
                     "status": "validated" if ready else "incomplete",
                     "execution_status":"complete" if execution_complete else "incomplete",
                     "pose_qc_status":poses.get('status','unavailable'),
                     "scientific_status":"consistent_record" if not reasoning_gaps else "requires_interpretation_or_revision",
                     "review_errors": sorted(set(errors)), "setup_validation": setup,
                     "pose_validation": poses, "geometry_metrics": metrics,
                     "proxy_evaluation": candidate.get("proxy_evaluation"),
                     "proxy_integrity": proxy_check,
                     "proxy_plan_registration": candidate.get("proxy_plan_registration"),
                     "scientific_reasoning": candidate.get("scientific_reasoning"),
                     "reasoning_registration": candidate.get("reasoning_registration"),
                     "reasoning_assessment": reasoning_assessment,
                     "reasoning_review": {"status": "incomplete" if reasoning_gaps else "consistent_record",
                                          "errors": reasoning_gaps},
                     "reasoning_response": candidate.get("reasoning_response"),
                     "reasoning_history": candidate.get("reasoning_history", []),
                     "modeling_input": candidate.get("modeling_input"),
                     "sequence_identity_source": identity_source or None,
                     "sequence_sha256": candidate.get("sequence_sha256"),
                     "experimental_activity": "not_established_by_modeling"})
    return {"schema_version": "3.0", "kind": "modeling_validation", "candidates": rows,
            "target": report.get("target"), "task_sha256": report.get("task_sha256"),
            "decision_origin":report.get('decision_origin'),
            "source_discovery_run": report.get("source_discovery_run"),
            "source_report": {"path": str(path), "sha256": file_digest(path)},
            "note": "Raw setup/pose evidence for review; discovery ranks and activity claims are not changed."}


def export_modeling_evidence(report_path, output_path=None):
    output = Path(output_path) if output_path else Path(report_path).with_name("structural_modeling_evidence.json")
    atomic_write_json(output, modeling_evidence(report_path), allow_nan=False)
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export_modeling_evidence(args.report, args.output))


if __name__ == "__main__":
    main()
