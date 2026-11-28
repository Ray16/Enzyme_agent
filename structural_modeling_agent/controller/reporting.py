"""Completion gating and structured report/evidence export for ModelingAgent."""
from __future__ import annotations

import copy
from pathlib import Path

from enzyme_core.state_store import atomic_write_json
from ..scientific_reasoning import assess_reasoning, execution_reasoning_errors
from ..validation import file_digest, fingerprint, validate_pose_plan, validate_setup


_VALIDATION_PY = Path(__file__).resolve().parents[1] / "validation.py"


def completion_errors(agent):
    """Return every unmet provenance, reasoning, setup, or pose requirement."""
    missing = []
    for candidate in agent.task["candidates"]:
        acc = candidate["accession"]
        from ..barrier.workflow import completion_errors as barrier_errors

        missing.extend(
            f"{acc}: {error}"
            for error in barrier_errors(agent, acc, required=agent.task.get("objective") == "evaluate_barriers")
        )
        if agent.task.get("objective") == "evaluate_barriers":
            continue
        if agent.task.get("objective") == "prepare_sampling":
            try:
                from .dispatch_handlers import evaluate_preparation_study

                result = evaluate_preparation_study(agent, acc)
                if result["status"] != "ready_for_sampling_review":
                    missing.append(f"{acc}: preparation or validation remains unresolved")
            except (ValueError, OSError, KeyError) as exc:
                missing.append(f"{acc}: {exc}")
            continue
        existing = agent.state["candidates"][acc].get("existing_prediction_review")
        if existing:
            from .reuse import checked_evidence

            try:
                if checked_evidence(existing)["execution_status"] != "complete":
                    missing.append(f"{acc}: historical prediction ensemble is incomplete or unverified")
            except (OSError, ValueError, KeyError) as exc:
                missing.append(f"{acc}: cached evidence changed or is unavailable: {exc}")
            continue

        candidate = agent._candidate(acc)
        try:
            revision = agent._current(acc)
            missing.extend(f"{acc}: reasoning: {error}" for error in execution_reasoning_errors(revision))
            if validate_pose_plan(revision["modeling_input"], (revision.get("execution") or {}).get("entity_map"))["status"] != "pass":
                missing.append(f"{acc}: physical acceptance plan is incomplete or invalid")
            if revision.get("execution") and revision["execution"]["modeling_input"] != revision["modeling_input"]:
                missing.append(f"{acc}: current model differs from the rendered execution")
            identity_error = agent._sequence_identity_error(acc, revision.get("execution"))
            if identity_error:
                missing.append(f"{acc}: {identity_error}")
            if validate_setup(revision["modeling_input"], candidate.get("sequence"))["status"] != "pass":
                missing.append(f"{acc}: setup is not validated")
            if not revision.get("execution") or file_digest(revision["input_yaml"]) != revision["input_sha256"]:
                missing.append(f"{acc}: current rendered inputs are missing or changed")
            elif any(file_digest(m["path"]) != m["sha256"] for m in revision["execution"].get("msa_evidence", [])):
                missing.append(f"{acc}: input MSA changed after rendering")
            if revision.get("proxy_evaluation") is not None:
                from .integrity import proxy_integrity

                integrity = proxy_integrity(
                    revision.get("proxy_plan"),
                    revision["proxy_evaluation"],
                    revision.get("proxy_evaluation_artifact"),
                    (revision.get("pose_validation") or {}).get("poses", []),
                )
                if integrity["status"] != "verified":
                    missing.append(f"{acc}: proxy evaluation integrity is not verified: " + "; ".join(integrity["errors"]))
            if agent.task.get("objective", "model_and_validate") != "prepare_inputs":
                report = revision.get("pose_validation") or {}
                if report.get("status") != "pass" or report.get("execution_sha256") != fingerprint(revision.get("execution")):
                    missing.append(f"{acc}: current prediction ensemble is not validated")
                elif report.get("validator_sha256") != file_digest(_VALIDATION_PY):
                    missing.append(f"{acc}: prediction checks were produced by a different validator implementation")
                elif any(not p.get("sha256") or file_digest(p["path"]) != p["sha256"] for p in report["poses"]):
                    missing.append(f"{acc}: prediction files changed after validation")
                provenance = report.get("prediction_provenance") or {}
                if provenance.get("status") != "pass":
                    missing.append(f"{acc}: prediction provenance is not verified for the current inputs")
                elif file_digest(provenance["path"]) != provenance["sha256"]:
                    missing.append(f"{acc}: producer manifest changed after validation")
                if any(not s.get("sha256") or file_digest(s["path"]) != s["sha256"] for s in provenance.get("supporting_files", [])):
                    missing.append(f"{acc}: producer supporting files changed after validation")
        except (ValueError, OSError, KeyError, TypeError):
            missing.append(f"{acc}: decision or supporting files are missing")
    return missing


def write_report(agent):
    """Write report.json, export structural_modeling_evidence.json, and refresh results_index.json."""
    candidates = []
    for acc, state in agent.state["candidates"].items():
        current = state["revisions"][-1] if state["revisions"] else {}
        candidates.append({
            "accession": acc,
            "revision": current.get("revision"),
            "barrier_studies": copy.deepcopy(state.get("barrier_studies", {})),
            "barrier_preparations": copy.deepcopy(state.get("barrier_preparations", [])),
            "sampling_jobs": copy.deepcopy(state.get("sampling_jobs", {})),
            "catalytic_profile": copy.deepcopy(state.get("catalytic_profile")),
            "developability": copy.deepcopy(state.get("developability")),
            "preparation_studies": [
                {"id": key, "artifact": value["artifact"]}
                for key, value in state.get("preparation_studies", {}).items()
            ],
            "preparation_operators": [
                {"artifact": value["artifact"], "status": value["status"]}
                for value in state.get("preparation_operators", [])
            ],
            "existing_prediction_review": state.get("existing_prediction_review"),
            "input_derivation": current.get("input_derivation"),
            "bioemu2_preparation": current.get("bioemu2_preparation"),
            "followups": current.get("followups", {}),
            "rationale": current.get("rationale"),
            "alternatives": current.get("alternatives"),
            "scientific_reasoning": current.get("scientific_reasoning"),
            "reasoning_registration": current.get("reasoning_registration"),
            "reasoning_assessment": assess_reasoning(current) if current else None,
            "reasoning_response": current.get("reasoning_response"),
            "reasoning_observations": current.get("reasoning_observations", []),
            "reasoning_history": [
                {
                    "revision": r["revision"],
                    "scientific_reasoning": r.get("scientific_reasoning"),
                    "followups": r.get("followups", {}),
                    "registration": r.get("reasoning_registration"),
                    "assessment": assess_reasoning(r),
                    "responses": r.get("reasoning_response_history", []),
                    "decision_artifact": r.get("decision_artifact"),
                }
                for r in state["revisions"]
            ],
            "modeling_input": current.get("modeling_input"),
            "setup_validation": current.get("setup_validation"),
            "pose_plan_validation": current.get("pose_plan_validation"),
            "pose_validation": current.get("pose_validation"),
            "input_sha256": current.get("input_sha256"),
            "execution_sha256": fingerprint(current["execution"]) if current.get("execution") else None,
            "execution_reasoning_sha256": (current.get("execution") or {}).get("scientific_reasoning_sha256"),
            "msa_evidence": (current.get("execution") or {}).get("msa_evidence"),
            "sequence_identity_source": (current.get("execution") or {}).get("sequence_identity_source"),
            "sequence_sha256": (current.get("execution") or {}).get("sequence_sha256"),
            "proxy_evaluation": current.get("proxy_evaluation"),
            "proxy_plan": current.get("proxy_plan"),
            "proxy_plan_registration": current.get("proxy_plan_registration"),
            "proxy_evaluation_artifact": current.get("proxy_evaluation_artifact"),
            "input_yaml": current.get("input_yaml"),
            "job": current.get("job"),
            "decision_artifact": current.get("decision_artifact"),
            "validation_artifact": current.get("validation_artifact"),
        })
    report = {
        "schema_version": "modeling-agent-1.0",
        "status": agent.state["status"],
        "run_dir": str(agent.run_dir),
        "objective": agent.task.get("objective", "model_and_validate"),
        "decision_origin": agent.state.get("decision_origin"),
        "result_layout": agent.state.get("result_layout"),
        "results_index": str(agent.run_dir / "results_index.json"),
        "target": agent.task.get("target"),
        "task_sha256": agent.state["task_sha256"],
        "source_discovery_run": agent.task.get("source_run"),
        "conclusion": agent.state.get("conclusion"),
        "completion_gaps": agent.completion_errors(),
        "usage": agent.state["usage"],
        "candidates": candidates,
        "code_snapshots": agent.state.get("code_snapshots", []),
        "environment": agent.state.get("environment"),
        "evidence_scope": (
            "Barrier execution and reviewed constrained-energy observations; no activation-free-energy or activity claim."
            if agent.task.get("objective") == "evaluate_barriers"
            else "Preparation evidence and agent-reviewed validation; no production dynamics or chemical accuracy claim."
            if agent.task.get("objective") == "prepare_sampling"
            else "Setup consistency and declared pose checks; no activity prediction or automatic discovery reranking."
        ),
    }
    if agent.task.get("kind") == "discovery_handoff":
        report["sequence_identity_requirement"] = "exact_uniprot_primary"
        report["discovery_handoff"] = {
            "discovery_status": (agent.task.get("producer") or {}).get("discovery_status", "unknown"),
            "coverage": (agent.task.get("producer") or {}).get("coverage", {"status": "unknown"}),
            "snapshot_sha256": (agent.task.get("integrity") or {}).get("snapshot_sha256"),
            "currentness": agent.state.get("discovery_currentness", {"status": "unknown"}),
            "unresolved_leads": [
                {"accession": row.get("accession"), "reason": row.get("reason")}
                for row in agent.task.get("unresolved_leads", [])
            ],
            "excluded_candidates": [
                {"accession": row.get("accession"), "reason": row.get("reason")}
                for row in agent.task.get("excluded_candidates", [])
            ],
        }
    atomic_write_json(agent.run_dir / "report.json", report, allow_nan=False)
    from ..evidence import export_modeling_evidence

    export_modeling_evidence(agent.run_dir / "report.json")
    from ..result_layout import write_index

    write_index(agent.run_dir, agent.state)
    return report
