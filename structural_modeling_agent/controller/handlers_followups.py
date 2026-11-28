"""Follow-up (dynamics, design handoff) and scientific-reasoning handlers."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from enzyme_core.state_store import atomic_write_json

from ..scientific_reasoning import (
    assess_reasoning,
    compare_hypothesis_tests,
    validate_observation,
    validate_test,
)
from ..validation import file_digest, fingerprint
from .handlers_inspection import _text_page


def _h_plan_followup(agent, args, acc, candidate, revision):
    from ..result_layout import followup_paths
    from .workflow_policy import TOOL_PURPOSES, validate_followup

    plan = copy.deepcopy(args["plan"])
    validate_followup(plan, revision)
    identity = fingerprint(plan)[:16]
    if identity in revision.get("followups", {}):
        return revision["followups"][identity]
    followup = {
        "id": identity,
        "revision": revision["revision"],
        "plan": plan,
        "tool": TOOL_PURPOSES[plan["purpose"]]["tool"],
        "status": "planned",
    }
    followup["paths"] = followup_paths(agent.run_dir, acc, revision["revision"], plan["purpose"], identity)
    Path(followup["paths"]["root"]).mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        followup["paths"]["request"],
        {
            "purpose": plan["purpose"],
            "plan": plan,
            "task_sha256": agent.state["task_sha256"],
            "accession": acc,
            "revision": revision["revision"],
            "paths": followup["paths"],
        },
        allow_nan=False,
    )
    revision.setdefault("followups", {})[identity] = followup
    if plan["purpose"] in {"de_novo_design", "enzyme_design"}:
        requested_tool = plan.get("requested_tool")
        if requested_tool is None and plan["purpose"] == "de_novo_design":
            requested_tool = "rfd4-proteina"
        prefix = (
            "De novo enzyme design requested using RFD4-Proteina. "
            if plan["purpose"] == "de_novo_design" and requested_tool == "rfd4-proteina"
            else "Enzyme design requested from structural modeling diagnostics (choose between redesign and generative strategies). "
        )
        design_query = prefix + json.dumps(
            {
                "target": agent.task.get("target"),
                "scientific_question": plan,
                "reference_accession": acc,
                "note": "The reference cofold is evidence, not proof of catalysis or a mandatory scaffold.",
            }
        )
        followup["design_agent_arguments"] = {
            "query": design_query,
            "handoff": followup["paths"]["design_handoff"],
            "output_dir": followup["paths"]["design_runs_root"],
        }
        followup["review_command"] = [
            "python",
            "-m",
            "enzyme_design_agent",
            design_query,
            "--handoff",
            followup["paths"]["design_handoff"],
            "--output-dir",
            followup["paths"]["design_runs_root"],
        ]
        profile = candidate.get("catalytic_profile") or {}
        request = {
            "kind": "de_novo_design_request" if plan["purpose"] == "de_novo_design" else "enzyme_design_request",
            "purpose": "de_novo_enzyme_design" if plan["purpose"] == "de_novo_design" else "enzyme_design",
            "recipient": "enzyme_design_agent",
            "requested_tool": requested_tool,
            "target": agent.task.get("target"),
            "source_accession": acc,
            "reference_model": revision["modeling_input"],
            "cofold_evidence": revision.get("validation_artifact"),
            "question": plan,
            "source_task_sha256": agent.state["task_sha256"],
            "output_dir": followup["paths"]["design_runs_root"],
            "catalytic_profile_summary": (
                {
                    "evolvability_class": profile.get("evolvability_class"),
                    "axes": profile.get("axes"),
                }
                if profile
                else None
            ),
            "soft_defect_design_targets": list(profile.get("design_targets") or []),
            "failure_modes": list(profile.get("failure_modes") or []),
            "note": "Design request only; no generation launched and no new sequence is assumed validated.",
        }
        atomic_write_json(followup["paths"]["design_handoff"], request, allow_nan=False)
        followup["handoff"] = followup["paths"]["design_handoff"]
    followup["artifact"] = agent._artifact(f"followup_{acc}", copy.deepcopy(followup))
    return followup


def _h_prepare_bioemu2(agent, args, acc, candidate, revision):
    from ..scientific_reasoning import reasoning_errors
    from .bioemu2_adapter import prepare
    from .workflow_policy import validate_followup

    followup = revision.get("followups", {}).get(args["followup_id"])
    if not followup or followup["plan"]["purpose"] != "dynamics":
        raise ValueError("BioEmu2 requires a saved dynamics follow-up plan")
    validate_followup(followup["plan"], revision)
    if reasoning_errors(revision.get("scientific_reasoning"), revision["modeling_input"]):
        raise ValueError("scientific evidence is unresolved")
    if file_digest(revision["input_yaml"]) != revision["input_sha256"]:
        raise ValueError("rendered input changed")
    path = Path(revision["input_yaml"]).with_name("execution.json")
    if fingerprint(json.loads(path.read_text())) != fingerprint(revision["execution"]):
        raise ValueError("saved execution differs from the current revision")
    provenance = revision["pose_validation"]["prediction_provenance"]
    baseline_sources = revision["pose_validation"]["poses"] + [provenance] + provenance.get("supporting_files", [])
    preparation_dir = (
        (followup.get("paths") or {}).get("preparation_dir")
        or (
            str(Path(followup["preparation"]["job_path"]).parent)
            if followup.get("preparation")
            else str(path.parent / f"bioemu2_{args['followup_id']}")
        )
    )
    prepared = prepare(
        path,
        args["checkpoint_path"],
        preparation_dir,
        seed=args["seed"],
        mode="equilibrium",
        baseline_sources=[{"path": s["path"], "sha256": s["sha256"]} for s in baseline_sources],
        num_samples=followup["plan"].get("num_samples"),
    )
    followup["preparation"] = prepared
    return prepared


def _h_inspect_dynamics(agent, args, acc, candidate, revision):
    from .bioemu2_adapter import digest, inspect_result

    followup = revision.get("followups", {}).get(args["followup_id"])
    if not followup or followup["plan"]["purpose"] != "dynamics" or not followup.get("preparation"):
        raise ValueError("prepare the saved dynamics follow-up before inspecting results")
    job = followup["preparation"]
    if digest(job["job_path"]) != job["job_sha256"]:
        raise ValueError("prepared dynamics job changed")
    result = inspect_result(job["job_path"], args["prediction_manifest"])
    if followup.get("result") and followup["result"]["source_manifest"] != result["source_manifest"]:
        raise ValueError("inspected follow-up evidence is immutable; register a new study")
    followup.update(result=result, status="inspected")
    return result


def _h_register_design_run(agent, args, acc, candidate, revision):
    followup = revision.get("followups", {}).get(args["followup_id"])
    destination = ((followup or {}).get("paths") or {}).get("design_runs_root")
    if not followup or followup["plan"]["purpose"] not in {"de_novo_design", "enzyme_design"} or not destination:
        raise ValueError("this follow-up has no recorded design output root")
    actual = Path(args["design_run_dir"]).expanduser().resolve(strict=True)
    expected = Path(destination)
    if not actual.is_relative_to(expected) or actual == expected:
        raise ValueError(
            "design run must be a child of its allocated output root; external results require explicit migration/review"
        )
    manifest = actual / "manifest.json"
    report = actual / "DESIGN_REPORT.md"
    data = json.loads(manifest.read_text())
    if data.get("agent") != "enzyme_design_agent" or not report.is_file():
        raise ValueError("design-agent manifest/report are missing or inconsistent")
    output_handoff = actual / "design_output_handoff.json"
    registration = {
        "run_dir": str(actual),
        "agent_status": data.get("status"),
        "strategy": data.get("strategy"),
        "designs": data.get("designs"),
        "manifest": {"path": str(manifest), "sha256": file_digest(manifest)},
        "report": {"path": str(report), "sha256": file_digest(report)},
        "output_handoff": (
            {"path": str(output_handoff), "sha256": file_digest(output_handoff)}
            if output_handoff.is_file()
            else None
        ),
        "generation_validation": "not_established_by_agent_report",
        "note": "This links result locations; generation, refolding and catalytic assessments retain their own separate evidence.",
    }
    if followup.get("design_run") and followup["design_run"] != registration:
        raise ValueError("registered design evidence is immutable; create a new follow-up for another run")
    followup.update(design_run=registration, status="design_run_linked")
    return registration


def _h_inspect_reasoning(agent, args, acc, candidate, revision):
    return {
        "scientific_reasoning": revision.get("scientific_reasoning"),
        "assessment": assess_reasoning(revision),
        "response": revision.get("reasoning_response"),
    }


def _h_read_reasoning_section(agent, args, acc, candidate, revision):
    sections = {
        "scientific_reasoning": revision.get("scientific_reasoning"),
        "assessment": assess_reasoning(revision),
        "observations": revision.get("reasoning_observations", []),
        "response": revision.get("reasoning_response"),
        "followups": revision.get("followups", {}),
    }
    if args.get("section") not in sections:
        raise ValueError("unknown reasoning section")
    return _text_page(
        json.dumps(sections[args["section"]], allow_nan=False),
        args.get("offset", 0),
        section=args["section"],
        accession=acc,
        revision=revision["revision"],
    )


def _h_compare_hypothesis_tests(agent, args, acc, candidate, revision):
    assessment = assess_reasoning(revision)
    return {
        "assessment_sha256": assessment["assessment_sha256"],
        **compare_hypothesis_tests(revision.get("scientific_reasoning") or {}, assessment),
    }


def _h_record_reasoning_observation(agent, args, acc, candidate, revision):
    observation = copy.deepcopy(args["observation"])
    agent._authorize_reasoning_source(observation.get("source"))
    validate_observation(observation)
    observation.setdefault("applicability", "model" if observation["scope"] == "structural" else "candidate")
    observation["model_context_sha256"] = fingerprint(revision["modeling_input"])
    observation["recorded_revision"] = revision["revision"]
    observations = revision.setdefault("reasoning_observations", [])
    if any(o["id"] == observation["id"] for o in observations):
        raise ValueError("observation IDs are immutable within a revision")
    observations.append(observation)
    revision.setdefault("reasoning_observation_artifacts", []).append(
        agent._artifact(f"reasoning_observation_{acc}", observation)
    )
    agent._event("reasoning_observation", accession=acc, observation_id=observation["id"])
    return assess_reasoning(revision)


def _h_respond_to_reasoning_feedback(agent, args, acc, candidate, revision):
    assessment = assess_reasoning(revision)
    if args.get("assessment_sha256") != assessment["assessment_sha256"]:
        raise ValueError("reasoning assessment changed; inspect the current feedback")
    if args.get("action") not in {"retain", "revise", "investigate", "abstain"}:
        raise ValueError("unknown reasoning response action")
    if not isinstance(args.get("rationale"), str) or not args["rationale"].strip():
        raise ValueError("a reasoning response needs a rationale")
    validate_test(args.get("next_test"))
    hypotheses = set(assessment["response_hypothesis_ids"])
    addressed = args.get("addressed_hypotheses")
    required = {p["hypothesis_id"] for p in assessment["predictions"] if p["status"] == "contradicted"}
    selected = (revision.get("scientific_reasoning") or {}).get("selected_hypothesis_id") or "legacy_unrecorded"
    if selected:
        required.add(selected)
    if (
        not isinstance(addressed, list)
        or not addressed
        or any(not isinstance(h, str) or h not in hypotheses for h in addressed)
        or not required.issubset(addressed)
    ):
        raise ValueError("address the selected and all contradicted hypotheses by their declared IDs")
    response = {
        k: copy.deepcopy(args[k])
        for k in ("assessment_sha256", "action", "rationale", "addressed_hypotheses", "next_test")
    }
    revision["reasoning_response"] = response
    revision.setdefault("reasoning_response_history", []).append(response)
    revision["reasoning_response_artifact"] = agent._artifact(f"reasoning_response_{acc}", response)
    agent._event(
        "reasoning_response",
        accession=acc,
        action=response["action"],
        assessment_sha256=response["assessment_sha256"],
    )
    return {"status": "recorded", "action": response["action"], "assessment": assess_reasoning(revision)}
