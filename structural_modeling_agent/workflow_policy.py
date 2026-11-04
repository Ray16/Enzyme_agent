"""Purpose-aware routing: cofold first, optional scientific follow-ups later."""
from pathlib import Path

from .validation import file_digest, fingerprint, validate_pose_plan, _source_checks

TOOL_PURPOSES = {
    "cofolding": {"role": "default assessment after discovery handoff", "tool": "configured cofold provider",
                  "question": "Can the declared molecular complex yield physically consistent substrate/cofactor poses?"},
    "dynamics": {"role": "optional investigation after cofold inspection", "tool": "bioemu2",
                 "question": "Could conformational variability explain an observation or test a mechanistic hypothesis?",
                 "limitation": "Equilibrium samples characterize predicted variability; they do not establish time-dependent rates."},
    "de_novo_design": {"role": "optional new-enzyme generation", "tool": "rfd4-proteina",
                       "recipient": "enzyme_design_agent", "question": "Would a new scaffold or enzyme design address an unmet catalytic requirement?"},
}


def baseline_errors(revision):
    """Require inspected, traceable cofold evidence, not successful catalysis.

    A complete ensemble with nonproductive geometry can motivate dynamics.
    Missing/incorrect protein identity and stale inputs cannot authorize it.
    """
    report = revision.get("pose_validation") or {}
    execution = revision.get("execution") or {}
    record = revision.get("modeling_input") or {}
    expected = ((record.get("validation_plan") or {}).get("poses") or {}).get("expected_predictions")
    errors = []
    if execution.get("purpose", "cofolding") != "cofolding":
        errors.append("the baseline must be a cofolding execution")
    if validate_pose_plan(record, execution.get("entity_map"))["status"] != "pass":
        errors.append("the baseline physical acceptance plan is incomplete or stale")
    setup = revision.get("setup_validation") or {}
    if setup.get("status") != "pass" or setup.get("record_sha256") != fingerprint(record):
        errors.append("the baseline setup is not validated for this model")
    for declaration in ((record.get("validation_plan") or {}).get("setup") or {}).values():
        if any(c["status"] != "pass" for c in _source_checks(declaration.get("evidence"), "baseline.setup")):
            errors.append("baseline setup evidence is missing or changed")
    if not report or report.get("n_predictions") != expected or not expected:
        errors.append("inspect the complete declared cofold ensemble first")
    if report.get("execution_sha256") != fingerprint(execution):
        errors.append("cofold evidence does not match the current execution")
    if report.get("validator_sha256") != file_digest(Path(__file__).with_name("validation.py")):
        errors.append("cofold checks need revalidation with the current validator")
    provenance = report.get("prediction_provenance") or {}
    if provenance.get("status") != "pass":
        errors.append("cofold producer provenance is unresolved")
    references = report.get("poses", []) + provenance.get("supporting_files", [])
    if provenance.get("path"):
        references = references + [provenance]
    try:
        if not execution or record != execution.get("modeling_input") or file_digest(revision["input_yaml"]) != revision["input_sha256"]:
            errors.append("cofold input/model identity changed")
        for source in references + execution.get("msa_evidence", []):
            if not source.get("sha256") or file_digest(source["path"]) != source["sha256"]:
                errors.append("cofold supporting files changed")
    except (OSError, KeyError, TypeError):
        errors.append("cofold supporting files are unavailable")
    for pose in report.get("poses", []):
        checks = pose.get("checks", [])
        identity = [c for c in checks if c.get("check", "").startswith(("identity.", "protein_completeness.", "chain_inventory", "structure"))]
        if not identity or any(c["status"] != "pass" for c in identity):
            errors.append("cofold molecular identity/completeness is unresolved")
    return sorted(set(errors))


def validate_followup(plan, revision):
    if not isinstance(plan, dict) or plan.get("purpose") not in {"dynamics", "de_novo_design"}:
        raise ValueError("follow-up purpose must be dynamics or de_novo_design")
    errors = baseline_errors(revision)
    if errors:
        raise ValueError("cofolding is the default prerequisite: " + "; ".join(errors))
    for key in ("question", "why_cofold_insufficient", "expected_information", "decision_if_supported", "decision_if_refuted"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"follow-up requires {key}")
    from .scientific_reasoning import assess_reasoning
    assessment = assess_reasoning(revision)
    if plan.get("assessment_sha256") != assessment["assessment_sha256"]:
        raise ValueError("follow-up must reference the current reasoning assessment")
    hypotheses = {h["id"] for h in (revision.get("scientific_reasoning") or {}).get("hypotheses", [])}
    selected = plan.get("hypothesis_ids")
    if not isinstance(selected, list) or not selected or any(h not in hypotheses for h in selected):
        raise ValueError("follow-up must identify the hypotheses it investigates")
    if "num_samples" in plan and (type(plan["num_samples"]) is not int or plan["num_samples"] < 1):
        raise ValueError("follow-up num_samples must be positive")
    return plan
