"""Versioned review gates for spending modeling effort, without composite scores."""


def review_priority(row, policy):
    reasons = []
    if row.get("previously_modeled"):
        decision = "review_existing_models"
        reasons.append("Indexed predictions already exist; inspect their setup and applicability before new compute")
    elif row.get("latest_status") == "rejected":
        decision = "exclude_current_target"
        reasons.append("The source run currently rejects this candidate")
    elif row.get("assessment_quality"):
        decision = "resolve_relevance_and_scoring_evidence"
        reasons.append("The reported source rank has evidence or substrate-comparison review flags")
    elif row.get("family_already_modeled"):
        decision = "justify_advantage_over_existing_representative"
        reasons.append("This family already has modeled representatives; document a functional, host-expression or pocket advantage")
    elif row.get("expression_preferred_representative") not in {None, row.get("accession")}:
        decision = "compare_expression_preferred_representative"
        reasons.append("Another member with the same recorded reaction has stronger host evidence or existing modeling coverage")
    elif row.get("host_evidence") in {None, "unknown", "expression_problem_reported"}:
        decision = "resolve_host_and_construct_feasibility"
        reasons.append("Preferred-host expression or construct suitability remains unresolved")
    else:
        decision = "define_modeling_question_and_validate_setup"
        reasons.append("Host evidence is available; an explicit modeling question, native control and independently checked molecular setup are still required")
    return {"modeling_priority_state": decision, "priority_reasons": " | ".join(reasons),
            "policy_version": policy["version"], "ready_for_submission": False}
