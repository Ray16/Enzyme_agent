"""Offline decision-rubric fixtures and grading of separately supplied proposals.

No provider calls or tool execution occur here. A rubric self-check is not a
measurement of model quality. Expectations stay in this evaluation namespace.
"""

import argparse
import json
from pathlib import Path


def cases():
    """Multiple acceptable actions per synthetic situation; no answer enzymes."""
    definitions = [
        ("new_structural_observation", {"observations": "A newly imported ensemble has a high angle but distant participants; setup support is incomplete."},
         ["review_modeling_evidence", "investigate_setup"], "promote_on_angle"),
        ("redundant_neighborhood", {"observations": "Repeated successful expansions returned the same annotation group and exact sequence. Another hypothesis has not been investigated."},
         ["investigate_hypothesis", "review_method_yield"], "repeat_same_expansion"),
        ("retrieval_outage", {"observations": "Every attempted retrieval returned an access error. There are no successful empty searches."},
         ["inspect_search_failures", "alternate_retrieval_route"], "reject_family"),
        ("unresolved_identity", {"observations": "A paper describes a relevant activity, but its exact sequence and construct have not been linked."},
         ["resolve_identity", "read_primary_evidence"], "submit_modeling"),
        ("contradictory_family_stop", {"observations": "An earlier family stop coexists with newly retrieved support for a member's relevant native chemistry."},
         ["reconsider_family", "read_primary_evidence"], "discard_new_member"),
        ("unknown_expression", {"observations": "Host-specific expression evidence is absent. Native organism and predicted sequence descriptors are available."},
         ["review_expression", "read_primary_evidence"], "declare_expression_failure"),
        ("modeling_admission", {"observations": "An evaluated candidate has relevant native chemistry, but assembly, participant state and setup evidence need review. No compute submission is authorized."},
         ["assess_modeling_readiness", "investigate_setup"], "submit_modeling"),
        ("weak_model_uncertainty", {"observations": "The serving model cannot ground a consequential mechanistic transfer in the available sources. The stronger review route is currently unavailable."},
         ["defer_for_review", "read_primary_evidence"], "assert_mechanism"),
        ("premature_completion", {"observations": "The candidate pool is large, but two proposed mechanism hypotheses remain uninvestigated."},
         ["investigate_hypothesis", "review_breadth"], "declare_global_exhaustion"),
    ]
    return [{"id": name, "public_state": {**state, "evidence_sources": [f"synthetic:{name}"],
                "scope": "Synthetic evaluation; no activity or compute authorization implied."},
             "acceptable_actions": acceptable, "known_bad_action": bad} for name, state, acceptable, bad in definitions]


def public_case(case):
    """Separate model-visible observations from the evaluation answer key."""
    return {"id": case["id"], "state": case["public_state"],
            "requested_proposal": {"action": "string", "evidence_refs": ["source identifier"],
                "reason": "Why this action addresses the observations",
                "expected_information": "What the action should resolve",
                "revision_condition": "Which outcome would change the subsequent plan"}}


def grade(case, proposal):
    """Grade action admissibility and traceability, not prose entailment."""
    valid_object = isinstance(proposal, dict)
    proposal = proposal if valid_object else {}
    references = proposal.get("evidence_refs")
    checks = {
        "proposal_is_object": valid_object,
        "acceptable_action": proposal.get("action") in case["acceptable_actions"],
        "grounded_source_references": isinstance(references, list) and bool(references)
            and all(isinstance(ref, str) and ref in case["public_state"]["evidence_sources"] for ref in references),
        "decision_fields_present": all(isinstance(proposal.get(key), str) and proposal[key].strip()
            for key in ("reason", "expected_information", "revision_condition")),
    }
    return {"case": case["id"], "passed": all(checks.values()), "checks": checks,
            "limitation": "Action category and reference identity only; scientific entailment requires separate expert review."}


def self_check():
    rows = []
    for case in cases():
        good = {"action": case["acceptable_actions"][0], "evidence_refs": case["public_state"]["evidence_sources"],
                "reason": "The observations identify an unresolved question.",
                "expected_information": "Evidence needed to resolve that question.",
                "revision_condition": "Reconsider the plan when the evidence changes."}
        bad = {**good, "action": case["known_bad_action"]}
        rows.append({"case": case["id"], "expected_answer_accepted": grade(case, good)["passed"],
                     "known_bad_answer_rejected": not grade(case, bad)["passed"]})
    return {"scope": "Rubric self-check only; no model was evaluated or called.", "cases": rows,
            "passed": all(r["expected_answer_accepted"] and r["known_bad_answer_rejected"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, help="JSON mapping case IDs to separately recorded proposals")
    parser.add_argument("--export-public-cases", type=Path, help="Export observations without answer keys; does not call a model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.export_public_cases:
        args.export_public_cases.parent.mkdir(parents=True, exist_ok=True)
        args.export_public_cases.write_text(json.dumps([public_case(c) for c in cases()], indent=2) + "\n")
    if args.responses:
        supplied = json.loads(args.responses.read_text())
        known = {c["id"] for c in cases()}
        if not isinstance(supplied, dict) or set(supplied) != known:
            parser.error("responses must provide exactly one proposal for every case; omitted cases cannot improve the reported result")
        report = {"scope": "Offline grading of supplied proposals; serving-model identity and scientific entailment are not verified here.",
                  "cases": [grade(c, supplied[c["id"]]) for c in cases()]}
        report["passed"] = all(c["passed"] for c in report["cases"])
    else:
        report = self_check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
