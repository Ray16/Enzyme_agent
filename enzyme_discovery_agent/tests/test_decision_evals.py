"""The offline rubric exposes no answer keys and cannot claim model quality."""

import copy
from enzyme_discovery_agent.eval.decision_evals import cases, public_case, grade, self_check


def test_rubric_accepts_expected_and_rejects_known_bad_actions():
    report = self_check()
    assert report["passed"] is True
    assert len(report["cases"]) == 9
    assert "no model was evaluated" in report["scope"]


def test_model_visible_case_has_observations_but_no_answer_keys():
    for case in cases():
        public = public_case(case)
        assert "acceptable_actions" not in public
        assert "known_bad_action" not in public
        assert public["state"] == case["public_state"]


def test_traceability_and_missing_revision_conditions_are_not_optional():
    case = cases()[0]
    good = {"action": case["acceptable_actions"][0], "evidence_refs": case["public_state"]["evidence_sources"],
            "reason": "Inspect the unresolved observation.", "expected_information": "Check the supporting record.",
            "revision_condition": "Revise the interpretation if checks fail."}
    assert grade(case, good)["passed"] is True
    invented = {**good, "evidence_refs": ["synthetic:invented"]}
    assert grade(case, invented)["passed"] is False
    for field in ("reason", "expected_information", "revision_condition"):
        missing = copy.deepcopy(good)
        del missing[field]
        assert grade(case, missing)["passed"] is False
    assert grade(case, [good])["passed"] is False
