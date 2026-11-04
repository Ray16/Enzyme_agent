"""Check public scientific claims and compare predictions with saved observations.

These checks establish consistency and traceability, not chemical truth, causal
identification, or calibrated probabilities. No LLM-generated confidence is scored.
"""
from __future__ import annotations

import math
import string
from itertools import combinations

from .validation import _source_checks, fingerprint, file_digest
from .integrity import proxy_integrity


CENTERS = {"atp_alpha", "atp_gamma", "carbonyl", "other"}
READOUT_FIELDS = ("metric", "scope", "unit", "endpoint", "conditions")
LIMITATION = ("Prediction intervals and source interpretations are declarations. "
              "Agreement is consistency with a prediction, not confirmation of a mechanism; "
              "contradiction does not identify its cause or establish inactivity.")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _texts(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    for item in value:
        _text(item, label)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _readout(value):
    for key in ("metric", "unit", "endpoint"):
        _text(value.get(key), key)
    if value.get("scope") not in {"structural", "experimental"}:
        raise ValueError("readout scope must be structural or experimental")
    if not isinstance(value.get("conditions"), dict):
        raise ValueError("readout conditions must be an explicit object")
    fingerprint(value["conditions"])  # Reject nonfinite/unserializable conditions before saving state.


def _readout_key(value):
    return fingerprint({key: value.get(key) for key in READOUT_FIELDS})


def compare_hypothesis_tests(reasoning, assessment):
    """Identify measurements that separate declared numerical predictions.

    This is interval discrimination, not expected information gain or a causal
    conclusion. It neither invents priors nor assumes a missing assay is feasible.
    """
    groups = {}
    statuses = {(p["hypothesis_id"], p["prediction_id"]): p["status"] for p in assessment["predictions"]}
    hypotheses = reasoning.get("hypotheses") or []
    hypothesis_ids = [h["id"] for h in hypotheses]
    separated = set()
    tests = []
    for hypothesis in hypotheses:
        for prediction in hypothesis["predictions"]:
            key = _readout_key(prediction)
            group = groups.setdefault(key, {"readout": {k: prediction[k] for k in READOUT_FIELDS}, "predictions": [], "controls": []})
            group["predictions"].append({"hypothesis_id": hypothesis["id"], "prediction_id": prediction["id"],
                                         "expected_interval": prediction["expected_interval"],
                                         "current_status": statuses[(hypothesis["id"], prediction["id"])]})
            group["controls"].extend(hypothesis["discriminating_test"]["controls"])
    for key, group in groups.items():
        distinguishable, unresolved = [], []
        for left, right in combinations(group["predictions"], 2):
            if left["hypothesis_id"] == right["hypothesis_id"]:
                continue
            a, b = left["expected_interval"], right["expected_interval"]
            if a[1] < b[0] or b[1] < a[0]:
                pair = sorted([left["hypothesis_id"], right["hypothesis_id"]])
                separated.add(tuple(pair))
                distinguishable.append(pair)
                if left["current_status"] == "unresolved" or right["current_status"] == "unresolved":
                    unresolved.append(pair)
        tests.append({"readout_id": key, **group, "controls": sorted(set(group["controls"])),
                      "distinguishable_pairs": [list(pair) for pair in sorted(set(map(tuple, distinguishable)))],
                      "unresolved_pairs": [list(pair) for pair in sorted(set(map(tuple, unresolved)))],
                      "status": "discriminating" if distinguishable else "overlapping_or_unshared_predictions"})
    tests.sort(key=lambda test: (-len(test["unresolved_pairs"]), -len(test["distinguishable_pairs"]), test["readout_id"]))
    contradicted = sorted({p["hypothesis_id"] for p in assessment["predictions"] if p["status"] == "contradicted"})
    return {"tests": tests,
            "pairs_needing_new_predictions": [list(pair) for pair in combinations(sorted(hypothesis_ids), 2) if pair not in separated],
            "contradicted_hypotheses": contradicted,
            "limitation": "Disjoint declared intervals can distinguish predictions, not prove mechanisms. Overlap, missing shared readouts and already contradictory hypotheses require new predictions or revised explanations. Feasibility, uncertainty and test costs are not inferred."}


def validate_test(test):
    if not isinstance(test, dict):
        raise ValueError("a discriminating test is required")
    for key in ("measurement", "expected_if_selected", "expected_if_alternative"):
        _text(test.get(key), f"test.{key}")
    _texts(test.get("controls"), "test.controls")
    if test["expected_if_selected"].strip() == test["expected_if_alternative"].strip():
        raise ValueError("a discriminating test needs different expected outcomes")


def validate_reasoning(reasoning, record):
    """Reject incomplete claims and descriptor/step mismatches before execution.

    Missing evidence is retained as a gap by reasoning_errors, so unsupported
    hypotheses can be recorded and investigated without being executed.
    """
    if not isinstance(reasoning, dict) or reasoning.get("schema_version") != "1.0":
        raise ValueError("a schema-1.0 scientific_reasoning record is required")
    hypotheses = reasoning.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) < 2:
        raise ValueError("declare at least two competing hypotheses")
    ids, statements, prediction_ids = set(), set(), set()
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            raise ValueError("hypotheses must be objects")
        for key in ("id", "statement", "mechanism", "step_description", "center_rationale"):
            _text(hypothesis.get(key), f"hypothesis.{key}")
        if hypothesis["id"] in ids or hypothesis["statement"].strip() in statements:
            raise ValueError("hypotheses need distinct IDs and statements")
        ids.add(hypothesis["id"])
        statements.add(hypothesis["statement"].strip())
        if hypothesis.get("reaction_center") not in CENTERS:
            raise ValueError(f"reaction_center must be one of {sorted(CENTERS)}")
        if type(hypothesis.get("step_index")) is not int or hypothesis["step_index"] < 1:
            raise ValueError("hypothesis.step_index must be positive")
        for key in ("assumptions", "uncertainties"):
            _texts(hypothesis.get(key), f"hypothesis.{key}")
        evidence = hypothesis.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(e, dict) or
                e.get("effect") not in {"supports", "contradicts", "context"} for e in evidence):
            raise ValueError("hypothesis evidence must list sources with supports/contradicts/context effects")
        predictions = hypothesis.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("every hypothesis needs a falsifiable prediction")
        for prediction in predictions:
            if not isinstance(prediction, dict):
                raise ValueError("predictions must be objects")
            _text(prediction.get("id"), "prediction.id")
            if prediction["id"] in prediction_ids:
                raise ValueError("prediction IDs must be unique across hypotheses")
            prediction_ids.add(prediction["id"])
            _readout(prediction)
            interval = prediction.get("expected_interval")
            if (not isinstance(interval, list) or len(interval) != 2 or
                    not all(_number(v) for v in interval) or interval[0] > interval[1]):
                raise ValueError("expected_interval must contain ordered finite bounds")
            if prediction["metric"] == "pose_pass_fraction" and (
                    prediction["scope"] != "structural" or prediction["unit"] != "fraction" or
                    prediction["endpoint"] != "declared_pose_checks" or
                    prediction["conditions"] != {"basis": "declared_ensemble"} or
                    not 0 <= interval[0] <= interval[1] <= 1):
                raise ValueError("pose_pass_fraction requires the declared_ensemble structural readout in [0, 1]")
            _text(prediction.get("decision_if_refuted"), "prediction.decision_if_refuted")
        validate_test(hypothesis.get("discriminating_test"))
    selected = selected_hypothesis(reasoning)
    step = record.get("modeled_step") or {}
    if (step.get("mechanism") != selected["mechanism"] or step.get("index") != selected["step_index"]
            or step.get("reaction_center") != selected["reaction_center"]):
        raise ValueError("selected hypothesis must match modeled_step mechanism, index and reaction_center")
    _text(reasoning.get("selection_rationale"), "selection_rationale")
    # ATP CCD atom names encode different elementary reaction centers. A free
    # text mechanism label cannot silently authorize reusing the other geometry.
    center = selected["reaction_center"]
    if center in {"atp_alpha", "atp_gamma"}:
        electrophile, reference = {"atp_alpha": ("PA", "O3A"), "atp_gamma": ("PG", "O3B")}[center]
        chain_index = len((record.get("enzyme_state") or {}).get("chains") or [])
        atp_chains = set()
        for cofactor in record.get("cofactors") or []:
            count = cofactor.get("count")
            if type(count) is not int or count < 1 or chain_index + count > len(string.ascii_uppercase):
                raise ValueError("ATP geometry requires explicit valid cofactor counts and chain allocation")
            if cofactor.get("ccd") == "ATP":
                atp_chains.update(string.ascii_uppercase[chain_index:chain_index + count])
            chain_index += count
        sites = ((record.get("validation_plan") or {}).get("poses") or {}).get("sites") or []
        if not sites:
            raise ValueError("ATP reaction centers require explicit pose sites")
        for site in sites:
            atom, axis = site.get("electrophile") or {}, site.get("angle_reference") or {}
            if (atom.get("atom") != electrophile or axis.get("atom") != reference or
                    atom.get("chain") not in atp_chains or atom.get("chain") != axis.get("chain") or
                    str(atom.get("residue")) != str(axis.get("residue"))):
                raise ValueError(f"{center} requires {electrophile}/{reference} on the same ATP residue at every site")
    return reasoning


def selected_hypothesis(reasoning):
    selected = next((h for h in reasoning.get("hypotheses", [])
                     if h.get("id") == reasoning.get("selected_hypothesis_id")), None)
    if selected is None:
        raise ValueError("selected_hypothesis_id must identify a declared hypothesis")
    return selected


def reasoning_errors(reasoning, record):
    """Recheck all cited source bytes, including contradictory evidence."""
    try:
        validate_reasoning(reasoning, record)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return [str(exc)]
    errors = []
    selected = selected_hypothesis(reasoning)
    if not any(e["effect"] == "supports" for e in selected["evidence"]):
        errors.append("selected hypothesis lacks supporting source evidence")
    for hypothesis in reasoning["hypotheses"]:
        if hypothesis["evidence"]:
            errors.extend(f"{hypothesis['id']}: {check['reason']}" for check in
                          _source_checks(hypothesis["evidence"], "hypothesis.evidence")
                          if check["status"] != "pass")
    return errors


def validate_observation(observation):
    if not isinstance(observation, dict):
        raise ValueError("observation must be an object")
    _text(observation.get("id"), "observation.id")
    if observation["id"].startswith("controller:"):
        raise ValueError("controller observation IDs are reserved")
    _readout(observation)
    if not _number(observation.get("value")):
        raise ValueError("observation.value must be finite")
    interval = observation.get("measurement_interval")
    if interval is not None and (not isinstance(interval, list) or len(interval) != 2 or
                                not all(_number(v) for v in interval) or
                                not interval[0] <= observation["value"] <= interval[1]):
        raise ValueError("measurement_interval must have finite ordered bounds containing the observed value")
    applicability = observation.get("applicability", "model" if observation["scope"] == "structural" else "candidate")
    if applicability not in {"model", "candidate"}:
        raise ValueError("observation applicability must be model or candidate")
    if observation["scope"] == "structural" and applicability == "candidate":
        _text(observation.get("transfer_rationale"), "structural observation transfer_rationale")
    _text(observation.get("interpretation"), "observation.interpretation")
    checks = _source_checks([observation.get("source")], "observation.source")
    errors = [c["reason"] for c in checks if c["status"] != "pass"]
    if errors:
        raise ValueError("; ".join(errors))


def _proxy_role_errors(revision):
    evaluation = revision.get("proxy_evaluation")
    if evaluation is None:
        return []
    try:
        from .modeling_reasoning.reactive_site import resolver_from_validation
        from .proxies.library import BY_ID
        for pid, measurement in evaluation.get("measurements", {}).items():
            if pid in BY_ID and BY_ID[pid].version != "1.0" and measurement.get("definition_version") != BY_ID[pid].version:
                return ["proxy descriptor definition changed; review/recalibrate and register a versioned plan"]
        record, plan = revision["modeling_input"], revision["proxy_plan"]
        validate_proxy_binding(plan, revision.get("scientific_reasoning") or {})
        _, expected = resolver_from_validation(record, (plan.get("context") or {}).get("site_id"))
        if evaluation.get("role_resolution") != expected or evaluation.get("resolved_record_sha256") != fingerprint(record):
            return ["proxy extraction is not bound to the current validated atoms and model; re-evaluate the plan"]
    except (ValueError, KeyError, TypeError) as exc:
        return ["proxy atom binding is unresolved: " + str(exc)]
    return []


def assess_reasoning(revision):
    """Compare commensurate readouts and preserve conflicting observations.

    Repeated observations never vote one another away. Missing inputs, changed
    sources, and mismatched endpoints/units/conditions remain unresolved.
    """
    reasoning = revision.get("scientific_reasoning") or {}
    pose_plan = ((revision.get("modeling_input") or {}).get("validation_plan") or {}).get("poses") or {}
    expected = pose_plan.get("expected_predictions")
    observations = list(revision.get("reasoning_observations") or [])
    observation_errors = {}
    for observation in observations:
        try:
            validate_observation(observation)
        except (ValueError, OSError) as exc:
            observation_errors[observation["id"]] = [str(exc)]
    poses = revision.get("pose_validation") or {}
    if poses:
        integrity_errors = []
        for entry in poses.get("poses", []):
            try:
                if file_digest(entry["path"]) != entry["sha256"]:
                    integrity_errors.append("prediction changed: " + entry["path"])
            except (OSError, KeyError):
                integrity_errors.append("prediction source unavailable")
        # A failed or unresolved sample is retained in the ensemble. Missing
        # atoms/samples must not turn into an observed zero fraction.
        complete = (poses.get("n_predictions") == expected
                    and poses.get("n_unresolved") == 0
                    and poses.get("n_unique_file_contents") == poses.get("n_predictions")
                    and poses.get("n_predictions", 0) > 0 and not integrity_errors
                    and not any(c.get("status") == "abstain" for c in poses.get("checks", []))
                    and not any(c.get("status") == "abstain" for pose in poses.get("poses", [])
                                for c in pose.get("checks", [])))
        observations.append({"id": "controller:pose_pass_fraction", "metric": "pose_pass_fraction",
                             "scope": "structural", "unit": "fraction", "endpoint": "declared_pose_checks",
                             "conditions": {"basis": "declared_ensemble"},
                             "value": poses.get("passing_fraction") if complete else None,
                             "integrity_errors": integrity_errors,
                             "provenance": poses.get("prediction_provenance", {"status": "abstain"})})
    proxy = revision.get("proxy_evaluation") or {}
    proxy_check = proxy_integrity(revision.get("proxy_plan"), revision.get("proxy_evaluation"),
                                  revision.get("proxy_evaluation_artifact"), poses.get("poses", []))
    proxy_role_errors = _proxy_role_errors(revision)
    for pid, measurement in proxy.get("measurements", {}).items():
        complete = (proxy_check["status"] == "verified" and not proxy_role_errors and measurement.get("status") == "computed"
                    and measurement.get("n_input_poses") == expected
                    and measurement.get("n_computed_poses", 0) >= (expected or 1))
        observations.append({"id": "controller:proxy:" + pid, "metric": "proxy:" + pid,
                             "scope": "structural", "unit": measurement.get("units"),
                             "endpoint": "declared_proxy_plan", "conditions": {"parameters": measurement.get("parameters") or {}},
                             "value": measurement.get("value") if complete else None,
                             "integrity": proxy_check, "coverage_status": measurement.get("status"),
                             "role_errors": proxy_role_errors,
                             "provenance": poses.get("prediction_provenance", {"status": "abstain"})})
    evaluations = []
    model_hash = fingerprint(revision.get("modeling_input") or {})
    inapplicable = {o["id"]: "bound to another model or missing its original model context"
                    for o in observations if not o["id"].startswith("controller:") and
                    o.get("applicability", "model" if o["scope"] == "structural" else "candidate") == "model" and
                    o.get("model_context_sha256") != model_hash}
    for hypothesis in reasoning.get("hypotheses", []):
        for prediction in hypothesis["predictions"]:
            matches = []
            for observation in observations:
                if observation["id"] in inapplicable:
                    continue
                if _readout_key(prediction) != _readout_key(observation):
                    continue
                errors = observation_errors.get(observation["id"], [])
                value = observation.get("value")
                lo, hi = prediction["expected_interval"]
                bounds = observation.get("measurement_interval") or [value, value]
                status = ("unresolved" if errors or not _number(value) else
                          "consistent" if lo <= bounds[0] <= bounds[1] <= hi else
                          "contradicted" if bounds[1] < lo or bounds[0] > hi else "unresolved")
                matches.append({"observation_id": observation["id"], "status": status,
                                "value": value, "measurement_interval": observation.get("measurement_interval"),
                                "errors": errors})
            statuses = {m["status"] for m in matches}
            status = ("contradicted" if "contradicted" in statuses else
                      "unresolved" if not matches or "unresolved" in statuses else "consistent")
            evaluations.append({"hypothesis_id": hypothesis["id"], "prediction_id": prediction["id"],
                                "status": status, "observations": matches,
                                "decision_if_refuted": prediction["decision_if_refuted"]})
    feedback = {key: revision.get(key) for key in
                ("scientific_reasoning", "setup_validation", "pose_plan_validation", "pose_validation", "proxy_evaluation", "reasoning_observations")}
    feedback["reasoning_observations"] = revision.get("reasoning_observations") or []
    if (feedback.get("pose_plan_validation") or {}).get("status") == "pass":
        # Successful declaration checks add no new observation and must not
        # invalidate a response merely because rendering recomputed them.
        feedback["pose_plan_validation"] = None
    # Source invalidation changes the assessment too, invalidating old responses.
    errors = reasoning_errors(reasoning, revision.get("modeling_input") or {})
    matched = {o["observation_id"] for e in evaluations for o in e["observations"]}
    result = {"selected_hypothesis_id": reasoning.get("selected_hypothesis_id"),
              "response_hypothesis_ids": [h["id"] for h in reasoning.get("hypotheses", [])] or ["legacy_unrecorded"],
              "hypotheses": [{k: h.get(k) for k in ("id", "statement", "mechanism", "step_index", "step_description",
                                                    "reaction_center", "assumptions", "uncertainties")}
                             for h in reasoning.get("hypotheses", [])],
              "predictions": evaluations, "observations": observations, "source_errors": errors,
              "observation_errors": observation_errors,
              "inapplicable_observations": inapplicable,
              "unmatched_observation_ids": [o["id"] for o in observations if o["id"] not in matched],
              "limitation": LIMITATION}
    comparison = compare_hypothesis_tests(reasoning, result)
    result["test_priorities"] = [{"readout": test["readout"], "unresolved_pairs": test["unresolved_pairs"],
                                   "controls": test["controls"]}
                                  for test in comparison["tests"] if test["unresolved_pairs"]][:3]
    result["pairs_needing_new_predictions"] = comparison["pairs_needing_new_predictions"]
    result["assessment_sha256"] = fingerprint({"feedback": feedback, "assessment": result})
    result["requires_response"] = bool(poses or revision.get("proxy_evaluation") is not None or
                                       observations or (revision.get("setup_validation") or {}).get("status") != "pass" or
                                       revision.get("pose_plan_validation") is not None and revision["pose_plan_validation"].get("status") != "pass")
    response = revision.get("reasoning_response") or {}
    result["response_current"] = response.get("assessment_sha256") == result["assessment_sha256"]
    result["next_actions"] = sorted({p["decision_if_refuted"] for p in evaluations if p["status"] == "contradicted"})
    return result


def execution_reasoning_errors(revision):
    errors = reasoning_errors(revision.get("scientific_reasoning"), revision["modeling_input"])
    errors.extend(_proxy_role_errors(revision))
    registration = revision.get("reasoning_registration") or {}
    if registration.get("reasoning_sha256") != fingerprint(revision.get("scientific_reasoning")):
        errors.append("registered scientific reasoning is missing or changed; create a new revision")
    execution = revision.get("execution")
    execution_hash = execution.get("scientific_reasoning_sha256") if execution else revision.get("execution_reasoning_sha256")
    if (execution or revision.get("input_yaml")) and execution_hash != fingerprint(revision.get("scientific_reasoning")):
        errors.append("execution is not bound to the current scientific reasoning record")
    assessment = assess_reasoning(revision)
    if assessment["observation_errors"]:
        errors.append("observation sources are unavailable or changed")
    if assessment["requires_response"]:
        if not assessment["response_current"]:
            errors.append("respond to the current reasoning assessment before execution or completion")
        elif revision["reasoning_response"]["action"] != "retain":
            errors.append("feedback requires revision, investigation or abstention before execution or completion")
    return errors


def validate_proxy_binding(plan, reasoning):
    selected = selected_hypothesis(reasoning)
    if plan.get("mechanism") != selected["mechanism"] or plan.get("reaction_step") != selected["step_description"]:
        raise ValueError("proxy mechanism and reaction_step must match the selected hypothesis")
    if selected["reaction_center"] in {"atp_alpha", "atp_gamma"} and (
            plan.get("context") or {}).get("reaction_center") != selected["reaction_center"]:
        raise ValueError("ATP proxy context.reaction_center must match the selected hypothesis")
