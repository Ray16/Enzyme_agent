"""Link transport and chemistry through an explicitly sourced kinetic network.

This solver evaluates a declared continuous-time Markov model. It does not infer
rates from pose scores, count biased MD time as physical time, or establish that
an enzyme is processive. Unknown rates remain unknown.
"""
from __future__ import annotations

import math
import numpy as np

from ..validation import file_digest, fingerprint


def contract():
    return {
        "schema_version": "processivity-network-1.0",
        "required": ["schema_version", "question", "conditions", "start", "minimum_extensions",
                     "extension_definition", "association_definition", "states", "transitions"],
        "agent_binding": "Agent tool additionally requires exact candidate sequence_sha256 and requested target_sha256",
        "state": {"id": "unique state name", "associated": "boolean: growing chain still associated with the same enzyme",
                  "extensions": "completed chain-extension cycles since start, not arbitrary bonds or ATP cleavage",
                  "terminal": "none, release, or inactive", "description": "chemical/conformational state and its evidence"},
        "transition": {"from": "state ID", "to": "state ID", "event": "transport, binding, reorientation, chemistry, extension, exchange, or release",
                       "rate_s_inv": "finite positive first-order/effective rate, or null when unresolved",
                       "source": "path, sha256, quote, interpretation, reviewed, kind",
                       "source_kind": "experiment, validated_rate_estimate, or synthetic",
                       "conditions_sha256": "fingerprint of the exact network conditions",
                       "assumptions": "explicit list; include concentration dependence, protonation and Hamiltonian transfer limits"},
        "rules": [
            "Define association independently of active-pocket occupancy; a tethered chain can leave the catalytic site without dissociation.",
            "Every extension edge needs chemical evidence for actual chain growth; productive geometry alone is insufficient.",
            "Use physical rates from validated kinetic estimators; umbrella PMFs or raw biased trajectory times are not rates.",
            "Include a competing committed-release path. Forced retention cannot establish processivity.",
            "First release is absorbing for the processivity calculation; rebinding cannot count as same-association continuation.",
            "Any synthetic input makes the result a software demonstration only.",
            "Even sourced rates yield a conditional kinetic prediction, not atomistic or experimental proof.",
        ],
    }


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit")


def evaluate(network):
    """Probability of reaching repeated growth before committed first release."""
    if network.get("schema_version") != "processivity-network-1.0":
        raise ValueError("A processivity-network-1.0 declaration is required")
    for key in ("question", "extension_definition", "association_definition"):
        _text(network.get(key), key)
    if not isinstance(network.get("conditions"), dict):
        raise ValueError("Explicit network conditions are required")
    minimum = network.get("minimum_extensions")
    if type(minimum) is not int or minimum < 2:
        raise ValueError("Processivity requires at least two declared extension cycles")
    states = {}
    for state in network["states"]:
        _text(state.get("id"), "state ID")
        _text(state.get("description"), "state description")
        if state["id"] in states or type(state.get("associated")) is not bool:
            raise ValueError("State IDs must be unique and association explicit")
        if type(state.get("extensions")) is not int or state["extensions"] < 0:
            raise ValueError("Extension counts must be nonnegative integers")
        if state.get("terminal") not in {None, "release", "inactive"}:
            raise ValueError("Unknown terminal-state definition")
        if state["terminal"] == "release" and state["associated"]:
            raise ValueError("A committed-release state cannot remain associated")
        states[state["id"]] = state
    start = network["start"]
    if start not in states or not states[start]["associated"] or states[start]["extensions"] != 0:
        raise ValueError("Start from an associated chain before counting extension cycles")
    conditions_sha = fingerprint(network["conditions"])
    adjacency = {key: [] for key in states}
    gaps, synthetic = [], False
    edges = []
    seen = set()
    for index, edge in enumerate(network["transitions"]):
        left, right = edge["from"], edge["to"]
        if left not in states or right not in states or left == right or (left, right) in seen:
            raise ValueError("Transitions need distinct known endpoints and unique directed pairs")
        seen.add((left, right))
        event = edge["event"]
        if event not in {"transport", "binding", "reorientation", "chemistry", "extension", "exchange", "release"}:
            raise ValueError("Unknown transition event")
        delta = states[right]["extensions"] - states[left]["extensions"]
        if (event == "extension" and delta != 1) or (event != "extension" and delta != 0):
            raise ValueError("Only a declared extension event increments the completed-cycle count")
        if event == "extension" and not (states[left]["associated"] and states[right]["associated"]):
            raise ValueError("Represent bound chemical extension separately from committed release")
        if states[left]["associated"] != states[right]["associated"] and event not in {"binding", "release"}:
            raise ValueError("Association changes need explicit binding/release events")
        if edge.get("conditions_sha256") != conditions_sha:
            gaps.append({"transition": index, "reason": "rate conditions do not match the network"})
        rate = edge.get("rate_s_inv")
        if rate is None:
            gaps.append({"transition": index, "reason": "physical transition rate remains unresolved"})
        elif type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Rates must be finite and positive; missing observations cannot become zero rates")
        if not isinstance(edge.get("assumptions"), list):
            raise ValueError("Transition assumptions must be explicit")
        source = edge.get("source") or {}
        kind = source.get("kind")
        if kind not in {"experiment", "validated_rate_estimate", "synthetic"}:
            gaps.append({"transition": index, "reason": "no supported physical rate provenance"})
        synthetic |= kind == "synthetic"
        try:
            for field in ("path", "sha256", "quote", "interpretation"):
                _text(source.get(field), f"source.{field}")
            if source.get("reviewed") is not True or file_digest(source["path"]) != source["sha256"]:
                raise ValueError("Rate evidence is unreviewed or changed")
            from pathlib import Path
            if source["quote"] not in Path(source["path"]).read_text(errors="replace"):
                raise ValueError("Rate evidence passage is absent")
        except (OSError, ValueError) as exc:
            gaps.append({"transition": index, "reason": str(exc)})
        adjacency[left].append(right)
        edges.append((left, right, rate))
    identity = fingerprint(network)
    if gaps:
        return {"status": "needs_rate_investigation", "network_sha256": identity, "gaps": gaps,
                "probability_repeated_growth_before_release": None, "processivity_demonstrated": False}
    # First release stays absorbing even if the full catalytic network includes rebinding.
    reachable, stack = set(), [start]
    while stack:
        key = stack.pop()
        if key in reachable:
            continue
        reachable.add(key)
        state = states[key]
        if state["terminal"] or not state["associated"] or state["extensions"] >= minimum:
            continue
        stack.extend(adjacency[key])
    success = {key for key in reachable if states[key]["associated"] and states[key]["extensions"] >= minimum}
    release = {key for key in reachable if not states[key]["associated"] or states[key]["terminal"] == "release"}
    if not success or not release:
        raise ValueError("Both repeated-growth and competing committed-release outcomes must be represented")
    transient = sorted(key for key in reachable if key not in success and key not in release
                       and states[key]["terminal"] != "inactive")
    index = {key: i for i, key in enumerate(transient)}
    q = np.zeros((len(index), len(index)))
    rhs = np.zeros(len(index))
    for left, right, rate in edges:
        if left not in index:
            continue
        i = index[left]
        q[i, i] -= rate
        if right in index:
            q[i, index[right]] += rate
        elif right in success:
            rhs[i] += rate
    try:
        probability = np.linalg.solve(-q, rhs)
        time_to_outcome = np.linalg.solve(-q, np.ones(len(index)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Network has an unresolved nonabsorbing trap or inconsistent state partition") from exc
    p = float(probability[index[start]])
    if not -1e-9 <= p <= 1 + 1e-9:
        raise ValueError("Invalid absorption probability")
    return {"status": "software_demonstration_only" if synthetic else "conditional_kinetic_prediction",
            "network_sha256": identity, "minimum_extensions": minimum,
            "probability_repeated_growth_before_release": min(1.0, max(0.0, p)),
            "mean_time_to_first_outcome_s": float(time_to_outcome[index[start]]),
            "transient_states": transient, "success_states": sorted(success), "release_states": sorted(release),
            "processivity_demonstrated": False,
            "limitations": "This is a conditional Markov-model result. Rates, state definitions, chemistry, common reference conditions and uncertainty require independent validation; no molecular demonstration is inferred."}
