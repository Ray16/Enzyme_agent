"""Evidence ledger for agent-selected MD protocols; no chemistry recipe is inferred.

The reasoning agent chooses values and alternatives. This module preserves its
decisions, verifies source snapshots and reports unresolved work. A complete
ledger is preparation readiness, never permission or scientific proof to run MD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..validation import file_digest, fingerprint

DECISIONS = {
    "reaction_step": "Which molecular/reaction states answer the question, which conditions generation, and how are the other required states evaluated?",
    "assembly": "Which construct and functional assembly contain the complete site?",
    "conditions": "Which pH, temperature and solution conditions apply, and which remain uncertain?",
    "participants": "Which exact substrates, intermediates and cofactors are present, with what chemical states?",
    "protonation": "Which residues/ligands need review, which microstates are justified, and how are ambiguities tested?",
    "metals": "Which metals, stoichiometry and coordination/hydration hypotheses are supported?",
    "solvent_boundary": "Which solvent, ions, boundaries and system extent are scientifically informative and feasible?",
    "model_backend": "Which potential(s), checkpoint, precision and runtime match the chemistry and hardware?",
    "sampling": "Is ordinary MD enough, which sampler answers the question, and what representative comparison justifies its efficiency and kinetic interpretation?",
    "collective_variables": "Which mapped coordinates resolve the relevant motions and hidden slow modes?",
    "validation": "Which preparation, force, timestep, stability, uncertainty and convergence checks must pass?",
    "resources": "What bounded CPU/GPU, memory, wall-time and resumability plan is justified?",
}
STATUSES = {"resolved", "conditional", "unresolved", "not_applicable"}
SOURCE_CATEGORIES = {"primary_literature", "curated_annotation", "experimental_structure",
                     "user_conditions", "method_output", "structural_observation"}


def contract():
    return {
        "schema_version": "sampling-setup-1.0", "decisions": DECISIONS,
        "top_level_required": ["schema_version", "question", "sequence_sha256", "target_sha256", "decisions", "sources"],
        "decision_required": ["status", "choice", "rationale", "evidence_ids", "alternatives", "assumptions", "uncertainties"],
        "statuses": sorted(STATUSES),
        "source_required": ["id", "category", "path", "sha256", "quote", "interpretation", "reviewed"],
        "source_categories": sorted(SOURCE_CATEGORIES),
        "agent_identity_binding": "declare_sampling_setup fills sequence_sha256 and target_sha256 when omitted. Do not calculate or invent hashes; explicit mismatches are rejected.",
        "conditional_required": "A discriminating_test explaining how the conditional choice will be challenged",
        "execution_choice_formats": {
            "conditions": "choice contains explicit pH and temperature_K",
            "model_backend": "choice.profile_ids lists exact calculator profile IDs",
            "sampling": "choice.method names the selected method; choice.parameters contains the exact run settings",
            "resources": "choice is the exact run resource_budget, including max_workers and max_wall_seconds_per_task",
        },
        "rules": [
            "Agent determines all scientific choices from the candidate, requested reaction and evidence; no case defaults are supplied.",
            "Unresolved decisions remain pending; not_applicable needs a scientific rationale.",
            "Separate primary evidence, user conditions, computational proposals and structural observations.",
            "Method/structure outputs cannot independently establish native catalytic states.",
            "Share family protocols only after exact-sequence, mechanism and molecular-setup applicability review.",
            "A ready ledger permits preparation planning only; prepared-atom and runtime gates remain separate.",
            "An unavailable chosen method stays pending; never silently substitute a supported engine or sampling method.",
        ],
        "readiness_checks_are": "Traceability/completeness checks, not pKa accuracy, chemical truth, or activity validation",
    }


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _strings(value, name):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f"{name} must be an explicit list of nonempty strings")


def assess(setup, sequence, target):
    """Return gaps rather than filling them with a standard protonation/MD recipe."""
    if not isinstance(setup, dict) or setup.get("schema_version") != "sampling-setup-1.0":
        raise ValueError("A sampling-setup-1.0 declaration is required")
    _text(setup.get("question"), "question")
    if setup.get("sequence_sha256") != hashlib.sha256(sequence.encode()).hexdigest():
        raise ValueError("Sampling setup does not match the exact candidate sequence")
    if setup.get("target_sha256") != fingerprint(target):
        raise ValueError("Sampling setup does not match the exact requested target")
    fingerprint(setup)  # Finite JSON only; preserves agent-selected scientific values.
    if not isinstance(setup.get("sources"), list) or not isinstance(setup.get("decisions"), dict):
        raise ValueError("Explicit source and decision collections are required")
    sources, source_checks = {}, []
    for source in setup["sources"]:
        for field in ("id", "path", "sha256", "quote", "interpretation"):
            _text(source.get(field), f"source.{field}")
        if source["id"] in sources:
            raise ValueError(f"Duplicate source id {source['id']!r}; each source needs a unique id")
        if source.get("category") not in SOURCE_CATEGORIES:
            raise ValueError(f"source {source['id']!r} category must be one of {sorted(SOURCE_CATEGORIES)}")
        errors = []
        if source.get("reviewed") is not True:
            errors.append("source interpretation not reviewed")
        try:
            path = Path(source["path"])
            if file_digest(path) != source["sha256"]:
                errors.append("source bytes changed")
            elif source["quote"] not in path.read_text(errors="replace"):
                errors.append("supporting passage absent")
        except OSError:
            errors.append("source unavailable")
        sources[source["id"]] = source
        source_checks.append({"id": source["id"], "category": source["category"], "errors": errors})
    valid_sources = {r["id"] for r in source_checks if not r["errors"]}
    gaps = []
    for name in DECISIONS:
        decision = setup["decisions"].get(name)
        if not isinstance(decision, dict):
            gaps.append({"decision": name, "reason": "decision not investigated"})
            continue
        if decision.get("status") not in STATUSES:
            raise ValueError(f"Unknown {name} decision status")
        _text(decision.get("rationale"), f"{name}.rationale")
        for key in ("evidence_ids", "assumptions", "uncertainties"):
            _strings(decision.get(key), f"{name}.{key}")
        if not isinstance(decision.get("alternatives"), list):
            raise ValueError(f"{name}.alternatives must be explicitly recorded")
        for alternative in decision["alternatives"]:
            if not isinstance(alternative, dict) or "choice" not in alternative:
                raise ValueError("Each alternative needs a choice and rationale")
            _text(alternative.get("rationale"), "alternative.rationale")
        if "choice" not in decision:
            raise ValueError(f"{name}.choice must be explicit, including null when unresolved")
        if decision["status"] == "unresolved":
            gaps.append({"decision": name, "reason": decision["rationale"]})
        elif decision["status"] != "not_applicable" and decision["choice"] is None:
            gaps.append({"decision": name, "reason": "selected choice is missing"})
        if decision["status"] == "conditional":
            _text(decision.get("discriminating_test"), f"{name}.discriminating_test")
            if not decision["uncertainties"]:
                raise ValueError("Conditional choices must state their uncertainties")
        if decision["status"] in {"resolved", "conditional"}:
            if not decision["evidence_ids"] or any(s not in valid_sources for s in decision["evidence_ids"]):
                gaps.append({"decision": name, "reason": "supporting evidence is missing, unreviewed or changed"})
    return {"status": "needs_investigation" if gaps else "ready_for_preparation",
            "setup_sha256": fingerprint(setup), "gaps": gaps, "sources": source_checks,
            "execution_authorized": False,
            "limitations": "Source attribution and scientific interpretation remain agent assertions; this is not a chemistry or pKa accuracy test."}


def validate_execution_binding(config, preparation, config_path):
    """Require the prepared run to implement the agent's actual selected protocol."""
    path = (Path(config_path).resolve().parent / config["setup_declaration"]).resolve()
    if file_digest(path) != config["setup_declaration_sha256"]:
        raise ValueError("Agent sampling setup artifact changed")
    saved = json.loads(path.read_text())
    setup = saved["setup"]
    report = assess(setup, preparation["sequence"], preparation["target"])
    if report["status"] != "ready_for_preparation":
        raise ValueError("Agent sampling setup still has unresolved decisions or source evidence")
    if (preparation.get("agent_setup_sha256") != report["setup_sha256"] or
            preparation["sequence_sha256"] != setup["sequence_sha256"]):
        raise ValueError("Prepared molecular system is not bound to this exact agent setup")
    choices = {key: value["choice"] for key, value in setup["decisions"].items()}
    conditions = choices["conditions"]
    if (conditions["pH"] != preparation["assay_conditions"]["pH"] or
            conditions["temperature_K"] != config["temperature_K"]):
        raise ValueError("Run conditions differ from the agent's selected conditions")
    if sorted(choices["model_backend"]["profile_ids"]) != sorted(m["id"] for m in config["models"]):
        raise ValueError("Calculator profiles differ from the agent's selected models")
    sampling = choices["sampling"]
    enabled = config["umbrella"]["enabled"]
    if sampling["method"] not in {"unbiased_md", "umbrella_distance"}:
        raise ValueError("The selected method needs a validated execution adapter; no automatic substitution")
    if enabled != (sampling["method"] == "umbrella_distance"):
        raise ValueError("Bias method differs from the agent's sampling decision")
    for key in ("timestep_fs", "friction_per_fs", "equilibration_steps", "production_steps",
                "sample_interval", "checkpoint_interval", "seeds", "umbrella"):
        if key not in sampling["parameters"] or sampling["parameters"][key] != config[key]:
            raise ValueError(f"Run {key} differs from the agent's selected sampling parameters")
    if config["resource_budget"] != choices["resources"]:
        raise ValueError("Run resource budget differs from the agent's selected budget")
    budget = config["resource_budget"]
    if (type(budget["max_workers"]) is not int or budget["max_workers"] < 1 or
            config["max_wall_seconds"] > budget["max_wall_seconds_per_task"]):
        raise ValueError("Execution exceeds the agent's declared resource bounds")
    return report


def declaration_schema():
    """Expose the actual ledger shape to tool callers, not an opaque object."""
    decision = {"type": "object", "properties": {
        "status": {"type": "string", "enum": sorted(STATUSES)}, "choice": {},
        "rationale": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "alternatives": {"type": "array", "items": {"type": "object", "required": ["choice", "rationale"]}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "discriminating_test": {"type": "string"}},
        "required": ["status", "choice", "rationale", "evidence_ids", "alternatives", "assumptions", "uncertainties"]}
    return {"type": "object", "description": "Exact ledger; identity hashes are filled by the tool when omitted. Unresolved decisions may have null choice and empty evidence_ids.",
            "properties": {"schema_version": {"type": "string", "enum": ["sampling-setup-1.0"]},
                "question": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "object", "required": ["id", "category", "path", "sha256", "quote", "interpretation", "reviewed"]}},
                "decisions": {"type": "object", "required": list(DECISIONS), "additionalProperties": decision}},
            "required": ["schema_version", "question", "sources", "decisions"]}
