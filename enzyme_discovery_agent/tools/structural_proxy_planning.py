"""Agent tools for selecting and declaring mechanism-relevant structural proxies."""
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.structural_proxy_planning import proxy_guidance, validate_proxy_plan
import hashlib
import json
from pathlib import Path


@tool(description="Get structural-proxy guidance organized by physical process, explicit assay/molecular-state context, input requirements and missing-state followups. Use after selecting candidates for modeling. Declare context rather than inferring bottlenecks from family names; choose physical hypotheses before descriptors. Supports small confirmatory panels or explicitly documented discovery searches and physically coupled pairs. Availability does not establish predictive utility.",
      category="reasoning", parallel_safe=True,
      params={"mechanism": {"type": "string", "description": "The mechanism supported by the candidate's evidence."},
              "reaction_step": {"type": "string", "description": "The specific modeled chemical step and molecular state."},
              "assay_endpoint": {"type": "string", "description": "The experimental quantity against which proxies will be assessed."},
              "available_inputs": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Available input capabilities, such as reactive_roles, substrate_chains, protein_alignment, confidence_sidecars, or charges."},
              "assay_context": {"type": "object", "default": {}, "description": "Explicit substrate_state (soluble/insoluble_polymer/surface_bound/membrane/unknown), measurement_basis (crude_lysate/whole_cell/enzyme_normalized/active_enzyme_normalized/unknown), modeled_state, requires_processing boolean, assembly_status, candidate limiting_processes, substrate_representation, substrate_is_assayed boolean and temperature_C/pH/duration_hours. Candidate limiting processes remain hypotheses."}})
def get_structural_proxy_guidance(mechanism: str, reaction_step: str,
                                  assay_endpoint: str, available_inputs: list[str] | None = None,
                                  assay_context: dict | None = None) -> dict:
    try:
        data = proxy_guidance(mechanism, reaction_step, assay_endpoint, available_inputs or [], assay_context)
    except ValueError as exc:
        return {"status": "error", "summary": str(exc), "data": None, "count": 0, "suggestions": [str(exc)]}
    return {"status": "success", "summary": "Physical hypotheses, contextual priorities and uncovered states; no activity-based selection performed.",
            "data": data, "count": len(data["descriptors"]),
            "suggestions": [g["followup"] for g in data["modeled_state_gaps"]]}


@tool(description="Validate a proposed structural-proxy plan before evaluating activity associations. Requires an explicit mechanism, modeled reaction step, assay endpoint, per-proxy rationale, expected direction and evidence identifiers. Does not validate the chemistry or claim predictive performance.",
      category="reasoning", parallel_safe=True,
      params={"plan": {"type": "object", "description": "Proxy plan with schema_version 1.0, mechanism, reaction_step, assay_endpoint, hypotheses and context."}})
def validate_structural_proxy_plan(plan: dict) -> dict:
    errors = validate_proxy_plan(plan)
    return {"status": "error" if errors else "success", "summary": "Plan requires revision." if errors else "Declared proxy plan is structurally valid.",
            "data": {"valid": not errors, "errors": errors}, "count": len(plan.get("hypotheses", [])) if isinstance(plan, dict) else 0,
            "suggestions": errors}


@tool(description="Inspect a user-supplied experimental calibration bundle after candidates are selected for modeling. Returns its frozen descriptor/model definition, assay endpoint, applicability scope, validation status and source fingerprints. Use to assess whether an existing proxy can be applied or requires a new hypothesis. Does not score candidates or refit a model.",
      category="reasoning", parallel_safe=True,
      params={"bundle_path": {"type": "string", "description": "Path to a frozen calibration bundle produced by the active structural modeling workflow."}})
def inspect_experimental_proxy_bundle(bundle_path: str) -> dict:
    try:
        bundle = json.loads(Path(bundle_path).read_text())
        if bundle.get("schema_version") != "1.0":
            raise ValueError("unsupported bundle schema")
        checksum = hashlib.sha256(json.dumps(bundle["definition"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        if checksum != bundle.get("definition_sha256"):
            raise ValueError("frozen definition fingerprint mismatch")
        calibration = {k: v for k, v in bundle["calibration"].items() if k != "training_ids"}
        calibration["n_training_enzymes"] = len(bundle["calibration"].get("training_ids", []))
        return {"status": "success", "summary": "Frozen experimental proxy definition and scope; transfer requires review.",
                "data": {"bundle_id": bundle["bundle_id"], "definition": bundle["definition"],
                         "definition_sha256": checksum, "calibration": calibration, "sources": bundle["sources"]},
                "count": 1, "suggestions": ["Compare the declared candidate mechanism, molecular state, sampling and assay endpoint with this scope before application."]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "error", "summary": str(exc), "data": None, "count": 0, "suggestions": []}
