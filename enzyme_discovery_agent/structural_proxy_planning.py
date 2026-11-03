"""Reaction-independent catalog access and pre-label proxy-plan validation."""
from __future__ import annotations

from importlib.resources import files
import json
import math


def load_proxy_catalog() -> dict:
    return json.loads(files("enzyme_discovery_agent").joinpath(
        "structural_proxy_catalog.json").read_text())


ASSAY_CONTEXT_CHOICES = {
    "substrate_state": {"soluble", "insoluble_polymer", "surface_bound", "membrane", "unknown"},
    "measurement_basis": {"crude_lysate", "whole_cell", "enzyme_normalized", "active_enzyme_normalized", "unknown"},
    "modeled_state": {"precursor", "mature", "substrate_bound", "product_bound", "intermediate", "unknown"},
    "assembly_status": {"established", "uncertain", "incomplete", "unknown"},
}


def validate_assay_context(context: dict | None) -> list[str]:
    """Check explicit assay/state declarations; never infer them from names."""
    if context is None:
        return []
    if not isinstance(context, dict):
        return ["assay_context must be an object"]
    errors = []
    for key, choices in ASSAY_CONTEXT_CHOICES.items():
        if key in context and (not isinstance(context[key], str) or context[key] not in choices):
            errors.append(f"assay_context.{key} must be one of {sorted(choices)}")
    for key in ("requires_processing", "substrate_is_assayed"):
        if key in context and not isinstance(context[key], bool):
            errors.append(f"assay_context.{key} must be boolean")
    processes = {p["id"] for p in load_proxy_catalog().get("physical_processes", [])}
    if "limiting_processes" in context:
        selected = context["limiting_processes"]
        if (not isinstance(selected, list) or not all(isinstance(v, str) and v in processes for v in selected)
                or len(set(selected)) != len(selected)):
            errors.append("assay_context.limiting_processes must list distinct declared physical process IDs")
    for key in ("substrate_representation", "state_evidence", "assembly_evidence"):
        if key in context and (not isinstance(context[key], str) or not context[key].strip()):
            errors.append(f"assay_context.{key} must be a nonempty description")
    for key in ("temperature_C", "pH", "duration_hours"):
        if key not in context:
            continue
        value = context[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"assay_context.{key} must be a finite number")
        elif key == "pH" and not 0 <= value <= 14:
            errors.append("assay_context.pH must be between 0 and 14")
        elif key == "duration_hours" and value <= 0:
            errors.append("assay_context.duration_hours must be positive")
    return errors


def validate_search_design(design: dict | None) -> list[str]:
    """Validate an optional exploration declaration, not its scientific success."""
    if design is None:
        return []
    if not isinstance(design, dict):
        return ["search_design must be an object"]
    errors = []
    choices = {
        "mode": {"discovery", "confirmatory"},
        "objective": {"high_producer_recovery", "activity_discrimination", "quantitative_activity"},
        "biological_unit": {"enzyme", "enzyme_construct"},
        "validation": {"nested_enzyme_heldout", "external_heldout", "descriptive_only"},
        "selection_correction": {"repeat_full_search", "descriptive_only"},
        "combination_policy": {"decomposed_only", "physical_pairs"},
    }
    for key, allowed in choices.items():
        if key in design and (not isinstance(design[key], str) or design[key] not in allowed):
            errors.append(f"search_design.{key} must be one of {sorted(allowed)}")
    if "budgets" in design:
        values = design["budgets"]
        if (not isinstance(values, list) or not values or
                not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values)
                or len(set(values)) != len(values)):
            errors.append("search_design.budgets must list distinct positive integer candidate counts")
    if "record_negative_results" in design and design["record_negative_results"] is not True:
        errors.append("search_design.record_negative_results must be true when declared")
    if design.get("combination_policy") == "physical_pairs":
        rationale = design.get("combination_rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append("search_design.physical_pairs requires a combination_rationale describing the physical coupling")
    qc = design.get("qc_policy")
    if qc is not None:
        if not isinstance(qc, dict):
            errors.append("search_design.qc_policy must be an object")
        else:
            if "minimum_valid_poses" in qc and (not isinstance(qc["minimum_valid_poses"], int)
                    or isinstance(qc["minimum_valid_poses"], bool) or qc["minimum_valid_poses"] < 1):
                errors.append("qc_policy.minimum_valid_poses must be a positive integer")
            if "require_all_poses" in qc and not isinstance(qc["require_all_poses"], bool):
                errors.append("qc_policy.require_all_poses must be boolean")
            sensitivity = qc.get("sensitivity_analysis")
            if qc.get("require_all_poses") is True and (not isinstance(sensitivity, str) or not sensitivity.strip()):
                errors.append("qc_policy requiring all poses needs a nonempty sensitivity_analysis description")
    return errors


def validate_proxy_plan(plan: dict) -> list[str]:
    """Validate declared hypotheses, without using any experimental labels.

    Missing execution inputs remain reportable requirements rather than being
    guessed. A plan may include targeted follow-ups which the shared runner will
    report as requiring a separate extractor.
    """
    errors = []
    if not isinstance(plan, dict):
        return ["plan must be an object"]
    if plan.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    errors.extend(validate_assay_context(plan.get("assay_context")))
    errors.extend(validate_search_design(plan.get("search_design")))
    for key in ("mechanism", "reaction_step", "assay_endpoint"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            errors.append(f"{key} must be an explicit nonempty description")
    context = plan.get("context")
    if not isinstance(context, dict):
        errors.append("context must be an object")
        context = {}
    catalog = {d["id"]: d for d in load_proxy_catalog()["descriptors"]}
    hypotheses = plan.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        errors.append("hypotheses must be a nonempty list")
        hypotheses = []
    seen = set()
    for h in hypotheses:
        if not isinstance(h, dict):
            errors.append("each hypothesis must be an object")
            continue
        pid = h.get("proxy_id")
        if pid not in catalog:
            errors.append(f"unknown proxy_id: {pid}")
            continue
        if pid in seen:
            errors.append(f"duplicate proxy hypothesis: {pid}")
        seen.add(pid)
        if not isinstance(h.get("rationale"), str) or not h["rationale"].strip():
            errors.append(f"{pid}: specify a mechanistic or baseline rationale")
        if not isinstance(h.get("evidence_ids"), list) or not all(
                isinstance(v, str) and v.strip() for v in h["evidence_ids"]):
            errors.append(f"{pid}: evidence_ids must be a list of source/inference identifiers")
        direction = h.get("expected_direction")
        if direction not in ("+", "-", None):
            errors.append(f"{pid}: expected_direction must be +, -, or null")
        if "expected_direction" not in h:
            errors.append(f"{pid}: explicitly declare expected_direction")
        if catalog[pid]["category"] != "prediction_confidence" and direction is None:
            errors.append(f"{pid}: declare a direction before activity screening")
        params = h.get("parameters", {})
        if not isinstance(params, dict):
            errors.append(f"{pid}: parameters must be an object")
    for key in ("reactive_group_atoms", "distal_group_atoms", "catalytic_base_atoms",
                "catalytic_donor_atoms", "charges"):
        group = context.get(key, [])
        if not isinstance(group, list):
            errors.append(f"{key} must be a list of exact atom selectors")
            continue
        seen_atoms = set()
        for atom in group:
            if not isinstance(atom, dict) or not all(k in atom for k in ("chain", "residue", "atom")):
                errors.append(f"{key}: selectors require chain, residue, atom")
                continue
            identity = (str(atom["chain"]), str(atom["residue"]), str(atom["atom"]))
            if identity in seen_atoms:
                errors.append(f"{key}: duplicate selector {identity}")
            seen_atoms.add(identity)
            if key == "charges":
                try:
                    value = float(atom["charge"])
                    if not math.isfinite(value):
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    errors.append("charges: each selected atom requires a finite charge")
    if context.get("charges") and not all(context.get(k) for k in (
            "charge_model", "charge_provenance", "unlisted_atom_policy")):
        errors.append("charges require charge_model, charge_provenance and unlisted_atom_policy")
    if context.get("unlisted_atom_policy") not in (None, "zero", "error"):
        errors.append("unlisted_atom_policy must be zero or error")
    for key in ("substrate_chains", "protein_chains", "sasa_occluder_chains"):
        if key in context and (not isinstance(context[key], list) or not context[key]
                               or not all(isinstance(x, str) and x for x in context[key])
                               or len(set(context[key])) != len(context[key])):
            errors.append(f"{key} must list distinct nonempty chain identifiers")
    if "confidence_chain_map" in context and not isinstance(context["confidence_chain_map"], dict):
        errors.append("confidence_chain_map must map PDB chain IDs to confidence indices")
    if "confidence_pairs" in context:
        pairs = context["confidence_pairs"]
        if not isinstance(pairs, list) or not all(isinstance(p, dict) and
                isinstance(p.get("protein_chain"), str) and isinstance(p.get("substrate_chain"), str)
                for p in pairs):
            errors.append("confidence_pairs require protein_chain and substrate_chain")
    if "field_axis" in context and (not isinstance(context["field_axis"], list)
                                    or len(context["field_axis"]) != 2):
        errors.append("field_axis must name two reactive roles")
    return errors


def proxy_guidance(mechanism: str, reaction_step: str, assay_endpoint: str,
                   available_inputs: list[str], assay_context: dict | None = None) -> dict:
    errors = validate_assay_context(assay_context)
    if errors:
        raise ValueError("; ".join(errors))
    catalog = load_proxy_catalog()
    available = set(available_inputs)
    declared = dict(assay_context or {})
    first = {p: ["Explicitly declared candidate limiting process; not an established cause"]
             for p in declared.get("limiting_processes", [])}

    def consider(process: str, reason: str):
        first.setdefault(process, []).append(reason)

    if declared.get("substrate_state") in ("insoluble_polymer", "surface_bound", "membrane"):
        consider("substrate_access", "Declared substrate phase makes physical presentation/access worth testing")
        consider("surface_adsorption", "Declared material/interface makes productive surface engagement worth testing")
    if declared.get("requires_processing"):
        consider("maturation", "Active-state processing is explicitly required")
    if declared.get("assembly_status") in ("uncertain", "incomplete"):
        consider("assembly", "The functional assembly or its modeled completeness is explicitly unresolved")
    if any(key in declared for key in ("temperature_C", "pH", "duration_hours")):
        consider("stability", "Declared assay conditions motivate testing persistence of the active state")
    process_map = [{**p, "priority": "consider_first" if p["id"] in first else "consider",
                    "priority_reasons": first.get(p["id"], []),
                    "evidence_status": "physical_hypothesis"}
                   for p in catalog.get("physical_processes", [])]
    process_map.sort(key=lambda p: p["priority"] != "consider_first")
    descriptors = []
    for entry in catalog["descriptors"]:
        missing = sorted(set(entry["requirements"]) - available)
        related = [p["id"] for p in process_map if entry["id"] in p["candidate_descriptors"]]
        descriptors.append({**entry, "missing_requirements": missing,
                            "ready_from_declared_inputs": not missing and entry["execution"] == "builtin",
                            "physical_processes": related,
                            "process_priority": "consider_first" if any(p in first for p in related) else "consider"})
    gaps = []
    if declared.get("requires_processing") and declared.get("modeled_state") != "precursor":
        gaps.append({"process": "maturation", "reason": "The modeled state does not represent the required precursor processing step",
                     "followup": "Annotate/model the precursor or measure processed fraction; mature-state confidence cannot establish processing rate"})
    if declared.get("assembly_status") in ("uncertain", "incomplete"):
        gaps.append({"process": "assembly", "reason": "The complete functional assembly is not established in the declaration",
                     "followup": "Identify site-contributing partners and compare assemblies with mapping/coverage uncertainty retained"})
    if declared.get("substrate_is_assayed") is False:
        gaps.append({"process": "substrate_access", "reason": "The modeled ligand is explicitly a surrogate for the assayed substrate",
                     "followup": "Separate surrogate binding/chemistry from access or adsorption to the assayed material"})
    if "product_release" in first and declared.get("modeled_state") != "product_bound":
        gaps.append({"process": "product_release", "reason": "No product-bound state is declared",
                     "followup": "Resolve the product and its exit pathway; substrate binding alone does not measure release"})
    if declared.get("measurement_basis") in ("crude_lysate", "whole_cell"):
        gaps.append({"process": "active_enzyme_amount", "reason": "The endpoint may combine abundance,folded/processed fraction,assembly,persistence and chemistry",
                     "followup": "Record assay normalization and distinguish intrinsic mature-enzyme activity from whole-preparation productivity"})
    elif declared.get("measurement_basis") == "enzyme_normalized":
        gaps.append({"process": "active_enzyme_amount", "reason": "Total-enzyme normalization does not establish the active mature fraction",
                     "followup": "Retain processing,folded fraction,assembly and lifetime uncertainty unless measured"})
    return {"schema_version": "1.0", "mechanism": mechanism,
            "reaction_step": reaction_step, "assay_endpoint": assay_endpoint,
            "assay_context": declared, "physical_process_map": process_map,
            "modeled_state_gaps": gaps,
            "unresolved_context": [k for k in ("substrate_state", "measurement_basis", "modeled_state", "assembly_status")
                                   if declared.get(k, "unknown") == "unknown"],
            "principles": catalog["principles"], "descriptors": descriptors,
            "plan_schema": catalog["plan_schema"],
            "exploration_protocol": {
                "choose_process_before_descriptor": True,
                "objective": "Declare shortlist recovery,activity discrimination or quantitative prediction before comparing results",
                "physical_definition": "Record formula,units,roles,molecular state,aggregation,parameters,direction and evidence versus inference",
                "discovery_mode": "Record the frozen candidate feature/parameter/combination space and every failed hypothesis; distinguish retrospective adaptation from prospective declarations",
                "combinations": "Test small physically coupled pairs with component evidence retained; fit/select/transform inside training folds and repeat selection under the null",
                "validation": "Use enzyme-construct heldout validation,confidence/length/composition controls and independent assays for transfer; poses/sites are not biological replicates",
                "qc": "Require atom/sequence integrity,report usable-pose denominators and test assembly/quality-gate sensitivity; no universal all-pose or confidence cutoff",
                "parallelism": "Partition independent families/enzymes over capacity-checked disjoint CPU/GPU reservations with native thread caps and worker verification",
                "artifacts": "Keep feature catalog,exact extraction scripts/commands,source hashes,search protocol,coverage,all outcomes and validation scope together",
            },
            "selection_note": "Readiness is not relevance or validation. Priorities identify hypotheses worth investigating, not proven rate-limiting steps or favorable directions. Use explicit assay context; do not infer it from a family name."}
