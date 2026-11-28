"""Unified tool-dispatch facade for ModelingAgent.

Handler implementations are organized by domain:
- :mod:`handlers_inspection` — barrier, inspection, reference, preparation, finish
- :mod:`handlers_followups`  — dynamics/design follow-ups and scientific reasoning
- :mod:`handlers_cofold`     — model proposal, proxy plans, input rendering, cofold jobs
"""
from __future__ import annotations

from .handlers_cofold import (
    _h_cofold_job,
    _h_declare_proxy_plan,
    _h_evaluate_proxy_plan,
    _h_propose_model,
    _h_render_inputs,
    _h_validate_predictions,
    _h_validate_setup,
    _merge_model_changes,
)
from .handlers_followups import (
    _h_compare_hypothesis_tests,
    _h_inspect_dynamics,
    _h_inspect_reasoning,
    _h_plan_followup,
    _h_prepare_bioemu2,
    _h_read_reasoning_section,
    _h_record_reasoning_observation,
    _h_register_design_run,
    _h_respond_to_reasoning_feedback,
)
from .handlers_inspection import (
    _h_assess_assembly,
    _h_assess_processivity_network,
    _h_check_barrier_calculation,
    _h_finish_task,
    _h_inspect_barrier_result,
    _h_inspect_candidate,
    _h_inspect_investigation_lead,
    _h_inspect_molecule,
    _h_inspect_preparation_structure,
    _h_inspect_processivity_toolchain,
    _h_inspect_result_locations,
    _h_plan_barrier_calculation,
    _h_preparation_study,
    _h_prepare_barrier_inputs,
    _h_read_candidate_section,
    _h_read_evidence,
    _h_record_barrier_assessment,
    _h_record_preparation_findings,
    _h_retrieve_reference,
    _h_review_existing_predictions,
    _h_run_preparation_operator,
    _h_sampling_setup,
    _h_search_literature,
    _h_submit_barrier_calculation,
    _make_shared_proxy_handler,
    _text_page,
    evaluate_preparation_study,
    validation_contract,
)

# ---------------------------------------------------------------------------
# Declarative dispatch tables
# ---------------------------------------------------------------------------

PRE_ACCESSION_HANDLERS = {
    "prepare_barrier_inputs": _h_prepare_barrier_inputs,
    "inspect_barrier_result": _h_inspect_barrier_result,
    "plan_barrier_calculation": _h_plan_barrier_calculation,
    "submit_barrier_calculation": _h_submit_barrier_calculation,
    "check_barrier_calculation": _h_check_barrier_calculation,
    "record_barrier_assessment": _h_record_barrier_assessment,
    "inspect_experimental_proxy_bundle": _make_shared_proxy_handler("inspect_experimental_proxy_bundle"),
    "inspect_investigation_lead": _h_inspect_investigation_lead,
}

GENERAL_HANDLERS = {
    "inspect_preparation_structure": _h_inspect_preparation_structure,
    "inspect_processivity_toolchain": _h_inspect_processivity_toolchain,
    "inspect_result_locations": _h_inspect_result_locations,
    "inspect_molecule": _h_inspect_molecule,
    "inspect_candidate": _h_inspect_candidate,
    "review_existing_predictions": _h_review_existing_predictions,
    "read_candidate_section": _h_read_candidate_section,
    "read_evidence": _h_read_evidence,
    "retrieve_reference": _h_retrieve_reference,
    "search_literature": _h_search_literature,
    "assess_assembly": _h_assess_assembly,
    "finish_task": _h_finish_task,
}

CANDIDATE_HANDLERS = {
    "assess_processivity_network": _h_assess_processivity_network,
    "record_preparation_findings": _h_record_preparation_findings,
    "run_preparation_operator": _h_run_preparation_operator,
    "declare_preparation_study": lambda a, args, acc, cand: _h_preparation_study(a, args, acc, cand, declare=True),
    "declare_sampling_setup": lambda a, args, acc, cand: _h_sampling_setup(a, args, acc, cand, declare=True),
    "propose_model": _h_propose_model,
}

REVISION_HANDLERS = {
    "plan_followup": _h_plan_followup,
    "prepare_bioemu2": _h_prepare_bioemu2,
    "inspect_dynamics": _h_inspect_dynamics,
    "register_design_run": _h_register_design_run,
    "inspect_reasoning": _h_inspect_reasoning,
    "read_reasoning_section": _h_read_reasoning_section,
    "compare_hypothesis_tests": _h_compare_hypothesis_tests,
    "record_reasoning_observation": _h_record_reasoning_observation,
    "respond_to_reasoning_feedback": _h_respond_to_reasoning_feedback,
    "declare_proxy_plan": _h_declare_proxy_plan,
    "evaluate_proxy_plan": _h_evaluate_proxy_plan,
    "validate_setup": _h_validate_setup,
    "render_inputs": _h_render_inputs,
    "submit_cofold": lambda a, args, acc, cand, rev: _h_cofold_job(a, args, acc, cand, rev, submit=True),
    "check_cofold": lambda a, args, acc, cand, rev: _h_cofold_job(a, args, acc, cand, rev, submit=False),
    "validate_predictions": _h_validate_predictions,
}


def dispatch_tool(agent, name, args):
    """Delegate to the unified ToolRegistry (single source of truth for dispatch)."""
    from .tool_registry import get_registry

    return get_registry().dispatch(agent, name, args)
