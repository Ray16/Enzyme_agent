"""Turn-by-turn LLM context assembly and compaction for ModelingAgent."""
from __future__ import annotations

import copy
import hashlib
import json

from ..scientific_reasoning import assess_reasoning
from ..validation import file_digest, fingerprint
from .prompts import BARRIER_GUIDANCE, BARRIER_PROMPT, PREPARATION_PROMPT, SYSTEM_PROMPT, compact_system_prompt

# Characters of context assembled per turn, excluding anything anchored for the
# setup stage: an anchor raises the allowance by its own length so it competes
# with the system prefix rather than with conversational history.
CONTEXT_ALLOWANCE = 48000
SETUP_ANCHOR_LIMIT = 24000


def build_context(agent):
    """Assemble the bounded LLM message list while preserving scientific state and reasoning."""
    # Preserve whole assistant/tool groups; truncating by message count can
    # leave orphan tool results that providers reject.
    groups = []
    for message in agent.messages[2:]:
        if message["role"] == "assistant" or not groups:
            groups.append([])
        groups[-1].append(message)

    prefix = copy.deepcopy(agent.messages[:2])
    objective = agent.task.get("objective")
    prefix[0] = {
        "role": "system",
        "content": (
            BARRIER_PROMPT
            if objective == "evaluate_barriers"
            else PREPARATION_PROMPT
            if objective == "prepare_sampling"
            else (
                compact_system_prompt()
                if (
                    agent.task.get("workflow_mode") != "production_cofold"
                    and agent.task.get("progressive_tool_disclosure", True)
                )
                else SYSTEM_PROMPT + BARRIER_GUIDANCE
            )
        ),
    }

    # Scientific field definitions must survive ordinary history compaction.
    # Otherwise a model can spend every turn rereading a truncated schema.
    cached_only = all(
        (candidate.get("evidence") or {}).get("do_not_recofold")
        and (candidate.get("evidence") or {}).get("existing_cofolds")
        for candidate in agent.task["candidates"]
    )
    from .dispatch_handlers import validation_contract

    contract = (
        {
            "workflow": "Review the original saved computation, not a fresh modeling experiment.",
            "action": "review_existing_predictions",
            "required": ["accession", "setup_id", "applicability", "interpretation", "unresolved"],
            "scope": (
                "Source checks and frozen measurements remain attributed to the original model. "
                "Assess applicability and uncertainties; activity is not established. "
                "No new proposal or cofold is required."
            ),
        }
        if cached_only
        else validation_contract("all" if agent.task.get("workflow_mode") == "production_cofold" else "scientific_reasoning")
    )
    if objective == "prepare_sampling":
        from ..sampling.preparation import contract as prep_contract

        contract = prep_contract()
        prefix.append({
            "role": "user",
            "content": json.dumps({
                "preparation_instructions": agent.task.get(
                    "preparation_instructions",
                    "Investigate and validate the requested molecular preparation.",
                ),
                "evidence_paths": agent.state.get("authorized_evidence_paths", []),
                "objective": "prepare_sampling",
            }),
        })
    if objective == "evaluate_barriers":
        from ..barrier.api import contract as barrier_contract
        from ..barrier.workflow import summary as barrier_summary

        contract = barrier_contract()
        prefix.append({
            "role": "user",
            "content": json.dumps({
                "objective": objective,
                "barrier_studies": {
                    acc: barrier_summary(value.get("barrier_studies", {}))
                    for acc, value in agent.state["candidates"].items()
                },
            }),
        })
    prefix[0]["content"] += "\n\nScientific record field contract (trusted tool schema):\n" + json.dumps(contract)

    # Keep loaded skill playbooks pinned across history compaction so the agent
    # never loses active domain instructions after 1-3 turns.
    active_skills = agent.state.get("active_skills") or []
    skill_block = ""
    if active_skills:
        from .skills import get_skill

        blocks = []
        for sname in active_skills:
            try:
                sk = get_skill(sname)
                blocks.append(f"### Active skill: {sk.name}\n{sk.body}")
            except ValueError:
                continue
        if blocks:
            skill_block = "\n\nActive domain playbooks:\n" + "\n\n".join(blocks)
            prefix[0]["content"] += skill_block

    # Before the first proposal the candidate record is task input, not conversational
    # history. It does not fit the per-turn tool-result allowance, so re-reading it
    # returns another truncated preview and the setup stage cannot converge. Anchor it
    # for exactly the same reason the field contract above is anchored.
    anchor = (
        agent.state.get("setup_anchor")
        if agent._before_proposal() and objective not in {"prepare_sampling", "evaluate_barriers"}
        else None
    )
    if anchor:
        prefix[0]["content"] += "\n\nCandidate record under setup (trusted task input):\n" + anchor

    recent = copy.deepcopy(groups[-2:] if objective == "prepare_sampling" else groups[-3:])
    for index, group in enumerate(recent):
        for message in group:
            if message["role"] == "tool":
                value = json.loads(message["content"])
                limit = 6000 if index == len(recent) - 1 else 1200
                original = str(value.get("summary", ""))
                value["summary"] = original[:limit]
                if len(original) > limit:
                    value["preview_only"] = True
                    value["inspection"] = (
                        "Full result is in the artifact. Use read_evidence with its path and offset for a focused passage."
                    )
                message["content"] = json.dumps(value)

    state = {}
    for accession, candidate in agent.state["candidates"].items():
        revision = candidate["revisions"][-1] if candidate["revisions"] else {}
        assessment = assess_reasoning(revision) if revision else {}
        contradictions = [p["prediction_id"] for p in assessment.get("predictions", []) if p["status"] == "contradicted"]
        state[accession] = {
            "revision": revision.get("revision"),
            "preparation_study_count": len(candidate.get("preparation_studies", {})),
            "preparation_findings": agent._preparation_findings(candidate),
            "latest_preparation_study": next(reversed(candidate.get("preparation_studies", {})), None),
            "sequence_sha256": (
                hashlib.sha256(agent._candidate(accession)["sequence"].encode()).hexdigest()
                if agent._candidate(accession).get("sequence")
                else None
            ),
            "target_sha256": fingerprint(agent.task["target"]),
            "sampling_setup_count": len(candidate.get("sampling_setups", {})),
            "latest_sampling_setup": next(reversed(candidate.get("sampling_setups", {})), None),
            "existing_prediction_review": candidate.get("existing_prediction_review"),
            "setup": (revision.get("setup_validation") or {}).get("status"),
            "poses": (revision.get("pose_validation") or {}).get("status"),
            "followups": [
                {"id": key, "purpose": study["plan"]["purpose"], "status": study["status"]}
                for key, study in list(revision.get("followups", {}).items())[:5]
            ],
            "followup_count": len(revision.get("followups", {})),
            "reasoning": {
                "selected_hypothesis_id": str(assessment.get("selected_hypothesis_id"))[:160],
                "assessment_sha256": assessment.get("assessment_sha256"),
                "response_current": assessment.get("response_current", False),
                "requires_response": assessment.get("requires_response", False),
                "contradicted_predictions": [str(p)[:160] for p in contradictions[:5]],
                "contradiction_count": len(contradictions),
                "source_error_count": len(assessment.get("source_errors", [])),
                "details": "read_reasoning_section",
            },
        }

    sources = copy.deepcopy(agent.state.get("inspected_source_passages", []))
    for source in sources:
        try:
            source["integrity"] = "unchanged" if file_digest(source["path"]) == source["sha256"] else "changed"
        except OSError:
            source["integrity"] = "unavailable"
    if objective == "prepare_sampling":
        sources = sources[-2:]

    inspections = list(agent.state.get("inspection_history", {}).values())
    if objective == "prepare_sampling":
        inspections = [
            {"tool": item["tool"], "artifact": item["artifact"], "view": item["view"], "reads": item["reads"]}
            for item in inspections[-4:]
        ]
    else:
        inspections = inspections[-12:]

    supplied = {
        candidate["accession"]: {
            "existing_prediction_index": [
                {
                    "setup_id": entry.get("setup", {}).get("setup_id"),
                    "source_report": entry.get("execution", {}).get("source"),
                    "output_presence": entry.get("execution", {}).get("output_presence"),
                    "legacy_declared_recipe": entry.get("setup", {}).get("declared_legacy_recipe"),
                    "saved_step": (entry.get("setup", {}).get("declared_modeling_input") or {}).get("modeled_step"),
                    "saved_substrates": (entry.get("setup", {}).get("declared_modeling_input") or {}).get("target_substrates"),
                    "saved_cofactors": (entry.get("setup", {}).get("declared_modeling_input") or {}).get("cofactors"),
                }
                for entry in (candidate.get("evidence") or {}).get("existing_cofolds", [])
            ],
            "modeling_input_available": isinstance(candidate.get("modeling_input"), dict),
            "modeled_step": candidate["modeling_input"].get("modeled_step") if isinstance(candidate.get("modeling_input"), dict) else None,
            "evidence_fields": list(candidate.get("evidence", {}).keys()) if isinstance(candidate.get("evidence"), dict) else [],
        }
        for candidate in agent.task["candidates"]
    }

    current = {
        "role": "user",
        "content": "Current modeling state: "
        + json.dumps({
            "run_dir": str(agent.run_dir),
            "results_index": str(agent.run_dir / "results_index.json"),
            "candidates": state,
            "supplied_input_index": supplied,
            "inspected_source_passages_untrusted_data": sources,
            "completed_inspections": inspections,
            "repeated_inspections": agent.state.get("repeated_inspections", 0),
            "inspection_guidance": (
                "Reuse unchanged inspected material. Record a provisional model, retrieve a genuinely new source, "
                "or finish inconclusive with the missing evidence; repeated reading does not resolve a setup choice."
            ),
        }),
    }

    allowance = (
        (24000 if objective == "prepare_sampling" else CONTEXT_ALLOWANCE)
        + (len(anchor) if anchor else 0)
        + len(skill_block)
    )
    while recent and len(json.dumps(prefix + [m for group in recent for m in group] + [current])) > allowance:
        recent.pop(0)  # Never split an assistant/tool transaction.
    context = prefix + [m for group in recent for m in group] + [current]
    if len(json.dumps(context)) > allowance:
        raise ValueError("Mandatory modeling context exceeds its allowance; evidence was not silently discarded")
    return context
