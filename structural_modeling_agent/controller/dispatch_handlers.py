"""Modular tool-dispatch handlers for ModelingAgent.

Each handler receives ``(agent, args)`` (or ``(agent, args, acc, candidate)`` /
``(agent, args, acc, candidate, revision)`` depending on its scope) and enforces
the exact scientific, provenance, and resource gates previously inlined in
``ModelingAgent.dispatch()``.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

from enzyme_discovery_agent.state_store import atomic_write_json
from ..scientific_reasoning import (
    assess_reasoning,
    compare_hypothesis_tests,
    execution_reasoning_errors,
    validate_observation,
    validate_proxy_binding,
    validate_reasoning,
    validate_test,
)
from ..validation import (
    entity_map_for,
    file_digest,
    fingerprint,
    validate_ensemble,
    validate_msa,
    validate_pose_plan,
    validate_setup,
)


def _text_page(content, offset, **metadata):
    """Return a complete JSON page small enough for the newest context preview."""
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")

    def page(size):
        end = min(offset + size, len(content))
        return {
            **metadata,
            "offset": offset,
            "characters": len(content),
            "next_offset": end if end < len(content) else None,
            "text": content[offset:end],
        }

    low, high = 0, min(3000, max(0, len(content) - offset))
    if len(json.dumps(page(0))) > 4800:
        raise ValueError("page metadata exceeds the inspection allowance")
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(page(middle))) <= 4800:
            low = middle
        else:
            high = middle - 1
    return page(low)


def _merge_model_changes(base, changes):
    """Apply an explicit JSON merge patch to a copied modeling proposal."""
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict):
            result[key] = _merge_model_changes(
                result.get(key) if isinstance(result.get(key), dict) else {}, value
            )
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Pre-accession handlers (barrier & shared discovery proxy tools)
# ---------------------------------------------------------------------------

def _h_inspect_barrier_contract(agent, args):
    from ..barrier.api import contract
    from ..barrier.workflow import backend_config

    result = contract()
    try:
        result["configured_backend"] = backend_config(agent)
    except ValueError as exc:
        result["backend_gap"] = str(exc)
    return result


def _h_prepare_barrier_inputs(agent, args):
    from ..barrier.api import prepare_for_agent
    return prepare_for_agent(agent, args)


def _h_inspect_barrier_result(agent, args):
    from ..barrier.api import review_result
    return review_result(agent._authorized_evidence(args["path"]))


def _h_plan_barrier_calculation(agent, args):
    from ..barrier.workflow import plan
    return plan(agent, args)


def _h_submit_barrier_calculation(agent, args):
    from ..barrier.workflow import reconcile
    return reconcile(agent, args["accession"], args["study_id"], "submit")


def _h_check_barrier_calculation(agent, args):
    from ..barrier.workflow import reconcile
    return reconcile(agent, args["accession"], args["study_id"], "status")


def _h_record_barrier_assessment(agent, args):
    from ..barrier.workflow import assess
    return assess(agent, args)


def _make_shared_proxy_handler(tool_name):
    def _handler(agent, args):
        from enzyme_discovery_agent.tools import structural_proxy_planning
        if tool_name == "inspect_experimental_proxy_bundle":
            args = {**args, "bundle_path": str(agent._authorized_evidence(args["bundle_path"]))}
        return getattr(structural_proxy_planning, tool_name)(**args)
    return _handler


# ---------------------------------------------------------------------------
# General / contract / inspection handlers (after accession defaulting)
# ---------------------------------------------------------------------------

def _h_inspect_validation_contract(agent, args):
    contract = json.loads((Path(__file__).resolve().parents[1] / "validation_contract.json").read_text())
    section = args.get("section", "all")
    if section == "all":
        return contract
    if section == "scientific_reasoning":
        return {k: v for k, v in contract["scientific_reasoning"].items() if k != "execution_rules"}
    if section == "reasoning_rules":
        return contract["scientific_reasoning"]["execution_rules"]
    if section in {"setup", "poses"}:
        return contract["validation_plan"][section]
    if section in {"source_record", "proposal"}:
        return contract[section]
    raise ValueError("unknown contract section")


def _h_inspect_mlmm_engine_options(agent, args):
    from importlib.metadata import PackageNotFoundError, version

    result = json.loads((Path(__file__).resolve().parents[1] / "knowledge" / "mlmm_engine_options.json").read_text())
    try:
        current = version("fairchem-core")
    except PackageNotFoundError:
        current = None
    return {
        **result,
        "controller_environment_fairchem_version": current,
        "execution_environment": "External engine environments must be inspected independently; this tool does not test runtime capability.",
    }


def _h_inspect_preparation_contract(agent, args):
    from ..sampling.preparation import contract
    return contract()


def _h_inspect_preparation_structure(agent, args):
    from ..sampling.preparation import inspect_pdb
    result = inspect_pdb(agent._authorized_evidence(args["path"]))
    return {"artifact": agent._artifact("preparation_structure", result), **result}


def _h_inspect_sampling_setup_contract(agent, args):
    from ..sampling.setup import contract
    return contract()


def _h_inspect_processivity_toolchain(agent, args):
    from ..sampling.toolchain import inspect_toolchain
    return inspect_toolchain()


def _h_inspect_processivity_contract(agent, args):
    from ..sampling.kinetics import contract
    return contract()


def _h_inspect_result_locations(agent, args):
    from ..result_layout import results_index
    index = results_index(agent.run_dir, agent.state)
    accession = args.get("accession")
    if accession is not None:
        agent._candidate(accession)
        index["candidates"] = {accession: index["candidates"][accession]}
    return index


def _h_inspect_molecule(agent, args):
    from ..modeling_reasoning.chemistry import characterize_smiles, resolve_ligand_atom
    result = characterize_smiles(args["smiles"])
    if args.get("smarts"):
        result["reactive_site_matches"] = resolve_ligand_atom(args["smiles"], args["smarts"])
    return result


def _h_inspect_candidate(agent, args):
    acc = args["accession"]
    return {
        "candidate": agent._candidate(acc),
        "modeling_state": agent.state["candidates"][acc],
        "evidence_sources": agent.state.get("bundle_evidence", []),
    }


def _h_review_existing_predictions(agent, args):
    from .reuse import review_existing
    return review_existing(agent, **args)


def _h_read_candidate_section(agent, args):
    candidate = agent._candidate(args["accession"])
    section = args.get("section")
    if section not in {"identity", "evidence", "modeling_input", "selection"}:
        raise ValueError("unknown candidate section")
    value = candidate.get(section)
    pointer = args.get("pointer", "")
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("pointer must be empty or a JSON pointer starting with /")
    for part in pointer.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        try:
            value = value[int(key)] if isinstance(value, list) and key.isdigit() else value[key]
        except (KeyError, IndexError, TypeError):
            raise ValueError("candidate section pointer does not exist") from None
    content = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return _text_page(content, args.get("offset", 0), section=section, pointer=pointer)


def _h_inspect_investigation_lead(agent, args):
    from .handoff_inspect import lead_page
    return lead_page(agent.task, args["accession"], args.get("offset", 0))


def _h_read_evidence(agent, args):
    path = agent._authorized_evidence(args["path"])
    offset = args.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("evidence exceeds the 16 MiB inspection limit")
    text = path.read_text()
    page = _text_page(text, offset, path=str(path.resolve()), sha256=file_digest(path))
    if not path.is_relative_to((agent.run_dir / "artifacts").resolve()):
        notes = agent.state.setdefault("inspected_source_passages", [])
        note = {k: page[k] for k in ("path", "sha256", "offset")}
        note["text"] = page["text"][:1000]
        note["next_offset"] = offset + len(note["text"]) if offset + len(note["text"]) < len(text) else None
        notes[:] = [old for old in notes if (old["path"], old["offset"]) != (note["path"], note["offset"])]
        notes.append(note)
        del notes[:-4]
    return page


def _h_retrieve_reference(agent, args):
    return agent._retrieve_reference(**args)


def _h_search_literature(agent, args):
    from enzyme_discovery_agent.tools.literature import search_europepmc
    return search_europepmc(args["query"], max_results=8)


def _h_assess_assembly(agent, args):
    from ..modeling_reasoning.modeling_unit import assess, gather_evidence
    identifier = args.get("uniprot_accession", args["accession"])
    if not re.fullmatch(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})", identifier):
        return {
            "status": "needs_identity_mapping",
            "accession": args["accession"],
            "next_action": "Resolve an exact-sequence UniProt mapping or investigate assembly from explicitly identified experimental structures; no monomer assumption is made.",
        }
    evidence, facts = gather_evidence(
        identifier,
        catalytic_position=args.get("catalytic_position"),
        cache_dir=str(agent.run_dir / "reference_structures"),
    )
    if facts.get("sequence") != agent._candidate(args["accession"])["sequence"]:
        return {
            "status": "sequence_mismatch",
            "accession": args["accession"],
            "mapped_accession": identifier,
            "next_action": "Resolve mapping/construct differences before transferring assembly evidence.",
        }
    return {
        "evidence": asdict(evidence),
        "facts": facts,
        "provisional_assessment": assess(evidence, args["ligand_heavy_atoms"]),
    }


def _h_finish_task(agent, args):
    if not args.get("rationale", "").strip():
        raise ValueError("a grounded interpretation is required")
    if args["outcome"] == "complete":
        missing = agent.completion_errors()
        if missing:
            return {"status": "blocked", "missing": missing}
    elif args["outcome"] != "inconclusive":
        raise ValueError("unknown finish outcome")
    if any(
        r.get("job", {}).get("status") in {"submission_pending", "queued", "running", "unknown"}
        for c in agent.state["candidates"].values()
        for r in c["revisions"]
    ):
        agent.state.update(status="waiting_for_backend", conclusion=args["rationale"])
        return {"status": "waiting_for_backend", "reason": "job ownership must be reconciled on resume"}
    from ..barrier.workflow import ACTIVE
    if any(
        s.get("job", {}).get("status") in ACTIVE
        for c in agent.state["candidates"].values()
        for s in c.get("barrier_studies", {}).values()
    ):
        agent.state.update(status="waiting_for_backend", conclusion=args["rationale"])
        return {"status": "waiting_for_backend", "reason": "Barrier job ownership must be reconciled on resume"}
    agent.state.update(status=args["outcome"], conclusion=args["rationale"])
    return {"status": agent.state["status"], "rationale": args["rationale"]}


# ---------------------------------------------------------------------------
# Candidate-scoped handlers (require accession & candidate, no current revision)
# ---------------------------------------------------------------------------

def _h_assess_processivity_network(agent, args, acc, candidate):
    from ..sampling.kinetics import evaluate
    network = copy.deepcopy(args["network"])
    if (
        network.get("sequence_sha256") != hashlib.sha256(candidate["sequence"].encode()).hexdigest()
        or network.get("target_sha256") != fingerprint(agent.task["target"])
    ):
        raise ValueError("Kinetic network must match the exact candidate sequence and requested target")
    for transition in network.get("transitions", []):
        agent._authorize_reasoning_source(transition.get("source", {}))
    result = evaluate(network)
    artifact = agent._artifact(f"kinetic_network_{acc}", {"network": network, "assessment": result})
    agent.state["candidates"][acc].setdefault("kinetic_networks", {})[result["network_sha256"]] = {
        "artifact": artifact,
        "status": result["status"],
        "processivity_demonstrated": False,
    }
    return {**result, "artifact": artifact}


def _h_inspect_workflow_policy(agent, args, acc, candidate):
    from .workflow_policy import TOOL_PURPOSES, baseline_errors
    revisions = agent.state["candidates"][acc]["revisions"]
    errors = baseline_errors(revisions[-1]) if revisions else ["no primary cofold decision yet"]
    return {
        "default": "cofolding",
        "tool_purposes": TOOL_PURPOSES,
        "cofold_baseline_errors": errors,
        "followups_available": not errors,
    }


def _h_record_preparation_findings(agent, args, acc, candidate):
    summary = args["summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 6000 or not args["next_action"].strip():
        raise ValueError("Provide compact findings and a concrete next action")
    for source in args["sources"]:
        path = agent._authorized_evidence(source["path"])
        if not source.get("sha256"):
            source["sha256"] = file_digest(path)
    note = {"summary": summary, "sources": copy.deepcopy(args["sources"]), "next_action": args["next_action"]}
    state = agent.state["candidates"][acc]
    previous = state.get("preparation_findings", {})
    if all(previous.get(key) == note[key] for key in note):
        return {"status": "unchanged", **previous}
    note["artifact"] = agent._artifact(f"preparation_findings_{acc}", note)
    state.setdefault("preparation_findings_history", []).append(note["artifact"])
    state["preparation_findings"] = note
    return note


def _h_run_preparation_operator(agent, args, acc, candidate):
    from ..sampling.preparation import run_operator
    folder = agent.run_dir / "artifacts" / f"preparation_{acc}_{uuid.uuid4().hex}"
    result = run_operator(
        folder, args["python"], args["code"], args["inputs"],
        args["outputs"], args["seconds"], agent._authorized_evidence,
    )
    agent.state["candidates"][acc].setdefault("preparation_operators", []).append(result)
    return result


def _h_preparation_study(agent, args, acc, candidate, *, declare: bool):
    from ..sampling.preparation import evaluate
    state = agent.state["candidates"][acc]
    studies = state.setdefault("preparation_studies", {})
    if declare:
        if not isinstance(args.get("study"), dict):
            raise ValueError("declare_preparation_study requires a 'study' object; inspect_preparation_contract shows its shape")
        study = copy.deepcopy(args["study"])
    else:
        if not studies:
            raise ValueError("No preparation study declared")
        study = copy.deepcopy(next(reversed(studies.values()))["study"])
    result = evaluate(
        study, state.get("sampling_setups", {}), candidate["sequence"],
        agent.task["target"], agent._authorized_evidence,
    )
    revisions = state["revisions"]
    basis = fingerprint(revisions[-1]["modeling_input"]) if revisions else None
    for variant in result["variants"]:
        if state["sampling_setups"][variant["setup_id"]]["basis_model_sha256"] != basis:
            result["status"] = "needs_preparation"
            variant["ready"] = False
            variant["gaps"].append({"stage": "setup", "reason": "Model context changed; redeclare setup applicability"})
    identity = fingerprint(study)[:16]
    if declare and identity not in studies:
        artifact = agent._artifact(f"preparation_{acc}", {"study": study, "assessment": result})
        studies[identity] = {"study": study, "artifact": artifact}
    return {"id": identity, "artifact": studies[identity]["artifact"], **result}


def _h_sampling_setup(agent, args, acc, candidate, *, declare: bool):
    from ..sampling.setup import assess
    from ..result_layout import revision_dir

    state = agent.state["candidates"][acc]
    revisions = state["revisions"]
    basis = fingerprint(revisions[-1]["modeling_input"]) if revisions else None
    setups = state.setdefault("sampling_setups", {})
    if declare:
        if not isinstance(args.get("setup"), dict):
            raise ValueError("declare_sampling_setup requires a 'setup' object; inspect_sampling_setup_contract shows its shape")
        setup = copy.deepcopy(args["setup"])
        identities = {
            "sequence_sha256": hashlib.sha256(candidate["sequence"].encode()).hexdigest(),
            "target_sha256": fingerprint(agent.task["target"]),
        }
        for key, expected in identities.items():
            if key in setup and setup[key] != expected:
                raise ValueError(f"{key} mismatch; use {expected} or omit this bookkeeping field for automatic binding")
            setup[key] = expected
        for source in setup.get("sources", []):
            agent._authorize_reasoning_source(source)
        assessment = assess(setup, candidate["sequence"], agent.task["target"])
        identity = fingerprint({"setup": setup, "basis_model_sha256": basis})[:16]
        if identity in setups:
            return {"id": identity, "artifact": setups[identity]["artifact"], **assessment}
        record = {
            "id": identity,
            "setup": setup,
            "assessment": assessment,
            "basis_model_sha256": basis,
            "revision": len(revisions) or None,
        }
        destination = revision_dir(agent.run_dir, acc, max(1, len(revisions))) / "sampling_setups" / identity
        destination.mkdir(parents=True, exist_ok=True)
        record["artifact"] = str(destination / "decision.json")
        atomic_write_json(record["artifact"], record, allow_nan=False)
        setups[identity] = record
        return {"id": identity, "artifact": record["artifact"], **assessment}
    if not setups:
        raise ValueError("No sampling setup has been declared for this candidate")
    identity = args.get("setup_id") or next(reversed(setups))
    record = copy.deepcopy(setups[identity])
    for source in record["setup"]["sources"]:
        agent._authorize_reasoning_source(source)
    record["assessment"] = assess(record["setup"], candidate["sequence"], agent.task["target"])
    if record["basis_model_sha256"] != basis:
        record["assessment"]["status"] = "needs_investigation"
        record["assessment"]["gaps"].append({
            "decision": "model_context",
            "reason": "Baseline model changed; review setup applicability and redeclare",
        })
    return _text_page(
        json.dumps(record, ensure_ascii=False, allow_nan=False),
        args.get("offset", 0),
        setup_id=identity,
        status=record["assessment"]["status"],
    )


def _h_propose_model(agent, args, acc, candidate):
    if not args.get("rationale", "").strip() or not args.get("alternatives") or any(
        not a.get("description") or not a.get("reason_not_selected") for a in args["alternatives"]
    ):
        raise ValueError("record rationale and alternatives with the exact fields description and reason_not_selected")
    derivation = None
    if "modeling_input" in args:
        if "modeling_input_from" in args or "modeling_input_changes" in args:
            raise ValueError("supply modeling_input or modeling_input_from with changes, not both")
        record = copy.deepcopy(args["modeling_input"])
    else:
        origin = args.get("modeling_input_from")
        if origin not in {"candidate", "current"}:
            raise ValueError("provide modeling_input or an explicit modeling_input_from base")
        base = candidate.get("modeling_input") if origin == "candidate" else agent._current(acc)["modeling_input"]
        changes = args.get("modeling_input_changes", {})
        if not isinstance(base, dict) or not isinstance(changes, dict):
            raise ValueError("model base and modeling_input_changes must be objects")
        record = _merge_model_changes(base, changes)
        derivation = {"base": origin, "base_sha256": fingerprint(base), "changes": copy.deepcopy(changes)}
    if not isinstance(record, dict):
        raise ValueError("modeling_input must be an object")
    plan = record.get("validation_plan") or {}
    declarations = list((plan.get("setup") or {}).values()) + [plan.get("poses") or {}]
    atom_policy = (plan.get("poses") or {}).get("protein_atom_policy") or {}
    declarations.extend(atom_policy.get("allowed_missing", []))
    declarations.extend(atom_policy.get("modified_residues", []))
    for declaration in declarations:
        for source in declaration.get("evidence") or []:
            if source.get("path"):
                source["path"] = str(agent._authorized_evidence(source["path"]))
    revisions = agent.state["candidates"][acc]["revisions"]
    if revisions and revisions[-1].get("job", {}).get("status") in {"submission_pending", "queued", "running", "unknown"}:
        raise ValueError("reconcile the existing job before revising; it still occupies the resource budget")
    reasoning = copy.deepcopy(args.get("scientific_reasoning"))
    validate_reasoning(reasoning, record)
    for hypothesis in reasoning["hypotheses"]:
        for source in hypothesis["evidence"]:
            agent._authorize_reasoning_source(source)
    if revisions:
        assessment = assess_reasoning(revisions[-1])
        if assessment["requires_response"] and not assessment["response_current"]:
            raise ValueError("respond to the previous revision's current reasoning assessment before revising")
        response = revisions[-1].get("reasoning_response") or {}
        if response.get("action") == "revise" and (
            revisions[-1]["modeling_input"] == record
            and revisions[-1].get("scientific_reasoning") == reasoning
        ):
            raise ValueError("a revise response requires a changed model or scientific hypothesis record")
    revision = {
        "revision": len(revisions) + 1,
        "modeling_input": record,
        "rationale": args["rationale"],
        "alternatives": args["alternatives"],
        "scientific_reasoning": reasoning,
        "input_derivation": derivation,
        "previous_reasoning_response": copy.deepcopy(revisions[-1].get("reasoning_response")) if revisions else None,
    }
    revision["reasoning_observations"] = copy.deepcopy(revisions[-1].get("reasoning_observations", [])) if revisions else []
    revision["reasoning_registration"] = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "reasoning_sha256": fingerprint(reasoning),
        "mode": "revised_after_feedback" if revisions else "initial",
        "known_observation_ids": [o["id"] for o in revision["reasoning_observations"]],
        "blinding": "not_verified",
    }
    revision["setup_validation"] = validate_setup(record, candidate.get("sequence"))
    revisions.append(revision)
    revision["decision_artifact"] = agent._artifact(f"decision_{acc}", copy.deepcopy(revision))
    return revision


# ---------------------------------------------------------------------------
# Revision-scoped handlers (require current revision)
# ---------------------------------------------------------------------------

def _h_plan_followup(agent, args, acc, candidate, revision):
    from .workflow_policy import TOOL_PURPOSES, validate_followup
    from ..result_layout import followup_paths

    plan = copy.deepcopy(args["plan"])
    validate_followup(plan, revision)
    identity = fingerprint(plan)[:16]
    if identity in revision.get("followups", {}):
        return revision["followups"][identity]
    followup = {
        "id": identity,
        "revision": revision["revision"],
        "plan": plan,
        "tool": TOOL_PURPOSES[plan["purpose"]]["tool"],
        "status": "planned",
    }
    followup["paths"] = followup_paths(agent.run_dir, acc, revision["revision"], plan["purpose"], identity)
    Path(followup["paths"]["root"]).mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        followup["paths"]["request"],
        {
            "purpose": plan["purpose"],
            "plan": plan,
            "task_sha256": agent.state["task_sha256"],
            "accession": acc,
            "revision": revision["revision"],
            "paths": followup["paths"],
        },
        allow_nan=False,
    )
    revision.setdefault("followups", {})[identity] = followup
    if plan["purpose"] == "de_novo_design":
        design_query = "De novo enzyme design requested using RFD4-Proteina. " + json.dumps({
            "target": agent.task.get("target"),
            "scientific_question": plan,
            "reference_accession": acc,
            "note": "The reference cofold is evidence, not proof of catalysis or a mandatory scaffold.",
        })
        followup["design_agent_arguments"] = {
            "query": design_query,
            "output_dir": followup["paths"]["design_runs_root"],
        }
        followup["review_command"] = [
            "python", "-m", "enzyme_design_agent", design_query,
            "--output-dir", followup["paths"]["design_runs_root"],
        ]
        profile = candidate.get("catalytic_profile") or {}
        request = {
            "kind": "de_novo_design_request",
            "purpose": "de_novo_enzyme_design",
            "recipient": "enzyme_design_agent",
            "requested_tool": "rfd4-proteina",
            "target": agent.task.get("target"),
            "source_accession": acc,
            "reference_model": revision["modeling_input"],
            "cofold_evidence": revision.get("validation_artifact"),
            "question": plan,
            "source_task_sha256": agent.state["task_sha256"],
            "output_dir": followup["paths"]["design_runs_root"],
            "catalytic_profile_summary": {
                "evolvability_class": profile.get("evolvability_class"),
                "axes": profile.get("axes"),
            } if profile else None,
            "soft_defect_design_targets": list(profile.get("design_targets") or []),
            "failure_modes": list(profile.get("failure_modes") or []),
            "note": "Design request only; no generation launched and no new sequence is assumed validated.",
        }
        atomic_write_json(followup["paths"]["design_handoff"], request, allow_nan=False)
        followup["handoff"] = followup["paths"]["design_handoff"]
    followup["artifact"] = agent._artifact(f"followup_{acc}", copy.deepcopy(followup))
    return followup


def _h_prepare_bioemu2(agent, args, acc, candidate, revision):
    from .bioemu2_adapter import prepare
    from ..scientific_reasoning import reasoning_errors
    from .workflow_policy import validate_followup

    followup = revision.get("followups", {}).get(args["followup_id"])
    if not followup or followup["plan"]["purpose"] != "dynamics":
        raise ValueError("BioEmu2 requires a saved dynamics follow-up plan")
    validate_followup(followup["plan"], revision)
    if reasoning_errors(revision.get("scientific_reasoning"), revision["modeling_input"]):
        raise ValueError("scientific evidence is unresolved")
    if file_digest(revision["input_yaml"]) != revision["input_sha256"]:
        raise ValueError("rendered input changed")
    path = Path(revision["input_yaml"]).with_name("execution.json")
    if fingerprint(json.loads(path.read_text())) != fingerprint(revision["execution"]):
        raise ValueError("saved execution differs from the current revision")
    provenance = revision["pose_validation"]["prediction_provenance"]
    baseline_sources = revision["pose_validation"]["poses"] + [provenance] + provenance.get("supporting_files", [])
    preparation_dir = (
        (followup.get("paths") or {}).get("preparation_dir")
        or (
            str(Path(followup["preparation"]["job_path"]).parent)
            if followup.get("preparation")
            else str(path.parent / f"bioemu2_{args['followup_id']}")
        )
    )
    prepared = prepare(
        path, args["checkpoint_path"], preparation_dir, seed=args["seed"], mode="equilibrium",
        baseline_sources=[{"path": s["path"], "sha256": s["sha256"]} for s in baseline_sources],
        num_samples=followup["plan"].get("num_samples"),
    )
    followup["preparation"] = prepared
    return prepared


def _h_inspect_dynamics(agent, args, acc, candidate, revision):
    from .bioemu2_adapter import digest, inspect_result

    followup = revision.get("followups", {}).get(args["followup_id"])
    if not followup or followup["plan"]["purpose"] != "dynamics" or not followup.get("preparation"):
        raise ValueError("prepare the saved dynamics follow-up before inspecting results")
    job = followup["preparation"]
    if digest(job["job_path"]) != job["job_sha256"]:
        raise ValueError("prepared dynamics job changed")
    result = inspect_result(job["job_path"], args["prediction_manifest"])
    if followup.get("result") and followup["result"]["source_manifest"] != result["source_manifest"]:
        raise ValueError("inspected follow-up evidence is immutable; register a new study")
    followup.update(result=result, status="inspected")
    return result


def _h_register_design_run(agent, args, acc, candidate, revision):
    followup = revision.get("followups", {}).get(args["followup_id"])
    destination = ((followup or {}).get("paths") or {}).get("design_runs_root")
    if not followup or followup["plan"]["purpose"] != "de_novo_design" or not destination:
        raise ValueError("this follow-up has no recorded design output root")
    actual = Path(args["design_run_dir"]).expanduser().resolve(strict=True)
    expected = Path(destination)
    if not actual.is_relative_to(expected) or actual == expected:
        raise ValueError("design run must be a child of its allocated output root; external results require explicit migration/review")
    manifest = actual / "manifest.json"
    report = actual / "DESIGN_REPORT.md"
    data = json.loads(manifest.read_text())
    if data.get("agent") != "enzyme_design_agent" or not report.is_file():
        raise ValueError("design-agent manifest/report are missing or inconsistent")
    registration = {
        "run_dir": str(actual),
        "agent_status": data.get("status"),
        "manifest": {"path": str(manifest), "sha256": file_digest(manifest)},
        "report": {"path": str(report), "sha256": file_digest(report)},
        "generation_validation": "not_established_by_agent_report",
        "note": "This links result locations; generation, refolding and catalytic assessments retain their own separate evidence.",
    }
    if followup.get("design_run") and followup["design_run"] != registration:
        raise ValueError("registered design evidence is immutable; create a new follow-up for another run")
    followup.update(design_run=registration, status="design_run_linked")
    return registration


def _h_inspect_reasoning(agent, args, acc, candidate, revision):
    return {
        "scientific_reasoning": revision.get("scientific_reasoning"),
        "assessment": assess_reasoning(revision),
        "response": revision.get("reasoning_response"),
    }


def _h_read_reasoning_section(agent, args, acc, candidate, revision):
    sections = {
        "scientific_reasoning": revision.get("scientific_reasoning"),
        "assessment": assess_reasoning(revision),
        "observations": revision.get("reasoning_observations", []),
        "response": revision.get("reasoning_response"),
        "followups": revision.get("followups", {}),
    }
    if args.get("section") not in sections:
        raise ValueError("unknown reasoning section")
    return _text_page(
        json.dumps(sections[args["section"]], allow_nan=False),
        args.get("offset", 0),
        section=args["section"],
        accession=acc,
        revision=revision["revision"],
    )


def _h_compare_hypothesis_tests(agent, args, acc, candidate, revision):
    assessment = assess_reasoning(revision)
    return {
        "assessment_sha256": assessment["assessment_sha256"],
        **compare_hypothesis_tests(revision.get("scientific_reasoning") or {}, assessment),
    }


def _h_record_reasoning_observation(agent, args, acc, candidate, revision):
    observation = copy.deepcopy(args["observation"])
    agent._authorize_reasoning_source(observation.get("source"))
    validate_observation(observation)
    observation.setdefault("applicability", "model" if observation["scope"] == "structural" else "candidate")
    observation["model_context_sha256"] = fingerprint(revision["modeling_input"])
    observation["recorded_revision"] = revision["revision"]
    observations = revision.setdefault("reasoning_observations", [])
    if any(o["id"] == observation["id"] for o in observations):
        raise ValueError("observation IDs are immutable within a revision")
    observations.append(observation)
    revision.setdefault("reasoning_observation_artifacts", []).append(
        agent._artifact(f"reasoning_observation_{acc}", observation)
    )
    agent._event("reasoning_observation", accession=acc, observation_id=observation["id"])
    return assess_reasoning(revision)


def _h_respond_to_reasoning_feedback(agent, args, acc, candidate, revision):
    assessment = assess_reasoning(revision)
    if args.get("assessment_sha256") != assessment["assessment_sha256"]:
        raise ValueError("reasoning assessment changed; inspect the current feedback")
    if args.get("action") not in {"retain", "revise", "investigate", "abstain"}:
        raise ValueError("unknown reasoning response action")
    if not isinstance(args.get("rationale"), str) or not args["rationale"].strip():
        raise ValueError("a reasoning response needs a rationale")
    validate_test(args.get("next_test"))
    hypotheses = set(assessment["response_hypothesis_ids"])
    addressed = args.get("addressed_hypotheses")
    required = {p["hypothesis_id"] for p in assessment["predictions"] if p["status"] == "contradicted"}
    selected = (revision.get("scientific_reasoning") or {}).get("selected_hypothesis_id") or "legacy_unrecorded"
    if selected:
        required.add(selected)
    if (
        not isinstance(addressed, list)
        or not addressed
        or any(not isinstance(h, str) or h not in hypotheses for h in addressed)
        or not required.issubset(addressed)
    ):
        raise ValueError("address the selected and all contradicted hypotheses by their declared IDs")
    response = {
        k: copy.deepcopy(args[k])
        for k in ("assessment_sha256", "action", "rationale", "addressed_hypotheses", "next_test")
    }
    revision["reasoning_response"] = response
    revision.setdefault("reasoning_response_history", []).append(response)
    revision["reasoning_response_artifact"] = agent._artifact(f"reasoning_response_{acc}", response)
    agent._event(
        "reasoning_response",
        accession=acc,
        action=response["action"],
        assessment_sha256=response["assessment_sha256"],
    )
    return {"status": "recorded", "action": response["action"], "assessment": assess_reasoning(revision)}


def _h_declare_proxy_plan(agent, args, acc, candidate, revision):
    from enzyme_discovery_agent.structural_proxy_planning import validate_proxy_plan
    from ..proxies.library import BY_ID

    plan = copy.deepcopy(args["plan"])
    for hypothesis in plan.get("hypotheses", []):
        if hypothesis.get("proxy_id") in BY_ID:
            version = BY_ID[hypothesis["proxy_id"]].version
            if hypothesis.get("definition_version", version) != version:
                raise ValueError("proxy definition changed; review the current version before registration")
            hypothesis["definition_version"] = version
    validate_proxy_binding(plan, revision.get("scientific_reasoning") or {})
    errors = validate_proxy_plan(plan)
    if errors:
        return {"status": "blocked", "errors": errors}
    if revision.get("proxy_plan") is not None and revision["proxy_plan"] != plan:
        raise ValueError("proxy plans are immutable within a revision; propose a new revision")
    revision["proxy_plan"] = plan
    revision.setdefault("proxy_plan_registration", {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": fingerprint(plan),
        "blinding": "not_verified",
    })
    revision["proxy_plan_artifact"] = agent._artifact(f"proxy_plan_{acc}", plan)
    return {"status": "declared", "plan_artifact": revision["proxy_plan_artifact"]}


def _h_evaluate_proxy_plan(agent, args, acc, candidate, revision):
    from ..proxies.library import BY_ID
    from ..proxies.workflow import run_plan

    if not revision.get("proxy_plan"):
        raise ValueError("declare a proxy plan before evaluating measurements")
    for hypothesis in revision["proxy_plan"].get("hypotheses", []):
        proxy = BY_ID.get(hypothesis.get("proxy_id"))
        if proxy and proxy.version != "1.0" and hypothesis.get("definition_version") != proxy.version:
            raise ValueError("registered proxy definition is stale; create a reviewed revision with the current version")
    validate_proxy_binding(revision["proxy_plan"], revision.get("scientific_reasoning") or {})
    poses = (revision.get("pose_validation") or {}).get("poses") or []
    if not poses or any(file_digest(p["path"]) != p["sha256"] for p in poses):
        raise ValueError("inspect the current ensemble and preserve its fingerprints first")
    folders = {Path(p["path"]).parent for p in poses}
    if len(folders) != 1:
        raise ValueError("proxy runner requires one prediction directory")
    folder = folders.pop()
    if {p.resolve() for p in folder.glob("*_model_*.pdb")} != {Path(p["path"]).resolve() for p in poses}:
        raise ValueError("proxy runner supports the exact inspected PDB ensemble only")
    result = run_plan(
        folder,
        copy.deepcopy(revision["modeling_input"]),
        copy.deepcopy(revision["proxy_plan"]),
        use_validated_sites=True,
    )
    revision["proxy_evaluation"] = result
    revision["proxy_evaluation_artifact"] = agent._artifact(f"proxy_evaluation_{acc}", result)
    return result


def _h_validate_setup(agent, args, acc, candidate, revision):
    revision["setup_validation"] = validate_setup(revision["modeling_input"], candidate.get("sequence"))
    return revision["setup_validation"]


def _h_render_inputs(agent, args, acc, candidate, revision):
    from ..result_layout import cofold_dir
    from ..scripts.build_step1_inputs import enzyme_chains, readiness_errors, yaml_for

    record = revision["modeling_input"]
    identity_error = agent._sequence_identity_error(acc)
    if identity_error:
        return {"status": "blocked", "reason": identity_error}
    if record.get("method", "cofold") != "cofold":
        raise ValueError("this renderer supports cofolding; another method requires its own execution adapter")
    validation = validate_setup(record, candidate.get("sequence"))
    revision["setup_validation"] = validation
    pose_plan = validate_pose_plan(record)
    revision["pose_plan_validation"] = pose_plan
    if pose_plan["status"] != "pass":
        return {"status": "blocked", "pose_plan_validation": pose_plan}
    reasoning_gaps = execution_reasoning_errors(revision)
    if reasoning_gaps:
        return {"status": "blocked", "reasoning_errors": reasoning_gaps, "assessment": assess_reasoning(revision)}
    if validation["status"] != "pass" or readiness_errors(record):
        return {"status": "blocked", "validation": validation, "readiness_errors": readiness_errors(record)}
    evidence = candidate.get("evidence") or {}
    supplied_msa = evidence.get("msa_path") if isinstance(evidence, dict) else None
    msa_index = args.get("msa_index", {acc: supplied_msa} if supplied_msa else {})
    if not isinstance(msa_index, dict):
        raise ValueError("msa_index must map candidate accessions to MSA paths")
    msa = msa_index.get(acc)
    if not msa or not Path(msa).is_file():
        raise ValueError(
            f"an existing full-sequence MSA must be supplied explicitly under accession key {acc!r}, "
            "not an entity/chain ID; omit msa_index to use an explicitly supplied evidence.msa_path"
        )
    chains = enzyme_chains(candidate["sequence"], str(Path(msa).resolve()), record["enzyme_state"]["chains"])
    if any(not Path(c["msa"]).is_file() for c in chains):
        raise ValueError("processed chains require existing fragment MSAs")
    for chain in chains:
        chain["msa"] = str(Path(chain["msa"]).resolve())
    msa_evidence = [validate_msa(c["msa"], c["sequence"]) for c in chains]
    identity_error = agent._sequence_identity_error(acc)
    if identity_error:
        return {"status": "blocked", "reason": identity_error}
    substrates = [s for s in record["target_substrates"] if s.get("include_in_model", True)]
    folder = cofold_dir(agent.run_dir, agent.state, acc, revision["revision"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "input.yaml"
    rendered = yaml_for(chains, record.get("cofactors", []), substrates)
    if path.exists() and path.read_text() != rendered:
        raise ValueError("rendered inputs are immutable within a revision; propose a new revision")
    path.write_text(rendered)
    execution = {
        "accession": acc,
        "revision": revision["revision"],
        "modeling_input": copy.deepcopy(record),
        "purpose": "cofolding",
        "scientific_reasoning_sha256": fingerprint(revision["scientific_reasoning"]),
        "entity_map": entity_map_for(chains, record.get("cofactors", []), substrates),
        "setup_validation": validation,
        "msa_evidence": msa_evidence,
        "sequence_identity_source": copy.deepcopy(candidate.get("sequence_source")),
        "sequence_sha256": hashlib.sha256(candidate["sequence"].encode("ascii")).hexdigest(),
    }
    atomic_write_json(folder / "execution.json", execution)
    revision.update(execution=execution, input_yaml=str(path), input_sha256=file_digest(path))
    return {"status": "rendered", "input_yaml": str(path), "execution": execution}


def _h_cofold_job(agent, args, acc, candidate, revision, *, submit: bool):
    if not revision.get("execution"):
        raise ValueError("rendered execution identity is missing")
    if submit:
        engine = str((agent.backend or {}).get("engine") or (agent.backend or {}).get("provider") or "").lower()
        if any(tool in engine for tool in ("bioemu", "rfd4", "proteina")):
            raise ValueError("initial submission is cofolding; dynamics and de novo tools require their own follow-up purpose")
        if validate_pose_plan(revision["modeling_input"], revision["execution"].get("entity_map"))["status"] != "pass":
            raise ValueError("complete physical acceptance criteria and evidence must be declared before submission")
        if execution_reasoning_errors(revision):
            raise ValueError("reasoning evidence or response is unresolved; inspect_reasoning before submission")
        identity_error = agent._sequence_identity_error(acc, revision.get("execution"))
        if identity_error:
            raise ValueError(identity_error)
        if file_digest(revision["input_yaml"]) != revision["input_sha256"]:
            raise ValueError("rendered inputs are missing or changed")
        if any(file_digest(m["path"]) != m["sha256"] for m in revision["execution"].get("msa_evidence", [])):
            raise ValueError("MSA inputs changed after rendering")
        if validate_setup(revision["modeling_input"], candidate.get("sequence"))["status"] != "pass":
            raise ValueError("setup evidence no longer validates")
    return agent._backend_call("submit" if submit else "status", revision)


def _h_validate_predictions(agent, args, acc, candidate, revision):
    if not revision.get("execution"):
        raise ValueError("render current inputs first, so predicted entity identities can be checked")
    directory = Path(args["directory"]).resolve()
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".pdb", ".cif", ".mmcif"})
    previous = revision.get("pose_validation")
    if previous:
        old_paths = {str(Path(p["path"]).resolve()) for p in previous["poses"]}
        if not old_paths <= {str(p.resolve()) for p in paths} or any(
            file_digest(p["path"]) != p["sha256"] for p in previous["poses"] if p.get("sha256")
        ):
            raise ValueError("replacing an ensemble requires a new revision and sampling rationale")
    producer_path = args.get("prediction_manifest") or (revision.get("job") or {}).get("prediction_manifest")
    provenance = {
        "status": "abstain",
        "reason": "producer manifest is missing; physical checks alone do not establish input provenance",
    }
    component_sources = {}
    generic_names = []
    if producer_path:
        producer_path = Path(producer_path).resolve()
        producer = json.loads(producer_path.read_text())
        if producer.get("producer") == "bioemu2" or producer.get("purpose") in {"dynamics", "de_novo_design"}:
            raise ValueError("follow-up outputs cannot satisfy primary cofolding; use inspect_dynamics for BioEmu2")
        entries = producer.get("predictions", [])
        source_files = {
            str((Path(p["path"]) if Path(p["path"]).is_absolute() else producer_path.parent / p["path"]).resolve()): p["sha256"]
            for p in entries
        }
        current_files = {str(p.resolve()): file_digest(p) for p in paths}
        same = (
            producer.get("input_sha256") == revision["input_sha256"]
            and producer.get("execution_sha256") == fingerprint(revision["execution"])
            and len(entries) == len(source_files)
            and source_files == current_files
        )
        provenance = {
            "status": "pass" if same else "fail",
            "path": str(Path(producer_path).resolve()),
            "sha256": file_digest(producer_path),
            "reason": "producer input/execution identity and exact prediction file set compared",
        }
        supporting = producer.get("supporting_files", [])
        if any(not s.get("sha256") or file_digest(s["path"]) != s["sha256"] for s in supporting):
            provenance["status"] = "fail"
        provenance["supporting_files"] = supporting
        if provenance["status"] == "pass":
            component_sources = producer.get("chemical_component_sources") or {}
            generic_names = (producer.get("format_policy") or {}).get("generic_ligand_residue_names") or []
            provenance["supporting_files"] += list(component_sources.values())
    report = validate_ensemble(
        paths,
        revision["execution"],
        chemical_component_sources=component_sources,
        generic_ligand_names=generic_names,
    )
    report["prediction_provenance"] = provenance
    revision["pose_validation"] = report
    revision["validation_artifact"] = agent._artifact(f"poses_{acc}", report)
    revision.setdefault("validation_history", []).append(revision["validation_artifact"])
    return report


# ---------------------------------------------------------------------------
# Declarative dispatch tables
# ---------------------------------------------------------------------------

PRE_ACCESSION_HANDLERS = {
    "inspect_barrier_contract": _h_inspect_barrier_contract,
    "prepare_barrier_inputs": _h_prepare_barrier_inputs,
    "inspect_barrier_result": _h_inspect_barrier_result,
    "plan_barrier_calculation": _h_plan_barrier_calculation,
    "submit_barrier_calculation": _h_submit_barrier_calculation,
    "check_barrier_calculation": _h_check_barrier_calculation,
    "record_barrier_assessment": _h_record_barrier_assessment,
    "get_structural_proxy_guidance": _make_shared_proxy_handler("get_structural_proxy_guidance"),
    "validate_structural_proxy_plan": _make_shared_proxy_handler("validate_structural_proxy_plan"),
    "inspect_experimental_proxy_bundle": _make_shared_proxy_handler("inspect_experimental_proxy_bundle"),
    "inspect_investigation_lead": _h_inspect_investigation_lead,
}

GENERAL_HANDLERS = {
    "inspect_validation_contract": _h_inspect_validation_contract,
    "inspect_mlmm_engine_options": _h_inspect_mlmm_engine_options,
    "inspect_preparation_contract": _h_inspect_preparation_contract,
    "inspect_preparation_structure": _h_inspect_preparation_structure,
    "inspect_sampling_setup_contract": _h_inspect_sampling_setup_contract,
    "inspect_processivity_toolchain": _h_inspect_processivity_toolchain,
    "inspect_processivity_contract": _h_inspect_processivity_contract,
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
    "inspect_workflow_policy": _h_inspect_workflow_policy,
    "record_preparation_findings": _h_record_preparation_findings,
    "run_preparation_operator": _h_run_preparation_operator,
    "declare_preparation_study": lambda a, args, acc, cand: _h_preparation_study(a, args, acc, cand, declare=True),
    "inspect_preparation_study": lambda a, args, acc, cand: _h_preparation_study(a, args, acc, cand, declare=False),
    "declare_sampling_setup": lambda a, args, acc, cand: _h_sampling_setup(a, args, acc, cand, declare=True),
    "inspect_sampling_setup": lambda a, args, acc, cand: _h_sampling_setup(a, args, acc, cand, declare=False),
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
    """Route a tool call to its domain handler with consistent accession resolution."""
    if name in PRE_ACCESSION_HANDLERS:
        return PRE_ACCESSION_HANDLERS[name](agent, args)

    # Single-candidate runs default omitted accession and reject unknown ones clearly.
    if isinstance(args, dict):
        if "accession" not in args and len(agent.state["candidates"]) == 1:
            args = {**args, "accession": next(iter(agent.state["candidates"]))}
        elif args.get("accession") is not None and args["accession"] not in agent.state["candidates"]:
            raise ValueError(
                f"unknown accession {args['accession']!r}; choose one of {list(agent.state['candidates'])}"
            )

    if name in GENERAL_HANDLERS:
        return GENERAL_HANDLERS[name](agent, args)

    acc = args.get("accession")
    if acc is None:
        ids = list(agent.state["candidates"])
        if len(ids) != 1:
            raise ValueError(f"accession is required; choose one of {ids}")
        acc = ids[0]
    elif acc not in agent.state["candidates"]:
        raise ValueError(f"unknown accession {acc!r}; choose one of {list(agent.state['candidates'])}")

    candidate = agent._candidate(acc)
    if name in CANDIDATE_HANDLERS:
        return CANDIDATE_HANDLERS[name](agent, args, acc, candidate)

    if name in REVISION_HANDLERS:
        revision = agent._current(acc)
        return REVISION_HANDLERS[name](agent, args, acc, candidate, revision)

    raise ValueError(f"unknown tool {name}")
