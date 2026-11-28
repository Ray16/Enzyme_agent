"""Inspection, reference, barrier, preparation, and task-completion handlers."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import uuid

from enzyme_core.state_store import atomic_write_json

from ..validation import file_digest, fingerprint


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


# ---------------------------------------------------------------------------
# Pre-accession handlers (barrier & shared discovery proxy tools)
# ---------------------------------------------------------------------------


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
# General / contract / inspection handlers
# ---------------------------------------------------------------------------


def validation_contract(section: str = "all"):
    """Load the validation_contract.json section (used internally by agent._context)."""
    contract = json.loads((Path(__file__).resolve().parents[1] / "validation_contract.json").read_text())
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


def evaluate_preparation_study(agent, acc: str):
    """Re-evaluate the latest declared preparation study (used by completion_errors)."""
    candidate = agent._candidate(acc)
    return _h_preparation_study(agent, {}, acc, candidate, declare=False)


def _h_inspect_preparation_structure(agent, args):
    from ..sampling.preparation import inspect_pdb

    result = inspect_pdb(agent._authorized_evidence(args["path"]))
    return {"artifact": agent._artifact("preparation_structure", result), **result}


def _h_inspect_processivity_toolchain(agent, args):
    from ..sampling.toolchain import inspect_toolchain

    return inspect_toolchain()


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
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"evidence file {path.name!r} is binary or non-UTF-8 and cannot be inspected as text"
        ) from exc
    page = _text_page(text, offset, path=str(path.resolve()), sha256=file_digest(path))
    if not path.is_relative_to((agent.run_dir / "artifacts").resolve()):
        notes = agent.state.setdefault("inspected_source_passages", [])
        note = {k: page[k] for k in ("path", "sha256", "offset")}
        note["text"] = page["text"][:1000]
        note["next_offset"] = page["next_offset"]
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
    if not re.fullmatch(
        r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})",
        identifier,
    ):
        return {
            "status": "needs_identity_mapping",
            "accession": args["accession"],
            "next_action": (
                "Resolve an exact-sequence UniProt mapping or investigate assembly from explicitly "
                "identified experimental structures; no monomer assumption is made."
            ),
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
    from ..sampling.workflow import ACTIVE as SAMPLING_ACTIVE

    if any(
        (j.get("job") or {}).get("status") in SAMPLING_ACTIVE
        for c in agent.state["candidates"].values()
        for j in c.get("sampling_jobs", {}).values()
    ):
        agent.state.update(status="waiting_for_backend", conclusion=args["rationale"])
        return {"status": "waiting_for_backend", "reason": "Sampling job ownership must be reconciled on resume"}
    agent.state.update(status=args["outcome"], conclusion=args["rationale"])
    return {"status": agent.state["status"], "rationale": args["rationale"]}


# ---------------------------------------------------------------------------
# Candidate-scoped preparation & sampling-setup handlers
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
        folder,
        args["python"],
        args["code"],
        args["inputs"],
        args["outputs"],
        args["seconds"],
        agent._authorized_evidence,
    )
    agent.state["candidates"][acc].setdefault("preparation_operators", []).append(result)
    return result


def _h_preparation_study(agent, args, acc, candidate, *, declare: bool):
    from ..sampling.preparation import evaluate

    state = agent.state["candidates"][acc]
    studies = state.setdefault("preparation_studies", {})
    if declare:
        if not isinstance(args.get("study"), dict):
            raise ValueError("declare_preparation_study requires a 'study' object (see md-preparation skill)")
        study = copy.deepcopy(args["study"])
    else:
        if not studies:
            raise ValueError("No preparation study declared")
        study = copy.deepcopy(next(reversed(studies.values()))["study"])
    result = evaluate(
        study,
        state.get("sampling_setups", {}),
        candidate["sequence"],
        agent.task["target"],
        agent._authorized_evidence,
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
    from ..result_layout import revision_dir
    from ..sampling.setup import assess

    state = agent.state["candidates"][acc]
    revisions = state["revisions"]
    basis = fingerprint(revisions[-1]["modeling_input"]) if revisions else None
    setups = state.setdefault("sampling_setups", {})
    if declare:
        if not isinstance(args.get("setup"), dict):
            raise ValueError("declare_sampling_setup requires a 'setup' object (see md-preparation skill)")
        setup = copy.deepcopy(args["setup"])
        identities = {
            "sequence_sha256": hashlib.sha256(candidate["sequence"].encode()).hexdigest(),
            "target_sha256": fingerprint(agent.task["target"]),
        }
        for key, expected in identities.items():
            if key in setup and setup[key] != expected:
                raise ValueError(
                    f"{key} mismatch; use {expected} or omit this bookkeeping field for automatic binding"
                )
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
        record["assessment"]["gaps"].append(
            {
                "decision": "model_context",
                "reason": "Baseline model changed; review setup applicability and redeclare",
            }
        )
    return _text_page(
        json.dumps(record, ensure_ascii=False, allow_nan=False),
        args.get("offset", 0),
        setup_id=identity,
        status=record["assessment"]["status"],
    )
