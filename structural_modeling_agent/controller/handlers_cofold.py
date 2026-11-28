"""Cofold proposal, setup validation, input rendering, submission, and proxy handlers."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from enzyme_core.state_store import atomic_write_json

from ..scientific_reasoning import (
    assess_reasoning,
    execution_reasoning_errors,
    validate_proxy_binding,
    validate_reasoning,
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


def _merge_model_changes(base, changes):
    """Apply an explicit JSON merge patch to a copied modeling proposal."""
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict):
            result[key] = _merge_model_changes(
                result.get(key) if isinstance(result.get(key), dict) else {},
                value,
            )
        else:
            result[key] = copy.deepcopy(value)
    return result


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
    revision["reasoning_observations"] = (
        copy.deepcopy(revisions[-1].get("reasoning_observations", [])) if revisions else []
    )
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
    revision.setdefault(
        "proxy_plan_registration",
        {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": fingerprint(plan),
            "blinding": "not_verified",
        },
    )
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
            raise ValueError(
                "initial submission is cofolding; dynamics and de novo tools require their own follow-up purpose"
            )
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
        if producer.get("producer") == "bioemu2" or producer.get("purpose") in {"dynamics", "de_novo_design", "enzyme_design"}:
            raise ValueError("follow-up outputs cannot satisfy primary cofolding; use inspect_dynamics for BioEmu2")
        entries = producer.get("predictions", [])
        source_files = {
            str(
                (Path(p["path"]) if Path(p["path"]).is_absolute() else producer_path.parent / p["path"]).resolve()
            ): p["sha256"]
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
