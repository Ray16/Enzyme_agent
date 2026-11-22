"""Reservation-aware MD / umbrella sampling submission and reconciliation.

Follows the same fail-closed, idempotent architecture as ``barrier/workflow.py``:
the LLM declares a reviewed ``sampling_setup`` and submits bounded compute
through an explicitly configured reservation-aware backend. Completed
trajectories are hash-verified and expose structured ``analysis_inputs`` for
chaining into ``analyze_substrate_contacts``, ``analyze_side_reactions``,
``analyze_preorganization``, and ``build_catalytic_profile``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid
from typing import Any

from enzyme_discovery_agent.state_store import atomic_write_json
from ..validation import file_digest, fingerprint

ACTIVE = {"submission_pending", "queued", "running", "unknown"}
SUPPORTED_ENGINES = ("openmm_classical", "ase_mlip_pilot")


def sampling_jobs(agent, accession: str) -> dict[str, Any]:
    agent._candidate(accession)
    return agent.state["candidates"][accession].setdefault("sampling_jobs", {})


def _candidate_identity(agent, accession: str) -> dict[str, str]:
    candidate = agent._candidate(accession)
    sequence = candidate.get("sequence")
    if not isinstance(sequence, str) or not re.fullmatch(r"[A-Z]+", sequence):
        raise ValueError("Resolve the exact candidate sequence before submitting sampling")
    return {
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "target_sha256": fingerprint(agent.task.get("target")),
    }


def _verify_files(files: dict[str, str], authorize=None) -> None:
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing sampling input/output fingerprints")
    for name, digest in files.items():
        path = authorize(name) if authorize else Path(name).resolve(strict=True)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or file_digest(path) != digest:
            raise ValueError(f"Sampling artifact changed or invalid digest: {name}")


def sampling_backend_config(agent) -> dict[str, Any]:
    """Validate that a reservation-aware sampling backend is explicitly configured."""
    config = getattr(agent, "sampling_backend", None) or agent.state.get("sampling_backend")
    if config is None and isinstance(agent.backend, dict):
        config = agent.backend.get("sampling")
    if not isinstance(config, dict):
        raise ValueError("No sampling backend configured; provide backend.sampling or sampling_backend")
    policy = config.get("resource_policy") or {}
    if policy.get("reservation_managed") is not True:
        raise ValueError("Sampling backend must manage reservations (resource_policy.reservation_managed=true)")
    for field in ("max_jobs", "max_wall_seconds"):
        if type(policy.get(field)) is not int or policy[field] < 1:
            raise ValueError(f"Explicit positive sampling resource_policy.{field} required")
    if not config.get("provider"):
        raise ValueError("Sampling provider is missing")
    engines = config.get("supported_engines")
    if not isinstance(engines, list) or not engines or any(e not in SUPPORTED_ENGINES for e in engines):
        raise ValueError(f"Sampling backend must declare supported_engines subset of {list(SUPPORTED_ENGINES)}")
    argv = config.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v for v in argv):
        raise ValueError("Sampling adapter requires an explicit argv list")
    return config


def _invoke_adapter(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    proc = subprocess.run(
        config["argv"],
        input=json.dumps(payload, allow_nan=False),
        text=True,
        capture_output=True,
        timeout=int(config.get("adapter_timeout_seconds", 60)),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Sampling adapter exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:400]}")
    try:
        reply = json.loads(proc.stdout)
    except ValueError as exc:
        raise RuntimeError(f"Sampling adapter returned invalid JSON: {exc}") from exc
    if not isinstance(reply, dict) or not reply.get("status"):
        raise RuntimeError("Sampling adapter reply must be a JSON object with 'status'")
    return reply


def _accept_outputs(record: dict[str, Any], reply: dict[str, Any]) -> None:
    plan = record["plan"]
    work_dir = Path(plan["work_dir"]).resolve()
    outputs = reply.get("output_sha256")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("Completed sampling job must return output_sha256 fingerprints")
    _verify_files(outputs)
    for name in outputs:
        if not Path(name).resolve().is_relative_to(work_dir):
            raise ValueError(f"Sampling output lies outside work_dir: {name}")

    top_path = reply.get("topology_path")
    traj_paths = list(reply.get("trajectory_paths") or [])
    geom_path = reply.get("geometry_jsonl")
    for p in ([top_path] if top_path else []) + traj_paths + ([geom_path] if geom_path else []):
        if p and str(Path(p).resolve()) not in {str(Path(k).resolve()) for k in outputs}:
            raise ValueError(f"Declared sampling output missing from output_sha256: {p}")

    record["outputs"] = {
        "topology_path": top_path,
        "trajectory_paths": traj_paths,
        "geometry_jsonl": geom_path,
        "output_sha256": outputs,
        "completed_ns": reply.get("completed_ns"),
        "analysis_inputs": {
            "top_path": top_path,
            "traj_paths": traj_paths,
            "geometry_jsonl": geom_path,
        },
        "recommended_next_tools": [
            "analyze_substrate_contacts",
            "analyze_side_reactions",
            "analyze_preorganization",
            "build_catalytic_profile",
        ],
    }


def submit_sampling(agent, args: dict[str, Any]) -> dict[str, Any]:
    """Validate a reviewed sampling setup and submit one bounded job through the reservation backend."""
    accession = args["accession"]
    records = sampling_jobs(agent, accession)
    identity = _candidate_identity(agent, accession)
    identity_gap = agent._sequence_identity_error(accession)
    if identity_gap:
        raise ValueError(identity_gap)

    if agent.task.get("workflow_mode") == "production_cofold" and agent.task.get("objective") not in {"prepare_sampling", "evaluate_barriers"}:
        from ..controller.workflow_policy import baseline_errors
        errors = baseline_errors(agent._current(accession))
        if errors:
            raise ValueError("Inspect the primary cofold before sampling compute: " + "; ".join(errors))

    setup_id = args["setup_id"]
    candidate_state = agent.state["candidates"][accession]
    setup_record = (candidate_state.get("sampling_setups") or {}).get(setup_id)
    if not setup_record:
        raise ValueError(f"Unknown sampling setup_id {setup_id!r}; call declare_sampling_setup first")
    setup_status = (setup_record.get("assessment") or {}).get("status") or setup_record.get("status")
    if setup_status != "ready_for_preparation":
        raise ValueError(
            f"Sampling setup {setup_id!r} has status {setup_status!r}; "
            "resolve pending setup decisions before submitting compute"
        )

    engine = args["engine"]
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"Unsupported sampling engine {engine!r}; choose from {list(SUPPORTED_ENGINES)}")

    config = sampling_backend_config(agent)
    if engine not in config["supported_engines"]:
        raise ValueError(f"Configured sampling backend does not support engine {engine!r}; no silent fallback")

    wall_seconds = args["wall_seconds"]
    if type(wall_seconds) is not int or wall_seconds < 1 or wall_seconds > config["resource_policy"]["max_wall_seconds"]:
        raise ValueError("wall_seconds must be a positive integer within resource_policy.max_wall_seconds")

    active_count = sum(
        1 for rec in records.values() if (rec.get("job") or {}).get("status") in ACTIVE
    )
    if active_count >= config["resource_policy"]["max_jobs"]:
        raise ValueError(f"Sampling max_jobs ({config['resource_policy']['max_jobs']}) reached; reconcile active jobs first")

    label = args["label"]
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", label):
        raise ValueError("Sampling label must be a safe file component")

    # Idempotent check: if this exact label already exists, return its status instead of double-submitting
    for existing_id, rec in records.items():
        if rec["plan"]["label"] == label:
            return {
                "job_id": existing_id,
                "status": "already_requested",
                "job": rec.get("job"),
                "next_action": "check_sampling",
            }

    raw_inputs = args.get("inputs") or {}
    required_keys = ("prmtop", "coordinates", "spec_path") if engine == "openmm_classical" else ("config_path",)
    missing = [k for k in required_keys if not raw_inputs.get(k)]
    if missing:
        raise ValueError(f"Engine {engine!r} requires inputs keys: {missing}")

    hashed_inputs: dict[str, str] = {}
    resolved_inputs: dict[str, str] = {}
    for k, v in raw_inputs.items():
        p = agent._authorized_evidence(v)
        resolved_inputs[k] = str(p)
        hashed_inputs[str(p)] = file_digest(p)

    job_id = f"{label}-{uuid.uuid4().hex[:12]}"
    work_dir = agent.run_dir / "artifacts" / "sampling" / accession / job_id
    work_dir.mkdir(parents=True, exist_ok=False)

    plan = {
        "schema_version": "sampling-job-1.0",
        "id": job_id,
        "label": label,
        "accession": accession,
        "setup_id": setup_id,
        "setup_sha256": fingerprint(setup_record.get("setup")),
        "engine": engine,
        "wall_seconds": wall_seconds,
        "inputs": resolved_inputs,
        "input_sha256": hashed_inputs,
        "settings": dict(args.get("settings") or {}),
        "work_dir": str(work_dir),
        **identity,
    }
    plan_path = work_dir / "plan.json"
    atomic_write_json(plan_path, plan, allow_nan=False)
    plan_sha256 = file_digest(plan_path)
    idem_key = fingerprint({"plan_sha256": plan_sha256, "backend": config})

    record: dict[str, Any] = {
        "plan": plan,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "status": "submission_pending",
        "job": {
            "idempotency_key": idem_key,
            "provider": config["provider"],
            "status": "submission_pending",
        },
    }
    records[job_id] = record

    reply = _invoke_adapter(
        config,
        {
            "operation": "submit",
            "idempotency_key": idem_key,
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha256,
            "plan": plan,
        },
    )
    status = str(reply["status"])
    record["job"].update(
        status=status,
        remote_job_id=reply.get("remote_job_id"),
        message=reply.get("message"),
    )
    record["status"] = status
    if status == "complete":
        _accept_outputs(record, reply)
    return {
        "job_id": job_id,
        "status": status,
        "engine": engine,
        "plan_path": str(plan_path),
        "job": record["job"],
        "outputs": record.get("outputs"),
        "next_action": "analyze_completed_trajectory" if status == "complete" else "check_sampling",
    }


def check_sampling(agent, args: dict[str, Any]) -> dict[str, Any]:
    """Reconcile an existing sampling job receipt and verify returned trajectory hashes."""
    accession = args["accession"]
    job_id = args["job_id"]
    records = sampling_jobs(agent, accession)
    if job_id not in records:
        raise ValueError(f"Unknown sampling job_id {job_id!r}")
    record = records[job_id]
    plan = record["plan"]
    if file_digest(record["plan_path"]) != record["plan_sha256"]:
        raise ValueError("Sampling plan changed on disk")
    if any(plan[k] != v for k, v in _candidate_identity(agent, accession).items()):
        raise ValueError("Candidate sequence or target changed since sampling submission")
    _verify_files(plan["input_sha256"])

    job = record.get("job")
    if not job:
        return {"job_id": job_id, "status": "not_submitted"}
    if record.get("status") == "complete" and record.get("outputs"):
        _verify_files(record["outputs"]["output_sha256"])
        return {
            "job_id": job_id,
            "status": "complete",
            "job": job,
            "outputs": record["outputs"],
        }

    config = sampling_backend_config(agent)
    expected_key = fingerprint({"plan_sha256": record["plan_sha256"], "backend": config})
    if job.get("idempotency_key") != expected_key:
        raise ValueError("Sampling backend configuration changed; retain existing receipt and create a new job")

    reply = _invoke_adapter(
        config,
        {
            "operation": "status",
            "idempotency_key": job["idempotency_key"],
            "remote_job_id": job.get("remote_job_id"),
            "plan_path": record["plan_path"],
            "plan_sha256": record["plan_sha256"],
        },
    )
    status = str(reply["status"])
    job.update(
        status=status,
        remote_job_id=reply.get("remote_job_id", job.get("remote_job_id")),
        message=reply.get("message"),
    )
    record["status"] = status
    if status == "complete":
        _accept_outputs(record, reply)
    return {
        "job_id": job_id,
        "status": status,
        "job": job,
        "outputs": record.get("outputs"),
        "next_action": "analyze_completed_trajectory" if status == "complete" else "check_sampling",
    }
