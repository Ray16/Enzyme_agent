"""Deterministic destinations for new systems; legacy runs stay in place.

The checkpoint is authoritative. The results index is a regenerable view of
planned paths and recorded outputs, never proof that a calculation succeeded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import uuid

from enzyme_discovery_agent.state_store import atomic_write_json
from .validation import fingerprint

LAYOUT_VERSION = "structural-results-2.0"


def default_results_root():
    checkout = Path(__file__).resolve().parents[1]
    if (checkout / "pyproject.toml").is_file() and (checkout / "structural_modeling_agent" / "agent.py").is_file():
        return checkout / "results"
    # Installed packages must never write inside site-packages. Their caller
    # chooses the project, or configures a stable root explicitly.
    return Path.cwd() / "results"


def system_identity(task, explicit=None):
    target = task.get("target")
    data = target if isinstance(target, dict) else {}
    label = explicit if explicit is not None else task.get("system_id")
    if label is not None:
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", label):
            raise ValueError("system_id must be a safe 1-80 character label, not a path")
    else:
        name = next((data.get(key) for key in ("id", "target_id", "name", "reaction")
                     if isinstance(data.get(key), str) and data[key].strip()), "system")
        label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._-")[:60] or "system"
    # Empty targets cannot establish that different candidate tasks are one
    # system. Explicit labels still remain disambiguated by the task identity.
    digest = fingerprint(target if target else {"underspecified_task": task})
    return {"label": label, "key": f"{label}-{digest[:12]}", "target_sha256": fingerprint(target),
            "identity_sha256": digest}


def new_layout(task, *, run_dir=None, results_root=None, system_id=None):
    if run_dir is not None and results_root is not None:
        raise ValueError("choose output-dir or results-root, not both")
    system = system_identity(task, system_id)
    root = Path(results_root or os.environ.get("ENZYME_AGENT_RESULTS_ROOT") or default_results_root()).expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    path = Path(run_dir).expanduser().resolve() if run_dir is not None else root / system["key"] / "structural_modeling" / run_id
    path = path.resolve()
    return {"schema_version": LAYOUT_VERSION, "run_id": run_id, "system": system,
            "results_root": str(root) if run_dir is None else None, "run_dir": str(path),
            "explicit_output_directory": run_dir is not None}


def revision_dir(run_dir, accession, number):
    return Path(run_dir) / "models" / accession / f"revision_{number}"


def cofold_dir(run_dir, state, accession, number):
    path = revision_dir(run_dir, accession, number)
    return path / "cofold" if (state.get("result_layout") or {}).get("schema_version") == LAYOUT_VERSION else path


def followup_paths(run_dir, accession, number, purpose, identity):
    path = revision_dir(run_dir, accession, number) / "followups" / purpose / identity
    return {"root": str(path), "request": str(path / "request.json"),
            "preparation_dir": str(path / "bioemu2") if purpose == "dynamics" else None,
            "results_dir": str(path / "bioemu2" / "results") if purpose == "dynamics" else None,
            "design_runs_root": str(path / "design_runs") if purpose == "de_novo_design" else None,
            "design_handoff": str(path / "design_request.json") if purpose == "de_novo_design" else None}


def results_index(run_dir, state):
    root = Path(run_dir).resolve()
    layout = state.get("result_layout") or {"schema_version": "legacy", "run_dir": str(root),
                                             "system": system_identity(state["task"])}
    candidates = {}
    for accession, candidate in state["candidates"].items():
        revisions = []
        for record in candidate["revisions"]:
            number = record["revision"]
            cofold = Path(record["input_yaml"]).parent if record.get("input_yaml") else cofold_dir(root, state, accession, number)
            poses = (record.get("pose_validation") or {}).get("poses", [])
            studies = []
            for identity, study in record.get("followups", {}).items():
                paths = study.get("paths")
                # Old handoffs without destinations are unknown, not silently
                # moved to a newly invented location.
                studies.append({"id": identity, "purpose": study["plan"]["purpose"], "status": study["status"],
                    "paths": paths, "legacy_destination_unrecorded": paths is None,
                    "preparation": study.get("preparation"),
                    "recorded_result": {key: study["result"].get(key) for key in ("status", "purpose", "source_manifest")} if study.get("result") else None,
                    "handoff": study.get("handoff"), "review_command": study.get("review_command")})
                studies[-1]["registered_design_run"] = study.get("design_run")
            revisions.append({"revision": number, "root": str(revision_dir(root, accession, number)),
                "cofold": {"purpose": "cofolding", "root": str(cofold),
                    "input_yaml": record.get("input_yaml") or str(cofold / "input.yaml"),
                    "execution": str(cofold / "execution.json"), "managed_predictions_dir": str(cofold / "predictions"),
                    "rendered": bool(record.get("execution")),
                    "job_status": (record.get("job") or {}).get("status", "not_submitted"),
                    "pose_validation_status": (record.get("pose_validation") or {}).get("status", "not_inspected"),
                    "registered_prediction_files": [{"path": p["path"], "sha256": p.get("sha256")} for p in poses],
                    "validation_artifact": record.get("validation_artifact")}, "followups": studies})
        candidates[accession] = {"current_revision": revisions[-1]["revision"] if revisions else None,
                                 "revisions": revisions,
                                 "planned_next_revision": len(revisions) + 1,
                                 "planned_next_cofold_dir": str(cofold_dir(root, state, accession, len(revisions) + 1))}
    return {"schema_version": "structural-results-index-1.0", "layout": layout,
            "run_status": state["status"], "task_sha256": state["task_sha256"],
            "reports": {"report": str(root / "report.json"), "evidence": str(root / "structural_modeling_evidence.json")},
            "candidates": candidates,
            "note": "Paths may be planned. Status and source fingerprints remain required; folder presence is not successful execution. Checkpoint is authoritative; this index is a regenerable view."}


def write_index(run_dir, state):
    path = Path(run_dir) / "results_index.json"
    atomic_write_json(path, results_index(run_dir, state), allow_nan=False)
    return path


def read_run_index(run_dir):
    """Offline, fresh reconstruction, including runs created before the index."""
    import json
    root = Path(run_dir).expanduser().resolve()
    checkpoint = root / "checkpoint.json"
    if checkpoint.is_file():
        data = json.loads(checkpoint.read_text())
        if data.get("schema_version") != 1:
            raise ValueError("unsupported controller checkpoint")
        state = data["state"]
    else:
        state = json.loads((root / "agent_state.json").read_text())
    if fingerprint(state["task"]) != state["task_sha256"]:
        raise ValueError("saved task identity differs from the checkpoint")
    saved = state.get("result_layout") or {}
    if saved.get("run_dir") and Path(saved["run_dir"]) != root:
        raise ValueError("run directory moved; recorded output paths require explicit migration")
    return results_index(root, state)
