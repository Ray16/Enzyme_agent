"""Persistent, tool-using structural-modeling decision agent.

Discovery is an optional input producer. This controller owns its own task,
decisions, evidence, revisions, backend receipts, validation and completion gate.
Scientific calculations remain in callable tools, outside the model's prose.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

from enzyme_discovery_agent.state_store import atomic_write_json
from .controller.dispatch_handlers import _merge_model_changes, _text_page
from .controller.prompts import BARRIER_GUIDANCE, BARRIER_PROMPT, PREPARATION_PROMPT, SYSTEM_PROMPT, compact_system_prompt
from .controller.tool_registry import get_registry, select_offered_tools
from .controller.tool_schemas import tool_schemas
from .validation import (entity_map_for, file_digest, fingerprint, validate_ensemble,
                         validate_setup, validate_msa, validate_pose_plan)
from .scientific_reasoning import (assess_reasoning, execution_reasoning_errors,
                                   validate_reasoning, validate_observation,
                                   validate_test, validate_proxy_binding, compare_hypothesis_tests)


# Characters of context assembled per turn, excluding anything anchored for the
# setup stage: an anchor raises the allowance by its own length so it competes
# with the system prefix rather than with conversational history.
CONTEXT_ALLOWANCE = 48000
SETUP_ANCHOR_LIMIT = 24000
# Derived from the unified ToolRegistry so schema, dispatch, stage filtering,
# and repeated-inspection guards stay in sync automatically.
_REPEATABLE_READERS = get_registry().repeatable_readers()


def _task_evidence_paths(task, base):
    """Resolve explicitly supplied evidence paths once, before model interaction."""
    paths = set()

    def visit(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, name)
        elif isinstance(value, list):
            for child in value:
                visit(child, "path" if key == "evidence_paths" else key)
        elif isinstance(value, str) and (key == "path" or key.endswith("_path") or key.endswith("_dir")):
            path = Path(value)
            paths.add(str((path if path.is_absolute() else base / path).resolve()))

    visit(task)
    return sorted(paths)


def _discovery_currentness(task, run_path):
    """Compare only an explicitly supplied run with identity-bearing state."""
    from enzyme_discovery_agent.run_layout import run_file
    stamp = datetime.now(timezone.utc).isoformat()
    unknown = {"status": "unknown", "checked_at": stamp}
    if run_path is None:
        return {**unknown, "reason": "no trusted discovery run was supplied for comparison"}
    root = Path(run_path).resolve()
    producer = task.get("producer") or {}
    recorded = producer.get("source_artifact_sha256") or {}
    files = {"candidate_registry": run_file(root, "candidate_registry.json"),
             "target_context": run_file(root, "target_context.json"),
             "ranking": run_file(root, "pass1_ranking.json"),
             "sequence_cache": run_file(root, "embedding_store") / "sequences.json",
             "candidate_map": run_file(root, "candidate_map.json")}
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        if (not isinstance(manifest, dict) or not producer.get("run_id")
                or manifest.get("run_id") != producer["run_id"]
                or not recorded.get("candidate_registry") or not recorded.get("target_context")
                or not all(key in recorded for key in files)):
            return {**unknown, "reason": "run identity or source state is not established"}
        observed = {key: file_digest(path) if path.exists() else None for key, path in files.items()}
        differences = [key for key in files if recorded[key] != observed[key]]
        return {"status": "stale" if differences else "unchanged_checked_artifacts", "checked_at": stamp,
                "compared_artifacts": list(files), "changed": differences,
                "limitation": "Evaluation inputs, external evidence and coverage were not recomputed; overall currentness is not established."}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {**unknown, "reason": f"run identity or source state unavailable: {exc}"}


def _load_checkpoint(run_dir):
    """Read the coherent checkpoint; support runs created before its introduction."""
    path = run_dir / "checkpoint.json"
    if path.exists():
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("schema_version") != 1:
            raise ValueError("unsupported controller checkpoint")
        return checkpoint["state"], checkpoint["messages"]
    return (json.loads((run_dir / "agent_state.json").read_text()),
            json.loads((run_dir / "transcript.json").read_text()))


@contextmanager
def _controller_lock(run_dir):
    import fcntl
    with (run_dir / "controller.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ModelingAgent:
    def __init__(self, *, task_path=None, run_dir=None, model=None, backend_path=None,
                 resume=False, llm_client=None, discovery_run_path=None, results_root=None, system_id=None,
                 barrier_backend_path=None):
        self._discovery_run_path = discovery_run_path
        if resume:
            if results_root is not None or system_id is not None:
                raise ValueError("resume cannot replace the saved system identity or results root")
            if run_dir is None:
                raise ValueError("resume requires an existing run directory")
            self.run_dir = Path(run_dir).resolve()
            self.state, self.messages = _load_checkpoint(self.run_dir)
            self.task = self.state["task"]
            saved_layout = self.state.get("result_layout")
            if saved_layout and Path(saved_layout["run_dir"]) != self.run_dir:
                raise ValueError("run directory moved; use the recorded location or an explicit migration")
            if task_path is not None:
                raise ValueError("a resumed run cannot replace its target")
        else:
            if task_path is None:
                raise ValueError("a standalone task or discovery handoff is required")
            task_path = Path(task_path).resolve()
            from enzyme_discovery_agent.models.handoff_contract import (
                adapt_legacy_handoff, load_document, validate_handoff,
            )
            raw_task = load_document(task_path)
            if not isinstance(raw_task, dict):
                raise ValueError("task must be an object")
            handoff = (raw_task.get("kind") == "discovery_handoff" or "source_run" in raw_task
                       or raw_task.get("schema_version") in {"2.0", "3.0"})
            if handoff and raw_task.get("schema_version") == "3.0":
                self.task = validate_handoff(raw_task)
            elif handoff and raw_task.get("schema_version") == "2.0":
                self.task = adapt_legacy_handoff(raw_task, file_digest(task_path))
            elif handoff:
                raise ValueError("unsupported discovery handoff schema")
            else:
                if raw_task.get("schema_version") not in {None, "1.0"}:
                    raise ValueError("unsupported standalone task schema")
                self.task = raw_task
            candidates = self.task.get("candidates")
            if not isinstance(candidates, list) or (not candidates and not handoff):
                raise ValueError("task requires a nonempty candidates list")
            ids = [c.get("accession") for c in candidates]
            if any(not isinstance(a, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", a) or a in {".", ".."} for a in ids) or len(set(ids)) != len(ids):
                raise ValueError("candidate identifiers must be unique safe path components")
            if self.task.get("objective", "model_and_validate") not in {"prepare_inputs", "prepare_sampling", "model_and_validate", "evaluate_barriers"}:
                raise ValueError("unknown modeling objective")
            from .result_layout import new_layout
            result_layout = new_layout(self.task, run_dir=run_dir, results_root=results_root, system_id=system_id)
            self.run_dir = Path(result_layout["run_dir"])
            if self.run_dir.exists() and any(self.run_dir.iterdir()):
                raise ValueError("new runs require an empty output directory; use --resume")
            self.state = {"schema_version": "1.0", "task": self.task, "task_sha256": fingerprint(self.task),
                          "source": str(task_path), "source_sha256": file_digest(task_path),
                          "invocation_cwd": str(Path.cwd()),
                          "authorized_evidence_paths": [] if handoff else _task_evidence_paths(self.task, Path.cwd()),
                          "source_bundle_root": str(task_path.parent) if handoff else None,
                          "model": model, "status": "running", "turn": 0,
                          "result_layout": result_layout,
                          "candidates": {a: {"revisions": []} for a in ids}, "usage": {}}
            if handoff:
                self.state["discovery_currentness"] = _discovery_currentness(self.task, discovery_run_path)
            initial = {"target": self.task.get("target"),
                       "objective": self.task.get("objective", "model_and_validate"), "candidates": ids,
                       "run_dir": str(self.run_dir), "system": result_layout["system"],
                       "results_index": str(self.run_dir / "results_index.json")}
            if handoff:
                initial["handoff_context"] = {
                    "discovery_status": (self.task.get("producer") or {}).get("discovery_status", "unknown"),
                    "coverage": (self.task.get("producer") or {}).get("coverage", {"status": "unknown"}),
                    "snapshot_sha256": (self.task.get("integrity") or {}).get("snapshot_sha256"),
                    "currentness": self.state["discovery_currentness"],
                    "investigation_leads": len(self.task.get("unresolved_leads") or []),
                    "excluded_candidates": len(self.task.get("excluded_candidates") or []),
                }
            initial_prompt = compact_system_prompt() if self.task.get("progressive_tool_disclosure", True) else SYSTEM_PROMPT
            self.messages = [{"role": "system", "content": initial_prompt},
                             {"role": "user", "content": json.dumps(initial)}]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._resume_requested = resume
        self.llm = llm_client
        self.model = model or self.state.get("model")
        self.backend = json.loads(Path(backend_path).read_text()) if backend_path else self.state.get("backend")
        if resume and backend_path and self.state.get("backend") and self.backend != self.state["backend"]:
            raise ValueError("resume cannot silently replace the saved provider adapter")
        self.state["backend"] = self.backend
        self.barrier_backend = (json.loads(Path(barrier_backend_path).read_text()) if barrier_backend_path
                                else self.state.get('barrier_backend'))
        if resume and barrier_backend_path and self.state.get('barrier_backend') and self.barrier_backend!=self.state['barrier_backend']:
            raise ValueError('Resume cannot silently replace the saved barrier backend')
        self.state['barrier_backend']=self.barrier_backend
        with _controller_lock(self.run_dir):
            if not resume and any((self.run_dir / name).exists() for name in ("checkpoint.json", "agent_state.json")):
                raise ValueError("another controller already initialized this run")
            if not resume:
                task_snapshot = self.run_dir / "task_snapshot.json"
                atomic_write_json(task_snapshot, self.task, allow_nan=False)
                atomic_write_json(self.run_dir / "run_manifest.json", {
                    "schema_version": "structural-run-manifest-2.0", "layout": self.state["result_layout"],
                    "source_task": {"path": self.state["source"], "sha256": self.state["source_sha256"]},
                    "task_snapshot": {"path": str(task_snapshot), "sha256": file_digest(task_snapshot)},
                    "task_sha256": self.state["task_sha256"],
                    "results_index": str(self.run_dir / "results_index.json")}, allow_nan=False)
                if self.task.get("kind") == "discovery_handoff":
                    from enzyme_discovery_agent.models.handoff_contract import read_bundle_source
                    bundled = []
                    for source in self.task.get("evidence_sources", []):
                        info = dict(source) if isinstance(source, dict) else {"availability": "invalid"}
                        if info.get("availability") == "available":
                            try:
                                content = read_bundle_source(task_path.parent, info)
                                folder = self.run_dir / "sources"
                                folder.mkdir(exist_ok=True)
                                destination = folder / (info["sha256"] + ".txt")
                                destination.write_bytes(content)
                                info["local_path"] = str(destination)
                            except (OSError, ValueError, KeyError) as exc:
                                info.update(availability="unavailable", reason=str(exc))
                        bundled.append(info)
                    self.state["bundle_evidence"] = bundled
                from enzyme_discovery_agent.provenance import snapshot_source_code, capture_environment
                self.state["code_snapshots"] = [snapshot_source_code(self.run_dir)]
                self.state["environment"] = capture_environment()
                self._save()

    def _save(self, *, compat: bool = True):
        # This single atomic file is authoritative. The two historical files are
        # compatibility views and can lag safely if a process dies between writes.
        atomic_write_json(self.run_dir / "checkpoint.json", {
            "schema_version": 1, "state": self.state, "messages": self.messages,
        }, allow_nan=False)
        if compat:
            atomic_write_json(self.run_dir / "agent_state.json", self.state, allow_nan=False)
            atomic_write_json(self.run_dir / "transcript.json", self.messages, allow_nan=False)
            from .result_layout import write_index
            write_index(self.run_dir, self.state)

    def _authorized_evidence(self, path, *, allow_dir: bool = False):
        resolved = Path(path).resolve(strict=True)
        allowed = set(self.state.get("authorized_evidence_paths", _task_evidence_paths(
            self.task, Path(self.state.get("invocation_cwd", ".")))))
        generated = any(resolved.is_relative_to(self.run_dir / name)
                        for name in ("sources", "artifacts", "models")) or resolved in {
                            self.run_dir / name for name in ("results_index.json", "run_manifest.json", "task_snapshot.json", "report.json", "structural_modeling_evidence.json")}
        if str(resolved) not in allowed and not generated:
            raise PermissionError("evidence path was not explicitly supplied in the task or produced by this run")
        if allow_dir:
            if not (resolved.is_file() or resolved.is_dir()):
                raise ValueError("evidence path must be a file or directory")
        elif not resolved.is_file():
            raise ValueError("evidence path must be a file")
        return resolved

    def _event(self, kind, **payload):
        with (self.run_dir / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "kind": kind,
                                     "turn": self.state["turn"], **payload}, allow_nan=False) + "\n")

    def _authorize_reasoning_source(self, source):
        if not isinstance(source, dict):
            raise ValueError("reasoning source must be an object")
        if source.get("path"):
            path = self._authorized_evidence(source["path"])
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("reasoning evidence exceeds the 16 MiB inspection limit")
            source["path"] = str(path)

    def _candidate(self, accession):
        original = next(c for c in self.task["candidates"] if c["accession"] == accession)
        return {**original, **self.state["candidates"][accession].get("retrieved_identity", {})}

    def _sequence_identity_error(self, accession, execution=None):
        """Require a separately retrieved, still-fingerprinted identity for handoffs."""
        if self.task.get("kind") != "discovery_handoff":
            return None
        state = self.state["candidates"][accession]
        if state.get("identity_conflict"):
            return "supplied and independently retrieved sequence identities conflict"
        identity = state.get("retrieved_identity") or {}
        if execution is not None:
            identity = {"sequence": self._candidate(accession).get("sequence"),
                        "sequence_source": execution.get("sequence_identity_source")}
            sequence = identity.get("sequence")
            if not isinstance(sequence, str) or hashlib.sha256(sequence.encode("ascii")).hexdigest() != execution.get("sequence_sha256"):
                return "rendered sequence identity differs from the current candidate"
        source = identity.get("sequence_source") or {}
        if not identity.get("sequence") or source.get("source_id") != f"uniprot:{accession}":
            return "independent sequence identity verification is required before rendering"
        try:
            from enzyme_discovery_agent.models.handoff_contract import read_bundle_source, _unique_object, _reject_constant
            path = Path(source["path"]).resolve(strict=True)
            if not path.is_relative_to((self.run_dir / "sources").resolve()):
                raise ValueError("identity source is outside the modeling run")
            raw = read_bundle_source(self.run_dir, {"bundle_relative_path": path.relative_to(self.run_dir).as_posix(),
                                                    "sha256": source["sha256"]})
            entry = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return f"verified sequence source unavailable or changed: {exc}"
        if (entry.get("primaryAccession") != accession or entry.get("inactiveReason")
                or str(entry.get("entryType") or "").casefold() == "inactive"
                or (entry.get("sequence") or {}).get("value") != identity["sequence"]):
            return "verified sequence source no longer establishes this identity"
        return None

    def _current(self, accession):
        revisions = self.state["candidates"][accession]["revisions"]
        if not revisions:
            raise ValueError("inspect the candidate and propose a modeling decision first")
        return revisions[-1]

    def _artifact(self, name, value):
        folder = self.run_dir / "artifacts"
        folder.mkdir(exist_ok=True)
        path = folder / f"{self.state['turn']:05d}_{name}_{uuid.uuid4().hex[:8]}.json"
        atomic_write_json(path, value, allow_nan=False)
        return str(path)

    def _source(self, text, kind, identifier):
        import hashlib
        digest = hashlib.sha256(text.encode()).hexdigest()
        folder = self.run_dir / "sources"
        folder.mkdir(exist_ok=True)
        path = folder / f"{digest}.txt"
        if not path.exists():
            path.write_text(text)
        return {"path": str(path), "sha256": digest, "source_id": f"{kind}:{identifier}",
                "characters": len(text), "excerpt": text[:12000]}

    def _retrieve_reference(self, kind, identifier):
        import requests
        import urllib.parse
        item = urllib.parse.quote(identifier, safe="")
        if kind == "uniprot":
            url = f"https://rest.uniprot.org/uniprotkb/{item}.json"
        elif kind == "pdb":
            url = f"https://data.rcsb.org/rest/v1/core/entry/{item}"
        elif kind == "paper":
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{item}/fullTextXML"
        else:
            raise ValueError("unknown reference kind")
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        text = response.text
        if kind == "paper":
            import xml.etree.ElementTree as ET
            text = " ".join(ET.fromstring(text).itertext())
        source = self._source(text, kind, identifier)
        if kind == "uniprot" and identifier in self.state["candidates"]:
            entry = response.json()
            sequence = (entry.get("sequence") or {}).get("value")
            if (entry.get("primaryAccession") == identifier and not entry.get("inactiveReason")
                    and str(entry.get("entryType") or "").casefold() != "inactive"
                    and isinstance(sequence, str) and re.fullmatch(r"[A-Z]+", sequence)):
                original = next(c for c in self.task["candidates"] if c["accession"] == identifier)
                state = self.state["candidates"][identifier]
                supplied = original.get("sequence") or (state.get("retrieved_identity") or {}).get("sequence")
                if state.get("identity_conflict"):
                    return {**source, "url": url, "identity_status": "conflict"}
                if supplied and supplied != sequence:
                    state["identity_conflict"] = {"reason": "supplied and retrieved sequences differ",
                        "supplied_sha256": hashlib.sha256(supplied.encode()).hexdigest(),
                        "retrieved_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                        "source": {k: source[k] for k in ("path", "sha256", "source_id")}}
                    state.pop("retrieved_identity", None)
                    return {**source, "url": url, "identity_status": "conflict"}
                state.pop("identity_conflict", None)
                state["retrieved_identity"] = {
                    "sequence": sequence, "sequence_source": {k: source[k] for k in ("path", "sha256", "source_id")}}
        return {**source, "url": url}

    def _backend_call(self, operation, revision):
        if not self.backend:
            raise ValueError("no provider adapter configured; preserve a checkpoint or validate existing predictions")
        config = self.backend
        policy = config.get("resource_policy") or {}
        argv = config.get("argv")
        if not config.get("provider") or not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("adapter requires an explicit provider and argv list")
        if policy.get("reservation_managed") is not True:
            raise ValueError("adapter must manage resource reservations with ownership and release")
        for key in ("max_jobs", "max_predictions_per_job"):
            if type(policy.get(key)) is not int or policy[key] < 1:
                raise ValueError(f"explicit positive resource_policy.{key} is required")
        execution = revision["execution"]
        size = execution["modeling_input"]["validation_plan"]["poses"].get("expected_predictions")
        if type(size) is not int or not 1 <= size <= policy["max_predictions_per_job"]:
            raise ValueError("declared prediction count exceeds provider policy or is unresolved")
        key = fingerprint({"execution": execution, "backend": config})
        request = {"operation": operation, "idempotency_key": key, "provider": config["provider"],
                   "input_yaml": revision["input_yaml"], "input_sha256": revision["input_sha256"],
                   "execution_sha256": fingerprint(execution),
                   "expected_predictions": size, "resource_policy": policy,
                   "work_dir": str(Path(revision["input_yaml"]).parent / "predictions")}
        existing = revision.get("job")
        if existing and existing.get("idempotency_key") != key:
            raise ValueError("job identity differs from the current inputs/provider; reconcile before continuing")
        if operation == "submit":
            if existing and existing.get("status") != "not_found":
                return {"status": "already_requested", "job": existing, "next_action": "check_cofold"}
            active = sum(r.get("job", {}).get("status") in {"submission_pending", "queued", "running", "unknown"}
                         for c in self.state["candidates"].values() for r in c["revisions"])
            if active >= policy["max_jobs"]:
                raise ValueError("provider job budget is occupied; inspect existing jobs")
            revision["job"] = {"idempotency_key": key, "status": "submission_pending"}
            self._save()  # A crash must lead to reconciliation, never blind resubmission.
        completed = subprocess.run(argv, input=json.dumps(request), text=True, capture_output=True,
                                   cwd=self.state.get("invocation_cwd", str(self.run_dir)),
                                   timeout=min(float(config.get("timeout_seconds", 60)), 60), check=True)
        reply = json.loads(completed.stdout)
        if reply.get("idempotency_key") != key or reply.get("status") not in {"queued", "running", "complete", "failed", "unknown", "not_found"}:
            raise ValueError("provider returned an invalid or mismatched receipt")
        revision["job"] = reply
        self._save()
        return reply

    def dispatch(self, name, args):
        """The model chooses tools; tools enforce scientific and resource boundaries."""
        return get_registry().dispatch(self, name, args)

    def completion_errors(self):
        missing = []
        for candidate in self.task["candidates"]:
            acc = candidate["accession"]
            from .barrier.workflow import completion_errors as barrier_errors
            missing.extend(f'{acc}: {error}' for error in barrier_errors(self,acc,
                required=self.task.get('objective')=='evaluate_barriers'))
            if self.task.get('objective')=='evaluate_barriers':continue
            if self.task.get("objective") == "prepare_sampling":
                try:
                    from .controller.dispatch_handlers import evaluate_preparation_study
                    result = evaluate_preparation_study(self, acc)
                    if result["status"] != "ready_for_sampling_review":
                        missing.append(f"{acc}: preparation or validation remains unresolved")
                except (ValueError, OSError, KeyError) as exc:
                    missing.append(f"{acc}: {exc}")
                continue
            existing=self.state['candidates'][acc].get('existing_prediction_review')
            if existing:
                from .controller.reuse import checked_evidence
                try:
                    if checked_evidence(existing)['execution_status']!='complete':
                        missing.append(f'{acc}: historical prediction ensemble is incomplete or unverified')
                except (OSError,ValueError,KeyError) as exc:
                    missing.append(f'{acc}: cached evidence changed or is unavailable: {exc}')
                continue
            candidate = self._candidate(acc)
            try:
                revision = self._current(acc)
                missing.extend(f"{acc}: reasoning: {error}" for error in execution_reasoning_errors(revision))
                if validate_pose_plan(revision["modeling_input"], (revision.get("execution") or {}).get("entity_map"))["status"] != "pass":
                    missing.append(f"{acc}: physical acceptance plan is incomplete or invalid")
                if revision.get("execution") and revision["execution"]["modeling_input"] != revision["modeling_input"]:
                    missing.append(f"{acc}: current model differs from the rendered execution")
                identity_error = self._sequence_identity_error(acc, revision.get("execution"))
                if identity_error:
                    missing.append(f"{acc}: {identity_error}")
                if validate_setup(revision["modeling_input"], candidate.get("sequence"))["status"] != "pass":
                    missing.append(f"{acc}: setup is not validated")
                if not revision.get("execution") or file_digest(revision["input_yaml"]) != revision["input_sha256"]:
                    missing.append(f"{acc}: current rendered inputs are missing or changed")
                elif any(file_digest(m["path"]) != m["sha256"] for m in revision["execution"].get("msa_evidence", [])):
                    missing.append(f"{acc}: input MSA changed after rendering")
                if revision.get("proxy_evaluation") is not None:
                    from .controller.integrity import proxy_integrity
                    integrity = proxy_integrity(revision.get("proxy_plan"), revision["proxy_evaluation"],
                                                revision.get("proxy_evaluation_artifact"),
                                                (revision.get("pose_validation") or {}).get("poses", []))
                    if integrity["status"] != "verified":
                        missing.append(f"{acc}: proxy evaluation integrity is not verified: " + "; ".join(integrity["errors"]))
                if self.task.get("objective", "model_and_validate") != "prepare_inputs":
                    report = revision.get("pose_validation") or {}
                    if report.get("status") != "pass" or report.get("execution_sha256") != fingerprint(revision.get("execution")):
                        missing.append(f"{acc}: current prediction ensemble is not validated")
                    elif report.get("validator_sha256") != file_digest(Path(__file__).with_name("validation.py")):
                        missing.append(f"{acc}: prediction checks were produced by a different validator implementation")
                    elif any(not p.get("sha256") or file_digest(p["path"]) != p["sha256"] for p in report["poses"]):
                        missing.append(f"{acc}: prediction files changed after validation")
                    provenance = report.get("prediction_provenance") or {}
                    if provenance.get("status") != "pass":
                        missing.append(f"{acc}: prediction provenance is not verified for the current inputs")
                    elif file_digest(provenance["path"]) != provenance["sha256"]:
                        missing.append(f"{acc}: producer manifest changed after validation")
                    if any(not s.get("sha256") or file_digest(s["path"]) != s["sha256"] for s in provenance.get("supporting_files", [])):
                        missing.append(f"{acc}: producer supporting files changed after validation")
            except (ValueError, OSError, KeyError, TypeError):
                missing.append(f"{acc}: decision or supporting files are missing")
        return missing

    def _context(self):
        # Preserve whole assistant/tool groups; truncating by message count can
        # leave orphan tool results that providers reject.
        groups = []
        for message in self.messages[2:]:
            if message["role"] == "assistant" or not groups:
                groups.append([])
            groups[-1].append(message)
        prefix=copy.deepcopy(self.messages[:2])
        objective=self.task.get('objective')
        prefix[0]={"role":"system","content":(BARRIER_PROMPT if objective=='evaluate_barriers' else
                    PREPARATION_PROMPT if objective == 'prepare_sampling' else (
                        compact_system_prompt() if (self.task.get('workflow_mode') != 'production_cofold' and self.task.get('progressive_tool_disclosure', True)) else SYSTEM_PROMPT + BARRIER_GUIDANCE
                    ))}
        # Scientific field definitions must survive ordinary history compaction.
        # Otherwise a model can spend every turn rereading a truncated schema.
        cached_only=all((candidate.get('evidence') or {}).get('do_not_recofold')
                        and (candidate.get('evidence') or {}).get('existing_cofolds') for candidate in self.task['candidates'])
        from .controller.dispatch_handlers import validation_contract
        contract = ({'workflow':'Review the original saved computation, not a fresh modeling experiment.',
                     'action':'review_existing_predictions','required':['accession','setup_id','applicability','interpretation','unresolved'],
                     'scope':'Source checks and frozen measurements remain attributed to the original model. Assess applicability and uncertainties; activity is not established. No new proposal or cofold is required.'}
                    if cached_only else validation_contract("all" if self.task.get('workflow_mode')=='production_cofold' else "scientific_reasoning"))
        if self.task.get("objective") == "prepare_sampling":
            from .sampling.preparation import contract as prep_contract
            contract = prep_contract()
            prefix.append({"role": "user", "content": json.dumps({
                "preparation_instructions": self.task.get("preparation_instructions", "Investigate and validate the requested molecular preparation."),
                "evidence_paths": self.state.get("authorized_evidence_paths", []),
                "objective": "prepare_sampling"})})
        if objective=='evaluate_barriers':
            from .barrier.api import contract as barrier_contract
            from .barrier.workflow import summary as barrier_summary
            contract=barrier_contract()
            prefix.append(dict(role='user',content=json.dumps(dict(
                objective=objective,barrier_studies={acc:barrier_summary(value.get('barrier_studies',{}))
                    for acc,value in self.state['candidates'].items()}))))
        prefix[0]["content"] += "\n\nScientific record field contract (trusted tool schema):\n" + json.dumps(contract)
        # Keep loaded skill playbooks pinned across history compaction so the agent
        # never loses active domain instructions after 1-3 turns.
        active_skills = self.state.get("active_skills") or []
        skill_block = ""
        if active_skills:
            from .controller.skills import get_skill
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
        anchor=self.state.get('setup_anchor') if self._before_proposal() and objective not in {'prepare_sampling','evaluate_barriers'} else None
        if anchor:
            prefix[0]["content"] += "\n\nCandidate record under setup (trusted task input):\n" + anchor
        recent=copy.deepcopy(groups[-2:] if self.task.get("objective") == "prepare_sampling" else groups[-3:])
        for index,group in enumerate(recent):
            for message in group:
                if message['role']=='tool':
                    value=json.loads(message['content'])
                    limit=6000 if index==len(recent)-1 else 1200
                    original=str(value.get('summary',''))
                    value['summary']=original[:limit]
                    if len(original)>limit:
                        value['preview_only']=True
                        value['inspection']='Full result is in the artifact. Use read_evidence with its path and offset for a focused passage.'
                    message['content']=json.dumps(value)
        state = {}
        for accession, candidate in self.state["candidates"].items():
            revision = candidate["revisions"][-1] if candidate["revisions"] else {}
            assessment = assess_reasoning(revision) if revision else {}
            contradictions = [p["prediction_id"] for p in assessment.get("predictions", []) if p["status"] == "contradicted"]
            state[accession] = {"revision": revision.get("revision"),
                "preparation_study_count": len(candidate.get("preparation_studies", {})),
                "preparation_findings": self._preparation_findings(candidate),
                "latest_preparation_study": next(reversed(candidate.get("preparation_studies", {})), None),
                "sequence_sha256": (hashlib.sha256(self._candidate(accession)["sequence"].encode()).hexdigest()
                                    if self._candidate(accession).get("sequence") else None),
                "target_sha256": fingerprint(self.task["target"]),
                "sampling_setup_count": len(candidate.get("sampling_setups", {})),
                "latest_sampling_setup": next(reversed(candidate.get("sampling_setups", {})), None),
                'existing_prediction_review':candidate.get('existing_prediction_review'),
                "setup": (revision.get("setup_validation") or {}).get("status"),
                "poses": (revision.get("pose_validation") or {}).get("status"),
                "followups": [{"id": key, "purpose": study["plan"]["purpose"], "status": study["status"]}
                              for key, study in list(revision.get("followups", {}).items())[:5]],
                "followup_count": len(revision.get("followups", {})),
                "reasoning": {"selected_hypothesis_id": str(assessment.get("selected_hypothesis_id"))[:160],
                    "assessment_sha256": assessment.get("assessment_sha256"),
                    "response_current": assessment.get("response_current", False),
                    "requires_response": assessment.get("requires_response", False),
                    "contradicted_predictions": [str(p)[:160] for p in contradictions[:5]],
                    "contradiction_count": len(contradictions), "source_error_count": len(assessment.get("source_errors", [])),
                    "details": "read_reasoning_section"}}
        sources = copy.deepcopy(self.state.get("inspected_source_passages", []))
        for source in sources:
            try:
                source["integrity"] = "unchanged" if file_digest(source["path"]) == source["sha256"] else "changed"
            except OSError:
                source["integrity"] = "unavailable"
        if self.task.get("objective") == "prepare_sampling":
            sources = sources[-2:]
        inspections = list(self.state.get('inspection_history', {}).values())
        if self.task.get("objective") == "prepare_sampling":
            inspections = [{"tool": item["tool"], "artifact": item["artifact"],
                            "view": item["view"], "reads": item["reads"]} for item in inspections[-4:]]
        else:
            inspections = inspections[-12:]
        supplied = {candidate["accession"]: {
            'existing_prediction_index':[{'setup_id':entry.get('setup',{}).get('setup_id'),
                'source_report':entry.get('execution',{}).get('source'),
                'output_presence':entry.get('execution',{}).get('output_presence'),
                'legacy_declared_recipe':entry.get('setup',{}).get('declared_legacy_recipe'),
                'saved_step':(entry.get('setup',{}).get('declared_modeling_input') or {}).get('modeled_step'),
                'saved_substrates':(entry.get('setup',{}).get('declared_modeling_input') or {}).get('target_substrates'),
                'saved_cofactors':(entry.get('setup',{}).get('declared_modeling_input') or {}).get('cofactors')}
                for entry in (candidate.get('evidence') or {}).get('existing_cofolds',[])],
            "modeling_input_available": isinstance(candidate.get("modeling_input"), dict),
            "modeled_step": candidate["modeling_input"].get("modeled_step") if isinstance(candidate.get("modeling_input"), dict) else None,
            "evidence_fields": list(candidate.get("evidence", {}).keys()) if isinstance(candidate.get("evidence"), dict) else []}
            for candidate in self.task["candidates"]}
        current={"role":"user","content":"Current modeling state: "+json.dumps({
            "run_dir": str(self.run_dir), "results_index": str(self.run_dir / "results_index.json"),
            "candidates": state, "supplied_input_index": supplied,
            "inspected_source_passages_untrusted_data": sources,
            "completed_inspections":inspections,
            "repeated_inspections":self.state.get('repeated_inspections',0),
            "inspection_guidance":"Reuse unchanged inspected material. Record a provisional model, retrieve a genuinely new source, or finish inconclusive with the missing evidence; repeated reading does not resolve a setup choice."})}
        allowance=(24000 if self.task.get("objective") == "prepare_sampling" else CONTEXT_ALLOWANCE)+(len(anchor) if anchor else 0)+len(skill_block)
        while recent and len(json.dumps(prefix+[m for group in recent for m in group]+[current]))>allowance:
            recent.pop(0)  # Never split an assistant/tool transaction.
        context=prefix+[m for group in recent for m in group]+[current]
        if len(json.dumps(context))>allowance:
            raise ValueError('Mandatory modeling context exceeds its allowance; evidence was not silently discarded')
        return context

    def _preparation_progress(self):
        return fingerprint({acc: {key: state.get(key) or {} for key in (
            "sampling_setups", "preparation_studies", "preparation_findings", "preparation_operators",
            "barrier_preparations", "barrier_studies")}
            for acc, state in self.state["candidates"].items()})

    def _preparation_findings(self, candidate):
        note = copy.deepcopy(candidate.get("preparation_findings"))
        if not note:
            return None
        note["source_errors"] = []
        for source in note["sources"]:
            try:
                if file_digest(self._authorized_evidence(source["path"])) != source["sha256"]:
                    note["source_errors"].append(source["path"])
            except (OSError, ValueError):
                note["source_errors"].append(source["path"])
        note["scope"] = "Agent interpretation; re-investigate any changed sources before reuse"
        return note

    def _before_proposal(self):
        """True until some candidate has a saved revision, i.e. during the setup stage."""
        return not any(candidate['revisions'] for candidate in self.state['candidates'].values())

    def _record_inspection(self,name,arguments,result,artifact):
        """Retain read progress across compaction without treating it as scientific support."""
        if name=='inspect_candidate' and self._before_proposal() and isinstance(result,dict):
            body=json.dumps(result,allow_nan=False)
            # Only anchor a record that is actually carried whole; a truncated anchor
            # would reproduce the very preview the model keeps trying to escape.
            self.state['setup_anchor']=body if len(body)<=SETUP_ANCHOR_LIMIT else None
        readers=_REPEATABLE_READERS
        if name not in readers:
            self.state['repeated_inspections']=0
            return
        history=self.state.setdefault('inspection_history',{})
        content=result
        if name=='read_evidence' and isinstance(result,dict):
            if result.get('offset',0)==0 and result.get('next_offset') is None:
                try:content=json.loads(result.get('text',''))
                except (ValueError,TypeError):content=result
            if content is result and result.get('sha256'):
                content={key:result.get(key) for key in ('sha256','offset','next_offset','text')}
        identity=fingerprint(content)
        previous=history.pop(identity,None)
        summary={key:result.get(key) for key in ('path','sha256','section','pointer','offset','next_offset','characters') if isinstance(result,dict) and key in result}
        history[identity]={'tool':name,'arguments':arguments,'artifact':artifact,'result_sha256':fingerprint(result),
                           'view':summary,'reads':(previous or {}).get('reads',0)+1}
        self.state['repeated_inspections']=self.state.get('repeated_inspections',0)+1 if previous else 0
        while len(history)>64:history.pop(next(iter(history)))

    def _repair_interrupted_transcript(self):
        repaired, index = [], 0
        while index < len(self.messages):
            message = self.messages[index]
            repaired.append(message)
            index += 1
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            completed = set()
            while index < len(self.messages) and self.messages[index].get("role") == "tool":
                result = self.messages[index]
                completed.add(result["tool_call_id"])
                repaired.append(result)
                index += 1
            for call in message["tool_calls"]:
                if call["id"] not in completed:
                    repaired.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({
                        "status": "interrupted", "summary": "Tool completion was not checkpointed. Inspect saved decisions and reconcile provider state before retrying mutations."})})
                    self._event("interrupted_tool_recovered", tool=call["function"]["name"])
        self.messages = repaired

    def run(self, *, max_turns=30, token_budget=100000, runtime_budget_seconds=1800):
        if type(max_turns) is not int or max_turns < 1 or token_budget < 1 or runtime_budget_seconds <= 0:
            raise ValueError("turn, token and runtime allowances must be positive")
        from enzyme_discovery_agent.llm.client import LLMClient, TokenUsage
        # Separate-process lock enforces one decision owner per modeling run.
        with _controller_lock(self.run_dir):
            if self._resume_requested:
                self.state, self.messages = _load_checkpoint(self.run_dir)
                self.task = self.state["task"]
                if fingerprint(self.task) != self.state["task_sha256"]:
                    raise ValueError("saved target/candidate task changed after initialization")
                self.state["backend"] = self.backend
                self.state['barrier_backend']=self.barrier_backend
                self._repair_interrupted_transcript()
                if self.task.get("kind") == "discovery_handoff":
                    previous = self.state.get("discovery_currentness", {})
                    if self._discovery_run_path is not None:
                        currentness = _discovery_currentness(self.task, self._discovery_run_path)
                    else:
                        currentness = {"status": "unknown", "reason": "not rechecked on resume",
                                       "previous_check": previous}
                    self.state["discovery_currentness"] = currentness
                    try:
                        initial = json.loads(self.messages[1]["content"])
                        initial.setdefault("handoff_context", {})["currentness"] = currentness
                        self.messages[1]["content"] = json.dumps(initial)
                    except (ValueError, KeyError, IndexError, TypeError):
                        self._event("handoff_currentness_context_unavailable", currentness=currentness)
            if self._resume_requested or not self.state.get("code_snapshots"):
                from enzyme_discovery_agent.provenance import snapshot_source_code, capture_environment
                snapshot = snapshot_source_code(self.run_dir)
                self.state["environment"] = capture_environment()
                if not any(s["source_sha256"] == snapshot["source_sha256"] for s in self.state.get("code_snapshots", [])):
                    self.state.setdefault("code_snapshots", []).append(snapshot)
            if self.state["status"] == "complete" and not self.completion_errors():
                self._save()
                return self._write_report()
            if self.task.get("kind") == "discovery_handoff" and not self.task.get("candidates"):
                self.state["status"] = "no_admitted_candidates"
                self._save()
                return self._write_report()
            if self.llm is None:
                self.llm = LLMClient(model=self.model) if self.model else LLMClient()
            self.state["model"] = self.model or getattr(self.llm, "requested_model", None)
            usage = TokenUsage.from_dict(self.state["usage"]) if self.state["usage"] else TokenUsage(model=self.model or "")
            start_tokens, started = usage.total_tokens, time.monotonic()
            def budget_reason():
                if usage.total_tokens-start_tokens >= token_budget:
                    return "model token allowance exhausted"
                if time.monotonic()-started >= runtime_budget_seconds:
                    return "runtime allowance exhausted"
                return None
            if isinstance(self.llm, LLMClient):
                self.llm.attach_usage(usage, self.run_dir)
                self.llm.budget_check = budget_reason
            # A resumed controller reconciles outstanding provider receipts before
            # asking the model for another decision. Status calls never submit.
            if self._resume_requested and self.backend:
                for candidate_state in self.state["candidates"].values():
                    for revision in candidate_state["revisions"]:
                        if budget_reason():
                            break
                        if revision.get("job", {}).get("status") not in {"submission_pending", "queued", "running", "unknown"}:
                            continue
                        try:
                            receipt = self._backend_call("status", revision)
                            self._event("job_reconciled", status=receipt["status"],
                                        idempotency_key=receipt["idempotency_key"])
                        except Exception as exc:
                            # Preserve the previous pending state: an unavailable
                            # status endpoint is not evidence that a job vanished.
                            self._event("job_reconciliation_failed", error=f"{type(exc).__name__}: {exc}")
            if self._resume_requested:
                from .barrier.workflow import ACTIVE,reconcile
                for accession,candidate_state in self.state['candidates'].items():
                    for identifier,study in candidate_state.get('barrier_studies',{}).items():
                        if budget_reason():break
                        if study.get('job',{}).get('status') not in ACTIVE:continue
                        try:
                            reply=reconcile(self,accession,identifier,'status')
                            self._event('barrier_job_reconciled',study_id=identifier,status=reply['status'])
                        except Exception as exc:
                            self._event('barrier_job_reconciliation_failed',study_id=identifier,error=str(exc))
                from .sampling.workflow import ACTIVE as SAMPLING_ACTIVE, check_sampling
                for accession, candidate_state in self.state["candidates"].items():
                    for job_id, job_rec in candidate_state.get("sampling_jobs", {}).items():
                        if budget_reason():
                            break
                        if (job_rec.get("job") or {}).get("status") not in SAMPLING_ACTIVE:
                            continue
                        try:
                            reply = check_sampling(self, {"accession": accession, "job_id": job_id})
                            self._event("sampling_job_reconciled", job_id=job_id, status=reply["status"])
                        except Exception as exc:
                            self._event("sampling_job_reconciliation_failed", job_id=job_id, error=str(exc))
            self.state["status"] = "running"
            try:
                for _ in range(max_turns):
                    reason = budget_reason()
                    if reason:
                        self.state.update(status="budget_exhausted", checkpoint_reason=reason)
                        break
                    self.state["turn"] += 1
                    offered = select_offered_tools(self)
                    if offered is None:
                        break
                    offered_names = {tool["function"]["name"] for tool in offered}
                    response = self.llm.call(messages=self._context(), tools=offered, timeout=120,
                                             usage_label=f"modeling_{self.state['turn']}")
                    if response is None:
                        self.state.update(status="budget_exhausted" if budget_reason() else "provider_unavailable")
                        break
                    usage.record(self.state["turn"], response)
                    message = response.choices[0].message
                    calls, content, _ = LLMClient.extract_tool_calls(message)
                    dumped = message.model_dump()
                    if calls:
                        dumped["tool_calls"] = [c.model_dump() if hasattr(c, "model_dump") else c for c in calls]
                    self.messages.append(dumped)
                    self._event("decision", rationale=content or "", tools=[
                        c.function.name if hasattr(c, "function") else c["function"]["name"] for c in calls])
                    preparation_before = self._preparation_progress()
                    for call in calls:
                        function = call.function if hasattr(call, "function") else call["function"]
                        name = function.name if hasattr(function, "name") else function["name"]
                        raw = function.arguments if hasattr(function, "arguments") else function["arguments"]
                        call_id = call.id if hasattr(call, "id") else call["id"]
                        try:
                            arguments = json.loads(raw) if isinstance(raw, str) else raw
                            if name not in offered_names:
                                if self.state.get("repeated_inspections", 0) >= 3 and name in _REPEATABLE_READERS:
                                    result={'status':'blocked','reason':'Choose a new evidence action, provisional proposal, or explicit inconclusive outcome after repeated unchanged inspections.'}
                                    self.state.update(status='inspection_stalled',checkpoint_reason=result['reason'])
                                else:
                                    registry = get_registry()
                                    hint = f"Tool {name!r} is not currently offered in this stage."
                                    if name in registry.names():
                                        domain = registry.get(name).domain
                                        from .controller.skills import list_skills
                                        matching = [s["name"] for s in list_skills() if domain in s["domains"]]
                                        if matching and "load_skill" in offered_names:
                                            hint = (f"Tool {name!r} belongs to the {domain!r} domain, which is not yet unlocked. "
                                                    f"Call load_skill(name=...) with one of {matching} first.")
                                    result = {"status": "blocked", "reason": hint, "offered_tools": sorted(offered_names)}
                            elif self.state["status"] != "running" or (budget_reason() and not (
                                    self.task.get("objective") == "prepare_sampling" and name in {
                                        "record_preparation_findings", "declare_sampling_setup", "declare_preparation_study", "finish_task"})):
                                result = {"status": "deferred", "reason": "resource boundary or task already finished"}
                            else:
                                result = self.dispatch(name, arguments)
                        except Exception as exc:
                            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                        artifact = self._artifact(name, result)
                        self._record_inspection(name,arguments,result,artifact)
                        body = json.dumps(result, allow_nan=False)
                        payload = {"artifact": artifact, "summary": body[:20000]}
                        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)})
                        self._event("tool_result", tool=name, artifact=artifact)
                        self.state["usage"] = usage.to_dict()
                        self._save(compat=False)
                        if (self.task.get('workflow_mode')=='production_cofold' and name=='render_inputs'
                                and result.get('status')=='rendered'):
                            self.state.update(status='ready_for_native_execution')
                    if calls:
                        self._save(compat=True)
                    if self.task.get('objective') == 'prepare_sampling':
                        changed = self._preparation_progress() != preparation_before
                        self.state['preparation_read_turns'] = 0 if changed else self.state.get('preparation_read_turns', 0) + 1
                    if not calls:
                        self.messages.append({"role": "user", "content": "Use tools to investigate/revise, or finish_task with a supported outcome. Prose alone does not complete this task."})
                    if self.state["status"] != "running":
                        break
                else:
                    self.state.update(status="budget_exhausted", checkpoint_reason="decision-turn allowance exhausted")
            except Exception as exc:
                self.state.update(status="failed", checkpoint_reason=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if isinstance(self.llm, LLMClient):
                    usage.model = self.llm.active_label
                self.state["usage"] = usage.to_dict()
                self._save()
                report = self._write_report()
            return report

    def _write_report(self):
        candidates = []
        for acc, state in self.state["candidates"].items():
            current = state["revisions"][-1] if state["revisions"] else {}
            candidates.append({"accession": acc, "revision": current.get("revision"),
                               "barrier_studies":copy.deepcopy(state.get('barrier_studies',{})),
                               "barrier_preparations":copy.deepcopy(state.get('barrier_preparations',[])),
                               "sampling_jobs": copy.deepcopy(state.get("sampling_jobs", {})),
                               "catalytic_profile": copy.deepcopy(state.get("catalytic_profile")),
                               "developability": copy.deepcopy(state.get("developability")),
                               "preparation_studies": [{"id": key, "artifact": value["artifact"]}
                                                       for key, value in state.get("preparation_studies", {}).items()],
                               "preparation_operators": [{"artifact": value["artifact"], "status": value["status"]}
                                                         for value in state.get("preparation_operators", [])],

                               'existing_prediction_review':state.get('existing_prediction_review'),
                               "input_derivation": current.get("input_derivation"),
                               "bioemu2_preparation": current.get("bioemu2_preparation"),
                               "followups": current.get("followups", {}),
                               "rationale": current.get("rationale"), "alternatives": current.get("alternatives"),
                               "scientific_reasoning": current.get("scientific_reasoning"),
                               "reasoning_registration": current.get("reasoning_registration"),
                               "reasoning_assessment": assess_reasoning(current) if current else None,
                               "reasoning_response": current.get("reasoning_response"),
                               "reasoning_observations": current.get("reasoning_observations", []),
                               "reasoning_history": [{"revision": r["revision"], "scientific_reasoning": r.get("scientific_reasoning"),
                                                      "followups": r.get("followups", {}),
                                                      "registration": r.get("reasoning_registration"),
                                                      "assessment": assess_reasoning(r),
                                                      "responses": r.get("reasoning_response_history", []),
                                                      "decision_artifact": r.get("decision_artifact")}
                                                     for r in state["revisions"]],
                               "modeling_input": current.get("modeling_input"),
                               "setup_validation": current.get("setup_validation"),
                               "pose_plan_validation": current.get("pose_plan_validation"),
                               "pose_validation": current.get("pose_validation"),
                               "input_sha256": current.get("input_sha256"),
                               "execution_sha256": fingerprint(current["execution"]) if current.get("execution") else None,
                               "execution_reasoning_sha256": (current.get("execution") or {}).get("scientific_reasoning_sha256"),
                               "msa_evidence": (current.get("execution") or {}).get("msa_evidence"),
                               "sequence_identity_source": (current.get("execution") or {}).get("sequence_identity_source"),
                               "sequence_sha256": (current.get("execution") or {}).get("sequence_sha256"),
                               "proxy_evaluation": current.get("proxy_evaluation"),
                               "proxy_plan": current.get("proxy_plan"),
                               "proxy_plan_registration": current.get("proxy_plan_registration"),
                               "proxy_evaluation_artifact": current.get("proxy_evaluation_artifact"),
                               "input_yaml": current.get("input_yaml"), "job": current.get("job"),
                               "decision_artifact": current.get("decision_artifact"),
                               "validation_artifact": current.get("validation_artifact")})
        report = {"schema_version": "modeling-agent-1.0", "status": self.state["status"], "run_dir": str(self.run_dir),
                  "objective":self.task.get('objective','model_and_validate'),
                  "decision_origin":self.state.get('decision_origin'),
                  "result_layout": self.state.get("result_layout"), "results_index": str(self.run_dir / "results_index.json"),
                  "target": self.task.get("target"), "task_sha256": self.state["task_sha256"],
                  "source_discovery_run": self.task.get("source_run"),
                  "conclusion": self.state.get("conclusion"), "completion_gaps": self.completion_errors(),
                  "usage": self.state["usage"], "candidates": candidates,
                  "code_snapshots": self.state.get("code_snapshots", []),
                  "environment": self.state.get("environment"),
                  "evidence_scope": ("Barrier execution and reviewed constrained-energy observations; no activation-free-energy or activity claim."
                                     if self.task.get('objective')=='evaluate_barriers' else
                                     "Preparation evidence and agent-reviewed validation; no production dynamics or chemical accuracy claim."
                                     if self.task.get("objective") == "prepare_sampling" else
                                     "Setup consistency and declared pose checks; no activity prediction or automatic discovery reranking.")}
        if self.task.get("kind") == "discovery_handoff":
            report["sequence_identity_requirement"] = "exact_uniprot_primary"
            report["discovery_handoff"] = {
                "discovery_status": (self.task.get("producer") or {}).get("discovery_status", "unknown"),
                "coverage": (self.task.get("producer") or {}).get("coverage", {"status": "unknown"}),
                "snapshot_sha256": (self.task.get("integrity") or {}).get("snapshot_sha256"),
                "currentness": self.state.get("discovery_currentness", {"status": "unknown"}),
                "unresolved_leads": [{"accession": row.get("accession"), "reason": row.get("reason")}
                                     for row in self.task.get("unresolved_leads", [])],
                "excluded_candidates": [{"accession": row.get("accession"), "reason": row.get("reason")}
                                        for row in self.task.get("excluded_candidates", [])],
            }
        atomic_write_json(self.run_dir / "report.json", report, allow_nan=False)
        from .evidence import export_modeling_evidence
        export_modeling_evidence(self.run_dir / "report.json")
        from .result_layout import write_index
        write_index(self.run_dir, self.state)
        return report
