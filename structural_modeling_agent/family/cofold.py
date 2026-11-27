"""Compile shared family protocols and execute cofolds without per-member LLM calls.

Core logic lives in cohesive submodules:
- ``family.protocol``: protocol compilation, handoff grouping, ortholog dedup, Markdown export
- ``family.queue``: job rendering, prioritization, cross-run reuse indexing, result publishing
- ``family.executor``: reserved native-provider execution, ownership checks, contention retries
- ``family.pose_analysis``: per-pose identity, geometry, and holo-seating diagnostics

All public and compatibility symbols are re-exported here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess  # noqa: F401 - preserved for test monkeypatch compatibility
import time

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from ..scripts.build_step1_inputs import enzyme_chains, modeled_substrates, yaml_for
from .executor import execute_job as _execute_job, provider_owner_live
from .memory_retry import MemoryRetryPending, is_memory_failure
from .pose_analysis import (
    _METAL_ELEMENTS,
    _POLAR_COORD_ELEMENTS,
    _analyze_single_prediction,
    _holo_seating_metrics,
    _summarize_holo_seating,
    analyze_outputs,
)
from .protocol import (
    COFOLD_ARM_NAMES,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    _INTERFACIAL_ARCH_RE,
    _as_bool,
    _canonical_family,
    _family,
    _interfacial_assembly_override,
    _load_discovery_handoff_index,
    adopt_registered_protocol,
    build_handoff as _build_handoff,
    compile_protocol,
    deduplicate_handoff_orthologs,
    digest,
    file_hash,
    register_protocol,
    target_key,
    write_family_handoff_markdown,
)
from .queue import (
    _prepared_candidates,
    build_reusable_index,
    find_reusable as _find_reusable,
    prioritize_modeling_jobs,
    publish_results,
    render_jobs,
    write_input_index,
)


def build_handoff(run_dir, specification):
    return _build_handoff(run_dir, specification, registry_loader=load_candidate_registry)


def find_reusable(job, run, specification, *, reusable_index=None):
    return _find_reusable(job, run, specification, reusable_index=reusable_index, analyzer=analyze_outputs)


def execute_job(job, backend, directory, resource_attempt=0):
    return _execute_job(job, backend, directory, resource_attempt=resource_attempt, owner_live_fn=provider_owner_live)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot-accession")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pilot-steps", default="1,2")
    parser.add_argument("--max-protein-length", type=int)
    parser.add_argument("--expand-after-pilot", action="store_true")
    parser.add_argument("--worker-id", default="worker-0")
    parser.add_argument("--msa-cache", type=Path)
    parser.add_argument("--watch-seconds", type=int, default=0)
    parser.add_argument("--resolver-model", help="Optional LLM model for once-per-family protocol resolution of unresolved targets or pending families")
    parser.add_argument("--no-llm-resolver", action="store_true", help="Disable automatic once-per-family LLM protocol resolution")
    parser.add_argument("--memory-retries-only", action="store_true", help="Recover only saved GPU-memory failures; reuse the existing prepared queue")
    parser.add_argument("--pocket-guided-from", type=Path, help="Prepare a separate broad-pocket arm from this saved unrestrained family queue")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch("[A-Za-z0-9_.-]+", args.worker_id):
        raise ValueError("Unsafe worker identity")

    failed = 0
    seen = set()
    with state_lock(args.out / "workers" / args.worker_id / "lock"):
        atomic_write_json(
            args.out / "workers" / args.worker_id / "owner.json",
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "process_start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "started_at": time.time(),
            },
        )
        while True:
            resolver_model = None if args.no_llm_resolver else args.resolver_model
            setup_llm_calls = 0
            if not args.specification.exists():
                if adopt_registered_protocol(args.run, args.specification):
                    continue
                if resolver_model:
                    from .resolver import resolve_target_protocol

                    resolved_spec, res_info = resolve_target_protocol(
                        args.run, args.specification, args.out, model=resolver_model
                    )
                    setup_llm_calls += int(res_info.get("llm_calls") or 0)
                    if resolved_spec is not None:
                        continue
                atomic_write_json(
                    args.out / "workers" / args.worker_id / "current.json",
                    {"status": "awaiting_shared_family_protocol", "path": str(args.specification), "setup_llm_calls": setup_llm_calls},
                )
                if not args.watch_seconds:
                    return
                time.sleep(args.watch_seconds)
                continue

            specification = json.loads(args.specification.read_text())
            if args.pocket_guided_from:
                from .pockets import prepare

                jobs = prepare(
                    args.run,
                    args.pocket_guided_from,
                    args.out,
                    specification.get("pocket_max_distance_A", 10.0),
                    specification.get("pocket_rescue_all", False),
                )
            elif args.memory_retries_only:
                jobs = json.loads((args.out / "queue.json").read_text())["items"]
            else:
                handoff = build_handoff(args.run, specification)
                if resolver_model and handoff.get("pending_families"):
                    from .resolver import resolve_pending_families

                    specification, res_report = resolve_pending_families(
                        args.run, handoff, specification, args.specification, args.out, model=resolver_model
                    )
                    setup_llm_calls += int(res_report.get("llm_calls") or 0)
                    if res_report.get("resolved"):
                        handoff = build_handoff(args.run, specification)
                handoff["setup_llm_calls"] = setup_llm_calls
                from .sequences import prepare_members

                candidates = prepare_members(
                    handoff,
                    _prepared_candidates(args.run),
                    args.out / "sequences",
                    args.msa_cache or args.run.parent / ".sequence_msa_cache",
                )
                deduplicate_handoff_orthologs(
                    handoff,
                    candidates=candidates,
                    identity_threshold=specification.get("ortholog_identity_threshold", 0.85),
                )
                atomic_write_json(args.out / "family_handoff.json", handoff)
                write_family_handoff_markdown(handoff, args.out / "family_handoff.md")
                jobs, pending = render_jobs(args.run, handoff, args.out / "inputs", candidates)
                atomic_write_json(args.out / "queue.json", {"items": jobs, "pending": pending, "setup_llm_calls": setup_llm_calls})
                write_input_index(jobs, args.out)

            publish_results(args.out, jobs)
            if args.prepare_only:
                return

            pilot_steps = {int(value) for value in args.pilot_steps.split(",")}
            pilot_jobs = [j for j in jobs if j["accession"] == args.pilot_accession and j["step"] in pilot_steps]
            pilot_passed = bool(pilot_jobs) and all((args.out / "jobs" / j["id"] / "complete.json").is_file() for j in pilot_jobs)
            if pilot_passed:
                pilot_passed = all(
                    json.loads((args.out / "jobs" / j["id"] / "complete.json").read_text())["analysis"]["identity_pass_poses"]
                    == int(j.get("expected_predictions", DEFAULT_SAMPLES))
                    for j in pilot_jobs
                )
            selected = [
                j
                for j in jobs
                if not args.pilot_accession
                or (j["accession"] == args.pilot_accession and j["step"] in pilot_steps)
                or (args.expand_after_pilot and pilot_passed)
            ]
            if args.memory_retries_only:
                def memory_pending(job):
                    folder = args.out / "jobs" / job["id"]
                    receipt = folder / "receipt.json"
                    return not (folder / "complete.json").exists() and (
                        (folder / "memory_retry.json").exists()
                        or (receipt.exists() and is_memory_failure(json.loads(receipt.read_text())))
                    )

                selected = [j for j in selected if memory_pending(j)]

            backend_definition = json.loads(args.backend.read_text())
            provider_root = Path(backend_definition["argv"][backend_definition["argv"].index("--state-dir") + 1])
            owned_native = {
                p.parent.name
                for p in (provider_root / "backend_jobs").glob("*/receipt.json")
                if json.loads(p.read_text()).get("status") in {"queued", "running", "unknown", "complete"}
                and not (args.out / "jobs" / p.parent.name / "complete.json").exists()
            }
            if owned_native:
                selected = [j for j in selected if j["id"] in owned_native]
            if args.max_protein_length:
                import yaml

                selected = [
                    j
                    for j in selected
                    if max(len(e["protein"]["sequence"]) for e in yaml.safe_load(Path(j["input_yaml"]).read_text())["sequences"] if e.get("protein"))
                    <= args.max_protein_length
                ]
            selected = prioritize_modeling_jobs(selected)
            deferred = []
            reusable_index = build_reusable_index(args.run)
            for job in {j["id"]: j for j in selected}.values():
                if job["id"] in seen:
                    continue
                folder = args.out / "jobs" / job["id"]
                folder.mkdir(parents=True, exist_ok=True)
                complete = folder / "complete.json"
                if complete.exists():
                    saved = json.loads(complete.read_text())
                    if saved["job"]["input_sha256"] != job["input_sha256"]:
                        raise ValueError("Cached input identity changed")
                    analyze_outputs(saved.get("reuse_source_job") or job, saved["receipt"])
                    seen.add(job["id"])
                    continue
                reused = find_reusable(job, args.run, specification, reusable_index=reusable_index)
                if reused:
                    if reused["status"] == "reused":
                        atomic_write_json(
                            complete,
                            {
                                "job": job,
                                "reuse_source_job": reused["job"],
                                "receipt": reused["receipt"],
                                "analysis": reused["analysis"],
                                "reuse_source": reused["source"],
                            },
                        )
                    else:
                        atomic_write_json(folder / "reuse_pending_analysis.json", reused)
                    seen.add(job["id"])
                    continue

                claim = args.out / "claims" / job["id"]
                claim.parent.mkdir(exist_ok=True)
                try:
                    claim.mkdir()
                except FileExistsError:
                    owner_path = claim / "owner.json"
                    if owner_path.exists():
                        owner = json.loads(owner_path.read_text())
                        if owner.get("host") == socket.gethostname():
                            try:
                                actual = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()[19]
                            except FileNotFoundError:
                                actual = None
                            native = [
                                json.loads(p.read_text())
                                for p in args.out.glob("workers/*/provider/backend_jobs/" + job["id"] + "/receipt.json")
                            ]
                            if actual != owner.get("process_start_ticks") and not any(
                                r.get("status") in {"queued", "running", "unknown"} for r in native
                            ):
                                archive = args.out / "stale_claims"
                                archive.mkdir(exist_ok=True)
                                try:
                                    os.rename(claim, archive / (job["id"] + "_" + str(time.time_ns())))
                                except FileNotFoundError:
                                    pass
                    continue

                atomic_write_json(
                    claim / "owner.json",
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "worker_id": args.worker_id,
                        "process_start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                    },
                )
                atomic_write_json(
                    args.out / "workers" / args.worker_id / "current.json",
                    {
                        "status": "running",
                        "accession": job["accession"],
                        "enzyme_name": job["enzyme_name"],
                        "step": job["step"],
                        "job": job["id"],
                        "setup_llm_calls": 0,
                    },
                )
                shared_claim = None
                try:
                    from .inflight import acquire, release

                    shared_claim = acquire(args.run, job, args.worker_id)
                    if shared_claim is None:
                        atomic_write_json(
                            folder / "inflight_wait.json",
                            {"status": "waiting_for_exact_existing_system", "discovery_run_id": args.run.name, "job": job["id"]},
                        )
                        continue
                    receipt = execute_job(job, json.loads(args.backend.read_text()), folder)
                    analysis = analyze_outputs(job, receipt)
                    atomic_write_json(
                        complete,
                        {
                            "job": job,
                            "receipt": receipt,
                            "analysis": analysis,
                            "scope": "Complete ensemble with identity checks and step-specific geometry; not experimental activity.",
                        },
                    )
                    memory_state = folder / "memory_retry.json"
                    if memory_state.exists():
                        record = json.loads(memory_state.read_text())
                        record["status"] = "complete"
                        atomic_write_json(memory_state, record)
                    publish_results(args.out, jobs)
                    failed = 0
                    seen.add(job["id"])
                except MemoryRetryPending as exc:
                    deferred.append({"job": job["id"], "reason": str(exc)})
                except Exception as exc:
                    atomic_write_json(folder / "failure.json", {"error": str(exc), "job": job})
                    failed += 1
                    if (args.pilot_accession and not pilot_passed) or failed >= 3:
                        raise
                finally:
                    if shared_claim is not None:
                        release(shared_claim)
                    (claim / "owner.json").unlink(missing_ok=True)
                    try:
                        claim.rmdir()
                    except OSError:
                        pass

            atomic_write_json(
                args.out / "workers" / args.worker_id / "current.json",
                {
                    "status": "waiting_for_new_ranked_candidates" if args.watch_seconds else "finished",
                    "completed_jobs": len(seen),
                    "memory_deferred": deferred,
                    "setup_llm_calls": 0,
                },
            )
            if args.memory_retries_only and not selected and not args.watch_seconds:
                return
            if not args.watch_seconds:
                return
            time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
