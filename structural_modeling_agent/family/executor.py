"""Reserved native-provider execution, ownership checks, and contention retry for family cofolds."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from enzyme_discovery_agent.state_store import atomic_write_json
from ..env_config import EXACT_BOLTZ_PYTHON
from .memory_retry import MemoryRetryPending, is_memory_failure, record_failure, retry_backend
from .protocol import DEFAULT_SAMPLES, DEFAULT_SEED, digest


def provider_owner_live(receipt):
    """A terminal receipt still needs PID/start-time reconciliation before retry."""
    code = (
        "from pathlib import Path;import json,sys;r=json.load(sys.stdin);"
        "p=Path('/proc/'+str(r['pid'])+'/stat');"
        "print(json.dumps((p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==r['process_start_ticks']) "
        "or (bool(r.get('predictor_pid')) and Path('/proc/'+str(r['predictor_pid'])).exists())))"
    )
    if receipt.get("host") == socket.gethostname():
        path = Path("/proc/" + str(receipt["pid"]) + "/stat")
        return (
            (path.exists() and path.read_text().rsplit(")", 1)[1].split()[19] == receipt["process_start_ticks"])
            or bool(receipt.get("predictor_pid") and Path("/proc/" + str(receipt["predictor_pid"])).exists())
        )
    command = shlex.join([EXACT_BOLTZ_PYTHON, "-c", code])
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", receipt["host"], command],
            input=json.dumps(receipt),
            text=True,
            capture_output=True,
            timeout=20,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        raise MemoryRetryPending(
            "Cannot verify previous producer ownership on " + receipt["host"] + "; preserve attempt and retry the ownership check"
        ) from exc


def execute_job(job, backend, directory, resource_attempt=0, *, owner_live_fn=None):
    """Use the existing reserved native provider, without a modeling-agent session."""
    directory = Path(directory)
    receipt_path = directory / "receipt.json"
    check_owner = owner_live_fn or provider_owner_live
    backend = retry_backend(directory, backend, check_owner, job)
    state_dir = Path(backend["argv"][backend["argv"].index("--state-dir") + 1])
    if not receipt_path.exists():
        for path in directory.parents[1].glob("workers/*/provider/backend_jobs/" + job["id"] + "/receipt.json"):
            native = json.loads(path.read_text())
            request_path = path.parent / "request.json"
            if native.get("status") not in {"queued", "running", "unknown", "complete"} or not request_path.exists():
                continue
            if json.loads(request_path.read_text()).get("input_sha256") != job["input_sha256"]:
                continue
            native["provider_state_dir"] = str(path.parents[2])
            atomic_write_json(receipt_path, native)
            break
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous.get("status") == "failed" and previous.get("provider_state_dir") not in {None, str(state_dir)}:
            attempts = directory / "backend_attempts"
            attempts.mkdir(exist_ok=True)
            os.rename(receipt_path, attempts / (str(time.time_ns()) + ".json"))

    expected_samples = int(job.get("expected_predictions", DEFAULT_SAMPLES))
    run_seed = int(job.get("seed", DEFAULT_SEED))
    request = {
        "operation": "submit",
        "idempotency_key": job["id"],
        "input_yaml": job["input_yaml"],
        "input_sha256": job["input_sha256"],
        "expected_predictions": expected_samples,
        "seed": run_seed,
        "resource_policy": backend["resource_policy"],
        "execution_sha256": digest({
            "job": job["id"],
            "input": job["input_sha256"],
            "msa": job["msa_sha256"],
            "profile": "exact",
            "samples": expected_samples,
        }),
        "msa_evidence": [{"path": job["msa_path"], "sha256": job["msa_sha256"]}],
    }
    if not receipt_path.exists():
        result = subprocess.run(backend["argv"], input=json.dumps(request), text=True, capture_output=True, timeout=90)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        receipt = json.loads(result.stdout)
        receipt["provider_state_dir"] = str(state_dir)
        atomic_write_json(receipt_path, receipt)
    else:
        receipt = json.loads(receipt_path.read_text())

    producer_state = Path(receipt.get("provider_state_dir", state_dir))
    live_receipt = producer_state / "backend_jobs" / job["id"] / "receipt.json"
    while receipt.get("status") in {"queued", "running", "unknown"}:
        if live_receipt.exists():
            receipt = {**json.loads(live_receipt.read_text()), "provider_state_dir": str(producer_state)}
        atomic_write_json(receipt_path, receipt)
        if receipt.get("status") not in {"queued", "running", "unknown"}:
            break
        time.sleep(5)

    if (
        receipt.get("status") == "failed"
        and receipt.get("error") == "Foreign resident compute process appeared"
        and resource_attempt < backend["resource_policy"].get("max_resource_retries", 6)
    ):
        for _ in range(6):
            if not check_owner(receipt):
                break
            time.sleep(2)
        else:
            raise RuntimeError("Stopped native worker still owns its attempt; reconciliation required")
        stamp = str(time.time_ns())
        native = producer_state / "backend_jobs" / job["id"]
        archive = producer_state / "resource_stops" / job["id"] / stamp
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.rename(native, archive)
        local = directory / "resource_stops" / stamp
        local.mkdir(parents=True, exist_ok=True)
        os.rename(receipt_path, local / "receipt.json")
        atomic_write_json(
            local / "reason.json",
            {
                "reason": "Actual foreign GPU contention; no completed ensemble replaced.",
                "native_archive": str(archive),
                "resource_attempt": resource_attempt + 1,
            },
        )
        time.sleep(5)
        return execute_job(job, backend, directory, resource_attempt + 1, owner_live_fn=check_owner)

    if is_memory_failure(receipt):
        record_failure(directory, receipt, backend["resource_policy"])
        raise MemoryRetryPending("GPU memory exhausted; queued for a larger device")
    if receipt.get("status") != "complete":
        raise RuntimeError(receipt.get("error") or str(receipt))
    return receipt
