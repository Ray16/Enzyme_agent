import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

from structural_modeling_agent.cofold.queue_report import inspect_claim
from structural_modeling_agent.cofold import queue_worker


def test_claim_status_is_diagnostic_and_never_releases(tmp_path):
    claim = tmp_path / "item"
    claim.mkdir()
    pid = os.getpid()
    ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    owner = {"host": socket.gethostname(), "pid": pid, "start_ticks": ticks}
    (claim / "owner.json").write_text(json.dumps(owner))
    assert inspect_claim(claim)["status"] == "owner_process_active"
    (claim / "owner.json").write_text(json.dumps({**owner, "start_ticks": "0"}))
    assert inspect_claim(claim)["status"] == "owner_pid_reused"
    (claim / "owner.json").write_text(json.dumps({**owner, "host": "remote-invalid"}))
    assert inspect_claim(claim)["status"] == "remote_owner_unverified"
    assert claim.exists()


def test_worker_records_failure_terminal_state(tmp_path, monkeypatch):
    source = tmp_path / "input.yaml"
    source.write_text("synthetic input")
    msa = tmp_path / "input.csv"
    msa.write_text("synthetic data")
    manifest = {"cpu_slots_per_worker": 1, "maximum_attempts": 1, "failure_circuit_breaker": 2,
                "producer_argv": ["test-producer"], "items": [{"id": "item", "input_yaml": str(source),
                "input_sha256": queue_worker.digest(source), "msa_path": str(msa), "msa_sha256": queue_worker.digest(msa),
                "seed": 1, "expected_predictions": 1}]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["queue_worker", "--manifest", str(manifest_path), "--worker-id", "w1"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0})
    monkeypatch.setattr(os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.setattr(queue_worker.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    queue_worker.main()
    attempt = json.loads((tmp_path / "attempts/item/attempt_1.json").read_text())
    assert attempt["status"] == "failed"
    assert "Predictor exited" in attempt["error"]
    assert not (tmp_path / "claims/item").exists()
