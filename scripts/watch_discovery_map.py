"""Attach resumable, reservation-aware embedding batches to a discovery run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def remote_python(host, interpreter, code):
    result = subprocess.run(["ssh", host, shlex.join([interpreter, "-c", code])],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"Remote worker control failed: {result.stderr[-300:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def launch_batch(settings, run, job):
    slots = settings["cpu_set"]
    native = [f"{key}=1" for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
                                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")]
    argv = ["env", *native, f"PYTHONPATH={settings['project']}", f"TORCH_HOME={settings['torch_home']}",
            "taskset", "-c", ",".join(map(str, slots)), settings["gpu_reserve"], "run", str(settings["gpu"]),
            "--label", "rolling-discovery-map", "--", settings["python"], "-u", "-m",
            "enzyme_discovery_agent.output.refresh_run_map", str(run), "--embed"]
    code = '''import os,json,subprocess
from pathlib import Path
slots=set(SLOTS);job=Path(JOB);gpu=GPU
if not slots<=os.sched_getaffinity(0) or os.getloadavg()[0]+len(slots)>os.cpu_count():
 print(json.dumps({"busy":"CPU capacity"}));raise SystemExit(0)
for proc in Path("/proc").iterdir():
 if not proc.name.isdigit() or proc.name==str(os.getpid()):continue
 try:
  command=(proc/"cmdline").read_bytes()
  affinity=os.sched_getaffinity(int(proc.name))
  if b"python" in command and len(affinity)<=8 and affinity&slots:
   print(json.dumps({"busy":"CPU affinity"}));raise SystemExit(0)
 except (OSError,PermissionError):pass
uuid=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
resident=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
if any(row.startswith(uuid) for row in resident.splitlines()):
 print(json.dumps({"busy":"GPU resident process"}));raise SystemExit(0)
with (job/"worker.log").open("ab") as log:
 child=subprocess.Popen(ARGV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:start=(Path("/proc")/str(child.pid)/"stat").read_text().split()[21]
except FileNotFoundError:start="gone"
print(json.dumps({"pid":child.pid,"process_start":start,"gpu_uuid":uuid,"load":os.getloadavg()[0],"argv":ARGV}))
'''.replace("set(SLOTS)", f"set({slots!r})").replace("Path(JOB)", f"Path({str(job)!r})").replace("gpu=GPU", f"gpu={settings['gpu']!r}").replace("ARGV", repr(argv))
    return remote_python(settings["host"], settings["python"], code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("launcher", type=Path)
    parser.add_argument("settings", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    sys.path.insert(0, settings["project"])
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, UNIPROT_ACC_RE
    from enzyme_discovery_agent.models.embedding_store import load_store, pending_accessions
    from enzyme_discovery_agent.output.evolving_map import update_candidate_map
    from enzyme_discovery_agent.state_store import atomic_write_json
    status = {"status": "waiting_for_candidates", "pid": os.getpid(), "host": settings["host"], "gpu": settings["gpu"]}
    destination = args.launcher / "map_status.json"
    failures, last_batch = 0, 0.0
    batch_number = None
    while True:
        launch_status = args.launcher / "discovery_status.json"
        if not launch_status.exists():
            time.sleep(5); continue
        discovery = json.loads(launch_status.read_text())
        run = Path(discovery["run_dir"])
        if batch_number is None:
            batches = run / "provenance" / "map_batches"
            batch_number = max((int(p.name.split("_")[-1]) for p in batches.glob("batch_*")
                                if p.is_dir() and p.name.split("_")[-1].isdigit()), default=0)
        registry = load_candidate_registry(run, strict=True).get("candidates", {})
        directory = run / "artifacts" / "candidate_map"
        coverage = directory / "sequence_coverage.json"
        unresolved = set(json.loads(coverage.read_text()).get("unresolved", [])) if coverage.exists() else set()
        pending = [acc for acc in pending_accessions(run, registry) if UNIPROT_ACC_RE.fullmatch(acc) and acc not in unresolved]
        finished = discovery.get("status") != "running"
        update_candidate_map(run)
        status.update(run_dir=str(run), registered=len(registry), embedded=len(load_store(run)), pending=len(pending))
        atomic_write_json(destination, status)
        if finished and not pending:
            status["status"] = "complete"; break
        ready = len(pending) >= 32 or finished and len(pending) >= 3 or len(pending) >= 3 and time.monotonic() - last_batch >= 120
        if not ready:
            if finished:
                status["status"] = "complete_with_unembedded_candidates"; break
            time.sleep(15); continue
        job = run / "provenance" / "map_batches" / f"batch_{batch_number + 1:04d}"
        job.mkdir(parents=True, exist_ok=True)
        launched = launch_batch(settings, run, job)
        if launched.get("busy"):
            status.update(status="waiting_for_resources", reason=launched["busy"])
            atomic_write_json(destination, status);time.sleep(30);continue
        batch_number += 1
        atomic_write_json(job / "launch_manifest.json", {**settings, **launched, "run_dir": str(run),
            "pending_accessions": pending, "started_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {name: hashlib.sha256((Path(settings["project"]) / "enzyme_discovery_agent" / name).read_bytes()).hexdigest()
                              for name in ("output/refresh_run_map.py", "output/evolving_map.py", "tools/esm_embed_worker.py")},
            "pending_sha256": hashlib.sha256(json.dumps(pending).encode()).hexdigest()})
        status.update(status="embedding", batch=batch_number, worker_pid=launched["pid"])
        atomic_write_json(destination, status)
        count_before = len(load_store(run))
        while True:
            code = f'from pathlib import Path;import json;p=Path("/proc/{launched["pid"]}/stat");print(json.dumps({{"start":p.read_text().split()[21] if p.exists() else "gone"}}))'
            owner = remote_python(settings["host"], settings["python"], code)
            if owner["start"] != launched["process_start"]:
                break
            time.sleep(15)
        count_after = len(load_store(run))
        failures = failures + 1 if count_after <= count_before else 0
        atomic_write_json(job / "result.json", {"embedded_before": count_before, "embedded_after": count_after,
                                                "consecutive_failures": failures})
        if failures >= 3:
            status.update(status="failed", reason="Three embedding batches added no vectors; inspect worker logs");break
        status["status"] = "watching_discovery"
        last_batch = time.monotonic()
        if failures:
            time.sleep(30)
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(destination, status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if len(sys.argv) > 1:
            status_path = Path(sys.argv[1]) / "map_status.json"
            status_path.write_text(json.dumps({"status": "failed", "error": str(exc)}))
        raise
