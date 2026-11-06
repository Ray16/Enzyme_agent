"""Explicit, bounded Lambda assignments for persistent sampling workers.

Inspect with --dry-run first. Host policy and GPU ownership remain in gpu_reserve.
There is no automatic expansion to every free GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from .pilot import atomic_json, implementation_hash, load, resource_check

GATE = "/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve"
PREFLIGHT = "/nfs/lambda_stor_01/homes/rzhu/.codex-personal/skills/lambda-fleet/scripts/preflight_node.py"

# Payload is data on stdin; shell arguments are quoted with shlex.join.
REMOTE_LAUNCH = r'''
import json,os,pathlib,subprocess,sys
x=json.load(sys.stdin)
allowed=set(os.sched_getaffinity(0)); cpus=set(x['cpus'])
if not cpus<=allowed or max(os.getloadavg())+len(cpus)>os.cpu_count():
    raise RuntimeError('CPU capacity changed after preflight')
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        pid=int(proc.name)
        if pid==os.getpid():continue
        cmd=(proc/'cmdline').read_bytes().split(b'\0')[0]
        if b'python' not in pathlib.Path(cmd.decode()).name.encode():continue
        bound=set(os.sched_getaffinity(pid))
        if len(bound)<len(allowed) and bound&cpus:
            raise RuntimeError('Assigned CPUs overlap another Python process')
    except (OSError,UnicodeError):continue
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    env[key]='1'
env['PYTHONPATH']=x['project_root']
os.makedirs(x['logdir'],exist_ok=True)
with open(x['logdir']+'/stdout.log','ab') as log:
    p=subprocess.Popen(x['argv'],cwd=x['project_root'],env=env,stdin=subprocess.DEVNULL,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'argv':x['argv'],'host':os.uname().nodename,
                  'process_start':pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]}))
'''


def validate_assignments(assignments, c):
    if type(assignments["max_workers"]) is not int or not 1 <= len(assignments["workers"]) <= assignments["max_workers"]:
        raise ValueError("Worker count must fit the explicitly declared budget")
    if c.get("purpose") != "software_validation" and assignments["max_workers"] > c["resource_budget"]["max_workers"]:
        raise ValueError("Worker allocation exceeds the agent's resource decision")
    used_gpu, used_cpu = set(), set()
    for worker in assignments["workers"]:
        host = worker["host"]
        if not host or host.startswith("-") or any(x.isspace() for x in host):
            raise ValueError("Invalid host")
        if type(worker["gpu"]) is not int or worker["gpu"] < 0:
            raise ValueError("Invalid GPU index")
        gpu = host, worker["gpu"]
        if gpu in used_gpu:
            raise ValueError("Two workers cannot share a GPU")
        used_gpu.add(gpu)
        cpus = worker["cpus"]
        if not 1 <= len(cpus) <= 4 or len(set(cpus)) != len(cpus):
            raise ValueError("Assign 1-4 distinct CPUs per worker")
        for cpu in cpus:
            if type(cpu) is not int or cpu < 0 or (host, cpu) in used_cpu:
                raise ValueError("CPU sets overlap or contain invalid IDs")
            used_cpu.add((host, cpu))
        if worker["model"] not in {m["id"] for m in c["models"]}:
            raise ValueError("Unknown assigned model")
        if type(worker["max_tasks"]) is not int or worker["max_tasks"] < 1:
            raise ValueError("Explicit positive max_tasks is required")
        if not Path(worker["python"]).is_absolute():
            raise ValueError("Use an absolute node-specific interpreter")


def launch(config, output, assignment_file, dry_run=False):
    c, _ = load(config)
    assignments = json.loads(Path(assignment_file).read_text())
    validate_assignments(assignments, c)
    resource_check(False)
    coordinator = os.sched_getaffinity(0)
    for worker in assignments["workers"]:
        if worker["host"] == socket.gethostname().split(".")[0] and coordinator & set(worker["cpus"]):
            raise ValueError("Coordinator and workers require disjoint CPU sets")
    root = str(Path(__file__).resolve().parents[2])
    config, output = str(Path(config).resolve()), str(Path(output).resolve())
    records = []
    # Preflight every host/stack before any worker can claim work.
    for i, worker in enumerate(assignments["workers"]):
        model = next(m for m in c["models"] if m["id"] == worker["model"])
        modules = ["ase", "numpy", "psutil"]
        external = {"orbmol_v2": "orb_models", "so3lr": "so3lr", "uma": "fairchem.core"}
        name = model["factory"].split(":")[1]
        if name in external:
            modules.append(external[name])
        modules += worker.get("preflight_modules", [])
        argv = [sys.executable, PREFLIGHT, worker["host"], "--python", worker["python"],
                "--cpu-set", ",".join(map(str, worker["cpus"])), "--path", config,
                "--path", str(Path(config).parent / c["structure"]), "--bin", GATE,
                "--min-available-ram-mib", str(worker["min_available_ram_mib"])]
        for module in modules:
            argv.extend(["--module", module])
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=60, cwd=root)
        report = json.loads(proc.stdout)
        if proc.returncode or not report["ok"]:
            raise RuntimeError(f"Node preflight failed: {report}")
        cmd = ["taskset", "-c", ",".join(map(str, worker["cpus"])), GATE, "run", str(worker["gpu"]),
               "--label", "amide-sampling-pilot", "--", worker["python"], "-m",
               "structural_modeling_agent.sampling.pilot", "worker", config, "--output", output,
               "--model", worker["model"], "--max-tasks", str(worker["max_tasks"])]
        records.append({**worker, "preflight": report, "argv": cmd, "project_root": root,
                        "logdir": f"{output}/launch-{time.time_ns()}-{i}"})
    for host in {r["host"] for r in records}:
        group = [r for r in records if r["host"] == host]
        if max(max(r["preflight"]["load"], 0) for r in group) + sum(len(r["cpus"]) for r in group) > group[0]["preflight"]["online_cpus"]:
            raise RuntimeError("Aggregate worker CPU budget exceeds host capacity")
    free = {tuple(line.split()) for line in subprocess.check_output([GATE, "list"], text=True).splitlines()}
    if any((r["host"], str(r["gpu"])) not in free for r in records):
        raise RuntimeError("An assigned GPU is occupied or excluded by shared policy")
    Path(output).mkdir(parents=True, exist_ok=True)
    manifest = {"status": "preflight_only" if dry_run else "launching", "assignments": records,
                "code_sha256": implementation_hash(), "config": c}
    record_path = Path(output) / f"fleet-{time.time_ns()}.json"
    atomic_json(record_path, manifest)
    if not dry_run:
        for record in records:
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", record["host"],
                   shlex.join([record["python"], "-c", REMOTE_LAUNCH])]
            proc = subprocess.run(cmd, input=json.dumps(record), capture_output=True, text=True, timeout=20)
            if proc.returncode:
                manifest["status"] = "partial_launch_failed"
                record["launch_error"] = proc.stderr
                atomic_json(record_path, manifest)
                raise RuntimeError("Launch failed; retain and inspect the manifest before resuming")
            record["launcher"] = json.loads(proc.stdout)
            atomic_json(record_path, manifest)
            time.sleep(1)  # Stagger checkpoint/model loading on shared storage.
        manifest["status"] = "submitted_requires_runtime_verification"
        atomic_json(record_path, manifest)
    return str(record_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("assignments")
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(launch(args.config, args.output, args.assignments, args.dry_run))
