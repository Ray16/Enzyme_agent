#!/usr/bin/env python3
"""Rolling GPU scheduler for Boltz-2 predictions.

As each GPU finishes a job, it immediately picks the next one from the queue.
No GPU sits idle while work remains. Supports both apo folding and cofolding.

Usage:
    python run_boltz_rolling.py --mode apo
    python run_boltz_rolling.py --mode cofold
    python run_boltz_rolling.py --mode all    # apo then cofold then analysis
"""

import argparse
import glob
import os
import subprocess
import sys
import time
import threading
from queue import Queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOLTZ_BIN = os.path.expanduser("~/miniforge3/envs/boltz-2/bin/boltz")
LD_LIB_FIX = os.path.expanduser("~/miniforge3/envs/boltz-2/lib")


def detect_free_gpus(mem_threshold_mb=1024):
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in r.stdout.strip().split("\n"):
            idx, mem = line.split(",")
            if int(mem.strip()) < mem_threshold_mb:
                gpus.append(int(idx.strip()))
        return gpus
    except Exception:
        return [0]


def run_boltz(yaml_path, gpu_id, output_dir, use_msa=True):
    uid = os.path.basename(yaml_path).replace(".yaml", "")
    uid_out = os.path.join(output_dir, uid)
    os.makedirs(uid_out, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["LD_LIBRARY_PATH"] = f"{LD_LIB_FIX}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        BOLTZ_BIN, "predict", yaml_path,
        "--out_dir", uid_out,
        "--output_format", "pdb",
        "--devices", "1",
        "--recycling_steps", "3",
        "--sampling_steps", "200",
        "--diffusion_samples", "1",
    ]
    if use_msa:
        cmd.append("--use_msa_server")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
        pdbs = glob.glob(os.path.join(uid_out, "**", "*_model_0.pdb"), recursive=True)
        elapsed = time.time() - t0
        if result.returncode == 0 and pdbs:
            return uid, gpu_id, True, f"OK ({elapsed:.0f}s)"
        else:
            tail = result.stderr[-200:] if result.stderr else "no stderr"
            return uid, gpu_id, False, f"exit={result.returncode} ({elapsed:.0f}s): {tail}"
    except subprocess.TimeoutExpired:
        return uid, gpu_id, False, "TIMEOUT (2h)"
    except Exception as e:
        return uid, gpu_id, False, str(e)


def gpu_worker(gpu_id, job_queue, results, use_msa, lock):
    while True:
        item = job_queue.get()
        if item is None:
            job_queue.task_done()
            break
        yaml_path, output_dir = item
        uid = os.path.basename(yaml_path).replace(".yaml", "")
        with lock:
            print(f"  START {uid} -> GPU {gpu_id}", flush=True)
        result = run_boltz(yaml_path, gpu_id, output_dir, use_msa)
        uid, gid, success, msg = result
        tag = "OK" if success else "FAIL"
        with lock:
            print(f"  [{tag}] {uid} (GPU {gpu_id}): {msg}", flush=True)
        results.append(result)
        job_queue.task_done()


def run_rolling(jobs, use_msa=True):
    gpus = detect_free_gpus()
    if not gpus:
        print("ERROR: No free GPUs")
        return []

    print(f"Using {len(gpus)} GPUs: {gpus}")
    print(f"Jobs: {len(jobs)}")

    job_queue = Queue()
    for j in jobs:
        job_queue.put(j)
    for _ in gpus:
        job_queue.put(None)

    results = []
    lock = threading.Lock()
    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=gpu_worker, args=(gpu_id, job_queue, results, use_msa, lock))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    ok = sum(1 for _, _, s, _ in results if s)
    fail = sum(1 for _, _, s, _ in results if not s)
    print(f"\nDone: {ok} succeeded, {fail} failed out of {len(results)}")
    return results


def collect_apo_jobs():
    jobs = []
    for proj in ["amide_synthetase", "amide_hydrolase"]:
        inp_dir = os.path.join(BASE_DIR, proj, "boltz_input")
        out_dir = os.path.join(BASE_DIR, proj, "boltz_output")
        os.makedirs(out_dir, exist_ok=True)
        for yf in sorted(glob.glob(os.path.join(inp_dir, "*.yaml"))):
            uid = os.path.basename(yf).replace(".yaml", "")
            uid_out = os.path.join(out_dir, uid)
            pdbs = glob.glob(os.path.join(uid_out, "**", "*_model_0.pdb"), recursive=True)
            if pdbs:
                continue
            jobs.append((yf, out_dir))
    return jobs


def collect_cofold_jobs():
    jobs = []
    for proj in ["amide_synthetase", "amide_hydrolase"]:
        inp_dir = os.path.join(BASE_DIR, proj, "cofold_input")
        out_dir = os.path.join(BASE_DIR, proj, "cofold_output")
        os.makedirs(out_dir, exist_ok=True)
        for yf in sorted(glob.glob(os.path.join(inp_dir, "*.yaml"))):
            uid = os.path.basename(yf).replace(".yaml", "")
            uid_out = os.path.join(out_dir, uid)
            pdbs = glob.glob(os.path.join(uid_out, "**", "*_model_0.pdb"), recursive=True)
            if pdbs:
                continue
            jobs.append((yf, out_dir))
    return jobs


def run_analysis():
    print("\n=== Running post-analysis ===", flush=True)
    scripts = [
        ["boltz_apo_pocket_analysis.py"],
        ["apo_pocket_analysis.py"],
        ["analyze_cofolding.py", "amide_synthetase"],
        ["analyze_cofolding.py", "amide_hydrolase"],
        ["holo_pocket_analysis.py"],
        ["generate_comprehensive_table.py"],
        ["plot_pocket_chemistry.py"],
        ["plot_boltz_vs_alphafold.py"],
    ]
    for parts in scripts:
        script = os.path.join(BASE_DIR, parts[0])
        args = [os.path.join(BASE_DIR, a) if not a.startswith("/") and "/" not in a and a.endswith(("se", "se/")) else os.path.join(BASE_DIR, a) for a in parts[1:]]
        if not os.path.exists(script):
            print(f"  SKIP {parts[0]} (not found)", flush=True)
            continue
        cmd_args = [sys.executable, script] + args
        print(f"  Running {parts[0]} ...", flush=True)
        try:
            subprocess.run(cmd_args, timeout=600)
        except Exception as e:
            print(f"  ERROR {parts[0]}: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["apo", "cofold", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ("apo", "all"):
        print("=== APO FOLDING (with MSA) ===", flush=True)
        jobs = collect_apo_jobs()
        if jobs:
            print(f"Found {len(jobs)} apo jobs to run", flush=True)
            run_rolling(jobs, use_msa=True)
        else:
            print("All apo jobs already complete", flush=True)

    if args.mode in ("cofold", "all"):
        print("\n=== COFOLDING (with MSA) ===", flush=True)
        jobs = collect_cofold_jobs()
        if jobs:
            print(f"Found {len(jobs)} cofold jobs to run", flush=True)
            run_rolling(jobs, use_msa=True)
        else:
            print("All cofold jobs already complete", flush=True)

    if args.mode == "all":
        run_analysis()

    print(f"\n=== ALL DONE at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)


if __name__ == "__main__":
    main()
