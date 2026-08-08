#!/usr/bin/env python3
"""Run Boltz-2 apo structure prediction for all synthetases and hydrolases.

One job per GPU — no concurrent runs on the same GPU. Jobs are queued
per-GPU and processed sequentially within each queue.

Usage:
    python run_boltz_apo_folding.py
    python run_boltz_apo_folding.py --gpus 0,1,2,3
"""

import argparse
import glob
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOLTZ_BIN = os.path.expanduser("~/miniforge3/envs/boltz-2/bin/boltz")
LD_LIB_FIX = os.path.expanduser("~/miniforge3/envs/boltz-2/lib")

PROJECTS = [
    ("amide_synthetase", "boltz_input", "boltz_output"),
    ("amide_hydrolase", "boltz_input", "boltz_output"),
]


def detect_free_gpus(skip_gpus=None, mem_threshold_mb=1024):
    skip_gpus = set(skip_gpus or [])
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in r.stdout.strip().split("\n"):
            idx, mem = line.split(",")
            idx, mem = int(idx.strip()), int(mem.strip())
            if idx not in skip_gpus and mem < mem_threshold_mb:
                gpus.append(idx)
        return gpus
    except Exception as e:
        print(f"GPU detection failed ({e}), using GPU 0")
        return [0]


def run_boltz(yaml_path, gpu_id, output_dir):
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
        "--use_msa_server",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        pdbs = glob.glob(os.path.join(uid_out, "**", "*.pdb"), recursive=True)
        elapsed = time.time() - t0
        if result.returncode == 0 and pdbs:
            return uid, gpu_id, True, f"OK ({len(pdbs)} PDB, {elapsed:.0f}s)"
        else:
            tail = result.stderr[-300:] if result.stderr else "no stderr"
            return uid, gpu_id, False, f"exit={result.returncode} ({elapsed:.0f}s): {tail}"
    except subprocess.TimeoutExpired:
        return uid, gpu_id, False, "TIMEOUT (60 min)"
    except Exception as e:
        return uid, gpu_id, False, str(e)


def run_gpu_queue(gpu_id, jobs):
    results = []
    for yaml_path, output_dir in jobs:
        results.append(run_boltz(yaml_path, gpu_id, output_dir))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    # Collect all YAML inputs
    all_jobs = []
    for proj, inp, out in PROJECTS:
        inp_dir = os.path.join(BASE_DIR, proj, inp)
        out_dir = os.path.join(BASE_DIR, proj, out)
        os.makedirs(out_dir, exist_ok=True)
        for yf in sorted(glob.glob(os.path.join(inp_dir, "*.yaml"))):
            uid = os.path.basename(yf).replace(".yaml", "")
            uid_out = os.path.join(out_dir, uid)
            pdbs = glob.glob(os.path.join(uid_out, "**", "*.pdb"), recursive=True)
            if pdbs:
                print(f"  SKIP {uid} (already has {len(pdbs)} PDB)")
                continue
            all_jobs.append((yf, out_dir))

    if not all_jobs:
        print("All proteins already folded. Nothing to do.")
        return

    # Detect GPUs
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]
    else:
        gpus = detect_free_gpus()
    print(f"Using {len(gpus)} GPUs: {gpus}")
    print(f"Jobs to run: {len(all_jobs)}\n")

    # Assign jobs round-robin to GPUs
    gpu_queues = defaultdict(list)
    for i, (yf, out_dir) in enumerate(all_jobs):
        gid = gpus[i % len(gpus)]
        gpu_queues[gid].append((yf, out_dir))

    for gid in sorted(gpu_queues):
        uids = [os.path.basename(y).replace(".yaml", "") for y, _ in gpu_queues[gid]]
        print(f"  GPU {gid}: {len(uids)} jobs — {', '.join(uids[:3])}{'...' if len(uids) > 3 else ''}")

    print(f"\nStarting folding ...\n", flush=True)
    t0 = time.time()
    ok, fail = [], []

    with ThreadPoolExecutor(max_workers=len(gpu_queues)) as ex:
        futures = {
            ex.submit(run_gpu_queue, gid, jobs): gid
            for gid, jobs in gpu_queues.items()
        }
        for fut in as_completed(futures):
            for uid, gid, success, msg in fut.result():
                tag = "OK" if success else "FAIL"
                print(f"  [{tag}] {uid} (GPU {gid}): {msg}", flush=True)
                if success:
                    ok.append(uid)
                else:
                    fail.append((uid, msg))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Boltz-2 apo folding complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Succeeded: {len(ok)}/{len(all_jobs)}")
    print(f"  Failed:    {len(fail)}/{len(all_jobs)}")
    if fail:
        print(f"\n  Failed:")
        for uid, msg in fail:
            print(f"    {uid}: {msg[:120]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
