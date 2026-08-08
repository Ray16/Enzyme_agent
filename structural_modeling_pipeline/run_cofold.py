#!/usr/bin/env python3
"""
Run Boltz-2 cofolding (enzyme + substrates) in parallel across multiple GPUs.

Supports three cofolding strategies (auto-selected by prepare_cofold_inputs.py):
  - Cofactor-guided: enzyme + cofactor (CCD) + substrate(s), cofactor anchored
  - Pocket-constrained: enzyme + substrate(s), substrate anchored to pocket
  - Blind: enzyme + substrate(s), no constraints

For very large complexes (>2048 tokens), consider using NVIDIA Fold-CP for
multi-GPU context parallelism: https://github.com/NVIDIA-Digital-Bio/boltz-cp
Fold-CP distributes the O(N^2) pair representation across a sqrt(P) x sqrt(P)
GPU grid, achieving O(N^2/P) memory per device.

Usage:
    python run_cofold.py <project_dir>
    python run_cofold.py <project_dir> --gpus 1,2,3,4
    python run_cofold.py <project_dir> --skip-prepare --affinity
    python run_cofold.py <project_dir> --devices 4  # multi-GPU per job

Workflow:
    1. Runs prepare_cofold_inputs.py to generate cofolding YAML inputs.
    2. Detects available GPUs (skipping those with >1GB used).
    3. Assigns each protein-ligand complex to a GPU round-robin.
    4. Launches all jobs in parallel via subprocess.
    5. Collects results and prints summary.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from collections import defaultdict

import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LD_LIBRARY_FIX = os.environ.get("LD_LIBRARY_PATH_FIX", os.path.expanduser("~/miniforge3/lib"))
BOLTZ_CONDA_ENV = os.environ.get("BOLTZ_CONDA_ENV", "boltz-2")


def detect_free_gpus(skip_gpus=None, mem_threshold_mb=1024):
    """Return list of GPU indices with < mem_threshold_mb used memory."""
    skip_gpus = set(skip_gpus or [])
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, mem_used = line.split(",")
            idx = int(idx.strip())
            mem_used = int(mem_used.strip())
            if idx not in skip_gpus and mem_used < mem_threshold_mb:
                gpus.append(idx)
        return gpus
    except Exception as e:
        print(f"Warning: GPU detection failed ({e}), defaulting to GPU 0")
        return [0]


def estimate_tokens(yaml_path):
    """Estimate token count from a Boltz-2 YAML input.

    Proteins: ~1 token per residue. Ligands: ~1 token per heavy atom (estimated
    from SMILES length / 2 or CCD lookup).
    Returns estimated token count.
    """
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return 0

    tokens = 0
    for entry in data.get("sequences", []):
        if "protein" in entry:
            seq = entry["protein"].get("sequence", "")
            tokens += len(seq)
        elif "ligand" in entry:
            lig = entry["ligand"]
            if "smiles" in lig:
                # Rough estimate: ~1 heavy atom per 2 SMILES chars
                tokens += max(1, len(lig["smiles"]) // 2)
            elif "ccd" in lig:
                # Common cofactors: ATP~31, NAD~44, CoA~51, FAD~53
                tokens += 40  # conservative estimate
    return tokens


def check_gpu_memory(gpu_id, max_used_mb=2048):
    """Check if a GPU has enough free memory before launching a job."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=10,
        )
        mem_used, mem_total = result.stdout.strip().split(",")
        return int(mem_used.strip()) < max_used_mb
    except Exception:
        return True  # proceed if check fails


def run_boltz_on_gpu(yaml_path, gpu_id, output_dir, recycling_steps=3,
                     sampling_steps=200, use_msa_server=False, devices=1,
                     diffusion_samples=5):
    """Run Boltz-2 cofolding for a single YAML input on a specific GPU.

    Returns (yaml_path, gpu_id, success, message).
    """
    uid = os.path.basename(yaml_path).replace(".yaml", "")
    uid_output_dir = os.path.join(output_dir, uid)
    os.makedirs(uid_output_dir, exist_ok=True)

    # Check GPU has sufficient free memory before launching
    if not check_gpu_memory(gpu_id):
        # Wait up to 5 min for GPU to free up
        for attempt in range(10):
            time.sleep(30)
            if check_gpu_memory(gpu_id):
                break
        else:
            return yaml_path, gpu_id, False, f"GPU {gpu_id} memory still in use after 5 min wait"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["LD_LIBRARY_PATH"] = f"{LD_LIBRARY_FIX}:{env.get('LD_LIBRARY_PATH', '')}"

    # Call boltz directly from the conda env binary path.
    # conda run does not reliably propagate CUDA_VISIBLE_DEVICES.
    conda_prefix = os.environ.get("CONDA_PREFIX_BASE", os.path.expanduser("~/miniforge3/envs"))
    boltz_bin_path = os.path.join(conda_prefix, BOLTZ_CONDA_ENV, "bin", "boltz")
    cmd = [
        boltz_bin_path, "predict", yaml_path,
        "--out_dir", uid_output_dir,
        "--output_format", "pdb",
        "--devices", str(devices),
        "--recycling_steps", str(recycling_steps),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", str(diffusion_samples),
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=3600,  # 60 min max for cofolding
        )
        # Check for output PDB files
        pdb_files = glob.glob(os.path.join(uid_output_dir, "**", "*.pdb"), recursive=True)
        if result.returncode == 0 and pdb_files:
            return yaml_path, gpu_id, True, f"OK ({len(pdb_files)} PDB files)"
        else:
            stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
            return yaml_path, gpu_id, False, f"exit={result.returncode}: {stderr_tail}"
    except subprocess.TimeoutExpired:
        return yaml_path, gpu_id, False, "TIMEOUT (60 min)"
    except Exception as e:
        return yaml_path, gpu_id, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Run Boltz-2 cofolding (enzyme + substrates) in parallel across GPUs")
    parser.add_argument("project_dir", help="Project directory with config.yaml")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g. 1,2,3). Auto-detects if omitted.")
    parser.add_argument("--skip-gpu", type=str, default="0",
                        help="Comma-separated GPU IDs to skip (default: 0)")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip running prepare_cofold_inputs.py")
    parser.add_argument("--no-pocket-constraint", action="store_true",
                        help="Pass --no-pocket-constraint to prepare script")
    parser.add_argument("--affinity", action="store_true",
                        help="Request binding affinity prediction")
    parser.add_argument("--use-msa-server", action="store_true",
                        help="Use MSA server instead of single-sequence mode")
    parser.add_argument("--recycling-steps", type=int, default=3,
                        help="Number of recycling steps (default: 3)")
    parser.add_argument("--sampling-steps", type=int, default=200,
                        help="Number of sampling steps (default: 200)")
    parser.add_argument("-o", "--output-dir", default="cofold_input",
                        help="Input YAML directory name (default: cofold_input)")
    parser.add_argument("--results-dir", default="cofold_output",
                        help="Output results directory name (default: cofold_output)")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs per job (default: 1). Use >1 for large complexes.")
    parser.add_argument("--diffusion-samples", type=int, default=5,
                        help="Number of diffusion samples per candidate (default: 5). "
                             "Best model is auto-selected based on substrate-cofactor proximity.")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    input_dir = os.path.join(project_dir, args.output_dir)
    results_dir = os.path.join(project_dir, args.results_dir)

    # Step 1: Prepare inputs
    if not args.skip_prepare:
        print("Step 1: Preparing Boltz-2 cofolding input files ...", flush=True)
        prepare_cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "prepare_cofold_inputs.py"),
            project_dir,
            "-o", args.output_dir,
        ]
        if args.no_pocket_constraint:
            prepare_cmd.append("--no-pocket-constraint")
        if args.affinity:
            prepare_cmd.append("--affinity")
        if args.use_msa_server:
            prepare_cmd.append("--use-msa-server")
        subprocess.run(prepare_cmd, check=True)
        print()

    # Step 1b: Validate cofolding inputs
    print("Step 1b: Validating cofolding input files ...", flush=True)
    validate_cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "validate_cofold_inputs.py"),
        project_dir,
        "--input-dir", args.output_dir,
    ]
    val_result = subprocess.run(validate_cmd)
    if val_result.returncode != 0:
        sys.exit("ERROR: Validation failed. Fix errors before cofolding.")
    print()

    # Step 2: Collect YAML inputs
    yaml_files = sorted(glob.glob(os.path.join(input_dir, "*.yaml")))
    if not yaml_files:
        print(f"No YAML input files found in {input_dir}. Nothing to cofold.")
        return

    print(f"Step 2: Found {len(yaml_files)} protein-ligand complexes to cofold", flush=True)

    # Check token sizes and warn about large complexes
    large_complexes = []
    for yf in yaml_files:
        tokens = estimate_tokens(yf)
        if tokens > 2048:
            uid = os.path.basename(yf).replace(".yaml", "")
            large_complexes.append((uid, tokens))
    if large_complexes:
        print(f"\n  WARNING: {len(large_complexes)} complexes exceed 2048 tokens:")
        for uid, tokens in large_complexes:
            print(f"    {uid}: ~{tokens} tokens")
        print(f"  For large complexes, consider Fold-CP for multi-GPU inference:")
        print(f"    https://github.com/NVIDIA-Digital-Bio/boltz-cp\n")

    # Step 3: Detect GPUs
    if args.gpus:
        available_gpus = [int(g) for g in args.gpus.split(",")]
    else:
        skip = [int(g) for g in args.skip_gpu.split(",")] if args.skip_gpu else []
        available_gpus = detect_free_gpus(skip_gpus=skip)

    if not available_gpus:
        print("ERROR: No available GPUs found.")
        sys.exit(1)

    print(f"Step 3: Using GPUs: {available_gpus} ({len(available_gpus)} available)", flush=True)

    # Step 4: Assign GPUs round-robin and launch
    os.makedirs(results_dir, exist_ok=True)
    assignments = []
    for i, yaml_path in enumerate(yaml_files):
        gpu_id = available_gpus[i % len(available_gpus)]
        assignments.append((yaml_path, gpu_id))

    print(f"\nStep 4: Launching {len(assignments)} Boltz-2 cofolding jobs ...\n", flush=True)
    for yaml_path, gpu_id in assignments:
        uid = os.path.basename(yaml_path).replace(".yaml", "")
        print(f"  {uid} -> GPU {gpu_id}")

    # Group jobs by GPU so each GPU processes its queue sequentially
    gpu_queues = defaultdict(list)
    for yaml_path, gpu_id in assignments:
        gpu_queues[gpu_id].append(yaml_path)

    def run_gpu_queue(gpu_id, yaml_paths):
        """Run all jobs assigned to a single GPU sequentially."""
        results = []
        for yaml_path in yaml_paths:
            result = run_boltz_on_gpu(
                yaml_path, gpu_id, results_dir,
                recycling_steps=args.recycling_steps,
                sampling_steps=args.sampling_steps,
                use_msa_server=args.use_msa_server,
                devices=args.devices,
                diffusion_samples=args.diffusion_samples,
            )
            results.append(result)
        return results

    succeeded = []
    failed = []
    start_time = time.time()

    # One thread per GPU -- each thread processes its queue sequentially
    with ThreadPoolExecutor(max_workers=len(gpu_queues)) as executor:
        futures = {
            executor.submit(run_gpu_queue, gid, paths): gid
            for gid, paths in gpu_queues.items()
        }
        for future in as_completed(futures):
            for yaml_path, gpu_id, success, msg in future.result():
                uid = os.path.basename(yaml_path).replace(".yaml", "")
                status = "OK" if success else "FAILED"
                print(f"  [{status}] {uid} (GPU {gpu_id}): {msg}", flush=True)
                if success:
                    succeeded.append(uid)
                else:
                    failed.append((uid, msg))

    elapsed = time.time() - start_time

    # Step 5: Best model selection (when using multi-seed sampling)
    if args.diffusion_samples > 1 and succeeded:
        print(f"\nStep 5: Selecting best model from {args.diffusion_samples} "
              f"samples per candidate ...", flush=True)
        select_cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "select_best_model.py"),
            project_dir,
            "--results-dir", args.results_dir,
        ]
        sel_result = subprocess.run(select_cmd, capture_output=True, text=True)
        print(sel_result.stdout)
        if sel_result.returncode != 0 and sel_result.stderr:
            print(f"  Warning: best model selection had issues: {sel_result.stderr[:200]}")

    # Step 6: Summary
    print(f"\n{'='*60}")
    print(f"Boltz-2 cofolding complete in {elapsed:.0f}s")
    print(f"  Succeeded: {len(succeeded)}/{len(assignments)}")
    print(f"  Failed:    {len(failed)}/{len(assignments)}")
    if failed:
        print(f"\n  Failed complexes:")
        for uid, msg in failed:
            print(f"    {uid}: {msg[:100]}")
    print(f"\n  Output directory: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
