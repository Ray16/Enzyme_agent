#!/usr/bin/env python3
"""
Run MD simulations on prepared AMBER systems using OpenMM.

GPU-parallel execution following the run_cofold.py pattern: detect free GPUs,
assign candidates round-robin, run one simulation per GPU thread.

Usage:
    python run_md.py <project_dir>
    python run_md.py <project_dir> --gpus 1,2,3
    python run_md.py <project_dir> --sim-time-ns 50
    python run_md.py <project_dir> --candidates P31663 P0A6W9
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONDA_ENV = os.environ.get("BOLTZ_CONDA_ENV", "boltz-2")
LD_LIBRARY_FIX = os.environ.get("LD_LIBRARY_PATH_FIX",
                                 os.path.expanduser("~/miniforge3/lib"))


def detect_free_gpus(skip_gpus=None, mem_threshold_mb=1024):
    """Return list of GPU indices with < mem_threshold_mb used memory."""
    skip_gpus = set(skip_gpus or [])
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
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


def run_md_subprocess(prmtop, inpcrd, uid, gpu_id, output_dir,
                      sim_time_ns=20.0):
    """Run MD simulation as a subprocess with CUDA_VISIBLE_DEVICES set."""
    os.makedirs(output_dir, exist_ok=True)

    # Copy topology file upfront so prmtop is always paired with any trajectory
    prmtop_dst = os.path.join(output_dir, "system.prmtop")
    if not os.path.exists(prmtop_dst):
        shutil.copy2(prmtop, prmtop_dst)

    # Write the simulation script as a temporary file
    sim_script = os.path.join(output_dir, "_run_openmm.py")
    write_simulation_script(sim_script)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if LD_LIBRARY_FIX:
        env["LD_LIBRARY_PATH"] = LD_LIBRARY_FIX

    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "python", sim_script,
        "--prmtop", prmtop,
        "--inpcrd", inpcrd,
        "--output-dir", output_dir,
        "--sim-time-ns", str(sim_time_ns),
    ]

    start = time.time()
    print(f"  [{uid}] Starting on GPU {gpu_id} ({sim_time_ns} ns)...",
          flush=True)

    # Timeout: estimate ~10 hours per 20 ns, scale linearly, add buffer
    timeout_s = max(int(sim_time_ns / 20.0 * 12 * 3600), 7200)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, env=env,
        )
        elapsed = time.time() - start
        elapsed_hr = elapsed / 3600

        if result.returncode == 0:
            print(f"  [{uid}] Completed in {elapsed_hr:.1f}h on GPU {gpu_id}",
                  flush=True)
            return uid, gpu_id, True, f"Completed in {elapsed_hr:.1f}h"
        else:
            msg = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
            print(f"  [{uid}] FAILED on GPU {gpu_id}: {msg}", flush=True)
            # Save error log
            err_path = os.path.join(output_dir, "error.log")
            with open(err_path, "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            return uid, gpu_id, False, msg
    except subprocess.TimeoutExpired:
        print(f"  [{uid}] TIMED OUT on GPU {gpu_id} after {timeout_s/3600:.1f}h",
              flush=True)
        return uid, gpu_id, False, f"Timeout after {timeout_s/3600:.1f}h"


def write_simulation_script(script_path):
    """Write the self-contained OpenMM simulation script."""
    script = '''#!/usr/bin/env python3
"""OpenMM simulation engine for MD binding stability validation."""

import argparse
import json
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prmtop", required=True)
    parser.add_argument("--inpcrd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sim-time-ns", type=float, default=20.0)
    args = parser.parse_args()

    from openmm.app import (
        AmberPrmtopFile, AmberInpcrdFile, Simulation,
        PME, HBonds, DCDReporter, StateDataReporter,
    )
    from openmm import (
        LangevinMiddleIntegrator, MonteCarloBarostat,
        CustomExternalForce, Platform, XmlSerializer,
    )
    from openmm.unit import (
        nanometers, picoseconds, femtoseconds, kelvin,
        bar, kilocalories_per_mole, angstroms,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Load system ---
    print("Loading AMBER files...", flush=True)
    prmtop = AmberPrmtopFile(args.prmtop)
    inpcrd = AmberInpcrdFile(args.inpcrd)

    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometers,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )

    # --- Select platform ---
    try:
        platform = Platform.getPlatformByName("CUDA")
        properties = {"CudaPrecision": "mixed"}
        print("Using CUDA platform (mixed precision)", flush=True)
    except Exception:
        platform = Platform.getPlatformByName("CPU")
        properties = {}
        print("Warning: CUDA unavailable, using CPU platform", flush=True)

    # --- Constants ---
    dt = 2.0 * femtoseconds
    temperature = 310.0 * kelvin
    friction = 1.0 / picoseconds
    pressure = 1.0 * bar

    # --- Step 1: Energy minimization ---
    print("Step 1: Energy minimization...", flush=True)
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    simulation = Simulation(prmtop.topology, system, integrator,
                            platform, properties)
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    simulation.minimizeEnergy(maxIterations=5000)
    min_state = simulation.context.getState(getEnergy=True)
    min_energy = min_state.getPotentialEnergy()
    print(f"  Minimized energy: {min_energy}", flush=True)

    positions_min = simulation.context.getState(getPositions=True).getPositions()

    # --- Step 2: NVT equilibration (100 ps) ---
    print("Step 2: NVT equilibration (100 ps, heavy atom restraints)...",
          flush=True)

    # Add heavy atom restraints
    k_restraint = 10.0 * kilocalories_per_mole / angstroms**2
    restraint = CustomExternalForce(
        "k*periodicdistance(x,y,z,x0,y0,z0)^2"
    )
    restraint.addGlobalParameter("k", k_restraint)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    heavy_atom_indices = []
    for i, atom in enumerate(prmtop.topology.atoms()):
        if atom.element is not None and atom.element.symbol != "H":
            pos = positions_min[i]
            restraint.addParticle(i, [pos.x, pos.y, pos.z])
            heavy_atom_indices.append(i)

    restraint_idx = system.addForce(restraint)

    # Re-create simulation with modified system
    del simulation
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    simulation = Simulation(prmtop.topology, system, integrator,
                            platform, properties)
    simulation.context.setPositions(positions_min)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    simulation.context.setVelocitiesToTemperature(temperature)

    # Equilibration log
    eq_log = os.path.join(output_dir, "equilibration.log")
    eq_reporter = StateDataReporter(
        eq_log, 5000, step=True, potentialEnergy=True,
        temperature=True, volume=True, speed=True,
    )
    simulation.reporters.append(eq_reporter)

    # Scale equilibration to production time:
    #   >= 20 ns production: full equilibration (100 ps NVT + 1 ns NPT)
    #   < 20 ns: scale proportionally (min 50 ps NVT + 150 ps NPT)
    sim_time_ns = args.sim_time_ns
    eq_scale = max(0.2, min(1.0, sim_time_ns / 20.0))

    nvt_steps = max(25000, int(50000 * eq_scale))  # 50-100 ps
    nvt_ps = nvt_steps * 2 / 1000
    simulation.step(nvt_steps)
    print(f"  NVT done ({nvt_ps:.0f} ps, {nvt_steps} steps)", flush=True)

    # --- Step 3: NPT equilibration (gradual restraint release) ---
    # Add barostat
    barostat = MonteCarloBarostat(pressure, temperature, 25)
    system.addForce(barostat)

    # Re-create simulation with barostat
    positions_nvt = simulation.context.getState(getPositions=True).getPositions()
    velocities_nvt = simulation.context.getState(getVelocities=True).getVelocities()
    box_nvt = simulation.context.getState().getPeriodicBoxVectors()

    del simulation
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    simulation = Simulation(prmtop.topology, system, integrator,
                            platform, properties)
    simulation.context.setPositions(positions_nvt)
    simulation.context.setVelocities(velocities_nvt)
    simulation.context.setPeriodicBoxVectors(*box_nvt)

    simulation.reporters.append(StateDataReporter(
        eq_log, 5000, step=True, potentialEnergy=True,
        temperature=True, volume=True, density=True, speed=True,
        append=True,
    ))

    # Gradual restraint release: 3 blocks (short) or 5 blocks (full)
    if eq_scale < 0.5:
        k_values = [5.0, 1.0, 0.0]
        steps_per_block = max(10000, int(25000 * eq_scale))
    else:
        k_values = [5.0, 3.0, 1.0, 0.5, 0.0]
        steps_per_block = max(25000, int(100000 * eq_scale))

    npt_ps = len(k_values) * steps_per_block * 2 / 1000
    print(f"Step 3: NPT equilibration ({npt_ps:.0f} ps, "
          f"{len(k_values)} blocks)...", flush=True)

    for k_val in k_values:
        simulation.context.setParameter(
            "k", k_val * kilocalories_per_mole / angstroms**2
        )
        simulation.step(steps_per_block)
        state = simulation.context.getState(getEnergy=True)
        print(f"    k={k_val:.1f} kcal/mol/A^2, "
              f"E={state.getPotentialEnergy()}", flush=True)

    print(f"  NPT equilibration done", flush=True)

    # --- Step 4: Production (configurable time, default 20 ns) ---
    sim_time_ns = args.sim_time_ns
    prod_steps = int(sim_time_ns * 1e6 / 2)  # ns -> fs -> steps at 2 fs
    save_interval = 5000  # 10 ps
    n_frames = prod_steps // save_interval

    print(f"Step 4: Production ({sim_time_ns} ns, {n_frames} frames)...",
          flush=True)

    # Remove restraints for production (k already 0 from equilibration)
    # Add production reporters
    traj_path = os.path.join(output_dir, "trajectory.dcd")
    prod_log = os.path.join(output_dir, "production.log")

    simulation.reporters.clear()
    simulation.reporters.append(DCDReporter(traj_path, save_interval))
    simulation.reporters.append(StateDataReporter(
        prod_log, save_interval, step=True, time=True,
        potentialEnergy=True, kineticEnergy=True,
        totalEnergy=True, temperature=True,
        volume=True, density=True, speed=True,
    ))

    prod_start = time.time()
    simulation.step(prod_steps)
    prod_elapsed = time.time() - prod_start

    # Save final state for restart
    state_xml = os.path.join(output_dir, "final_state.xml")
    state = simulation.context.getState(
        getPositions=True, getVelocities=True,
        getForces=True, getEnergy=True,
    )
    with open(state_xml, "w") as f:
        f.write(XmlSerializer.serialize(state))

    # Save simulation info
    final_state = simulation.context.getState(getEnergy=True)
    info = {
        "sim_time_ns": sim_time_ns,
        "production_steps": prod_steps,
        "n_frames": n_frames,
        "save_interval_steps": save_interval,
        "timestep_fs": 2.0,
        "temperature_K": 310.0,
        "production_walltime_hours": round(prod_elapsed / 3600, 2),
        "final_potential_energy_kJ_mol": float(
            final_state.getPotentialEnergy().value_in_unit(
                kilocalories_per_mole)) * 4.184,
        "platform": platform.getName(),
    }

    info_path = os.path.join(output_dir, "simulation_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Production done in {prod_elapsed/3600:.2f}h "
          f"({prod_steps/prod_elapsed:.0f} steps/s)", flush=True)
    print(f"  Trajectory: {traj_path} ({n_frames} frames)", flush=True)


if __name__ == "__main__":
    main()
'''
    with open(script_path, "w") as f:
        f.write(script)


def collect_prepared_systems(project_dir, prep_dir="md_prep",
                             candidate_filter=None):
    """Find all prepared systems with valid prmtop/inpcrd."""
    prep_base = os.path.join(project_dir, prep_dir)
    if not os.path.isdir(prep_base):
        return {}

    systems = {}
    for uid in sorted(os.listdir(prep_base)):
        if candidate_filter and uid not in candidate_filter:
            continue
        uid_dir = os.path.join(prep_base, uid)
        prmtop = os.path.join(uid_dir, "system.prmtop")
        inpcrd = os.path.join(uid_dir, "system.inpcrd")
        if os.path.exists(prmtop) and os.path.exists(inpcrd):
            systems[uid] = (prmtop, inpcrd)

    return systems


def md_completed(output_dir, uid):
    """Check if MD is already done for this UID."""
    traj = os.path.join(output_dir, uid, "trajectory.dcd")
    info = os.path.join(output_dir, uid, "simulation_info.json")
    return os.path.exists(traj) and os.path.exists(info)


def run_gpu_queue(gpu_id, uid_jobs, output_base, sim_time_ns):
    """Process a queue of MD jobs sequentially on a single GPU."""
    results = []
    for uid, prmtop, inpcrd in uid_jobs:
        output_dir = os.path.join(output_base, uid)
        result = run_md_subprocess(
            prmtop, inpcrd, uid, gpu_id, output_dir, sim_time_ns,
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run MD simulations on prepared AMBER systems")
    parser.add_argument("project_dir",
                        help="Project directory with md_prep output")
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Specific UIDs to simulate")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices (auto-detect if omitted)")
    parser.add_argument("--skip-gpu", type=str, default="",
                        help="GPUs to skip (default: none)")
    parser.add_argument("--sim-time-ns", type=float, default=10.0,
                        help="Production simulation time in ns (default: 10)")
    parser.add_argument("--prep-dir", default="md_prep",
                        help="Preparation directory (default: md_prep)")
    parser.add_argument("-o", "--output-dir", default="md_output",
                        help="Output directory (default: md_output)")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    output_base = os.path.join(project_dir, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Phase 6b: MD Simulations (OpenMM)")
    print(f"{'='*60}\n", flush=True)

    # Collect prepared systems
    systems = collect_prepared_systems(
        project_dir, args.prep_dir, args.candidates,
    )
    if not systems:
        sys.exit("ERROR: No prepared systems found in md_prep/")

    # Filter out already completed
    to_run = {uid: paths for uid, paths in systems.items()
              if not md_completed(output_base, uid)}
    skipped = len(systems) - len(to_run)

    if skipped:
        print(f"  Skipping {skipped} already completed simulations")
    if not to_run:
        print("  All simulations already completed")
        return

    print(f"  Simulations to run: {len(to_run)} "
          f"({', '.join(to_run.keys())})")
    print(f"  Production time: {args.sim_time_ns} ns each")

    # Detect GPUs
    if args.gpus:
        available_gpus = [int(g) for g in args.gpus.split(",")]
    else:
        skip = [int(g) for g in args.skip_gpu.split(",") if g]
        available_gpus = detect_free_gpus(skip_gpus=skip)

    if not available_gpus:
        sys.exit("ERROR: No GPUs available")

    print(f"  GPUs: {available_gpus}\n")

    # Assign jobs to GPUs round-robin
    os.makedirs(output_base, exist_ok=True)
    gpu_queues = defaultdict(list)
    for i, (uid, (prmtop, inpcrd)) in enumerate(to_run.items()):
        gpu_id = available_gpus[i % len(available_gpus)]
        gpu_queues[gpu_id].append((uid, prmtop, inpcrd))

    # Launch parallel GPU workers
    all_results = []
    with ThreadPoolExecutor(max_workers=len(gpu_queues)) as executor:
        futures = {
            executor.submit(
                run_gpu_queue, gpu_id, jobs,
                output_base, args.sim_time_ns,
            ): gpu_id
            for gpu_id, jobs in gpu_queues.items()
        }
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"  GPU {gpu_id} worker failed: {e}", flush=True)

    # Summary
    succeeded = [r for r in all_results if r[2]]
    failed = [r for r in all_results if not r[2]]

    print(f"\n{'='*60}")
    print(f"  MD Simulations Complete")
    print(f"{'='*60}")
    print(f"  Succeeded: {len(succeeded)}")
    if failed:
        print(f"  Failed:    {len(failed)}")
        for uid, gpu_id, _, msg in failed:
            print(f"    {uid} (GPU {gpu_id}): {msg}")
    print(f"  Output:    {output_base}\n")


if __name__ == "__main__":
    main()
