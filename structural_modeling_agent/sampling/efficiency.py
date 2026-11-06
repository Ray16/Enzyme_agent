"""Bounded performance checks on explicit, already validated OpenMM fixtures.

All output is computational cost evidence. No rate or convergence is inferred.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .pilot import atomic_json, digest, resource_check, runtime


def direct(profile_path, output, worker_index):
    started = time.monotonic()
    resource_check(True)
    import openmm as mm
    from openmm import app, unit
    import psutil

    profile = json.loads(Path(profile_path).read_text())
    profile_sha256 = digest(profile_path)
    if profile.get("purpose") != "software_performance_benchmark":
        raise ValueError("This runner accepts software-performance fixtures only")
    for key in ("system", "integrator", "state", "topology"):
        if digest(profile[key]["path"]) != profile[key]["sha256"]:
            raise ValueError("Benchmark input bytes changed")
    steps = profile["steps_per_segment"]
    repeats = profile["segments_per_case"]
    if (not steps or any(type(n) is not int or not 1 <= n <= 10000 for n in steps)
            or type(repeats) is not int or not 1 <= repeats <= 4):
        raise ValueError("Require a bounded benchmark schedule")
    system = mm.XmlSerializer.deserialize(Path(profile["system"]["path"]).read_text())
    monitoring = profile.get("plumed_monitor")
    if monitoring is not None:
        from openmmplumed import PlumedForce
        if any(f.getForceGroup() == 31 for f in system.getForces()):
            raise ValueError("PLUMED test group already belongs to another force")
        force = PlumedForce(monitoring["script"])
        force.setForceGroup(31)
        system.addForce(force)
    integrator = mm.XmlSerializer.deserialize(Path(profile["integrator"]["path"]).read_text())
    seeds = profile["worker_seeds"]
    if (len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 < s < 2**31 for s in seeds)
            or not 0 <= worker_index < len(seeds)):
        raise ValueError("Require distinct explicit positive seeds for each worker")
    integrator.setRandomNumberSeed(seeds[worker_index])
    barostat_seeds = profile.get("barostat_worker_seeds", [])
    for force in system.getForces():
        if isinstance(force, mm.MonteCarloBarostat):
            if len(barostat_seeds) != len(seeds) or len(set(barostat_seeds)) != len(barostat_seeds):
                raise ValueError("Explicit distinct barostat seeds are required for matched comparisons")
            force.setRandomNumberSeed(barostat_seeds[worker_index])
    state = mm.XmlSerializer.deserialize(Path(profile["state"]["path"]).read_text())
    t = time.monotonic()
    pdb = app.PDBFile(profile["topology"]["path"])
    simulation = app.Simulation(pdb.topology, system, integrator, mm.Platform.getPlatformByName("CUDA"),
                               {"DeviceIndex": "0", "Precision": "double", "DeterministicForces": "true"})
    context = simulation.context
    construction = time.monotonic() - t
    context.setState(state)
    if monitoring is not None:
        check = context.getState(getEnergy=True, getForces=True, groups={31})
        e = check.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f = check.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        if abs(e) > 1e-12 or np.max(abs(f)) > 1e-12:
            raise ValueError("Monitoring benchmark must leave the Hamiltonian unchanged")
    t = time.monotonic()
    context.getState(getEnergy=True, getForces=True)
    warmup = time.monotonic() - t
    rows = []
    for n in steps:
        for repeat in range(repeats):
            resource_check(True)
            context.setState(state)
            files = Path(output).parent / "segments" / f"steps-{n:06d}-repeat-{repeat}"
            files.mkdir(parents=True, exist_ok=False)
            with open(files / "seg.log", "w") as log:
                simulation.reporters = [app.DCDReporter(str(files / "seg.dcd"), max(1, n // 2)),
                    app.StateDataReporter(log, 100, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True)]
                t = time.monotonic()
                done = 0
                monitoring_seconds = 0.0
                while done < n:
                    chunk = min(500, n - done)
                    simulation.step(chunk)
                    done += chunk
                    monitor_started = time.monotonic()
                    resource_check(True)
                    monitoring_seconds += time.monotonic() - monitor_started
                    if time.monotonic() - started > profile["max_wall_seconds"]:
                        raise TimeoutError("Direct MD benchmark reached its wall budget")
                dynamics = time.monotonic() - t
                simulation.reporters = []
            t = time.monotonic()
            simulation.saveState(str(files / "seg.xml"))
            checkpoint_io = time.monotonic() - t
            t = time.monotonic()
            end = context.getState(getPositions=True, getEnergy=True)
            x = end.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            energy = end.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if not np.isfinite(x).all() or not np.isfinite(energy):
                raise ValueError("Nonfinite direct benchmark trajectory")
            validation = time.monotonic() - t
            rows.append(dict(steps=n, repeat=repeat, dynamics_and_monitoring_seconds=dynamics,
                             checkpoint_io_seconds=checkpoint_io,
                             resource_monitoring_seconds=monitoring_seconds,
                             dynamics_and_reporter_seconds=dynamics-monitoring_seconds,
                             endpoint_validation_seconds=validation, energy_kJ_per_mol=float(energy)))
            atomic_json(output, dict(status="partial", rows=rows))
    report = dict(status="passed", scope="software_performance_only_no_kinetic_claim",
                  runtime=runtime(), openmm_version=mm.__version__, atoms=system.getNumParticles(),
                  worker_index=worker_index, seed=seeds[worker_index], barostat_worker_seeds=barostat_seeds,
                  profile_sha256=profile_sha256, context_construction_seconds=construction,
                  plumed_monitoring=monitoring, monitoring_potential_zero_checked=monitoring is not None,
                  first_energy_force_seconds=warmup, rows=rows, persistent_contexts=1,
                  timestep_ps=integrator.getStepSize().value_in_unit(unit.picosecond),
                  entrypoint_seconds=time.monotonic() - started,
                  process_wall_at_case_end_seconds=time.time() - psutil.Process().create_time(),
                  peak_device_allocated_bytes=None, peak_device_reserved_bytes=None,
                  scheduler_wait_seconds=None, physical_rate=None,
                  rng_policy="One independent seeded stream per direct worker; portable state resets do not reset the worker stream.")
    atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-index", required=True, type=int)
    a = parser.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    profile = json.loads(Path(a.profile).read_text())
    if profile.get("purpose") == "paired_software_performance_benchmark":
        import psutil
        order = profile["orders"][a.worker_index]
        reports = {}
        for name in order:
            reports[name] = direct(profile["profiles"][name], Path(a.output).parent / name / "report.json", a.worker_index)
        report = dict(status="passed", scope="paired_software_cost_only", order=order, reports=reports,
                      process_wall_seconds=time.time()-psutil.Process().create_time())
        atomic_json(a.output, report)
    else:
        report = direct(a.profile, a.output, a.worker_index)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
