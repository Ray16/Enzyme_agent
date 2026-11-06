"""Bounded functional checks for MD and the public ML/MM ASE interface.

Inputs are explicit software benchmarks. These checks establish interface and
numerical consistency, never enzyme accuracy, physical rates or processivity.
Run under thread caps, taskset and (for CUDA) gpu_reserve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .pilot import atomic_json, digest, force_check, resource_check, runtime


def md_check(pdb, output, platform_name):
    import openmm as mm
    from openmm import app, unit
    from openmmplumed import PlumedForce

    resource_check(platform_name == "CUDA")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "software_benchmark_only", "runtime": runtime(),
              "pdb_sha256": digest(pdb), "platform": platform_name,
              "openmm_version": mm.__version__, "status": "failed"}
    platform = mm.Platform.getPlatformByName(platform_name)
    props = {"Threads": "1", "DeterministicForces": "true"} if platform_name == "CPU" else {
        "Precision": "double", "DeterministicForces": "true", "DeviceIndex": "0"}
    # Two-particle analytical check fixes PLUMED's 1-based atom mapping and units.
    system = mm.System()
    for _ in range(2):
        system.addParticle(12.0)
    system.addForce(PlumedForce("d: DISTANCE ATOMS=1,2 NOPBC\nb: RESTRAINT ARG=d AT=0.4 KAPPA=2000\n"))
    integ = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integ, platform, props)
    context.setPositions([[0, 0, 0], [0.35, 0, 0]])
    state = context.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    assert abs(e - 2.5) < 1e-8 and np.max(abs(f - [[-100, 0, 0], [100, 0, 0]])) < 1e-6
    report["plumed_analytical_energy_error_kJ_per_mol"] = abs(e - 2.5)
    report["plumed_analytical_force_max_error_kJ_per_mol_nm"] = float(np.max(abs(f - [[-100, 0, 0], [100, 0, 0]])))
    del context, integ
    # Official WESTPA NaCl/water fixture; no target-enzyme assumptions are used.
    structure = app.PDBFile(str(pdb))
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = ff.createSystem(structure.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1 * unit.nanometer, constraints=app.HBonds)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(81231)
    sim = app.Simulation(structure.topology, system, integ, platform, props)
    sim.context.setPositions(structure.positions)
    sim.minimizeEnergy(maxIterations=30)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 9182)
    sim.step(20)
    checkpoint = out.with_suffix(".chk")
    sim.saveCheckpoint(str(checkpoint))
    saved_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    start = time.monotonic()
    sim.step(20)
    first = sim.context.getState(getPositions=True, getEnergy=True)
    positions = first.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    energy = first.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    sim.loadCheckpoint(str(checkpoint))
    restored = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    restore_error = float(np.max(abs(saved_positions - restored)))
    assert restore_error < 1e-12
    sim.step(20)
    replay = sim.context.getState(getPositions=True, getEnergy=True)
    error = float(np.max(abs(positions - replay.getPositions(asNumpy=True).value_in_unit(unit.nanometer))))
    # The CPU platform uses single-precision nonbonded arithmetic. Rebuilding
    # its neighbor list on load can change roundoff; checkpoint restoration
    # itself must still reproduce coordinates exactly.
    replay_tolerance = 1e-5 if platform_name == "CPU" else 1e-7
    report.update(restart_position_max_error_nm=error, potential_energy_kJ_per_mol=energy,
                  checkpoint_restore_max_error_nm=restore_error, replay_tolerance_nm=replay_tolerance)
    atomic_json(out, report)
    assert np.isfinite(energy) and error < replay_tolerance, report
    report.update(status="passed", atoms=system.getNumParticles(), md_steps=60,
                  restart_position_max_error_nm=error, checkpoint_sha256=digest(checkpoint),
                  potential_energy_kJ_per_mol=energy, two_segments_seconds=time.monotonic() - start,
                  enzyme_accuracy=None, physical_rate=None, processivity_demonstrated=False)
    atomic_json(out, report)
    return report


def hybrid_check(pdb, parm7, model_pdb, output, device, profile, second_pdb=None):
    from ase.io import read
    from ase.md.verlet import VelocityVerlet
    from ase import units
    from mlmm.backends.mlmm_calc import MLMMCore, MLMMASECalculator
    import os

    resource_check(device == "cuda")
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["SAMPLING_HIGH_LEVEL_PROFILE"] = str(Path(profile).resolve(strict=True))
    spec = json.loads(Path(profile).read_text())
    # Fixture charge/layers come from the authors' smoke input, explicitly supplied.
    calc_file = Path(__file__).with_name("persistent_calculator.py")
    core = MLMMCore(input_pdb=str(Path(pdb).resolve(strict=True)),
                    real_parm7=str(Path(parm7).resolve(strict=True)),
                    model_pdb=str(Path(model_pdb).resolve(strict=True)),
                    model_charge=spec["charge"], model_mult=spec["multiplicity"],
                    backend="custom", calc_file=str(calc_file), ml_device=device,
                    mm_backend="openmm", mm_device="cpu", mm_threads=1,
                    xtb_ncores=1, embedcharge=False, print_timing=False, print_vram=False)
    calculator = MLMMASECalculator(core)
    atoms = read(pdb)
    atoms.calc = calculator
    try:
        first = time.monotonic()
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        warmup = time.monotonic() - first
        # Include every layer and a covalent boundary when present, without guessing atom names.
        selected = sorted(set([0, len(atoms) - 1, *core.selection_indices[:1],
                               *[i - 1 for pair in core.mlmm_links for i in pair]]))
        fd = force_check(atoms, calculator, selected, step=1e-3)
        assert np.isfinite(forces).all() and np.isfinite(energy) and fd < 0.02
        atoms.set_velocities(np.zeros((len(atoms), 3)))
        dyn = VelocityVerlet(atoms, timestep=0.05 * units.fs)
        before = atoms.get_total_energy()
        t = time.monotonic()
        dyn.run(3)
        drift = float(atoms.get_total_energy() - before)
        assert abs(drift) < 0.02
        report = dict(scope="author_fixture_software_numerical_check_only", status="passed",
                      runtime=runtime(), input_sha256=digest(pdb), parm7_sha256=digest(parm7),
                      model_pdb_sha256=digest(model_pdb), ml_atoms=len(core.selection_indices),
                      link_bonds_1based=core.mlmm_links,
                      model_profile_sha256=digest(profile), energy_eV=energy, atom_count=len(atoms),
                      selected_force_check_atoms=selected, force_check_max_error_eV_per_A=fd,
                      first_force_seconds=warmup, md_steps=3, timestep_fs=0.05,
                      nve_total_energy_drift_eV=drift, md_seconds=time.monotonic() - t,
                      embedding="mechanical_ONIOM", mm_backend="openmm", enzyme_accuracy=None,
                      activation_free_energy=None, processivity_demonstrated=False)
        if second_pdb is not None:
            from ase.mep import NEB
            second = read(second_pdb)
            if not np.array_equal(atoms.numbers, second.numbers):
                raise ValueError("Path endpoint atom identities/order differ")
            images = [read(pdb), read(pdb), second]
            for image in images:
                image.calc = calculator
            band = NEB(images, allow_shared_calculator=True, method="aseneb")
            band.interpolate()
            projected = band.get_forces()
            assert np.isfinite(projected).all()
            report["path_interface"] = dict(images=3, second_input_sha256=digest(second_pdb),
                finite_projected_forces=True, converged=False, activation_free_energy=None,
                scope="single_band_force_evaluation_only_no_transition_state_claim")
        atomic_json(out, report)
        return report
    finally:
        custom = getattr(getattr(core, "_ml_backend", None), "_ase_calc", None)
        if custom is not None:
            custom.close()
        core.cleanup()
        # The backend owns the custom ASE calculator; process exit also closes its pipe.


def main():
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    md = sub.add_parser("md")
    md.add_argument("--pdb", required=True)
    md.add_argument("--output", required=True)
    md.add_argument("--platform", choices=["CPU", "CUDA"], required=True)
    hybrid = sub.add_parser("hybrid")
    for arg in ("pdb", "parm7", "model-pdb", "output", "profile"):
        hybrid.add_argument(f"--{arg}", required=True)
    hybrid.add_argument("--device", choices=["cpu", "cuda"], required=True)
    hybrid.add_argument("--second-pdb")
    a = p.parse_args()
    if a.command == "md":
        report = md_check(a.pdb, a.output, a.platform)
    else:
        report = hybrid_check(a.pdb, a.parm7, a.model_pdb, a.output, a.device, a.profile, a.second_pdb)
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
