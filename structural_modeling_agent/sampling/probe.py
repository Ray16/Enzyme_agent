"""Bounded model/accelerator and native-batch probe, using one water molecule.

This is software compatibility evidence, not enzyme or solvent validation.
Run under taskset and gpu_reserve; all outputs include their exact scope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from ase.build import molecule

from . import backends
from .pilot import atomic_json, digest, force_check, resource_check, runtime


def probe(backend, checkpoint, output, reference_yaml=None):
    report = {"scope": "single_water_software_compatibility_only", "backend": backend,
              "runtime": runtime(), "status": "failed", "enzyme_accuracy": None,
              "checkpoint": str(Path(checkpoint).resolve()), "peak_device_memory_bytes": None}
    started = time.monotonic()
    try:
        resource_check(True)
        atoms = molecule("H2O")
        atoms.info.update(charge=0, spin=1)
        if backend == "uma":
            report["checkpoint_sha256"] = digest(checkpoint)
            report["reference_sha256"] = digest(reference_yaml)
            calc = backends.uma(checkpoint, charge=0, multiplicity=1, reference_yaml=reference_yaml)
            report["actual_inference_settings"] = calc.sampling_settings
        elif backend == "orbmol_v2":
            report["checkpoint_sha256"] = digest(checkpoint)
            calc = backends.orbmol_v2(checkpoint)
        else:
            report["checkpoint_artifacts"] = {str(p.relative_to(checkpoint)): digest(p)
                                               for p in sorted(Path(checkpoint).rglob("*")) if p.is_file()}
            calc = backends.so3lr(checkpoint, lr_cutoff_A=100.0)
        report["construction_seconds"] = time.monotonic() - started
        atoms.calc = calc
        t = time.monotonic()
        energy = float(np.asarray(atoms.get_potential_energy()).reshape(()))
        forces = atoms.get_forces()
        report["first_energy_force_seconds"] = time.monotonic() - t
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise RuntimeError("Nonfinite energy or forces")
        report["energy_eV"] = float(energy)
        report["force_check_max_error_eV_per_A"] = force_check(atoms, calc, [0, 1, 2], 1e-3)
        t = time.monotonic()
        for step in range(10):
            atoms.positions[0, 0] += 1e-5
            atoms.get_forces()
        report["steady_force_seconds_per_call"] = (time.monotonic() - t) / 10
        resource_check(True)
        if backend == "orbmol_v2":
            import torch
            import torch_sim as ts
            from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

            native = OrbTorchSimModel(calc.model, calc.adapter)
            copies = [atoms.copy() for _ in range(4)]
            state = ts.io.atoms_to_state(copies, "cuda", dtype=torch.float32,
                                         system_extras_map={"charge": "charge", "spin": "spin"})
            t = time.monotonic()
            prediction = native(state)
            torch.cuda.synchronize()
            report["native_batch_first_call_seconds"] = time.monotonic() - t
            e = prediction["energy"].detach().cpu().numpy()
            f = prediction["forces"].detach().cpu().numpy()
            report["native_batch_size"] = 4
            report["native_vs_ase_max_energy_error_eV"] = float(np.max(np.abs(e - atoms.get_potential_energy())))
            report["native_vs_ase_max_force_error_eV_per_A"] = float(np.max(np.abs(f.reshape(4, 3, 3) - atoms.get_forces())))
            t = time.monotonic()
            for _ in range(10):
                native(state)
            torch.cuda.synchronize()
            report["native_batch_steady_seconds_per_call"] = (time.monotonic() - t) / 10
            report["peak_device_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        report["status"] = "software_probe_completed_review_numerical_errors"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["entrypoint_seconds"] = time.monotonic() - started
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("so3lr", "orbmol_v2", "uma"))
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-yaml")
    args = parser.parse_args()
    if args.backend == "uma" and not args.reference_yaml:
        parser.error("UMA requires --reference-yaml")
    result = probe(args.backend, args.checkpoint, args.output, args.reference_yaml)
    print(json.dumps(result))
    raise SystemExit(0 if result["status"].startswith("software_probe_completed") else 2)
