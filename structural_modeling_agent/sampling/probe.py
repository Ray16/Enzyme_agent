"""Bounded model/accelerator and native-batch probe, using one water molecule.

This is software compatibility evidence, not enzyme or solvent validation.
Run under taskset and gpu_reserve; all outputs include their exact scope.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
from ase.build import molecule

from . import backends
from .pilot import atomic_json, digest, force_check, resource_check, runtime


def probe(backend, checkpoint, output, reference_yaml=None, structure_path=None, charge=None, multiplicity=None, uma_precision="float32"):
    scope = "prepared_input_software_resource_probe_only" if structure_path else "single_water_software_compatibility_only"
    report = {"scope": scope, "backend": backend,
              "runtime": runtime(), "status": "failed", "enzyme_accuracy": None,
              "checkpoint": str(Path(checkpoint).resolve()), "peak_device_memory_bytes": None}
    started = time.monotonic()
    stop_watchdog = threading.Event()
    watchdog = None
    try:
        resource_check(True)
        def watch_resources():
            while not stop_watchdog.wait(10):
                try:
                    resource_check(True)
                except Exception as exc:
                    report.update(status="stopped_resource_check", error=str(exc))
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(output, report)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
        watchdog = threading.Thread(target=watch_resources, daemon=True)
        watchdog.start()
        if structure_path:
            from ase.io import read
            if type(charge) is not int or type(multiplicity) is not int or multiplicity < 1:
                raise ValueError("A supplied structure requires explicit integer charge and multiplicity")
            atoms = read(structure_path)
            electrons = int(atoms.numbers.sum()) - charge
            if (electrons - multiplicity + 1) % 2:
                raise ValueError("Input electron parity disagrees with charge/spin")
            report.update(input_sha256=digest(structure_path), atom_count=len(atoms),
                          charge=charge, spin_multiplicity=multiplicity,
                          chemical_validation="not established by this numerical/resource probe")
        else:
            atoms = molecule("H2O")
            charge, multiplicity = 0, 1
        backends.bind_electronic_state(atoms, charge, multiplicity)
        if backend == "uma":
            report["checkpoint_sha256"] = digest(checkpoint)
            report["reference_sha256"] = digest(reference_yaml)
            calc = backends.uma(checkpoint, charge=charge, multiplicity=multiplicity,
                                reference_yaml=reference_yaml, precision=uma_precision)
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
            batch_size = 1 if structure_path else 4
            copies = [atoms.copy() for _ in range(batch_size)]
            state = ts.io.atoms_to_state(copies, "cuda", dtype=torch.float32,
                                         system_extras_map={"charge": "charge", "spin": "spin"})
            t = time.monotonic()
            prediction = native(state)
            torch.cuda.synchronize()
            report["native_batch_first_call_seconds"] = time.monotonic() - t
            e = prediction["energy"].detach().cpu().numpy()
            f = prediction["forces"].detach().cpu().numpy()
            report["native_batch_size"] = batch_size
            report["native_vs_ase_max_energy_error_eV"] = float(np.max(np.abs(e - atoms.get_potential_energy())))
            report["native_vs_ase_max_force_error_eV_per_A"] = float(np.max(np.abs(f.reshape(batch_size, len(atoms), 3) - atoms.get_forces())))
            t = time.monotonic()
            for _ in range(10):
                native(state)
            torch.cuda.synchronize()
            report["native_batch_steady_seconds_per_call"] = (time.monotonic() - t) / 10
            report["peak_device_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        report["status"] = "software_probe_completed_review_numerical_errors"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_watchdog.set()
        if watchdog is not None:
            watchdog.join(timeout=15)
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            report["peak_device_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            report["peak_device_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
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
    parser.add_argument("--structure")
    parser.add_argument("--charge", type=int)
    parser.add_argument("--multiplicity", type=int)
    parser.add_argument("--uma-precision", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    if args.backend == "uma" and not args.reference_yaml:
        parser.error("UMA requires --reference-yaml")
    result = probe(args.backend, args.checkpoint, args.output, args.reference_yaml,
                   args.structure, args.charge, args.multiplicity, args.uma_precision)
    print(json.dumps(result))
    raise SystemExit(0 if result["status"].startswith("software_probe_completed") else 2)
