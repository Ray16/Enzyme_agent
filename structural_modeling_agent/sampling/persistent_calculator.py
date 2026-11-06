"""Persistent ASE model bridge for the toolkit's documented --calc-file API.

The MM engine and each ML model keep their independently validated environments.
Coordinates/forces use ASE Angstrom/eV units throughout the pipe. No model is
loaded per MD step. This adapter supplies a potential, not chemical validation.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from structural_modeling_agent.sampling.backends import bind_electronic_state
from structural_modeling_agent.sampling.pilot import digest, resource_check


def read_profile(path):
    p = Path(path).resolve(strict=True)
    c = json.loads(p.read_text())
    if c.get("schema") != "ase-model-process-1.0":
        raise ValueError("Unsupported persistent calculator profile")
    for key in ("python", "factory", "kwargs", "charge", "multiplicity", "require_gpu",
                "artifacts", "log", "timeout_seconds"):
        if key not in c:
            raise ValueError(f"Missing model-process setting: {key}")
    if type(c["require_gpu"]) is not bool or not c["artifacts"]:
        raise ValueError("Require explicit GPU policy and hashed model artifacts")
    if not 0 < c["timeout_seconds"] < 3601:
        raise ValueError("Invalid per-request timeout")
    for artifact in c["artifacts"]:
        if digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Model-process artifact hash mismatch")
    return c


class PersistentCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, profile, **kwargs):
        super().__init__(**kwargs)
        self.profile_path = str(Path(profile).resolve(strict=True))
        self.profile = read_profile(profile)
        self.profile_sha256 = digest(profile)
        self.process = None
        self.log_handle = None
        self.last_resource_check = 0.0

    def _reply(self):
        ready, _, _ = select.select([self.process.stdout], [], [], self.profile["timeout_seconds"])
        if not ready:
            self.close()
            raise TimeoutError("Persistent model request timed out; inspect model log")
        raw = self.process.stdout.readline()
        if not raw:
            self.close()
            raise RuntimeError("Persistent model exited; inspect model log")
        reply = json.loads(raw)
        if "error" in reply:
            self.close()
            raise RuntimeError(f"Persistent model: {reply['error']}")
        return reply

    def _start(self):
        if digest(self.profile_path) != self.profile_sha256:
            raise ValueError("Model-process profile changed after construction")
        log = Path(self.profile["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = log.open("a")
        self.process = subprocess.Popen(
            [self.profile["python"], "-m", "structural_modeling_agent.sampling.persistent_calculator",
             "--serve", self.profile_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log_handle, text=True, bufsize=1,
        )
        if self._reply().get("ready") is not True:
            self.close()
            raise RuntimeError("Model server did not complete initialization")

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        # Electronic-state changes cannot silently reuse ASE's position cache.
        bind_electronic_state(atoms, self.profile["charge"], self.profile["multiplicity"])
        super().calculate(atoms, properties, system_changes)
        if time.monotonic() - self.last_resource_check > 10:
            resource_check(self.profile["require_gpu"])
            self.last_resource_check = time.monotonic()
        if self.process is None:
            self._start()
        self.process.stdin.write(json.dumps({"numbers": atoms.numbers.tolist(),
                                            "positions_A": atoms.positions.tolist(),
                                            "cell_A": atoms.cell.array.tolist(),
                                            "pbc": atoms.pbc.tolist()}) + "\n")
        self.process.stdin.flush()
        reply = self._reply()
        e, f = float(reply["energy_eV"]), np.asarray(reply["forces_eV_per_A"], dtype=float)
        if not np.isfinite(e) or f.shape != (len(atoms), 3) or not np.isfinite(f).all():
            self.close()
            raise ValueError("Invalid model-process energy/forces")
        self.results = {"energy": e, "forces": f}

    def get_property(self, name, atoms=None, allow_calculation=True):
        if atoms is not None:
            bind_electronic_state(atoms, self.profile["charge"], self.profile["multiplicity"])
        return super().get_property(name, atoms, allow_calculation)

    def close(self):
        p, self.process = getattr(self, "process", None), None
        if p is not None:
            try:
                p.stdin.close()
                p.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                p.kill()
                p.wait()
            p.stdout.close()
        if getattr(self, "log_handle", None) is not None:
            self.log_handle.close()
            self.log_handle = None

    def __del__(self):
        self.close()


def get_calculator(charge, spin, device):
    """ML/MM toolkit custom factory; spin is multiplicity in its public API."""
    profile = os.environ["SAMPLING_HIGH_LEVEL_PROFILE"]
    c = read_profile(profile)
    if charge != c["charge"] or spin != c["multiplicity"]:
        raise ValueError("QM/ML region charge/spin disagrees with model-process profile")
    if c["require_gpu"] != str(device).startswith("cuda"):
        raise ValueError("ML/MM device disagrees with model-process GPU policy")
    return PersistentCalculator(profile)


def serve(path):
    # Preserve a protocol descriptor; native model logging also goes to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        c = read_profile(path)
        resource_check(c["require_gpu"])
        module, name = c["factory"].rsplit(":", 1)
        calc = getattr(importlib.import_module(module), name)(**c["kwargs"])
        protocol.write(json.dumps({"ready": True}) + "\n")
        last_check = time.monotonic()
        for line in sys.stdin:
            if time.monotonic() - last_check > 10:
                resource_check(c["require_gpu"])
                last_check = time.monotonic()
            q = json.loads(line)
            atoms = Atoms(numbers=q["numbers"], positions=q["positions_A"],
                          cell=q["cell_A"], pbc=q["pbc"])
            bind_electronic_state(atoms, c["charge"], c["multiplicity"])
            atoms.calc = calc
            forces = atoms.get_forces()
            protocol.write(json.dumps({"energy_eV": float(atoms.get_potential_energy()),
                                       "forces_eV_per_A": forces.tolist()}, allow_nan=False) + "\n")
    except Exception as exc:
        protocol.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", required=True)
    serve(parser.parse_args().serve)
