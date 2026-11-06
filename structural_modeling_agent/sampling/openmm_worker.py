"""Experimental persistent OpenMM WESTPA propagator for software benchmarks.

Uses WESTPA's path/state bookkeeping and OpenMM's ordinary dynamics. A context
and independent RNG stream belong to one worker process, never to cloned walkers.
This adapter currently supports one mapped distance and endpoint WE recycling.
It is not qualified for enzyme kinetics or first-passage boundary detection.
"""
from pathlib import Path
import json
import os
import secrets
import time

import numpy as np
from westpa.core.propagators.executable import ExecutablePropagator
from westpa.core.segment import Segment

from .pilot import atomic_json, digest, resource_check, runtime


def claim_seed(folder, role):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        seed = secrets.randbelow(2**31 - 1) + 1
        path = folder / f"{seed}.json"
        try:
            with path.open("x") as f:
                json.dump(dict(seed=seed, role=role, runtime=runtime()), f)
            return seed
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique recorded worker RNG seed")


class PersistentOpenMMPropagator(ExecutablePropagator):
    def __init__(self, rc=None):
        super().__init__(rc)
        self.profile_path = Path(os.environ["SAMPLING_OPENMM_PROFILE"])
        self.profile = json.loads(self.profile_path.read_text())
        self.profile_hash = digest(self.profile_path)
        self.implementation_hash = digest(__file__)
        if self.profile.get("purpose") != "software_performance_benchmark":
            raise ValueError("Persistent OpenMM adapter is currently software-benchmark only")
        self.simulation = None
        self.owner_pid = None
        self.completed = 0
        self.last_resource_check = 0.0

    def _check_resources(self):
        # The reserved node also runs an independent ten-second watchdog.
        # Avoid spawning nvidia-smi for every short trajectory segment.
        if time.monotonic() - self.last_resource_check >= 10:
            resource_check(True)
            self.last_resource_check = time.monotonic()

    def _initialize(self):
        import openmm as mm
        from openmm import app

        self._check_resources()
        if self.simulation is not None:
            if self.owner_pid != os.getpid():
                raise RuntimeError("An OpenMM context cannot be inherited across a fork")
            return 0.0
        t = time.monotonic()
        for key in ("system", "integrator", "topology"):
            if digest(self.profile[key]["path"]) != self.profile[key]["sha256"]:
                raise ValueError("Persistent OpenMM input hash changed")
        system = mm.XmlSerializer.deserialize(Path(self.profile["system"]["path"]).read_text())
        integrator = mm.XmlSerializer.deserialize(Path(self.profile["integrator"]["path"]).read_text())
        folder = Path(os.environ["WEST_SIM_ROOT"]) / "worker_streams"
        self.integrator_seed = claim_seed(folder, "integrator")
        self.barostat_seed = claim_seed(folder, "barostat")
        integrator.setRandomNumberSeed(self.integrator_seed)
        for force in system.getForces():
            if isinstance(force, mm.MonteCarloBarostat):
                force.setRandomNumberSeed(self.barostat_seed)
        pdb = app.PDBFile(self.profile["topology"]["path"])
        self.simulation = app.Simulation(pdb.topology, system, integrator,
            mm.Platform.getPlatformByName("CUDA"),
            {"DeviceIndex": "0", "Precision": "double", "DeterministicForces": "true"})
        self.owner_pid = os.getpid()
        return time.monotonic() - t

    def _distance(self, state):
        from ase.geometry import find_mic
        from openmm import unit

        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        i, j = self.profile["distance_atoms"]
        vector = xyz[j] - xyz[i]
        if self.profile["periodic"]:
            cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            vector, _ = find_mic(vector, cell, pbc=True)
        value = float(np.linalg.norm(vector) * 10)  # Declared output unit: Angstrom.
        if not np.isfinite(value):
            raise ValueError("Nonfinite persistent-worker progress coordinate")
        return value

    def propagate(self, segments):
        import openmm as mm
        from openmm import app, unit

        for segment in segments:
            started, cpu_started = time.monotonic(), time.process_time()
            if digest(self.profile_path) != self.profile_hash:
                raise ValueError("Worker profile changed during propagation")
            if digest(__file__) != self.implementation_hash:
                raise ValueError("Persistent worker implementation changed during propagation")
            args, env = {}, {}
            self.update_args_env_iter(args, env, segment.n_iter)
            self.update_args_env_segment(args, env, segment)
            work = Path(env[self.ENV_CURRENT_SEG_DATA_REF])
            work.mkdir(parents=True, exist_ok=True)
            parent = Path(env[self.ENV_PARENT_DATA_REF])
            parent = parent / "seg.xml" if parent.is_dir() else parent
            parent_hash = digest(parent)
            marker = work / "complete.json"
            if marker.exists():
                old = json.loads(marker.read_text())
                if old["profile_sha256"] != self.profile_hash or old["parent_sha256"] != parent_hash:
                    raise ValueError("Completed segment identity differs from requested propagation")
                if old["implementation_sha256"] != self.implementation_hash or old["openmm_version"] != mm.__version__:
                    raise ValueError("Completed segment implementation/runtime differs")
                for name, sha in old["artifacts"].items():
                    if digest(work / name) != sha:
                        raise ValueError("Completed segment artifact changed")
                segment.pcoord = np.load(work / "pcoord.npy", allow_pickle=False)
                segment.status = Segment.SEG_STATUS_COMPLETE
                segment.walltime = time.monotonic() - started
                segment.cputime = time.process_time() - cpu_started
                continue
            construction = self._initialize()
            sim = self.simulation
            sim.context.setState(mm.XmlSerializer.deserialize(parent.read_text()))
            steps = self.profile["segment_steps"]
            interval = self.profile["pcoord_interval_steps"]
            if (type(steps) is not int or type(interval) is not int or steps < 1 or interval < 1
                    or steps % interval or steps // interval + 1 != self.system.pcoord_len):
                raise ValueError("Persistent segment schedule does not match WESTPA coordinate shape")
            pcoord = [[self._distance(sim.context.getState(getPositions=True))]]
            t = time.monotonic()
            with (work / "seg.log").open("w") as log:
                sim.reporters = [app.DCDReporter(str(work / "seg.dcd"), interval),
                    app.StateDataReporter(log, 100, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True)]
                done = 0
                while done < steps:
                    count = min(500, interval - done % interval, steps - done)
                    sim.step(count)
                    done += count
                    self._check_resources()
                    if done % interval == 0:
                        pcoord.append([self._distance(sim.context.getState(getPositions=True))])
                sim.reporters = []
            propagation = time.monotonic() - t
            state = sim.context.getState(getPositions=True, getEnergy=True)
            xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            if not np.isfinite(xyz).all() or not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                raise ValueError("Persistent worker produced invalid coordinates/energy")
            sim.saveState(str(work / "seg.xml.tmp"))
            os.replace(work / "seg.xml.tmp", work / "seg.xml")
            segment.pcoord = np.asarray(pcoord, dtype=self.system.pcoord_dtype)
            with (work / "pcoord.npy.tmp").open("wb") as f:
                np.save(f, segment.pcoord, allow_pickle=False)
            os.replace(work / "pcoord.npy.tmp", work / "pcoord.npy")
            self.completed += 1
            report = dict(status="complete", runtime=runtime(), physical_gpu_index=os.environ["CUDA_VISIBLE_DEVICES"],
                profile_sha256=self.profile_hash, parent_sha256=parent_hash,
                implementation_sha256=self.implementation_hash, openmm_version=mm.__version__,
                context_construction_seconds=construction, context_reused=self.completed > 1,
                integrator_seed=self.integrator_seed, barostat_seed=self.barostat_seed,
                completed_by_context=self.completed, propagation_seconds=propagation,
                steps=steps, wall_seconds=time.monotonic()-started,
                artifacts={name:digest(work/name) for name in ("seg.xml","seg.dcd","pcoord.npy")},
                scope="software_performance_only_endpoint_resampling_no_first_passage_claim")
            atomic_json(marker, report)
            segment.status = Segment.SEG_STATUS_COMPLETE
            segment.walltime = time.monotonic() - started
            segment.cputime = time.process_time() - cpu_started
        return segments
