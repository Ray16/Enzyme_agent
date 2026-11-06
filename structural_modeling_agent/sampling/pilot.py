"""Small ASE NVT pilot with immutable checkpoints and explicit bias removal.

Run with ``python -m structural_modeling_agent.sampling.pilot --help``.
No force-field imports, downloads or GPU initialization occur during planning.
"""
from __future__ import annotations

import argparse
import hashlib
import functools
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

THREAD_VARS = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class ResourceUnavailable(RuntimeError):
    """Contention is a resumable stop, not a failed scientific input."""


def digest(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return _digest_file(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@functools.lru_cache(maxsize=256)
def _digest_file(path, size, mtime_ns, ctime_ns):
    # Cache immutable model/input identities per worker, invalidating on file changes.
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def implementation_hash():
    return canonical_hash({p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def positive(x, name):
    if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be finite and positive")


def integer(x, name, minimum=0):
    if type(x) is not int or x < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def resolve(config_path, value):
    return (Path(config_path).resolve().parent / value).resolve()


def load(config_path):
    c = json.loads(Path(config_path).read_text())
    if c["schema_version"] != 1 or c["purpose"] not in ("pose_accessibility", "software_validation"):
        raise ValueError("Only pose accessibility and software validation are implemented")
    for key in ("temperature_K", "timestep_fs", "friction_per_fs", "max_wall_seconds"):
        positive(c[key], key)
    if c["timestep_fs"] > 0.5:
        raise ValueError("Pilot timestep is capped at 0.5 fs; validate a larger step separately")
    for key in ("production_steps", "sample_interval", "checkpoint_interval"):
        integer(c[key], key, 1)
    integer(c["equilibration_steps"], "equilibration_steps")
    if c["checkpoint_interval"] % c["sample_interval"]:
        raise ValueError("checkpoint_interval must be a multiple of sample_interval")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("Provide distinct seeds")
    for seed in c["seeds"]:
        integer(seed, "seed")
    for key in ("min_effective_samples_per_window",):
        integer(c["analysis"][key], key, 2)
    if not 0 < c["analysis"]["min_adjacent_overlap"] < 1:
        raise ValueError("Invalid minimum adjacent overlap")
    centers = c["umbrella"]["centers_A"]
    if len(centers) < 2 or any(not math.isfinite(x) for x in centers) or any(
            b <= a for a, b in zip(centers, centers[1:])):
        raise ValueError("Provide at least two finite, strictly increasing umbrella centers")
    positive(c["umbrella"]["spring_eV_per_A2"], "spring_eV_per_A2")
    if not c["models"] or len({m["id"] for m in c["models"]}) != len(c["models"]):
        raise ValueError("Model IDs must be nonempty and distinct")
    p = resolve(config_path, c["structure"])
    if digest(p) != c["structure_sha256"]:
        raise ValueError("Prepared structure hash mismatch")
    atoms = read(p)
    if atoms.constraints:
        raise ValueError("Input constraints require a separate reviewed protocol")
    if not np.isfinite(atoms.positions).all() or len(atoms) < 2:
        raise ValueError("Invalid coordinates")
    if atoms.pbc.any() and (not atoms.pbc.all() or atoms.cell.volume <= 0):
        raise ValueError("Require a finite 3D periodic cell or a nonperiodic system")
    integer(c["spin_multiplicity"], "spin_multiplicity", 1)
    if type(c["charge"]) is not int:
        raise ValueError("charge must be an explicit integer")
    electrons = int(atoms.numbers.sum()) - c["charge"]
    if electrons <= 0 or (electrons - (c["spin_multiplicity"] - 1)) % 2:
        raise ValueError("Charge/spin electron parity is inconsistent")
    for key, value in (("charge", c["charge"]), ("spin", c["spin_multiplicity"])):
        if key in atoms.info and atoms.info[key] != value:
            raise ValueError(f"Structure {key} disagrees with configuration")
        atoms.info[key] = value
    for selection, count in ((c["distance_atoms"], 2), (c["angle_atoms"], 3)):
        if len(selection) != count or len({x["index"] for x in selection}) != count:
            raise ValueError("Invalid or repeated CV atom indices")
        for atom in selection:
            i = atom["index"]
            integer(i, "atom index")
            if i >= len(atoms) or atoms[i].symbol != atom["element"]:
                raise ValueError("CV atom element/index mismatch")
    for guard in c["bond_guards"]:
        i, j = guard["atoms"]
        if type(i) is not int or type(j) is not int or i == j or not 0 <= min(i, j) <= max(i, j) < len(atoms):
            raise ValueError("Invalid guard indices")
        if not 0 < guard["min_A"] < guard["max_A"] < math.inf:
            raise ValueError("Invalid bond guard range")
    for site in c["metal_sites"]:
        i = site["index"]
        if type(i) is not int or not 0 <= i < len(atoms) or atoms[i].symbol != site["element"]:
            raise ValueError("Invalid metal identity")
        positive(site["cutoff_A"], "metal cutoff")
        donors = site["donor_indices"]
        if not donors or len(set(donors)) != len(donors) or any(
                type(j) is not int or not 0 <= j < len(atoms) or j == i for j in donors):
            raise ValueError("Invalid metal donors")
    if c["purpose"] == "software_validation":
        if any(m["factory"] != "structural_modeling_agent.sampling.pilot:toy_calculator" for m in c["models"]):
            raise ValueError("Software validation only permits the explicit toy potential")
    else:
        prep_path = resolve(config_path, c["preparation_report"])
        if digest(prep_path) != c["preparation_sha256"]:
            raise ValueError("Preparation report hash mismatch")
        prep = json.loads(prep_path.read_text())
        if prep["structure_sha256"] != c["structure_sha256"] or prep["reviewed"] is not True:
            raise ValueError("Preparation report does not review this exact structure")
        for key in ("sequence_sha256", "protonation_evidence", "solvent_evidence", "metal_evidence",
                    "mechanism_evidence", "reactive_atom_mapping_evidence"):
            if not isinstance(prep.get(key), str) or not prep[key].strip():
                raise ValueError(f"Missing preparation evidence: {key}")
        if not prep.get("protein_lengths") or not all(type(x) is int and x > 0 for x in prep["protein_lengths"]):
            raise ValueError("Preparation must declare protein lengths")
        if not prep.get("components") or sum(x["formal_charge"] * x["count"] for x in prep["components"]) != c["charge"]:
            raise ValueError("Prepared component charge inventory must sum to total charge")
        if not (atoms.numbers == 1).any() or not c["bond_guards"]:
            raise ValueError("Require explicit hydrogens and reviewed reactant-state bond guards")
        from .protonation import validate_protonation_review
        validate_protonation_review(atoms, prep)
        for m in c["models"]:
            if not m.get("source_revision") or not m.get("precision") or not m.get("artifacts"):
                raise ValueError("Pin model revision, numerical profile and checkpoint artifacts")
            declared = set()
            for artifact in m["artifacts"]:
                path = resolve(config_path, artifact["path"])
                if digest(path) != artifact["sha256"]:
                    raise ValueError("Model artifact hash mismatch")
                declared.add(path)
            kwargs = m.get("kwargs", {})
            if "checkpoint" in kwargs:
                if not Path(kwargs["checkpoint"]).is_absolute() or Path(kwargs["checkpoint"]).resolve() not in declared:
                    raise ValueError("Calculator checkpoint must be an explicitly hashed absolute path")
            elif "model_directory" in kwargs:
                directory = Path(kwargs["model_directory"])
                if not directory.is_absolute() or not directory.is_dir():
                    raise ValueError("Use an absolute, existing model directory")
                actual = {p.resolve() for p in directory.rglob("*") if p.is_file()}
                if not actual or actual != declared:
                    raise ValueError("Hash the exact complete model directory used by the calculator")
            else:
                raise ValueError("Scientific runs require a pinned local checkpoint or model directory")
            if m["factory"].endswith((":orbmol_v2", ":so3lr")) and kwargs.get("precision") != m["precision"]:
                raise ValueError("Declared precision disagrees with calculator arguments")
            if m["factory"].endswith(":uma") and (kwargs.get("charge") != c["charge"] or kwargs.get("multiplicity") != c["spin_multiplicity"] or m["precision"] != "ieee-fp32-general"):
                raise ValueError("UMA charge/spin/precision arguments disagree with the prepared system")
            if m["factory"].endswith(":uma") and Path(kwargs["reference_yaml"]).resolve() not in declared:
                raise ValueError("UMA atomic reference table must be hashed alongside its checkpoint")
    return c, atoms


def tasks(c):
    return [dict(model=m["id"], seed=seed, window=window, center_A=center)
            for m in c["models"] for seed in c["seeds"]
            for window, center in [(-1, None), *enumerate(c["umbrella"]["centers_A"])]]


def measurement(atoms, c):
    pair = [x["index"] for x in c["distance_atoms"]]
    angle = [x["index"] for x in c["angle_atoms"]]
    r = atoms.get_distance(*pair, mic=bool(atoms.pbc.any()))
    theta = atoms.get_angle(*angle, mic=bool(atoms.pbc.any()))
    cn = [sum(atoms.get_distances(s["index"], s["donor_indices"], mic=bool(atoms.pbc.any())) < s["cutoff_A"])
          for s in c["metal_sites"]]
    return r, theta, cn


def check_bonds(atoms, c):
    for g in c["bond_guards"]:
        r = atoms.get_distance(*g["atoms"], mic=bool(atoms.pbc.any()))
        if not g["min_A"] <= r <= g["max_A"]:
            raise RuntimeError(f"Reactant-state bond guard failed: {g['atoms']}, {r:.5f} A")


class DistanceUmbrella(Calculator):
    """U_bias = k/2 (r-r0)^2, eV and Angstrom; no orientation restraint."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, base, pair, center, spring):
        super().__init__()
        self.base, self.pair, self.center, self.spring = base, pair, center, spring

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e = float(np.asarray(self.base.get_potential_energy(atoms)).reshape(()))
        f = self.base.get_forces(atoms).copy()
        if self.center is not None:
            i, j = self.pair
            v = atoms.get_distance(i, j, mic=bool(atoms.pbc.any()), vector=True)
            r = np.linalg.norm(v)
            if r < 1e-10:
                raise ValueError("Umbrella atoms coincide")
            delta = r - self.center
            e += 0.5 * self.spring * delta ** 2
            force = self.spring * delta * v / r
            f[i] += force
            f[j] -= force
        if not np.isfinite(e) or not np.isfinite(f).all():
            raise RuntimeError("Nonfinite potential energy/forces")
        self.results = {"energy": e, "forces": f}


def toy_calculator(**kwargs):
    """Two-atom harmonic fixture. Never evidence about enzyme chemistry."""
    class Toy(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            # Use the first pair; a third atom permits an angle monitor.
            v = atoms.get_distance(0, 1, vector=True, mic=bool(atoms.pbc.any()))
            r = np.linalg.norm(v)
            f = np.zeros((len(atoms), 3))
            f[0] = 0.2 * (r - 3.2) * v / r
            f[1] = -f[0]
            self.results = {"energy": 0.1 * (r - 3.2) ** 2, "forces": f}
    return Toy()


@functools.lru_cache(maxsize=1)
def _factory_cached(model_json):
    model = json.loads(model_json)
    module, name = model["factory"].split(":")
    return getattr(importlib.import_module(module), name)(**model.get("kwargs", {}))


def factory(model):
    if "source_revision" in model:
        name = model["factory"].split(":")[1]
        package = {"orbmol_v2": "orb-models", "so3lr": "so3lr", "uma": "fairchem-core"}.get(name)
        if package:
            distribution = importlib.metadata.distribution(package)
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            revision = direct.get("vcs_info", {}).get("commit_id", distribution.version)
            if revision != model["source_revision"]:
                raise ValueError(f"Installed {package} revision differs from the declared model stack")
    return _factory_cached(json.dumps(model, sort_keys=True))


def runtime():
    packages = {}
    for key in ("numpy", "ase", "torch", "jax", "jaxlib", "so3lr", "orb-models", "fairchem-core", "pymbar"):
        try:
            packages[key] = importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:
            packages[key] = None
    return dict(python=sys.executable, packages=packages, host=socket.gethostname(), pid=os.getpid(),
                process_start=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                affinity=sorted(os.sched_getaffinity(0)), threads={k: os.environ.get(k) for k in THREAD_VARS})


def resource_check(gpu=False):
    """Run inside taskset and gpu_reserve, before initializing accelerator libraries."""
    affinity = os.sched_getaffinity(0)
    if not 1 <= len(affinity) <= 4 or any(os.environ.get(k) != "1" for k in THREAD_VARS):
        raise RuntimeError("Require taskset with 1-4 CPU slots and all five thread caps set to 1")
    if os.getloadavg()[0] + len(affinity) > os.cpu_count():
        raise ResourceUnavailable("CPU capacity unavailable")
    if not gpu:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("Require one GPU assigned by gpu_reserve run")
    import psutil
    ancestors = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    owner = Path("/nfs/lambda_stor_01/homes/rzhu/.gpu_locks") / f"{socket.gethostname().split('.')[0]}_gpu{visible}" / "owner"
    fields = owner.read_text().split()
    pid = int(fields[0])
    if pid not in ancestors or len(fields) < 6 or Path(f"/proc/{pid}/stat").read_text().split()[21] != fields[5]:
        raise RuntimeError("No live ancestral GPU reservation")
    uuid = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    allowed = ancestors | {p.pid for p in psutil.Process().children(recursive=True)}
    for line in processes.splitlines():
        other_uuid, other_pid = (x.strip() for x in line.split(","))
        other_pid = int(other_pid)
        if other_uuid == uuid and other_pid not in allowed:
            raise ResourceUnavailable(f"GPU contention: PID {other_pid}; preserving checkpoints")
        if other_pid not in allowed:
            try:
                other_affinity = set(psutil.Process(other_pid).cpu_affinity())
                if len(other_affinity) <= 8 and affinity & other_affinity:
                    raise ResourceUnavailable(f"CPU affinity overlaps GPU process {other_pid}")
            except psutil.NoSuchProcess:
                pass


def force_check(atoms, calc, pair, step=1e-3):
    """Central differences on selected Cartesian coordinates; restore geometry."""
    atoms = atoms.copy()
    atoms.calc = calc
    analytic = atoms.get_forces().copy()
    errors = []
    for i in pair:
        for axis in range(3):
            x = atoms.positions[i, axis]
            atoms.positions[i, axis] = x + step
            ep = atoms.get_potential_energy()
            atoms.positions[i, axis] = x - step
            em = atoms.get_potential_energy()
            atoms.positions[i, axis] = x
            errors.append(abs(-(ep - em) / (2 * step) - analytic[i, axis]))
    return float(max(errors))


def run(config_path, output, task_index, resume=False):
    started = time.monotonic()
    c, atoms = load(config_path)
    task = tasks(c)[task_index]
    model = next(m for m in c["models"] if m["id"] == task["model"])
    gpu = c["purpose"] != "software_validation"
    resource_check(gpu)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    work = output / f"task-{task_index:05d}"
    work.mkdir(exist_ok=True)
    claim = work / "claim"
    claim.mkdir()  # Never reclaim an existing owner by age.
    atomic_json(claim / "owner.json", runtime())
    try:
        return _run_locked(c, atoms, task, model, config_path, work, resume, started, gpu)
    finally:
        (claim / "owner.json").unlink()
        claim.rmdir()


def _run_locked(c, atoms, task, model, config_path, work, resume, started, gpu):
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    identity = canonical_hash(c)
    provenance = work / "provenance.json"
    state_files = sorted(work.glob("chunk-*.npz"))
    old = json.loads(provenance.read_text()) if provenance.exists() else None
    if old and (old["config_hash"] != identity or old["task"] != task or old["code_sha256"] != implementation_hash()):
        raise ValueError("Resume identity mismatch (configuration/task/code)")
    if old and not resume:
        raise ValueError("Existing output: use --resume; never overwrite a run")
    if state_files and not old:
        raise ValueError("Checkpoints lack provenance")
    env = runtime()
    if old and old["runtime"]["packages"] != env["packages"]:
        raise ValueError("Resume dependency versions differ")
    rng = np.random.default_rng(task["seed"])
    step = 0
    if state_files:
        with np.load(state_files[-1], allow_pickle=False) as state:
            step = int(state["step"])
            atoms.positions[:] = state["positions"]
            atoms.set_momenta(state["momenta"])
            rng.bit_generator.state = json.loads(str(state["rng"]))
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=c["temperature_K"], rng=rng)
    total = c["equilibration_steps"] + c["production_steps"]
    if step >= total:
        report = {"status": "complete", "step": step, "requested_steps": total,
                  "recovered_from_complete_checkpoint": True}
        atomic_json(work / "status.json", report)
        return report
    pair = [x["index"] for x in c["distance_atoms"]]
    t = time.monotonic()
    base = factory(model)
    construction_seconds = time.monotonic() - t
    atoms.calc = DistanceUmbrella(base, pair, task["center_A"], c["umbrella"]["spring_eV_per_A2"])
    check_bonds(atoms, c)
    t = time.monotonic()
    atoms.get_forces()
    first_force_seconds = time.monotonic() - t
    t = time.monotonic()
    # Check both displacement scales: a noisy force can accidentally agree at one scale.
    force_errors = {str(h): force_check(atoms, atoms.calc, pair, h) for h in (1e-3, 5e-4)}
    tolerance = model["force_check_tolerance_eV_per_A"]
    positive(tolerance, "force_check_tolerance_eV_per_A")
    if max(force_errors.values()) > tolerance:
        raise RuntimeError(f"Finite-difference force check failed: {force_errors}")
    validation_seconds = time.monotonic() - t
    if gpu:
        resource_check(True)
    if not old:
        atomic_json(provenance, dict(config_hash=identity, config=c, task=task, runtime=env,
                                    code_sha256=implementation_hash(), structure_sha256=c["structure_sha256"],
                                    construction_seconds=construction_seconds, first_force_seconds=first_force_seconds,
                                    force_check_max_errors_eV_per_A=force_errors,
                                    validation_seconds=validation_seconds, import_seconds=None,
                                    atom_count=len(atoms), protein_lengths=None if c["purpose"] == "software_validation" else
                                    json.loads(resolve(config_path, c["preparation_report"]).read_text())["protein_lengths"],
                                    batch_size=1, scheduler_wait_seconds=None,
                                    peak_device_allocated_bytes=None, peak_device_reserved_bytes=None))
    dyn = Langevin(atoms, c["timestep_fs"] * units.fs, temperature_K=c["temperature_K"],
                   friction=c["friction_per_fs"] / units.fs, fixcm=False, rng=rng)
    stop = [False]
    handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    samples = []
    start_step = step
    md_started = time.monotonic()
    last_resource = md_started
    chunk_number = len(state_files)
    status = "partial"

    def checkpoint():
        nonlocal chunk_number, samples
        if not samples:
            return
        path = work / f"chunk-{chunk_number:06d}.npz"
        if path.exists():
            raise RuntimeError("Checkpoint collision")
        with open(path.with_suffix(".tmp"), "wb") as f:
            np.savez(f, step=step, positions=atoms.positions, momenta=atoms.get_momenta(),
                     rng=json.dumps(rng.bit_generator.state), samples=np.asarray(samples),
                     config_hash=identity)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path.with_suffix(".tmp"), path)
        chunk_number += 1
        samples = []

    try:
        while step < total:
            count = min(c["sample_interval"] - step % c["sample_interval"], total - step)
            # If integration/force evaluation fails, retain only previously committed chunks.
            dyn.run(count)
            step += count
            check_bonds(atoms, c)
            r, angle, cn = measurement(atoms, c)
            samples.append([step, r, angle, atoms.get_temperature(), atoms.get_potential_energy(), *cn])
            if step % c["checkpoint_interval"] == 0 or stop[0] or time.monotonic() - started >= c["max_wall_seconds"]:
                checkpoint()
            if time.monotonic() - last_resource >= 30:
                resource_check(gpu)
                last_resource = time.monotonic()
            if stop[0] or time.monotonic() - started >= c["max_wall_seconds"]:
                break
        checkpoint()
        status = "complete" if step == total else "partial"
    except ResourceUnavailable as exc:
        checkpoint()
        status = "partial_resource_contention"
        atomic_json(work / f"contention-{time.time_ns()}.json", {"reason": str(exc), "step": step})
    except Exception as exc:
        atomic_json(work / f"failure-{time.time_ns()}.json", {"error": str(exc), "last_committed_chunks": chunk_number})
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    elapsed = time.monotonic() - md_started
    report = dict(status=status, step=step, requested_steps=total, completed_this_invocation=step - start_step,
                  md_wall_seconds=elapsed, task_wall_seconds=time.monotonic() - started,
                  seconds_per_step=elapsed / max(1, step - start_step),
                  ns_per_day=(step - start_step) * c["timestep_fs"] * 86400 / (elapsed * 1e6),
                  timing_definition="MD includes monitoring and checkpoint I/O; excludes factory, first force, validation",
                  validation_seconds=validation_seconds,
                  construction_seconds=construction_seconds, first_force_seconds=first_force_seconds,
                  scientific_status="software_validation_only" if c["purpose"] == "software_validation" else "exploratory_pose_sampling")
    atomic_json(work / f"timing-{time.time_ns()}.json", report)
    atomic_json(work / "status.json", report)
    return report


def analyze(config_path, output, model_id):
    """Bias-only MBAR: the shared physical U cancels at identical T and Hamiltonian."""
    from pymbar import MBAR, timeseries
    from scipy.special import logsumexp

    c, _ = load(config_path)
    identity = canonical_hash(c)
    centers, springs, series = [], [], []
    metadata = []
    all_tasks = tasks(c)
    for task_index, task in enumerate(all_tasks):
        if task["model"] != model_id or task["window"] < 0:
            continue
        work = Path(output) / f"task-{task_index:05d}"
        p = json.loads((work / "provenance.json").read_text())
        status = json.loads((work / "status.json").read_text())
        if p["config_hash"] != identity or p["task"] != task or status["status"] != "complete":
            raise ValueError("Require matching completed windows; missing/partial windows cannot be hidden")
        if status["requested_steps"] != c["equilibration_steps"] + c["production_steps"]:
            raise ValueError("Completion steps disagree with requested protocol")
        chunks = []
        previous_step = 0
        for file in sorted(work.glob("chunk-*.npz")):
            with np.load(file, allow_pickle=False) as chunk:
                rows = chunk["samples"]
                if str(chunk["config_hash"]) != identity or not len(rows) or rows[0, 0] <= previous_step:
                    raise ValueError("Corrupt or mismatched checkpoint chronology")
                previous_step = int(chunk["step"])
                chunks.append(rows)
        if previous_step != status["requested_steps"]:
            raise ValueError("Completion status lacks matching checkpoint")
        data = np.concatenate(chunks)
        production = data[data[:, 0] > c["equilibration_steps"]]
        if len(production) < 10 or not np.isfinite(production).all():
            raise ValueError("Too few finite production samples")
        # Include angle and coordination relaxation, not just the biased distance.
        start, inefficiency = 0, 1.0
        for column in (1, 2, *range(5, production.shape[1])):
            x = production[:, column]
            if np.std(x) > 1e-12:
                t0, g, _ = timeseries.detect_equilibration(x)
                start, inefficiency = max(start, int(t0)), max(inefficiency, float(g))
        retained = production[start:][timeseries.subsample_correlated_data(production[start:, 1], g=inefficiency), 1]
        if len(retained) < c["analysis"]["min_effective_samples_per_window"]:
            raise ValueError("Insufficient decorrelated samples; extend sampling")
        series.append(retained)
        centers.append(task["center_A"])
        springs.append(c["umbrella"]["spring_eV_per_A2"])
        metadata.append(dict(task=task_index, seed=task["seed"], center_A=task["center_A"],
                             discarded_samples=start, statistical_inefficiency=inefficiency, retained=len(retained),
                             chunk_sha256={f.name: digest(f) for f in sorted(work.glob("chunk-*.npz"))}))
    if len(set(centers)) != len(c["umbrella"]["centers_A"]):
        raise ValueError("Missing umbrella centers")
    beta = 1 / (units.kB * c["temperature_K"])
    x = np.concatenate(series)
    nk = np.array([len(a) for a in series])
    u_kn = beta * 0.5 * np.asarray(springs)[:, None] * (x[None, :] - np.asarray(centers)[:, None]) ** 2
    mbar = MBAR(u_kn, nk, solver_protocol="robust")
    overlap = mbar.compute_overlap()["matrix"]
    # Replicates have duplicate centers; evaluate overlap between each pair of neighboring centers.
    ordered = sorted(set(centers))
    adjacent = []
    for left, right in zip(ordered, ordered[1:]):
        a, b = np.where(np.array(centers) == left)[0], np.where(np.array(centers) == right)[0]
        adjacent.append(float(min(overlap[np.ix_(a, b)].sum(axis=1).mean(),
                                  overlap[np.ix_(b, a)].sum(axis=1).mean())))
    if min(adjacent) < c["analysis"]["min_adjacent_overlap"]:
        raise ValueError("Disconnected/poorly overlapping umbrella windows; no PMF reported")
    log_weights = -logsumexp(np.log(nk)[:, None] + mbar.f_k[:, None] - u_kn, axis=0)
    weights = np.exp(log_weights - logsumexp(log_weights))
    edges = np.asarray(c["analysis"]["bin_edges_A"], dtype=float)
    if not np.isfinite(edges).all() or len(edges) < 3 or (np.diff(edges) <= 0).any():
        raise ValueError("Invalid histogram bin edges")
    hist, _ = np.histogram(x, edges, weights=weights)
    density = hist / np.diff(edges)
    pmf = np.full(len(hist), np.nan)
    valid = density > 0
    pmf[valid] = -np.log(density[valid]) / beta
    pmf[valid] -= pmf[valid].min()
    # A diagnostic profile is intentionally not an uncertainty-qualified scientific result.
    result = dict(status="exploratory_profile_requires_replica_and_block_convergence_review",
                  model=model_id, units="eV", bin_centers_A=((edges[:-1] + edges[1:]) / 2).tolist(),
                  pmf_eV=[float(v) if np.isfinite(v) else None for v in pmf],
                  adjacent_overlap=adjacent, effective_unbiased_samples=float(1 / (weights @ weights)),
                  windows=metadata, uncertainty_eV=None, chemical_barrier=None, activity_prediction=None,
                  limitations=["No bootstrap uncertainty or replica/split-time agreement yet",
                               "A distance PMF can miss slow orthogonal coordinates",
                               "No radial Jacobian correction: this is the marginal distance distribution",
                               "Model accuracy and preparation require independent reference validation"])
    atomic_json(Path(output) / f"analysis-{model_id}.json", result)
    return result


def complete(work, config_hash, task):
    """Validate output evidence, rather than trusting a done marker."""
    work = Path(work)
    if not (work / "status.json").exists():
        return False
    status = json.loads((work / "status.json").read_text())
    if status["status"] != "complete":
        return False
    p = json.loads((work / "provenance.json").read_text())
    if p["config_hash"] != config_hash or p["task"] != task:
        raise ValueError("Completed output identity mismatch")
    chunks = sorted(work.glob("chunk-*.npz"))
    if not chunks:
        raise ValueError("Complete marker without chunks")
    for number, path in enumerate(chunks):
        if path.name != f"chunk-{number:06d}.npz":
            raise ValueError("Missing checkpoint chunk")
    with np.load(chunks[-1], allow_pickle=False) as state:
        if str(state["config_hash"]) != config_hash or int(state["step"]) != status["requested_steps"]:
            raise ValueError("Complete marker disagrees with last checkpoint")
    return True


def worker(config_path, output, model_id, max_tasks):
    """One persistent model per process/GPU; atomic task claims shared over NFS.

    An idle worker exits with explicit peer/partial/terminal state; it never
    polls indefinitely, steals a claim, or retries a broken environment.
    """
    c, _ = load(config_path)
    if model_id not in {m["id"] for m in c["models"]}:
        raise ValueError("Unknown model")
    resource_check(c["purpose"] != "software_validation")
    # Dependency/checkpoint errors must be discovered before claiming an input.
    factory(next(m for m in c["models"] if m["id"] == model_id))
    identity = canonical_hash(c)
    results = []
    consecutive_failures = 0
    peer_claims = []
    for i, task in enumerate(tasks(c)):
        if task["model"] != model_id:
            continue
        work = Path(output) / f"task-{i:05d}"
        if complete(work, identity, task):
            continue
        if (work / "claim").exists():
            peer_claims.append(i)
            continue
        # One actual attempt per worker pass; at most two across resumed passes.
        if len(list(work.glob("failure-*.json"))) >= 2:
            results.append({"task": i, "status": "terminal_failure"})
            continue
        previous_failures = len(list(work.glob("failure-*.json")))
        try:
            result = run(config_path, output, i, resume=(work / "provenance.json").exists())
            results.append({"task": i, **result})
            consecutive_failures = 0
            if result["status"].startswith("partial"):
                break  # wall limit or signal: preserve and release the GPU promptly.
        except FileExistsError:
            peer_claims.append(i)  # Lost an atomic-claim race, not an inference attempt.
            continue
        except ResourceUnavailable as exc:
            results.append({"task": i, "status": "resource_unavailable", "reason": str(exc)})
            break
        except Exception as exc:
            work.mkdir(parents=True, exist_ok=True)
            # Includes failures before first-force/provenance creation.
            if len(list(work.glob("failure-*.json"))) == previous_failures:
                atomic_json(work / f"failure-{time.time_ns()}.json", {"error": str(exc), "worker": runtime()})
            results.append({"task": i, "status": "failed", "error": str(exc)})
            consecutive_failures += 1
            if consecutive_failures >= 3 or isinstance(exc, (ImportError, ModuleNotFoundError)):
                break
        if len([r for r in results if r["status"] != "terminal_failure"]) >= max_tasks:
            break
    summary = {"worker": runtime(), "results": results, "peer_or_unresolved_claims": peer_claims}
    Path(output).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(output) / f"worker-{socket.gethostname()}-{os.getpid()}-{time.time_ns()}.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "worker", "analyze"))
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--max-tasks", type=int, default=1)
    args = parser.parse_args()
    if args.command == "plan":
        c, atoms = load(args.config)
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        manifest = dict(config_hash=canonical_hash(c), config=c, atom_count=len(atoms), tasks=tasks(c),
                        integration_steps=sum(c["equilibration_steps"] + c["production_steps"] for _ in tasks(c)),
                        status="planned_not_submitted", runtime=runtime())
        atomic_json(path / "manifest.json", manifest)
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("config", "runtime", "tasks")}))
    elif args.command == "run":
        if args.task is None or args.task < 0:
            parser.error("run requires --task >= 0")
        print(json.dumps(run(args.config, args.output, args.task, args.resume)))
    elif args.command == "worker":
        if not args.model or args.max_tasks < 1:
            parser.error("worker requires --model and --max-tasks >= 1")
        print(json.dumps(worker(args.config, args.output, args.model, args.max_tasks)))
    else:
        if not args.model:
            parser.error("analyze requires --model")
        print(json.dumps(analyze(args.config, args.output, args.model)))


if __name__ == "__main__":
    main()
