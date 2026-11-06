"""Numerical and recovery tests; toy data are never enzyme validation evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
import pytest
from ase import Atoms, units
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from structural_modeling_agent.sampling import pilot as p
from structural_modeling_agent.sampling.fleet import validate_assignments


def fixture_config(tmp_path):
    atoms = Atoms("Ar3", positions=[[0, 0, 0], [3.2, 0.2, 0], [5, 0.8, 0.3]])
    structure = tmp_path / "prepared.extxyz"
    write(structure, atoms)
    c = dict(schema_version=1, purpose="software_validation", structure=str(structure),
             structure_sha256=p.digest(structure), charge=0, spin_multiplicity=1,
             temperature_K=300, timestep_fs=0.5, friction_per_fs=0.01,
             max_wall_seconds=60, equilibration_steps=10, production_steps=50,
             sample_interval=5, checkpoint_interval=10, seeds=[17, 31],
             distance_atoms=[{"index": 0, "element": "Ar"}, {"index": 1, "element": "Ar"}],
             angle_atoms=[{"index": i, "element": "Ar"} for i in (0, 1, 2)],
             umbrella={"centers_A": [3.0, 3.2, 3.4], "spring_eV_per_A2": 0.1},
             models=[{"id": "toy", "factory": "structural_modeling_agent.sampling.pilot:toy_calculator",
                      "force_check_tolerance_eV_per_A": 1e-5}],
             bond_guards=[], metal_sites=[],
             analysis={"min_effective_samples_per_window": 20, "min_adjacent_overlap": 0.03,
                       "bin_edges_A": np.linspace(2.5, 3.9, 20).tolist()})
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(c))
    return path, c, atoms


@pytest.fixture
def cpu_run(monkeypatch):
    # Resource enforcement has separate tests. No tests reserve or initialize a GPU.
    monkeypatch.setattr(p, "resource_check", lambda *_: None)


def test_umbrella_matches_energy_gradient_in_periodic_cell(tmp_path):
    _, _, atoms = fixture_config(tmp_path)
    atoms.cell = [10, 10, 10]
    atoms.pbc = True
    atoms.positions[0] = [9, 0, 0]
    atoms.positions[1] = [1, 0.3, 0.2]
    calc = p.DistanceUmbrella(p.toy_calculator(), [0, 1], 3.0, 0.5)
    error = p.force_check(atoms, calc, [0, 1], 1e-4)
    assert error < 1e-7
    assert type(error) is float  # GPU FP32 results must remain JSON serializable.


def test_unbiased_potential_unchanged(tmp_path):
    _, _, atoms = fixture_config(tmp_path)
    base = p.toy_calculator()
    wrapper = p.DistanceUmbrella(base, [0, 1], None, 0.5)
    assert wrapper.get_potential_energy(atoms) == base.get_potential_energy(atoms)
    np.testing.assert_array_equal(wrapper.get_forces(atoms), base.get_forces(atoms))


def test_preparation_and_identity_fail_closed(tmp_path):
    path, c, _ = fixture_config(tmp_path)
    c["distance_atoms"][0]["element"] = "O"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="element/index"):
        p.load(path)
    c["distance_atoms"][0]["element"] = "Ar"
    c["purpose"] = "pose_accessibility"
    path.write_text(json.dumps(c))
    with pytest.raises(KeyError, match="preparation_report"):
        p.load(path)


def test_scientific_input_requires_key_residue_protonation_review(tmp_path):
    path, c, atoms = fixture_config(tmp_path)
    atoms += Atoms("H", positions=[[10, 10, 10]])
    write(c["structure"], atoms)
    c.update(purpose="pose_accessibility", charge=1, structure_sha256=p.digest(c["structure"]),
             bond_guards=[{"atoms": [0, 1], "min_A": 2.0, "max_A": 4.0}])
    prep = {"reviewed": True, "structure_sha256": c["structure_sha256"], "protein_lengths": [1],
            "sequence_sha256": "synthetic", "components": [{"formal_charge": 1, "count": 1}],
            "assay_conditions": {"pH": 7.5, "pH_source": "synthetic fixture"}}
    for key in ("protonation_evidence", "solvent_evidence", "metal_evidence", "mechanism_evidence",
                "reactive_atom_mapping_evidence"):
        prep[key] = "synthetic fixture"
    prep_path = tmp_path / "preparation.json"
    prep_path.write_text(json.dumps(prep))
    c.update(preparation_report=str(prep_path), preparation_sha256=p.digest(prep_path))
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="coverage of catalytic"):
        p.load(path)


def test_resume_reproduces_uninterrupted_langevin(tmp_path, monkeypatch, cpu_run):
    from ase.md.langevin import Langevin
    path, _, _ = fixture_config(tmp_path)
    full = tmp_path / "full"
    split = tmp_path / "split"
    assert p.run(path, full, 1)["status"] == "complete"
    original = Langevin.run
    calls = [0]

    def stop_after_block(self, steps):
        result = original(self, steps)
        calls[0] += 1
        if calls[0] == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(Langevin, "run", stop_after_block)
    assert p.run(path, split, 1)["status"] == "partial"
    monkeypatch.setattr(Langevin, "run", original)
    assert p.run(path, split, 1, resume=True)["status"] == "complete"
    a = np.load(sorted((full / "task-00001").glob("chunk-*.npz"))[-1])
    b = np.load(sorted((split / "task-00001").glob("chunk-*.npz"))[-1])
    np.testing.assert_array_equal(a["positions"], b["positions"])
    np.testing.assert_array_equal(a["momenta"], b["momenta"])
    assert str(a["rng"]) == str(b["rng"])


def test_claim_and_existing_outputs_are_preserved(tmp_path, cpu_run):
    path, c, _ = fixture_config(tmp_path)
    out = tmp_path / "out"
    p.run(path, out, 0)
    with pytest.raises(ValueError, match="Existing output"):
        p.run(path, out, 0)
    claim = out / "task-00001" / "claim"
    claim.mkdir(parents=True)
    (claim / "owner.json").write_text('{"host":"unreachable"}')
    with pytest.raises(FileExistsError):
        p.run(path, out, 1)
    assert claim.exists()
    assert p.complete(out / "task-00000", p.canonical_hash(c), p.tasks(c)[0])


def test_worker_reuses_model_and_schedules_independent_tasks(tmp_path, cpu_run, monkeypatch):
    path, c, _ = fixture_config(tmp_path)
    p._factory_cached.cache_clear()
    result = p.worker(path, tmp_path / "out", "toy", max_tasks=3)
    assert len(result["results"]) == 3
    assert all(r["status"] == "complete" for r in result["results"])
    assert p._factory_cached.cache_info().misses == 1
    assert p._factory_cached.cache_info().hits == 3
    # A peer skips finished work and claims the next independent window.
    result = p.worker(path, tmp_path / "out", "toy", max_tasks=1)
    assert result["results"][0]["task"] == 3


def test_thread_and_cpu_guards_fail_before_model_import(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1})
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    with pytest.raises(RuntimeError, match="thread caps"):
        p.resource_check()


def test_fleet_refuses_gpu_and_cpu_sharing(tmp_path):
    _, c, _ = fixture_config(tmp_path)
    a = {"host": "lambda2", "gpu": 0, "cpus": [78], "model": "toy", "max_tasks": 2,
         "python": "/absolute/python", "min_available_ram_mib": 1024}
    b = {**a, "gpu": 1, "cpus": [79]}
    assignments = {"max_workers": 2, "workers": [a, b]}
    validate_assignments(assignments, c)
    b["cpus"] = [78]
    with pytest.raises(ValueError, match="CPU sets overlap"):
        validate_assignments(assignments, c)
    b["cpus"] = [79]
    b["gpu"] = 0
    with pytest.raises(ValueError, match="share a GPU"):
        validate_assignments(assignments, c)


def test_completed_marker_without_checkpoint_is_not_completion(tmp_path, cpu_run):
    path, c, _ = fixture_config(tmp_path)
    out = tmp_path / "out"
    p.run(path, out, 0)
    for file in (out / "task-00000").glob("chunk-*.npz"):
        file.unlink()
    with pytest.raises(ValueError, match="without chunks"):
        p.complete(out / "task-00000", p.canonical_hash(c), p.tasks(c)[0])


def test_reactant_bond_guard_stops_chemical_progress(tmp_path, cpu_run):
    path, c, _ = fixture_config(tmp_path)
    c["bond_guards"] = [{"atoms": [0, 1], "min_A": 3.5, "max_A": 4.5}]
    path.write_text(json.dumps(c))
    with pytest.raises(RuntimeError, match="bond guard failed"):
        p.run(path, tmp_path / "out", 0)
    assert not list((tmp_path / "out" / "task-00000").glob("chunk-*.npz"))


def test_contention_stops_worker_without_spending_input_attempt(tmp_path, cpu_run, monkeypatch):
    path, _, _ = fixture_config(tmp_path)

    def contended(*args, **kwargs):
        raise p.ResourceUnavailable("Another process occupies the assigned GPU")

    monkeypatch.setattr(p, "run", contended)
    result = p.worker(path, tmp_path / "out", "toy", max_tasks=4)
    assert len(result["results"]) == 1
    assert result["results"][0]["status"] == "resource_unavailable"
    assert not list((tmp_path / "out").glob("task-*/failure-*.json"))


def test_mbar_recovers_known_harmonic_profile_and_rejects_missing_window(tmp_path):
    path, c, _ = fixture_config(tmp_path)
    c["production_steps"] = 50000
    path.write_text(json.dumps(c))
    out = tmp_path / "out"
    out.mkdir()
    rng = np.random.default_rng(912)
    beta = 1 / (units.kB * c["temperature_K"])
    k0, ku, r0 = 0.2, c["umbrella"]["spring_eV_per_A2"], 3.2
    for i, task in enumerate(p.tasks(c)):
        if task["window"] < 0:
            continue
        work = out / f"task-{i:05d}"
        work.mkdir()
        p.atomic_json(work / "provenance.json", {"config_hash": p.canonical_hash(c), "task": task})
        center = task["center_A"]
        mean = (k0 * r0 + ku * center) / (k0 + ku)
        x = rng.normal(mean, np.sqrt(1 / (beta * (k0 + ku))), 10000)
        samples = np.column_stack([np.arange(15, 50011, 5), x, np.full(len(x), 160),
                                   np.full(len(x), 300), np.zeros(len(x))])
        np.savez(work / "chunk-000000.npz", step=50010, samples=samples, config_hash=p.canonical_hash(c))
        p.atomic_json(work / "status.json", {"status": "complete", "requested_steps": 50010})
    report = p.analyze(path, out, "toy")
    centers = np.array(report["bin_centers_A"])
    estimated = np.array(report["pmf_eV"], dtype=float)
    expected = k0 / 2 * (centers - r0) ** 2
    expected -= expected.min()
    assert np.max(np.abs(estimated - expected)) < 0.006
    assert report["chemical_barrier"] is None
    assert report["uncertainty_eV"] is None
    (out / "task-00001" / "status.json").unlink()
    with pytest.raises(FileNotFoundError):
        p.analyze(path, out, "toy")
