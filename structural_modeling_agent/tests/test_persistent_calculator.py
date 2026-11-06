import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from structural_modeling_agent.sampling.persistent_calculator import PersistentCalculator, read_profile


def test_persistent_environment_bridge_preserves_forces_and_process(tmp_path, monkeypatch):
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    artifact = tmp_path / "fixture.txt"
    artifact.write_text("CPU interface test only")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(dict(schema="ase-model-process-1.0", python=sys.executable,
        factory="ase.calculators.emt:EMT", kwargs={}, charge=0, multiplicity=1, require_gpu=False,
        artifacts=[dict(path=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())],
        log=str(tmp_path / "model.log"), timeout_seconds=30)))
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.4, 0.2, 0]])
    reference = atoms.copy()
    reference.calc = EMT()
    bridge = PersistentCalculator(profile)
    atoms.calc = bridge
    try:
        np.testing.assert_allclose(atoms.get_forces(), reference.get_forces(), atol=1e-12)
        pid = bridge.process.pid
        atoms.positions[1, 0] += 0.1
        reference.positions[:] = atoms.positions
        assert atoms.get_potential_energy() == pytest.approx(reference.get_potential_energy(), abs=1e-12)
        assert bridge.process.pid == pid
        atoms.info["multiplicity"] = 2
        with pytest.raises(ValueError, match="disagrees"):
            atoms.get_forces()
    finally:
        process = bridge.process
        bridge.close()
    assert process.poll() is not None
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_profile(profile)
