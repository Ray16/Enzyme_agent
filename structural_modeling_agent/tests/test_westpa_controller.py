import h5py
import numpy as np
import pytest

from structural_modeling_agent.sampling.westpa_controller import validate_completed_run


@pytest.mark.parametrize("problem", [None, "status", "nan", "weight", "iteration"])
def test_zero_process_exit_cannot_substitute_for_committed_segments(tmp_path, problem):
    path = tmp_path / "west.h5"
    seg = np.zeros(2, dtype=[("status", "u1"), ("weight", "f8")])
    seg["status"], seg["weight"] = 2, .5
    if problem == "status": seg["status"][0] = 1
    if problem == "weight": seg["weight"][0] = .25
    with h5py.File(path, "w") as h:
        h.attrs["west_current_iteration"] = 1 if problem == "iteration" else 2
        g = h.create_group("iterations/iter_00000001")
        g["seg_index"] = seg
        pcoord = np.zeros((2, 3, 1))
        if problem == "nan": pcoord[0, 0, 0] = np.nan
        g["pcoord"] = pcoord
    result = validate_completed_run(path, 1, expected_total_probability=1.0)
    assert result["complete"] == (problem is None)
