"""WESTPA master with a completion check independent of CLI exit status.

WESTPA can catch a propagation exception and return zero. Require the requested
iterations, completed segments, finite coordinates and declared weight total.
Only the master opens/writes HDF5. Failed/partial runs remain resumable.
"""
import os
from pathlib import Path
import time

import numpy as np

from .pilot import atomic_json, resource_check, runtime


def validate_completed_run(path, requested_iterations, *, expected_total_probability):
    import h5py
    problems, iterations = [], []
    with h5py.File(path, "r") as h:
        current = int(h.attrs["west_current_iteration"])
        if current <= requested_iterations:
            problems.append("Requested final iteration is not committed")
        for i in range(1, requested_iterations + 1):
            name = f"iterations/iter_{i:08d}"
            if name not in h:
                problems.append(f"Missing iteration {i}")
                continue
            g = h[name]
            seg = g["seg_index"][:]
            total = float(np.sum(seg["weight"]))
            complete = int(np.count_nonzero(seg["status"] == 2))
            if complete != len(seg) or not len(seg):
                problems.append(f"Incomplete segments in iteration {i}")
            if not np.isfinite(g["pcoord"][:]).all() or not np.isfinite(seg["weight"]).all():
                problems.append(f"Nonfinite coordinate/weight in iteration {i}")
            if np.any(seg["weight"] <= 0) or abs(total - expected_total_probability) > 1e-10:
                problems.append(f"Weight total differs from declared protocol in iteration {i}")
            iterations.append(dict(iteration=i, segments=len(seg), completed=complete, weight_sum=total))
    return dict(complete=not problems, problems=problems, iterations=iterations,
                current_iteration=current, requested_iterations=requested_iterations)


def main():
    from westpa.cli.core.w_run import entry_point
    import westpa

    resource_check(False)
    expected = float(os.environ["WESTPA_EXPECTED_TOTAL_PROBABILITY"])
    if not np.isfinite(expected) or expected <= 0:
        raise ValueError("Explicit positive expected total probability is required")
    start = time.monotonic()
    entry_point()
    config = westpa.rc.config
    requested = int(config["west", "propagation", "max_total_iterations"])
    data_file = Path(config["west", "data", "west_data_file"])
    report = validate_completed_run(data_file, requested, expected_total_probability=expected)
    report.update(status="complete" if report["complete"] else "partial_requires_resume",
                  runtime=runtime(), controller_wall_seconds=time.monotonic() - start,
                  scope="execution_integrity_only_no_scientific_convergence_claim")
    folder = Path(os.environ["WEST_SIM_ROOT"]) / "controller_attempts"
    folder.mkdir(exist_ok=True)
    atomic_json(folder / f"{os.getpid()}.json", report)
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
