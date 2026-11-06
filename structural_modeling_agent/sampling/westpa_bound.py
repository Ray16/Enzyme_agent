"""Local WESTPA process workers with explicit, disjoint CPU assignments.

Set WESTPA_WORKER_CPUS to comma-separated CPU IDs, one per worker. Launch the
coordinator under taskset on a separate CPU and use --work-manager processes
--n-workers matching the list. GPU workers instead need separately reserved
worker processes; this local CPU launcher does not allocate GPUs.
"""
import json
import os
from pathlib import Path

from .pilot import resource_check


def main():
    import westpa.work_managers as managers
    from westpa.work_managers.processes import ProcessWorkManager
    from westpa.cli.core.w_run import entry_point

    resource_check(False)
    cpus = [int(x) for x in os.environ["WESTPA_WORKER_CPUS"].split(",")]
    if len(set(cpus)) != len(cpus) or set(cpus) & os.sched_getaffinity(0):
        raise ValueError("WESTPA coordinator and workers require disjoint CPUs")
    if any(c < 0 or c >= os.cpu_count() for c in cpus):
        raise ValueError("WESTPA worker CPU is not online")
    if os.getloadavg()[0] + len(cpus) + len(os.sched_getaffinity(0)) > os.cpu_count():
        raise RuntimeError("Insufficient CPU capacity")

    class BoundProcessWorkManager(ProcessWorkManager):
        def startup(self):
            if self.n_workers != len(cpus):
                raise ValueError("WESTPA worker count differs from explicit CPU allocation")
            super().startup()

        def task_loop(self):
            index = int(os.environ["WM_PROCESS_INDEX"])
            os.sched_setaffinity(0, {cpus[index]})
            resource_check(False)
            log = Path(os.environ["WEST_SIM_ROOT"]) / "worker_bindings"
            log.mkdir(exist_ok=True)
            (log / f"{os.getpid()}.json").write_text(json.dumps({
                "pid": os.getpid(), "worker": index, "affinity": sorted(os.sched_getaffinity(0)),
                "scope": "local_CPU_workers_only"}) + "\n")
            super().task_loop()

    managers._available_work_managers["processes"] = BoundProcessWorkManager
    entry_point()


if __name__ == "__main__":
    main()
