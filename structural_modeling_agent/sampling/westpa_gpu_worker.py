"""One reserved GPU per WESTPA ZeroMQ worker node; master owns HDF5 writes.

Run through taskset and gpu_reserve with explicit --work-manager zmq
--zmq-mode node --n-workers 1. The propagator must select CUDA device 0 inside
the reservation and preserve its own segment outputs for WESTPA restart.
"""
import os
from pathlib import Path
import signal
import sys
import threading

from .pilot import atomic_json, resource_check, runtime


def enable_subreaper():
    """Adopt grandchildren if WESTPA kills an intermediate helper at shutdown."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot supervise WESTPA descendants")


def cleanup_descendants():
    import psutil
    children = psutil.Process().children(recursive=True)
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)
    return [p.pid for p in children]


def main():
    from westpa.cli.core.w_run import entry_point

    args = sys.argv[1:]
    for key, value in (("--work-manager", "zmq"), ("--zmq-mode", "node"), ("--n-workers", "1")):
        if args.count(key) != 1 or args.index(key) + 1 >= len(args) or args[args.index(key) + 1] != value:
            raise ValueError("GPU WESTPA requires explicit ZMQ node mode and exactly one worker")
    resource_check(True)
    enable_subreaper()
    out = Path(os.environ["WEST_SIM_ROOT"]) / "gpu_workers"
    out.mkdir(exist_ok=True)
    record = runtime()
    record["physical_gpu_index"] = os.environ["CUDA_VISIBLE_DEVICES"]
    record["status"] = "started"
    path = out / f"{record['host']}-{os.getpid()}.json"
    atomic_json(path, record)
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(10):
            try:
                resource_check(True)
            except Exception as exc:
                record.update(status="stopped_resource_check", error=str(exc))
                atomic_json(path, record)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def interrupt(*_):
        raise KeyboardInterrupt("WESTPA worker interrupted")

    previous_signals = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        entry_point()
        # ZMQNode.run() starts a communication thread and returns immediately.
        # Keep the reservation until the master's shutdown reaches that thread.
        import westpa
        westpa.rc.work_manager.join()
        if record["status"] == "started":
            record["status"] = "worker_exited"
    except BaseException:
        if record["status"] == "started":
            record["status"] = "worker_interrupted"
        raise
    finally:
        stopped.set()
        watcher.join(timeout=1)
        # Reparented ZMQ helper processes remain ours until explicitly reaped.
        record["cleanup_descendant_pids"] = cleanup_descendants()
        atomic_json(path, record)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
