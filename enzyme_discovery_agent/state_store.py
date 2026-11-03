"""Atomic JSON snapshots and process-safe updates for shared discovery state."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import threading

_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}
_local = threading.local()


@contextmanager
def controller_lock(run_dir: str | Path):
    """Give one controller exclusive ownership while allowing tool-state locks."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            try:
                fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
                atomic_write_json(root / "agent_state" / "discovery_owner.json", {
                    "hostname": socket.gethostname(), "pid": os.getpid(),
                    "process_start_ticks": fields[19],
                })
            except (OSError, ValueError, IndexError):
                pass  # Owner telemetry is optional; controller locking remains authoritative.
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def state_lock(path: str | Path):
    """Lock a read-modify-write transaction across threads and POSIX processes.

    Readers of atomic snapshots need no lock. Nested transactions in one thread
    reuse the outer advisory lock rather than opening a competing descriptor.
    """
    path = Path(path).resolve()
    key = str(path)
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        depths = getattr(_local, "depths", None)
        if depths is None:
            depths = _local.depths = {}
        if depths.get(key, 0):
            depths[key] += 1
            try:
                yield
            finally:
                depths[key] -= 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_name(path.name + ".lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            depths[key] = 1
            try:
                yield
            finally:
                depths.pop(key)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write(path: str | Path, writer) -> None:
    """Publish a complete, flushed snapshot without exposing a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: str | Path, payload, *, indent=2, allow_nan=True) -> None:
    def write(handle):
        json.dump(payload,handle,indent=indent,default=str,allow_nan=allow_nan)
        handle.write('\n')
    _atomic_write(path,write)


def atomic_write_text(path: str | Path, text: str) -> None:
    _atomic_write(path,lambda handle:handle.write(text))


def write_tool_artifact(run_dir: str | Path, category: str, tool_name: str,
                        payload: dict) -> Path:
    """Allocate and publish a numbered artifact without replacing prior evidence."""
    run_dir = Path(run_dir)
    sequence_path = run_dir / "agent_state" / "artifact_sequence.json"
    with state_lock(sequence_path):
        try:
            counter = int(json.loads(sequence_path.read_text())["counter"])
        except FileNotFoundError:
            counter = 0
            for existing in (run_dir / "artifacts").rglob("*.json"):
                match = re.search(r"_(\d+)\.json$", existing.name)
                if match:
                    counter = max(counter, int(match.group(1)))
        destination = run_dir / "artifacts" / category
        while True:
            counter += 1
            path = destination / f"{tool_name}_{counter:03d}.json"
            if not path.exists():
                break
        # Reserve first: a crash can leave a gap, but cannot reuse an allocation.
        atomic_write_json(sequence_path, {"counter": counter})
        atomic_write_json(path, payload)
        return path
