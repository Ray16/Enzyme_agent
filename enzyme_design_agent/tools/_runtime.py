"""Shared stdlib-only runtime, GPU/CPU reservation, and process helpers for design tools.

Centralizes the hardware-courtesy and process-lifecycle primitives shared by
AlphaProtein Novo, DISCO, GENzyme, and RFD4-Proteina so resource-safety rules
(5-var thread caps, CUDA LD_LIBRARY_PATH stripping, verified CPU-slot checks,
gpu_reserve listing/claiming, /proc start-ticks liveness, and atomic JSON)
cannot drift across backends.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import threading

THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def short_hostname() -> str:
    return socket.gethostname().split(".")[0]


def strip_cuda_library_path(ld_library_path: str) -> str | None:
    """Remove system CUDA directories that can shadow a backend venv's CUDA wheels."""
    if not ld_library_path:
        return None
    kept = [p for p in ld_library_path.split(":") if p and "cuda" not in p.lower()]
    return ":".join(kept) if kept else None


def base_subprocess_env(
    threads: int,
    *,
    cuda_visible_devices: str | None = None,
    extra: dict[str, str] | None = None,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a sanitized environment with explicit native thread caps."""
    env = dict(os.environ if base is None else base)
    t = str(int(threads))
    for key in THREAD_ENV_KEYS:
        env[key] = t
    env["TOKENIZERS_PARALLELISM"] = "false"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cleaned_ld = strip_cuda_library_path(env.get("LD_LIBRARY_PATH", ""))
    if cleaned_ld:
        env["LD_LIBRARY_PATH"] = cleaned_ld
    else:
        env.pop("LD_LIBRARY_PATH", None)
    if extra:
        env.update(extra)
    return env


def atomic_write_json(path: Path, data) -> None:
    """Atomically write JSON to ``path`` via a process+thread-unique temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)


def process_start_ticks(pid: int) -> str:
    """Read /proc/<pid>/stat field 22 (start_ticks) to guard against PID reuse."""
    return Path(f"/proc/{pid}/stat").read_text().split()[21]


def is_process_alive(pid: int, expected_start_ticks: str) -> bool:
    """True iff ``pid`` is still running with the same start_ticks and not a zombie."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[21] == expected_start_ticks and fields[2] != "Z"
    except OSError:
        return False


def verified_cpu_slots(n: int) -> list[int]:
    """Return ``n`` unpinned CPU core ids after verifying host load capacity.

    Fails closed when online CPU count cannot be verified, load + n exceeds
    online CPUs, or fewer than ``n`` unpinned cores are available.
    """
    allowed = set(os.sched_getaffinity(0))
    occupied: set[int] = set()
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if not (p / "cmdline").read_bytes():
                continue
            affinity = set(os.sched_getaffinity(int(p.name)))
            if len(affinity) < len(allowed):
                occupied.update(affinity)
        except OSError:
            continue
    available = sorted(allowed - occupied)
    cpus_online = os.cpu_count()
    if cpus_online is None or max(os.getloadavg()) + n > cpus_online or len(available) < n:
        raise RuntimeError("Insufficient verified CPU capacity")
    return available[-n:]


def free_gpu_indices(
    gpu_reserve: str,
    max_n: int | None = None,
    *,
    host: str | None = None,
    min_memory_gib: float | None = None,
    host_filter_flag: bool = False,
    check: bool = False,
    timeout: int = 120,
    runner=subprocess.run,
) -> list[str]:
    """Return free GPU indices on ``host`` via ``gpu_reserve list``."""
    target_host = (host or short_hostname()).split(".")[0]
    cmd = [gpu_reserve, "list"]
    if host_filter_flag:
        cmd.extend(["--hosts", target_host])
    try:
        r = runner(cmd, capture_output=True, text=True, timeout=timeout, **({"check": True} if check else {}))
    except Exception:
        if check:
            raise
        return []
    out: list[str] = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == target_host and parts[1].isdigit():
            out.append(parts[1])
    if min_memory_gib is not None:
        mem_proc = runner(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=15, **({"check": True} if check else {}),
        )
        capacity = {
            parts[0].strip(): float(parts[1])
            for line in mem_proc.stdout.splitlines()
            if len(parts := line.split(",")) == 2
        }
        out = [gpu for gpu in out if capacity.get(gpu, 0.0) >= min_memory_gib * 1024.0]
    return out[:max_n] if max_n is not None else out


def acquire_gpus(
    gpu_reserve: str,
    n: int,
    label: str,
    *,
    candidates: list[str],
    runner=subprocess.run,
) -> list[str]:
    """Reserve up to ``n`` GPUs from ``candidates``, recording this PID as owner."""
    reserved: list[str] = []
    for idx in candidates:
        if len(reserved) >= n:
            break
        try:
            r = runner(
                [gpu_reserve, "acquire", idx, "--label", label, "--pid", str(os.getpid())],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:  # noqa: BLE001
            continue
        if r.returncode == 0:
            reserved.append(idx)
    return reserved


def release_gpus(gpu_reserve: str, idxs: list[str], *, runner=subprocess.run) -> None:
    """Release reservations owned by this process."""
    for idx in idxs:
        try:
            runner([gpu_reserve, "release", idx], capture_output=True, text=True, timeout=120)
        except Exception:  # noqa: BLE001
            pass


def local_gpu_capability(*, runner=subprocess.run) -> float | None:
    """Highest CUDA compute capability on this host, or None if undetectable."""
    try:
        r = runner(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:  # noqa: BLE001
        return None
    caps: list[float] = []
    for line in r.stdout.splitlines():
        try:
            caps.append(float(line.strip()))
        except ValueError:
            continue
    return max(caps) if caps else None
