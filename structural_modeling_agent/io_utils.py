"""Shared I/O, hashing, and CPU-safe concurrency helpers for structural_modeling_agent."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path

SINGLE_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

# Stat-keyed digest cache for larger files (>= 4 KiB) to avoid repeatedly
# re-hashing unchanged multi-MB PDBs/MSAs on NFS within a turn. Small files
# hash in microseconds and are always read fresh so fast in-place edits in
# tests or scripts cannot collide on coarse filesystem timestamps.
_DIGEST_CACHE_MIN_BYTES = 4096
_DIGEST_CACHE_MAX_ENTRIES = 2048
_DIGEST_CACHE: OrderedDict[tuple[str, int, int, int, int, int], str] = OrderedDict()


def file_digest(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the streaming SHA-256 hex digest of a file's raw bytes."""
    p = Path(path)
    st = p.stat()
    cache_key = None
    if st.st_size >= _DIGEST_CACHE_MIN_BYTES:
        cache_key = (str(p.resolve()), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        cached = _DIGEST_CACHE.get(cache_key)
        if cached is not None:
            _DIGEST_CACHE.move_to_end(cache_key)
            return cached
    digest = hashlib.sha256()
    with p.open("rb") as source:
        for block in iter(lambda: source.read(chunk_size), b""):
            digest.update(block)
    result = digest.hexdigest()
    if cache_key is not None:
        _DIGEST_CACHE[cache_key] = result
        if len(_DIGEST_CACHE) > _DIGEST_CACHE_MAX_ENTRIES:
            _DIGEST_CACHE.popitem(last=False)
    return result


def available_cpus() -> int:
    """Return the number of CPUs available to this process (respecting affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def bounded_workers(
    requested: int | None = None,
    *,
    n_items: int = 1,
    min_items_for_pool: int = 4,
    max_cap: int = 8,
) -> int:
    """Return a safe worker count bounded by item count, CPU affinity, and max_cap."""
    if n_items < min_items_for_pool:
        return 1
    limit = max(1, min(available_cpus(), max_cap, n_items))
    if requested is None:
        return limit
    if requested < 1:
        return 1
    return min(int(requested), limit)


def init_single_thread_worker() -> None:
    """Pool initializer enforcing single-threaded native math libraries per worker."""
    os.environ.update(SINGLE_THREAD_ENV)
