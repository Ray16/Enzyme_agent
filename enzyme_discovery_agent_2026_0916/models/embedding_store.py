"""Persistent, incremental protein-embedding store for the candidate pool.

Numpy-only (no torch): the heavy protein-language-model forward pass lives in a
separate GPU subprocess (:mod:`enzyme_discovery_agent.tools.esm_embed_worker`);
this module only persists the resulting vectors and answers nearest-neighbor
queries over them.

Layout under ``run_dir/embedding_store/``:
  - ``vectors.npy``    -- float32 (N, D) matrix, rows L2-normalized (so cosine =
                          dot product, mirroring tools.literature._embed_texts).
  - ``index.json``     -- {model, dim, created_at, entries:[{accession, seq_sha1,
                          added_at, row}]} -- row-aligned to vectors.npy.
  - ``sequences.json`` -- {accession: sequence} cache (sequences are NOT kept in
                          the candidate registry, so they are cached here to avoid
                          re-fetching across firings).

All reads tolerate a missing/partial store (returns an empty store); all writes
are atomic (temp file + os.replace).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def seq_sha1(sequence: str) -> str:
    """Stable content hash for a sequence (used as a cache key)."""
    return hashlib.sha1((sequence or "").encode("utf-8")).hexdigest()


def _store_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "embedding_store"


@dataclass
class EmbeddingStore:
    """In-memory view of the persisted store."""

    model: str = ""
    dim: int = 0
    accessions: list[str] = field(default_factory=list)
    seq_sha1s: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None  # (N, D) float32, L2-normalized rows
    acc_to_row: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.accessions)


def load_store(run_dir: Path) -> EmbeddingStore:
    """Load the store, or return an empty one if absent/corrupt."""
    sdir = _store_dir(run_dir)
    index_path = sdir / "index.json"
    vectors_path = sdir / "vectors.npy"
    if not index_path.exists() or not vectors_path.exists():
        return EmbeddingStore()
    try:
        index = json.loads(index_path.read_text())
        matrix = np.load(vectors_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return EmbeddingStore()
    entries = index.get("entries", [])
    accessions = [e.get("accession", "") for e in entries]
    seq_hashes = [e.get("seq_sha1", "") for e in entries]
    # Guard against an index/matrix row-count mismatch (partial write): trust the
    # smaller of the two so we never index out of bounds.
    n = min(len(accessions), matrix.shape[0]) if matrix.ndim == 2 else 0
    accessions, seq_hashes = accessions[:n], seq_hashes[:n]
    matrix = matrix[:n] if n else None
    return EmbeddingStore(
        model=index.get("model", ""),
        dim=int(index.get("dim", matrix.shape[1] if matrix is not None else 0)),
        accessions=accessions,
        seq_sha1s=seq_hashes,
        matrix=matrix,
        acc_to_row={acc: i for i, acc in enumerate(accessions)},
    )


def pending_accessions(run_dir: Path, candidate_accessions) -> list[str]:
    """Registered candidates that are not yet embedded (order-preserving, deduped)."""
    store = load_store(run_dir)
    embedded = set(store.accessions)
    seen: set[str] = set()
    pending: list[str] = []
    for acc in candidate_accessions:
        if acc and acc not in embedded and acc not in seen:
            seen.add(acc)
            pending.append(acc)
    return pending


def load_sequences(run_dir: Path) -> dict[str, str]:
    path = _store_dir(run_dir) / "sequences.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def cache_sequences(run_dir: Path, mapping: dict[str, str]) -> None:
    """Merge ``mapping`` into the persisted sequence cache (atomic)."""
    if not mapping:
        return
    sdir = _store_dir(run_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    existing = load_sequences(run_dir)
    existing.update({k: v for k, v in mapping.items() if v})
    _atomic_write_text(sdir / "sequences.json", json.dumps(existing, indent=2))


def append_vectors(run_dir: Path, model: str, rows: list[tuple[str, str, np.ndarray]]) -> int:
    """Append (accession, seq_sha1, vector) rows to the store; returns rows added.

    Vectors are L2-normalized here (idempotent if already normalized). Accessions
    already present are skipped so repeated firings are safe. Atomic on disk.
    """
    if not rows:
        return 0
    sdir = _store_dir(run_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    store = load_store(run_dir)
    have = set(store.accessions)

    new_accs, new_hashes, new_vecs = [], [], []
    for acc, sha, vec in rows:
        if not acc or acc in have:
            continue
        have.add(acc)
        v = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        new_accs.append(acc)
        new_hashes.append(sha)
        new_vecs.append(v)
    if not new_vecs:
        return 0

    new_matrix = np.vstack(new_vecs).astype(np.float32)
    if store.matrix is not None and store.matrix.shape[0]:
        if store.matrix.shape[1] != new_matrix.shape[1]:
            raise ValueError(
                f"embedding dim mismatch: store={store.matrix.shape[1]} new={new_matrix.shape[1]}"
            )
        combined = np.vstack([store.matrix, new_matrix])
    else:
        combined = new_matrix

    accessions = store.accessions + new_accs
    seq_hashes = store.seq_sha1s + new_hashes
    index = {
        "model": model or store.model,
        "dim": int(combined.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {"accession": a, "seq_sha1": h, "row": i}
            for i, (a, h) in enumerate(zip(accessions, seq_hashes))
        ],
    }
    _atomic_write_npy(sdir / "vectors.npy", combined)
    _atomic_write_text(sdir / "index.json", json.dumps(index, indent=2))
    return len(new_vecs)


def knn(
    store: EmbeddingStore,
    query_rows: list[int],
    k: int,
    exclude: set[int] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Cosine nearest neighbors for each query row.

    Rows are L2-normalized, so similarity is a plain dot product. Returns
    ``{query_accession: [(neighbor_accession, distance), ...]}`` where
    distance = 1 - cosine, ascending (closest first). ``exclude`` rows (plus the
    query itself) are never returned as neighbors.
    """
    if store.matrix is None or not len(query_rows):
        return {}
    exclude = set(exclude or set())
    M = store.matrix  # (N, D), normalized
    out: dict[str, list[tuple[str, float]]] = {}
    for qr in query_rows:
        if qr < 0 or qr >= M.shape[0]:
            continue
        sims = M @ M[qr]
        order = np.argsort(-sims)
        neighbors: list[tuple[str, float]] = []
        for r in order:
            r = int(r)
            if r == qr or r in exclude:
                continue
            neighbors.append((store.accessions[r], float(1.0 - sims[r])))
            if len(neighbors) >= k:
                break
        out[store.accessions[qr]] = neighbors
    return out


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    # Temp name ends in ".npy" so np.save writes exactly it (no extra suffix).
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)
