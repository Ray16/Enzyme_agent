"""Agent-invoked tool: map candidates in a protein-language-model space and
return the nearest neighbors of the most-promising candidates.

The agent decides *when* to call this (after enough promising candidates have
accumulated) and *how much* to retrieve (``k`` neighbors, ``n_promising``
candidates) -- the config values are only fallback defaults it can override.

Design:
  - This module and the store are numpy-only. The heavy embedding forward pass
    runs in a GPU subprocess launched through the cluster reservation gate
    (:mod:`enzyme_discovery_agent.tools.esm_embed_worker`); if no GPU is free the
    tool degrades gracefully (embeds nothing this call, answers from whatever is
    already embedded) and never blocks or errors.
  - Exploit-first steering: neighbors are drawn for the highest-scoring
    candidates, using the run's OWN scores -- no answer identifiers anywhere
    (data-leakage firewall).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from enzyme_discovery_agent import config
from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status,
    family_display,
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent.models import embedding_store as store_mod
from enzyme_discovery_agent.tool_registry import execute_tool, tool

_ACTIVE_STATUSES = {"evaluated", "ranked"}


def _substrate_fit(meta: dict) -> float:
    try:
        return float(meta.get("substrate_fit") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _select_promising(candidates: dict, n: int) -> list[str]:
    """Highest-scoring candidates, using existing scores only (no new scoring).

    Primary: candidates that reached ``evaluated``/``ranked``, not rejected, with
    ``claim_validation.promotion_eligible`` not False -- ordered by ``substrate_fit``
    desc (accession tiebreak). Fallback (nothing evaluated yet): the
    substrate-fit-promoted set (``source_tool`` starting ``family_substrate_fit``).
    """
    primary: list[tuple[str, float]] = []
    for acc, info in candidates.items():
        status = current_candidate_status(info)
        if status not in _ACTIVE_STATUSES:
            continue
        meta = info.get("metadata", {})
        validation = meta.get("claim_validation") or {}
        if validation and validation.get("promotion_eligible") is False:
            continue
        primary.append((acc, _substrate_fit(meta)))
    if primary:
        primary.sort(key=lambda x: (-x[1], x[0]))
        return [acc for acc, _ in primary[:n]]

    promoted = [
        (acc, _substrate_fit(info.get("metadata", {})))
        for acc, info in candidates.items()
        if str(info.get("metadata", {}).get("source_tool") or "").startswith("family_substrate_fit")
    ]
    promoted.sort(key=lambda x: (-x[1], x[0]))
    return [acc for acc, _ in promoted[:n]]


def _fetch_sequences(run_dir: Path, accessions: list[str]) -> dict[str, str]:
    """Return {accession: sequence} for the given accessions, using the store's
    sequence cache first and the existing fetch_uniprot_sequence tool for misses."""
    cache = store_mod.load_sequences(run_dir)
    resolved: dict[str, str] = {}
    fetched: dict[str, str] = {}
    for acc in accessions:
        seq = cache.get(acc)
        if seq:
            resolved[acc] = seq
            continue
        result = execute_tool(
            "fetch_uniprot_sequence", {"accession": acc}, {"run_dir": str(run_dir)}
        )
        seq = ((result or {}).get("data") or {}).get("sequence") if result else None
        if seq:
            resolved[acc] = seq
            fetched[acc] = seq
    if fetched:
        store_mod.cache_sequences(run_dir, fetched)
    return resolved


def _run_embed_subprocess(run_dir: Path, items: list[dict], model: str):
    """Embed ``items`` on a reserved GPU. Returns ``(rows, note)`` where ``rows``
    is a list of ``(accession, seq_sha1, vector)`` or None if embedding was
    skipped (no GPU free, gate missing, subprocess failure, timeout)."""
    gpu_reserve = config.CANDIDATE_MAP_GPU_RESERVE
    if not Path(gpu_reserve).exists():
        return None, "gpu_reserve gate not found; skipped embedding"

    host = socket.gethostname().split(".")[0]
    try:
        listing = subprocess.run(
            [gpu_reserve, "list"], capture_output=True, text=True, timeout=60
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"gpu_reserve list failed ({exc}); skipped embedding"

    free = []
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == host and parts[1].isdigit():
            free.append(parts[1])
    if not free:
        return None, "no free GPU on this host; deferred embedding"
    idx = free[0]

    work = Path(run_dir) / "embedding_store" / "_work"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(work)) as tmp:
        pending_json = str(Path(tmp) / "pending.json")
        out_npy = str(Path(tmp) / "vectors.npy")
        out_meta = str(Path(tmp) / "meta.json")
        Path(pending_json).write_text(json.dumps({"model": model, "items": items}))
        cmd = [
            gpu_reserve, "run", idx, "--label", "esm_embed", "--",
            config.CANDIDATE_MAP_EMBED_PYTHON, "-m",
            "enzyme_discovery_agent.tools.esm_embed_worker",
            pending_json, out_npy, out_meta,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=config.CANDIDATE_MAP_EMBED_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"embed subprocess error ({exc}); skipped"
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-300:]
            return None, f"embed subprocess rc={proc.returncode}; skipped ({tail})"
        try:
            vectors = np.load(out_npy)
            meta = json.loads(Path(out_meta).read_text())
        except Exception as exc:  # noqa: BLE001
            return None, f"embed output unreadable ({exc}); skipped"

    accs = meta.get("accessions", [])
    hashes = meta.get("seq_sha1", [""] * len(accs))
    rows = [
        (accs[i], hashes[i] if i < len(hashes) else "", vectors[i])
        for i in range(min(len(accs), vectors.shape[0]))
    ]
    return rows, f"embedded {len(rows)} sequence(s) on GPU {host}:{idx}"


def _free_gpu_indices(max_n: int) -> list[str]:
    """Free GPU indices on this host, via the cluster reservation gate."""
    host = socket.gethostname().split(".")[0]
    try:
        r = subprocess.run([config.CANDIDATE_MAP_GPU_RESERVE, "list"],
                           capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return []
    idxs = [p.split()[1] for p in r.stdout.splitlines()
            if len(p.split()) >= 2 and p.split()[0] == host and p.split()[1].isdigit()]
    return idxs[:max_n]


def embed_sequences(run_dir: Path, items: list[dict], model: str, max_gpus: int = 8):
    """Embed ``items`` by sharding across up to ``max_gpus`` free GPUs in parallel
    (one ``gpu_reserve run`` worker each), falling back to the single-GPU path when
    only one GPU is free. Returns ``(rows, note)`` exactly like
    :func:`_run_embed_subprocess` (``rows`` = ``(accession, seq_sha1, vector)`` or
    None if embedding was skipped). Shared by the agent expansion tool and the
    standalone figure so both use one implementation.

    Sharding is round-robin (``items[i::n]``) so long/short sequences balance across
    workers, and the merged result is identical to embedding the whole list on one
    GPU (order-independent: each row is keyed by accession) -- fast path == slow path.
    """
    if not items:
        return [], "nothing to embed"
    gpus = _free_gpu_indices(max_gpus)
    if len(gpus) <= 1:
        return _run_embed_subprocess(run_dir, items, model)

    import numpy as np

    n = len(gpus)
    shards = [items[i::n] for i in range(n)]
    workdir = Path(run_dir) / "embedding_store" / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4"})

    procs = []
    for idx, shard in zip(gpus, shards):
        if not shard:
            continue
        d = Path(tempfile.mkdtemp(dir=str(workdir)))
        pj, ov, om = d / "pending.json", d / "vec.npy", d / "meta.json"
        pj.write_text(json.dumps({"model": model, "items": shard}))
        cmd = [config.CANDIDATE_MAP_GPU_RESERVE, "run", idx, "--label", "esm_embed_mg", "--",
               config.CANDIDATE_MAP_EMBED_PYTHON, "-m",
               "enzyme_discovery_agent.tools.esm_embed_worker", str(pj), str(ov), str(om)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=env)
        procs.append((p, ov, om, idx))

    rows: list[tuple] = []
    used: list[str] = []
    for p, ov, om, idx in procs:
        try:
            out = p.communicate(timeout=config.CANDIDATE_MAP_EMBED_TIMEOUT)[0]
        except Exception as exc:  # noqa: BLE001
            p.kill()
            sys.stderr.write(f"  shard GPU {idx} error ({exc})\n")
            continue
        if p.returncode != 0 or not ov.exists():
            sys.stderr.write(f"  shard GPU {idx} failed rc={p.returncode}: {(out or '')[-200:]}\n")
            continue
        v = np.load(ov)
        meta = json.loads(om.read_text())
        accs = meta.get("accessions", [])
        hs = meta.get("seq_sha1", [""] * len(accs))
        for i in range(min(len(accs), v.shape[0])):
            rows.append((accs[i], hs[i] if i < len(hs) else "", v[i]))
        used.append(idx)
    if not rows:
        return None, "multi-GPU embedding produced no rows"
    return rows, f"embedded {len(rows)} via {len(used)} GPU(s): {','.join(used)}"


@tool(
    description=(
        "Embed the discovered candidate protein sequences in a protein "
        "language-model representation space and return, for the highest-scoring "
        "candidates (or a candidate you name), their nearest neighbors among the "
        "discovered pool. Neighbors are registered as new leads to expand or to "
        "advance to structural modeling. Call this once several promising "
        "candidates have accumulated; you choose how many candidates and how many "
        "neighbors to retrieve."
    ),
    category="database",
    parallel_safe=False,
    injected_params=["run_dir"],
    params={
        "accession": {
            "type": "string",
            "description": (
                "Optional. Map neighbors of this one candidate accession; omit to "
                "use the current highest-scoring candidates."
            ),
            "default": "",
        },
        "k": {
            "type": "integer",
            "description": "Number of nearest neighbors to return per candidate.",
            "default": config.CANDIDATE_MAP_TOP_K,
        },
        "n_promising": {
            "type": "integer",
            "description": "How many top candidates to draw neighbors from.",
            "default": config.CANDIDATE_MAP_PROMISING_N,
        },
        "run_dir": {"type": "string", "description": "Injected run directory."},
    },
)
def map_candidate_neighbors(
    accession: str = "",
    k: int | None = None,
    n_promising: int | None = None,
    run_dir: str = "",
) -> dict:
    if not config.ENABLE_CANDIDATE_MAP:
        return {
            "status": "skipped",
            "summary": "Embedding map disabled (set ENZYME_AGENT_ENABLE_CANDIDATE_MAP=1).",
            "data": None,
        }

    k = int(k) if k else config.CANDIDATE_MAP_TOP_K
    n_promising = int(n_promising) if n_promising else config.CANDIDATE_MAP_PROMISING_N
    run_path = Path(run_dir)
    model = config.CANDIDATE_MAP_ESM_MODEL

    registry = load_candidate_registry(run_path)
    candidates = registry.get("candidates", {})
    if not candidates:
        return {"status": "partial", "summary": "No candidates to map yet.", "data": None, "count": 0}

    # 1) Embed candidates that are not embedded yet (GPU subprocess).
    pending = store_mod.pending_accessions(run_path, list(candidates.keys()))
    note = "no new candidates to embed"
    if pending:
        seqs = _fetch_sequences(run_path, pending)
        items = [
            {"accession": acc, "seq_sha1": store_mod.seq_sha1(seq), "sequence": seq}
            for acc, seq in seqs.items()
        ]
        if items:
            rows, note = _run_embed_subprocess(run_path, items, model)
            if rows:
                store_mod.append_vectors(run_path, model, rows)

    # 2) Nearest neighbors of the promising set (exploit-first).
    store = store_mod.load_store(run_path)
    if store.matrix is None or len(store) == 0:
        return {
            "status": "partial",
            "summary": f"Embedding map has no vectors yet ({note}).",
            "data": None,
            "count": 0,
        }

    if accession:
        promising = [accession] if accession in store.acc_to_row else []
    else:
        promising = [a for a in _select_promising(candidates, n_promising) if a in store.acc_to_row]
    if not promising:
        return {
            "status": "partial",
            "summary": f"No promising candidates are embedded yet ({note}).",
            "data": None,
            "count": 0,
        }

    query_rows = [store.acc_to_row[a] for a in promising]
    neighbor_map = store_mod.knn(store, query_rows, k=k, exclude=set(query_rows))

    # 3) Register neighbors as discovered leads + build the output map.
    promising_out = []
    registered = 0
    for prom in promising:
        prom_meta = candidates.get(prom, {}).get("metadata", {})
        nbrs = neighbor_map.get(prom, [])
        nbr_out = []
        for nbr_acc, dist in nbrs:
            nbr_info = candidates.get(nbr_acc, {})
            nbr_meta = nbr_info.get("metadata", {})
            nbr_out.append({
                "accession": nbr_acc,
                "distance": round(float(dist), 4),
                "family": family_display(nbr_meta, nbr_acc),
                "status": current_candidate_status(nbr_info),
            })
            try:
                update_candidate_registry(
                    run_path, nbr_acc, "discovered",
                    metadata={"source_tool": f"esm_neighbor:{prom}", "esm_distance": round(float(dist), 4)},
                )
                registered += 1
            except Exception:  # noqa: BLE001 - registration must not break the tool
                pass
        promising_out.append({
            "accession": prom,
            "substrate_fit": _substrate_fit(prom_meta),
            "family": family_display(prom_meta, prom),
            "neighbors": nbr_out,
        })

    map_payload = {
        "schema_version": "1.0",
        "model": model,
        "steer_mode": config.CANDIDATE_MAP_STEER_MODE,
        "n_embedded": len(store),
        "promising": promising_out,
    }
    try:
        (run_path / "candidate_map.json").write_text(json.dumps(map_payload, indent=2))
    except Exception:  # noqa: BLE001
        pass

    n_nbrs = sum(len(p["neighbors"]) for p in promising_out)
    return {
        "status": "success",
        "summary": (
            f"Mapped {len(promising)} promising candidate(s) in ESM space "
            f"({len(store)} embedded); surfaced {n_nbrs} neighbor(s) as new leads. {note}."
        ),
        "data": map_payload,
        "count": n_nbrs,
    }
