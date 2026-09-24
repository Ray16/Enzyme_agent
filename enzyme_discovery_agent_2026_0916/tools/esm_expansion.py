"""Agent-invoked tool: expand the candidate pool OUTSIDE what has been discovered.

Where :mod:`enzyme_discovery_agent.tools.embedding_map` finds neighbors *within*
the already-discovered pool, this tool reaches *out*: it takes the highest-scoring
candidates, gathers the wider membership of the families they belong to (sequences
not yet in the pool), embeds that background in the same protein language-model
space, and keeps -- as new leads -- the background sequences NEAREST the top
candidates. The embedding is the decision layer that picks which of the broadly
retrieved sequences are worth pursuing.

Design mirrors the structural (Foldseek) expansion gate but in sequence/embedding
space, and shares the numpy-only store + GPU-subprocess embedding of the map tool:

  - The family membership comes from the families the agent's OWN top candidates
    resolved to -- clean search provenance, no answer identifiers anywhere
    (data-leakage firewall).
  - The heavy forward pass runs in a reserved-GPU subprocess; if no GPU is free the
    tool degrades gracefully (embeds nothing, returns a partial status) and never
    blocks or errors.
  - Selected leads are registered as ``discovered`` candidates and written into
    ``candidate_map.json`` in the same schema the structural-modeling handoff
    already consumes, so they flow into the cofold panel with no other changes.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
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
from enzyme_discovery_agent.tools.embedding_map import (
    _fetch_sequences,
    _select_promising,
    _substrate_fit,
    embed_sequences,
)
from enzyme_discovery_agent.tool_registry import execute_tool, tool

_IPR_RE = re.compile(r"IPR\d{6}")
_UNIPROT_ACC_RE = re.compile(r"^[A-NR-Z0-9][A-Z0-9]{5,9}$")


# --------------------------------------------------------------------------- #
# Family resolution (which families to pull wider membership from)
# --------------------------------------------------------------------------- #
def _seed_families(meta: dict) -> list[str]:
    """InterPro family accessions a candidate belongs to, best-first.

    Reads the metadata a screened/evaluated candidate already carries; the family
    that promoted it is the most relevant, so it is tried first."""
    families: list[str] = []

    def _add(value):
        for ipr in _IPR_RE.findall(str(value or "")):
            if ipr not in families:
                families.append(ipr)

    _add(meta.get("interpro_family_id"))
    _add(meta.get("screening_family_id"))
    _add(meta.get("source_tool"))  # e.g. "family_substrate_fit:IPR053158"
    for ipr in meta.get("interpro_ids") or []:
        _add(ipr)
    return families


def _resolve_families(run_dir: Path, accession: str, meta: dict) -> list[str]:
    """Families from metadata; fall back to a live domain lookup if none recorded."""
    families = _seed_families(meta)
    if families:
        return families
    try:
        result = execute_tool(
            "get_protein_domains", {"uniprot_accession": accession},
            {"run_dir": str(run_dir)},
        )
    except Exception:  # noqa: BLE001
        return []
    return list(dict.fromkeys(_IPR_RE.findall(json.dumps((result or {}).get("data") or {}))))


# --------------------------------------------------------------------------- #
# Outside-pool background: wider family membership from UniProt
# --------------------------------------------------------------------------- #
def _parse_fasta(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    acc, seq = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc and seq:
                out[acc] = "".join(seq)
            parts = line[1:].split("|")  # sp|ACC|NAME or tr|ACC|NAME
            acc = parts[1] if len(parts) >= 2 else line[1:].split()[0]
            seq = []
        else:
            seq.append(line.strip())
    if acc and seq:
        out[acc] = "".join(seq)
    return out


def _uniprot_family_fasta(query: str, size: int) -> dict[str, str]:
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        {"query": query, "format": "fasta", "size": max(1, min(size, 500))}
    )
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return _parse_fasta(resp.read().decode())
    except Exception:  # noqa: BLE001 - network hiccup should not break the tool
        return {}


def _fetch_family_background(ipr: str, exclude: set[str], cap: int) -> dict[str, str]:
    """Members of an InterPro family NOT already in the pool, as {acc: sequence}.

    Reviewed (Swiss-Prot) first for quality; if that is thin (narrow family), widen
    to include unreviewed entries so a promising but sparse family still expands."""
    reviewed = _uniprot_family_fasta(f"(xref:interpro-{ipr}) AND (reviewed:true)", cap)
    out = {a: s for a, s in reviewed.items()
           if a not in exclude and _UNIPROT_ACC_RE.match(a) and s}
    if len(out) < config.CANDIDATE_MAP_EXPANSION_MIN_REVIEWED:
        for a, s in _uniprot_family_fasta(f"xref:interpro-{ipr}", cap).items():
            if len(out) >= cap:
                break
            if a not in exclude and a not in out and _UNIPROT_ACC_RE.match(a) and s:
                out[a] = s
    return dict(list(out.items())[:cap])


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #
@tool(
    description=(
        "Expand the candidate pool BEYOND what has been discovered so far. For the "
        "highest-scoring candidates (or a candidate you name), this fetches the "
        "wider membership of the families they belong to, embeds those sequences in "
        "a protein language-model space, and registers as new leads the ones that "
        "sit NEAREST the top candidates in that space. Use this to grow the search "
        "into unexplored but closely related sequence space once several promising "
        "candidates have accumulated; you choose how many candidates to expand "
        "around (n_seeds) and how many new leads to keep per candidate (k)."
    ),
    category="database",
    parallel_safe=False,
    injected_params=["run_dir"],
    params={
        "accession": {
            "type": "string",
            "description": (
                "Optional. Expand around this one candidate accession; omit to use "
                "the current highest-scoring candidates."
            ),
            "default": "",
        },
        "k": {
            "type": "integer",
            "description": "New leads to keep per candidate (nearest in embedding space).",
            "default": config.CANDIDATE_MAP_EXPANSION_K,
        },
        "n_seeds": {
            "type": "integer",
            "description": "How many top candidates to expand around.",
            "default": config.CANDIDATE_MAP_EXPANSION_N_SEEDS,
        },
        "per_family": {
            "type": "integer",
            "description": "Max family members to fetch and screen per family.",
            "default": config.CANDIDATE_MAP_EXPANSION_BACKGROUND_PER_FAMILY,
        },
        "run_dir": {"type": "string", "description": "Injected run directory."},
    },
)
def expand_candidates_esm(
    accession: str = "",
    k: int | None = None,
    n_seeds: int | None = None,
    per_family: int | None = None,
    run_dir: str = "",
) -> dict:
    if not config.ENABLE_CANDIDATE_MAP:
        return {
            "status": "skipped",
            "summary": "Embedding expansion disabled (set ENZYME_AGENT_ENABLE_CANDIDATE_MAP=1).",
            "data": None,
        }

    k = int(k) if k else config.CANDIDATE_MAP_EXPANSION_K
    n_seeds = int(n_seeds) if n_seeds else config.CANDIDATE_MAP_EXPANSION_N_SEEDS
    per_family = int(per_family) if per_family else config.CANDIDATE_MAP_EXPANSION_BACKGROUND_PER_FAMILY
    run_path = Path(run_dir)
    model = config.CANDIDATE_MAP_ESM_MODEL

    registry = load_candidate_registry(run_path)
    candidates = registry.get("candidates", {})
    if not candidates:
        return {"status": "partial", "summary": "No candidates to expand from yet.", "data": None, "count": 0}

    seeds = [accession] if accession else _select_promising(candidates, n_seeds)
    seeds = [s for s in seeds if s in candidates]
    if not seeds:
        return {"status": "partial", "summary": "No promising candidates to expand around yet.", "data": None, "count": 0}

    pool = set(candidates.keys())

    # 1) Gather outside-pool background: wider membership of the seeds' families.
    background: dict[str, str] = {}          # acc -> sequence
    seed_families: dict[str, list[str]] = {}
    bg_family: dict[str, str] = {}           # background acc -> family it came from
    for seed in seeds:
        fams = _resolve_families(run_path, seed, candidates[seed].get("metadata", {}))
        seed_families[seed] = fams
        for ipr in fams:
            if len(background) >= config.CANDIDATE_MAP_EXPANSION_MAX_BACKGROUND:
                break
            members = _fetch_family_background(ipr, pool | set(background), per_family)
            for a, s in members.items():
                if len(background) >= config.CANDIDATE_MAP_EXPANSION_MAX_BACKGROUND:
                    break
                background[a] = s
                bg_family.setdefault(a, ipr)

    if not background:
        return {
            "status": "partial",
            "summary": (
                f"Found no outside-pool family members to embed for {len(seeds)} seed(s) "
                f"(families: {sorted({f for fs in seed_families.values() for f in fs}) or 'none resolved'})."
            ),
            "data": None,
            "count": 0,
        }

    # 2) Ensure seeds are embedded, then embed the background (reserved-GPU subprocess).
    note_parts: list[str] = []
    seed_pending = store_mod.pending_accessions(run_path, seeds)
    if seed_pending:
        seed_seqs = _fetch_sequences(run_path, seed_pending)
        items = [{"accession": a, "seq_sha1": store_mod.seq_sha1(s), "sequence": s}
                 for a, s in seed_seqs.items()]
        rows, note = embed_sequences(run_path, items, model)
        note_parts.append(note)
        if rows:
            store_mod.append_vectors(run_path, model, rows)

    store = store_mod.load_store(run_path)
    embedded_seeds = [s for s in seeds if s in store.acc_to_row]
    if not embedded_seeds:
        return {
            "status": "partial",
            "summary": f"Could not embed any seed candidate ({'; '.join(note_parts) or 'no GPU free'}).",
            "data": None,
            "count": 0,
        }

    store_mod.cache_sequences(run_path, background)
    bg_items = [{"accession": a, "seq_sha1": store_mod.seq_sha1(s), "sequence": s}
                for a, s in background.items()]
    bg_rows, bg_note = embed_sequences(run_path, bg_items, model)
    note_parts.append(bg_note)
    if not bg_rows:
        return {
            "status": "partial",
            "summary": f"Embedding of {len(bg_items)} family member(s) was skipped ({bg_note}).",
            "data": None,
            "count": 0,
        }

    # 3) ESM rerank: nearest background members to each seed (cosine on L2-normed).
    bg_acc = [r[0] for r in bg_rows]
    bg_vecs = []
    for _, _, vec in bg_rows:
        v = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        bg_vecs.append(v / norm if norm > 0 else v)
    bg_mat = np.vstack(bg_vecs).astype(np.float32)  # (B, D), rows L2-normalized
    bg_row_by_acc = {r[0]: r for r in bg_rows}

    promising_out = []
    registered = 0
    selected_rows: dict[str, tuple] = {}     # acc -> (acc, sha1, vector) to add to store
    for seed in embedded_seeds:
        qvec = store.matrix[store.acc_to_row[seed]]
        sims = bg_mat @ qvec
        order = np.argsort(-sims)
        seed_meta = candidates[seed].get("metadata", {})
        nbr_out = []
        for r in order:
            r = int(r)
            acc = bg_acc[r]
            if acc in pool:            # never re-register an existing pool member
                continue
            dist = round(float(1.0 - sims[r]), 4)
            ipr = bg_family.get(acc, "")
            nbr_out.append({
                "accession": acc, "distance": dist,
                "family": f"IPR:{ipr}" if ipr else "",
                "status": "discovered",
            })
            try:
                update_candidate_registry(
                    run_path, acc, "discovered",
                    metadata={
                        "source_tool": f"esm_expansion:{seed}",
                        "esm_distance": dist,
                        "expansion_background": f"family:{ipr}" if ipr else "family",
                        "interpro_family_id": ipr or None,
                    },
                )
                registered += 1
            except Exception:  # noqa: BLE001 - registration must not break the tool
                pass
            # Persist the new lead's vector so it appears on the map / is queryable.
            for br in bg_rows:
                if br[0] == acc:
                    selected_rows[acc] = br
                    break
            if len(nbr_out) >= k:
                break
        promising_out.append({
            "accession": seed,
            "substrate_fit": _substrate_fit(seed_meta),
            "family": family_display(seed_meta, seed),
            "neighbors": nbr_out,
        })

    if selected_rows:
        store_mod.append_vectors(run_path, model, list(selected_rows.values()))

    map_payload = {
        "schema_version": "1.0",
        "model": model,
        "steer_mode": "expand",
        "n_embedded": len(store) + len(selected_rows),
        "n_background_screened": len(bg_acc),
        "promising": promising_out,
    }
    try:
        (run_path / "candidate_map.json").write_text(json.dumps(map_payload, indent=2))
    except Exception:  # noqa: BLE001
        pass

    n_new = sum(len(p["neighbors"]) for p in promising_out)
    return {
        "status": "success",
        "summary": (
            f"Expanded {len(embedded_seeds)} candidate(s) into their families: screened "
            f"{len(bg_acc)} outside-pool member(s) in embedding space, registered "
            f"{n_new} new lead(s). {'; '.join(p for p in note_parts if p)}."
        ),
        "data": map_payload,
        "count": n_new,
    }
