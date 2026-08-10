"""Candidate registry: structured JSON tracking of discovered enzyme candidates.

Provides a thread-safe registry that tracks candidate status transitions
(discovered -> screened -> evaluated -> ranked / rejected) and metadata.
Also provides helper to extract and register candidates from search results.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("enzyme_agent")

_registry_lock = threading.Lock()

UNIPROT_ACC_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)


def _enzyme_name(metadata: dict) -> str:
    """Return the most useful human-readable label for a registry candidate.

    UniProt accessions are stable identifiers, but they are not useful as the
    only visible label in a run artifact.  Prefer the name the discovering paper
    gave the enzyme (an enzyme found only via a deposited accession often carries
    a generic, uninformative UniProt annotation while the paper calls it by a
    recognisable name); then the curated gene symbol; then the protein name.
    """
    lit_name = str(metadata.get("literature_name") or "").strip()
    if lit_name:
        return lit_name
    genes = metadata.get("gene_names") or []
    if isinstance(genes, str):
        genes = [genes]
    if genes and str(genes[0]).strip():
        return str(genes[0]).strip()
    return str(metadata.get("protein_name") or "").strip()


def _normalize_registry(registry: dict) -> dict:
    """Backfill display fields for registries written by older agent versions."""
    candidates = registry.setdefault("candidates", {})
    for key, entry in candidates.items():
        metadata = entry.setdefault("metadata", {})
        # Keep these top-level so a human can scan candidate_registry.json
        # without knowing the internal UniProt metadata schema.  ``accession`` is
        # the functional identity (the file may be keyed by enzyme name).
        entry.setdefault("accession", key)
        entry["enzyme_name"] = _enzyme_name(metadata)
    return registry


def _rekey_by_accession(registry: dict) -> dict:
    """Key candidates by their internal accession — the functional identity every
    accessor uses — tolerating both the enzyme-name-keyed file layout and the
    legacy accession-keyed one."""
    out: dict = {}
    for key, entry in registry.get("candidates", {}).items():
        acc = entry.get("accession") or key
        entry["accession"] = acc
        out[acc] = entry
    registry["candidates"] = out
    return registry


def _to_file_view(registry: dict) -> dict:
    """Enzyme-name-keyed view for on-disk readability: key each candidate by its
    enzyme name (accession when no name is known); a name shared by two
    accessions is disambiguated with the accession, so no entry is lost. The
    accession stays inside each entry, so :func:`_rekey_by_accession` restores
    the functional key on load."""
    out: dict = {}
    for acc, entry in registry.get("candidates", {}).items():
        entry.setdefault("accession", acc)
        name = (entry.get("enzyme_name") or "").strip()
        key = name or acc
        if key in out and out[key].get("accession") != acc:
            key = f"{name} ({acc})" if name else acc
        base, i = key, 2
        while key in out and out[key].get("accession") != acc:
            key, i = f"{base} ({i})", i + 1
        out[key] = entry
    return {**registry, "candidates": out}


def load_candidate_registry(run_dir: Path) -> dict:
    """Load or initialize the candidate registry from run_dir/candidate_registry.json.

    The file is keyed by enzyme name for readability; this returns it re-keyed by
    accession (the functional identity every caller expects)."""
    path = run_dir / "candidate_registry.json"
    if path.exists():
        try:
            with open(path) as f:
                return _rekey_by_accession(_normalize_registry(json.load(f)))
        except (json.JSONDecodeError, OSError):
            pass
    return {"candidates": {}}


def normalize_candidate_registry_file(run_dir: Path) -> dict:
    """Persist display-field backfills for an existing run registry.

    This migration is intentionally status-preserving: it neither adds a
    candidate nor appends a lifecycle transition.
    """
    run_dir = Path(run_dir)
    with _registry_lock:
        registry = load_candidate_registry(run_dir)
        path = run_dir / "candidate_registry.json"
        if path.exists():
            with open(path, "w") as f:
                json.dump(_to_file_view(registry), f, indent=2, default=str)
        return registry


def update_candidate_registry(
    run_dir: Path,
    accession: str,
    status: str | None,
    metadata: dict | None = None,
) -> None:
    """Update a candidate's status and/or metadata in the registry.

    Statuses are appended, never overwritten.
    Status transitions: discovered -> screened -> evaluated -> ranked / rejected
    Pass ``status=None`` for a metadata-only update (e.g. attaching the paper's
    enzyme name to an already-registered candidate) without appending a status.
    """
    now = datetime.now().isoformat()

    with _registry_lock:
        run_dir.mkdir(parents=True, exist_ok=True)
        registry = load_candidate_registry(run_dir)

        if accession not in registry["candidates"]:
            registry["candidates"][accession] = {
                "statuses": [],
                "first_seen": now,
                "last_updated": now,
                "metadata": {},
            }

        entry = registry["candidates"][accession]
        if status:
            entry["statuses"].append({"status": status, "timestamp": now})
        entry["last_updated"] = now
        if metadata:
            entry["metadata"].update(metadata)
        entry["enzyme_name"] = _enzyme_name(entry["metadata"])

        try:
            with open(run_dir / "candidate_registry.json", "w") as f:
                json.dump(registry, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to write candidate registry: %s", e)


def extract_and_register_from_search(
    run_dir: Path,
    tool_name: str,
    result: dict,
) -> list[str]:
    """Extract candidates from search results and register as discovered.

    For search_uniprot, parses structured data to capture gene names and
    sequence lengths. For other tools, falls back to regex extraction.

    Returns list of registered accessions.
    """
    from enzyme_discovery_agent.models.exploration_log import append_exploration_log

    query_desc = result.get("summary", "")[:80] or f"{tool_name} search"
    data = result.get("data", {})
    registered = []

    if tool_name == "search_uniprot" and isinstance(data, list):
        for entry in data:
            acc = entry.get("accession", "")
            if not acc or not UNIPROT_ACC_RE.match(acc):
                continue
            gene_names = entry.get("gene_names", [])
            update_candidate_registry(run_dir, acc, "discovered", metadata={
                "source_tool": tool_name,
                "gene_names": gene_names,
                "protein_name": entry.get("name", ""),
                "organism": entry.get("organism", ""),
                "seq_length": entry.get("seq_length", 0),
                "function": entry.get("function", ""),
                "catalytic_activities": entry.get("catalytic_activities", []),
            })
            registered.append(acc)
    else:
        result_str = json.dumps(data, default=str)
        accessions = set(UNIPROT_ACC_RE.findall(result_str))
        for acc in accessions:
            update_candidate_registry(run_dir, acc, "discovered", metadata={
                "source_tool": tool_name,
            })
            registered.append(acc)

    if registered:
        labeled = []
        if tool_name == "search_uniprot" and isinstance(data, list):
            acc_to_entry = {e.get("accession", ""): e for e in data}
            for acc in sorted(registered)[:10]:
                entry = acc_to_entry.get(acc, {})
                genes = entry.get("gene_names", [])
                label = genes[0] if genes else entry.get("name", "")[:30]
                labeled.append(f"{label} ({acc})" if label else acc)
        else:
            labeled = sorted(registered)[:10]

        append_exploration_log(
            run_dir,
            entry_type="SEARCH",
            title=f"{tool_name}: {query_desc}",
            details={
                "Accessions discovered": (
                    f"{len(registered)} — "
                    f"{', '.join(labeled)}"
                    f"{'...' if len(registered) > 10 else ''}"
                ),
            },
        )

    return registered


def register_structural_homologs(run_dir: Path, result: dict) -> list[str]:
    """Register Foldseek-derived candidates with auditable structural provenance."""
    data = result.get("data") or {}
    seed_accession = data.get("seed_accession", "")
    registered = []
    for hit in data.get("homologs", []):
        accession = hit.get("accession", "")
        if not accession or not UNIPROT_ACC_RE.match(accession):
            continue
        provenance = {
            "seed_accession": seed_accession,
            "search_method": data.get("search_method", "Foldseek structural-homolog search"),
            "tm_score": hit.get("tm_score"),
            "sequence_identity": hit.get("sequence_identity"),
            "source_database": hit.get("source_database"),
        }
        update_candidate_registry(run_dir, accession, "discovered", metadata={
            "source_tool": "search_structural_homologs",
            "structural_homology": provenance,
        })
        registered.append(accession)
    return registered
