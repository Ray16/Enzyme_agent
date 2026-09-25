"""Candidate registry: structured JSON tracking of discovered enzyme candidates.

Provides a thread-safe registry that tracks candidate status transitions
(discovered -> screened -> evaluated -> ranked / rejected) and metadata.
Also provides helper to extract and register candidates from search results.
"""

from __future__ import annotations

import hashlib
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


def family_key(metadata: dict, accession: str) -> str:
    """Return a stable, coarse family key without relying on one gene label.

    The canonical definition of "which enzyme family a candidate belongs to",
    used both for screening (orthologs collapse to one representative) and for
    search-novelty accounting (how many *distinct families* a search surfaced
    versus how many redundant orthologs).  Kept here in the low-level registry
    module so both the registry and the gates share one definition; the gates
    re-export it as ``_family_key`` for backward compatibility.
    """
    genes = metadata.get("gene_names", [])
    if genes:
        return f"gene:{genes[0].lower()}"
    name = re.sub(r"\b\d+\b", "", metadata.get("protein_name", "").lower())
    name = re.sub(r"\s+", " ", name).strip()
    return f"name:{name}" if name else f"accession:{accession}"


def family_label(metadata: dict, accession: str) -> str:
    """Best-available family partition key for a candidate.

    Prefers the catalytic InterPro family resolved at *evaluation*
    (``interpro_family_id`` -- multi-domain-aware, the honest family) and falls
    back to the coarse gene/name :func:`family_key` for candidates not yet
    evaluated.  A candidate's label therefore *sharpens* from gene-level to
    InterPro-level as the run invests attention in it, so accuracy lands on the
    ranked set without paying an InterPro lookup for the discarded ortholog tail.
    """
    ipr_id = str(
        metadata.get("interpro_family_id")
        or metadata.get("screening_family_id")
        or ""
    ).strip()
    if ipr_id:
        return f"ipr:{ipr_id}"
    return family_key(metadata, accession)


def family_display(metadata: dict, accession: str) -> str:
    """Human-readable family name for the index (InterPro name + id when known)."""
    ipr_id = str(
        metadata.get("interpro_family_id")
        or metadata.get("screening_family_id")
        or ""
    ).strip()
    name = str(metadata.get("interpro_family") or "").strip()
    if ipr_id:
        return f"{name} ({ipr_id})" if name else ipr_id
    return family_key(metadata, accession)


def current_candidate_status(entry: dict) -> str | None:
    """Return the current lifecycle state, not any historical state."""
    statuses = entry.get("statuses") or []
    return statuses[-1].get("status") if statuses else None


def evidence_fingerprint(metadata: dict | None) -> str:
    """Return a stable fingerprint for evidence behind a lifecycle decision."""
    encoded = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_families_index(candidates: dict) -> dict:
    """Group an accession-keyed candidate pool into a family index.

    Index *over* the shared pool -- candidates keep their accession identity; a
    family is a label with its member accessions and the furthest lifecycle
    status any member reached.  A candidate belongs to exactly one label (its
    best-available family), so the index is a clean partition of the pool.
    """
    families: dict = {}
    for acc, info in candidates.items():
        meta = info.get("metadata", {})
        label = family_label(meta, acc)
        fam = families.setdefault(label, {
            "family": family_display(meta, acc),
            "interpro_id": meta.get("interpro_family_id") or None,
            "members": [],
            "_statuses": set(),
        })
        fam["members"].append(acc)
        # A family's display sharpens to the InterPro name as soon as ANY member
        # is evaluated, even if the label is still gene-based for the others.
        if meta.get("interpro_family_id") and not fam["interpro_id"]:
            fam["interpro_id"] = meta.get("interpro_family_id")
            fam["family"] = family_display(meta, acc)
        for s in info.get("statuses", []):
            if s.get("status"):
                fam["_statuses"].add(s["status"])
    for fam in families.values():
        fam["n_members"] = len(fam["members"])
        fam["statuses"] = sorted(fam.pop("_statuses"))
    # Largest families first: the ortholog-heavy ones a reader wants to see.
    return dict(sorted(families.items(), key=lambda kv: -kv[1]["n_members"]))


def load_families(run_dir: Path) -> dict:
    """Return the family index from the on-disk registry (current as of last write)."""
    path = Path(run_dir) / "candidate_registry.json"
    if path.exists():
        try:
            with open(path) as f:
                fams = json.load(f).get("families")
                if isinstance(fams, dict):
                    return fams
        except (json.JSONDecodeError, OSError):
            pass
    # Fall back to computing from the pool if an older file lacks the index.
    return build_families_index(load_candidate_registry(run_dir).get("candidates", {}))


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
    # Fold a fresh family index into the file so the on-disk registry is itself
    # partitioned by family (index-over-pool: candidates keep identity, families
    # are a view).  Computed from the accession-keyed pool; the explicit key wins
    # over any stale ``families`` carried in from a prior load.
    families = build_families_index(registry.get("candidates", {}))
    return {**registry, "candidates": out, "families": families}


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
    now = datetime.now().astimezone().isoformat()

    with _registry_lock:
        run_dir.mkdir(parents=True, exist_ok=True)
        registry = load_candidate_registry(run_dir)

        if accession not in registry["candidates"]:
            registry["candidates"][accession] = {
                "accession": accession,
                "statuses": [],
                "first_seen": now,
                "last_updated": now,
                "metadata": {},
            }

        entry = registry["candidates"][accession]
        entry["accession"] = accession
        if status:
            current = current_candidate_status(entry)
            if current == "rejected" and status not in {"rejected", "reopened"}:
                logger.info(
                    "Ignoring %s transition for rejected candidate %s; reopen explicitly first",
                    status, accession,
                )
                status = None
        if status:
            transition = {"status": status, "timestamp": now}
            if status in {"rejected", "reopened"}:
                transition["evidence_fingerprint"] = evidence_fingerprint(metadata)
            entry["statuses"].append(transition)
        entry["last_updated"] = now
        if metadata:
            entry["metadata"].update(metadata)
        entry["enzyme_name"] = _enzyme_name(entry["metadata"])

        try:
            with open(run_dir / "candidate_registry.json", "w") as f:
                json.dump(_to_file_view(registry), f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to write candidate registry: %s", e)


def reopen_candidate(
    run_dir: Path,
    accession: str,
    *,
    reason: str,
    evidence: dict | None = None,
) -> None:
    """Explicitly reopen a rejected candidate after contradictory evidence arrives."""
    update_candidate_registry(run_dir, accession, "reopened", metadata={
        "reopen_reason": reason,
        "reopen_evidence": evidence or {},
        "rejection_reason": None,
    })


from enzyme_discovery_agent.models.target_profile import (  # noqa: E402
    is_opposite_direction as _is_opposite_direction,
)


def _target_direction(run_dir: Path) -> str | None:
    """Read the run's declared target bond direction (answer-agnostic).

    Prefers the shared chemistry contract so registration and screening agree;
    falls back to re-extraction for legacy runs.
    """
    from enzyme_discovery_agent.models.target_reasoning import target_claims
    direction, _, _ = target_claims(run_dir)
    if direction is not None:
        return direction
    try:
        from enzyme_discovery_agent.chemistry_logic import extract_target_bond_type
        ctx = json.loads((run_dir / "target_context.json").read_text())
        reaction = (ctx.get("target_profile") or {}).get("reaction") or {}
        text = " ".join(filter(None, [
            str(reaction.get("class", "")),
            str(reaction.get("direction", "")),
            str(ctx.get("query", "")),
        ]))
        return extract_target_bond_type(text).get("direction")
    except (OSError, json.JSONDecodeError, AttributeError, KeyError):
        return None


def extract_and_register_from_search(
    run_dir: Path,
    tool_name: str,
    result: dict,
) -> list[str]:
    """Extract candidates from search results and register as discovered.

    A search hit whose native reaction runs OPPOSITE the target's bond direction
    (a cleavage/hydrolysis enzyme when the target is bond formation, or vice
    versa) is chemically disqualified and is NOT registered as a lead — the call
    is EC-based, so it stays answer-agnostic and never names an enzyme class.

    For search_uniprot, parses structured data to capture gene names and
    sequence lengths. For other tools, falls back to regex extraction.

    Returns list of registered accessions.
    """
    from enzyme_discovery_agent.models.exploration_log import append_exploration_log

    query_desc = result.get("summary", "")[:80] or f"{tool_name} search"
    data = result.get("data", {})
    registered = []
    direction_dropped = 0

    # Snapshot the registry BEFORE this search so novelty can be measured: a
    # broad text search returns dozens of orthologs of the same enzyme, and the
    # raw hit count then reads as broad discovery when it is mostly redundant
    # re-import.  Recording new-vs-known accessions and distinct-vs-known
    # *families* turns "50 hits" into the honest "50 hits, 1 family, 0 new".
    _prior = load_candidate_registry(run_dir).get("candidates", {})
    _prior_accs = set(_prior)
    _prior_families = {
        family_key(info.get("metadata", {}), acc) for acc, info in _prior.items()
    }
    _hit_families: set[str] = set()

    ortholog_capped = 0
    if tool_name == "search_uniprot" and isinstance(data, list):
        from enzyme_discovery_agent import config
        from enzyme_discovery_agent.chemistry_logic import classify_reaction_chemistry
        target_direction = _target_direction(run_dir)

        # Existing representatives already registered per distinct enzyme
        # (family_key), so the ortholog cap is GLOBAL across the run — a family
        # that already has its quota gains no more orthologs from a later search.
        _fam_counts: dict[str, int] = {}
        for _acc, _info in _prior.items():
            _fam_counts[family_key(_info.get("metadata", {}), _acc)] = \
                _fam_counts.get(family_key(_info.get("metadata", {}), _acc), 0) + 1

        # Collect direction-eligible hits, grouped by enzyme family, so we can
        # keep only the best few representatives per family instead of every
        # ortholog the search returned.
        eligible: dict[str, list[dict]] = {}
        for entry in data:
            acc = entry.get("accession", "")
            if not acc or not UNIPROT_ACC_RE.match(acc):
                continue
            ec_numbers = entry.get("ec_numbers", [])
            native_direction = classify_reaction_chemistry(
                ec_numbers, entry.get("catalytic_activities", []),
            ).get("reaction_direction")
            if _is_opposite_direction(native_direction, target_direction):
                direction_dropped += 1
                append_exploration_log(
                    run_dir, entry_type="REJECT",
                    title=f"{acc} not registered — reaction-direction mismatch",
                    details={"Reason": (
                        f"native reaction direction is {native_direction}, "
                        f"opposite the target's {target_direction} direction")},
                )
                continue
            fk = family_key(
                {"gene_names": entry.get("gene_names", []),
                 "protein_name": entry.get("name", "")}, acc)
            eligible.setdefault(fk, []).append(entry)

        cap = getattr(config, "MAX_REPRESENTATIVES_PER_FAMILY", 2)
        for fk, entries in eligible.items():
            # Best representatives first: characterised (catalytic activity), then
            # annotated (EC), then a Swiss-Prot-style accession, then longer chain.
            entries.sort(key=lambda e: (
                1 if e.get("catalytic_activities") else 0,
                1 if e.get("ec_numbers") else 0,
                1 if re.match(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$", e.get("accession", "")) else 0,
                e.get("seq_length") or 0,
            ), reverse=True)
            slots = max(0, cap - _fam_counts.get(fk, 0))
            for entry in entries[:slots]:
                acc = entry.get("accession", "")
                update_candidate_registry(run_dir, acc, "discovered", metadata={
                    "source_tool": tool_name,
                    "gene_names": entry.get("gene_names", []),
                    "protein_name": entry.get("name", ""),
                    "organism": entry.get("organism", ""),
                    "seq_length": entry.get("seq_length", 0),
                    "function": entry.get("function", ""),
                    "ec_numbers": entry.get("ec_numbers", []),
                    "catalytic_activities": entry.get("catalytic_activities", []),
                })
                registered.append(acc)
                _hit_families.add(fk)
            ortholog_capped += max(0, len(entries) - slots)

        # Surface the direction filter back to the agent so it learns the search
        # targeted the OPPOSITE reaction type, and stops mining that class.
        if direction_dropped and isinstance(result, dict):
            result["summary"] = (
                f"{result.get('summary', '')} | {direction_dropped} hit(s) "
                f"excluded before registration: native reaction runs opposite the "
                f"target's {target_direction} direction (a search for the reverse "
                f"reaction type)."
            )
            result["direction_excluded"] = direction_dropped
    else:
        result_str = json.dumps(data, default=str)
        accessions = set(UNIPROT_ACC_RE.findall(result_str))
        for acc in accessions:
            update_candidate_registry(run_dir, acc, "discovered", metadata={
                "source_tool": tool_name,
            })
            registered.append(acc)
            _hit_families.add(family_key({}, acc))

    # Novelty accounting: distinguish genuinely new leads from re-imported
    # orthologs, and new families from ones already represented.  Attached to
    # ``result`` (so the agent sees novelty, not just volume) and emitted as a
    # structured event (so the replayable stream records the ortholog collapse).
    n_new = sum(1 for acc in registered if acc not in _prior_accs)
    novelty = {
        "hits": len(data) if isinstance(data, list) else len(registered),
        "registered": len(registered),
        "new_accessions": n_new,
        "known_accessions": len(registered) - n_new,
        "distinct_families": len(_hit_families),
        "new_families": len(_hit_families - _prior_families),
        # Members beyond the first per family in THIS search: the ortholog
        # redundancy the raw hit count hides.
        "redundant_orthologs": max(0, len(registered) - len(_hit_families)),
        # Surplus orthologs the family cap DROPPED (not registered): budget saved
        # for exploring different families rather than re-importing one enzyme.
        "dropped_orthologs": ortholog_capped,
    }
    if isinstance(result, dict):
        result["novelty"] = novelty
        if registered:
            dropped_note = (
                f"; {novelty['dropped_orthologs']} surplus ortholog(s) dropped by the "
                f"family cap (re-queryable on demand)" if novelty["dropped_orthologs"] else ""
            )
            result["summary"] = (
                f"{result.get('summary', '')} | {novelty['registered']} registered "
                f"across {novelty['distinct_families']} distinct famil"
                f"{'y' if novelty['distinct_families'] == 1 else 'ies'} "
                f"({novelty['new_families']} new to this run); "
                f"{novelty['known_accessions']} already-known{dropped_note}."
            )
    if registered:
        from enzyme_discovery_agent.models.event_log import append_event
        append_event(run_dir, "candidates_registered", tool=tool_name, **novelty)

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
