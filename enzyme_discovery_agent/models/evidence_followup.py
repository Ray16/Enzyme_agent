"""Targeted primary-literature substrate-scope follow-up for promising investigate candidates.

When a candidate passes initial mechanistic review as ``investigate`` but remains
blocked by missing substrate-scope, promiscuity, or kinetics evidence, a one-line
database reaction string is not enough to promote or exclude it. This module
performs a bounded, zero-LLM-token literature lookup (Europe PMC + PubMed) for
family-diverse ``investigate`` candidates and attaches checksummed primary
abstract passages to the registry so the next target review can reassess them
with actual literature evidence.

Reaction-agnostic: builds queries strictly from each candidate's own retrieved
gene/protein name and organism.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.models.candidate_evidence import candidate_record
from enzyme_discovery_agent.models.candidate_registry import (
    discovery_status,
    family_label,
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent.state_store import atomic_write_json, read_json

logger = logging.getLogger("enzyme_agent")

_SCOPE_CUES = re.compile(
    r"\b(?:substrate\s+specificit\w*|substrate\s+scope|promiscu\w*|kcat|km\b|k_m\b|"
    r"catalytic\s+efficienc\w*|active\s+site|mutagenesis|crystal\s+structure|"
    r"no\s+activity|inactive\s+with|tolerat\w*|analog\w*|characteriz\w*|kinetic\w*)\b",
    re.I,
)


def _ledger_path(run_dir: Path) -> Path:
    return Path(run_dir) / "agent_state" / "scope_followup.json"


def _has_missing_scope_uncertainty(assessment: dict) -> bool:
    if assessment.get("unknowns"):
        return True
    for constraint in assessment.get("constraints") or []:
        if isinstance(constraint, dict) and constraint.get("basis") == "missing_evidence":
            return True
    return False


def _short_query_for_candidate(data: dict) -> str | None:
    """Build a concise 3-5 word enzyme-specific query from retrieved metadata."""
    genes = [str(g).strip() for g in (data.get("gene_names") or []) if str(g).strip()]
    protein = str(data.get("protein_name") or "").strip()
    organism = str(data.get("organism") or "").strip()
    # Keep first two tokens of organism (Genus species) when available.
    org_short = " ".join(organism.split()[:2]) if organism else ""

    if genes and len(genes[0]) >= 3:
        anchor = genes[0]
    elif protein:
        # Strip parenthetical qualifiers and keep first 3 informative words.
        clean = re.sub(r"\([^)]*\)", "", protein).strip()
        anchor = " ".join(clean.split()[:3])
    else:
        return None
    if not anchor:
        return None
    if org_short:
        return f'"{anchor}" "{org_short}" substrate'
    return f'"{anchor}" substrate specificity'


def select_scope_followup_candidates(run_dir: str | Path, limit: int = 2) -> list[dict]:
    """Select family-diverse ``investigate`` candidates whose limiting gap is missing evidence."""
    root = Path(run_dir)
    ledger = read_json(_ledger_path(root), {"attempted": {}, "total_attempted": 0})
    cap = getattr(config, "SCOPE_FOLLOWUP_MAX_CANDIDATES", 12)
    if ledger.get("total_attempted", 0) >= cap:
        return []

    registry = load_candidate_registry(root, strict=True).get("candidates", {})
    attempted = ledger.get("attempted") or {}
    eligible_rows = []
    for acc, entry in sorted(registry.items()):
        if discovery_status(entry) == "rejected":
            continue
        meta = entry.get("metadata") or {}
        if meta.get("is_literature_lead"):
            continue
        assessment = meta.get("target_assessment") or {}
        if assessment.get("decision") != "investigate":
            continue
        if not _has_missing_scope_uncertainty(assessment):
            continue
        rec_hash = assessment.get("record_sha256") or ""
        prior = attempted.get(acc) or {}
        if prior.get("record_sha256") == rec_hash and prior.get("status") in {"found", "empty"}:
            continue
        data = candidate_record(meta)
        query = _short_query_for_candidate(data)
        if not query:
            continue
        eligible_rows.append({
            "accession": acc,
            "family": family_label(meta, acc),
            "protein_name": data.get("protein_name") or "",
            "gene_names": data.get("gene_names") or [],
            "organism": data.get("organism") or "",
            "query": query,
            "record_sha256": rec_hash,
        })

    # Diversify across families: at most one candidate per family per round.
    selected = []
    seen_families: set[str] = set()
    remaining_budget = max(0, cap - ledger.get("total_attempted", 0))
    target_n = min(limit, remaining_budget)
    for row in eligible_rows:
        if len(selected) >= target_n:
            break
        if row["family"] in seen_families:
            continue
        seen_families.add(row["family"])
        selected.append(row)
    return selected


def _extract_scope_passages(papers: list[dict], max_papers: int = 3) -> list[dict]:
    """Keep papers whose abstracts contain substrate-scope, kinetic, or active-site cues."""
    kept = []
    seen_ids: set[str] = set()
    for paper in papers or []:
        if not isinstance(paper, dict):
            continue
        abstract = str(paper.get("abstract") or "").strip()
        title = str(paper.get("title") or "").strip()
        if not abstract or not title:
            continue
        key = str(paper.get("doi") or paper.get("pmid") or title).casefold()
        if key in seen_ids:
            continue
        matches = _SCOPE_CUES.findall(abstract)
        if not matches:
            continue
        seen_ids.add(key)
        kept.append({
            "title": title[:220],
            "doi": paper.get("doi"),
            "pmid": paper.get("pmid"),
            "year": paper.get("year"),
            "matched_cues": sorted({m.lower() for m in matches})[:8],
            "abstract_excerpt": abstract[:900],
        })
        if len(kept) >= max_papers:
            break
    return kept


def run_scope_followup(
    run_dir: str | Path,
    *,
    limit: int = 2,
    max_papers_per_candidate: int = 3,
    search_fn=None,
) -> dict:
    """Fetch targeted primary-literature substrate-scope abstracts for top investigate leads.

    Makes zero LLM calls. Returns ``{"attempted": [...], "enriched": [...], "empty": [...], "errors": [...]}``.
    """
    root = Path(run_dir)
    selected = select_scope_followup_candidates(root, limit=limit)
    summary = {"attempted": [], "enriched": [], "empty": [], "errors": []}
    if not selected:
        return summary

    if search_fn is None:
        from enzyme_discovery_agent.tools.literature import search_europepmc

        def _default_search(query: str) -> list[dict]:
            res = search_europepmc(query=query, max_results=6)
            return res.get("data") if isinstance(res, dict) and isinstance(res.get("data"), list) else []

        search_fn = _default_search

    from concurrent.futures import ThreadPoolExecutor
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry_batch

    def _fetch_one(row: dict):
        try:
            papers = search_fn(row["query"]) or []
            return row, papers, None
        except Exception as exc:  # noqa: BLE001 - network lookup must never halt discovery
            return row, None, exc

    if len(selected) > 1:
        with ThreadPoolExecutor(max_workers=min(len(selected), 6)) as pool:
            fetched = list(pool.map(_fetch_one, selected))
    else:
        fetched = [_fetch_one(selected[0])]

    ledger = read_json(_ledger_path(root), {"attempted": {}, "total_attempted": 0})
    attempted_map = ledger.setdefault("attempted", {})
    registry_updates = []

    for row, papers, exc in fetched:
        acc = row["accession"]
        summary["attempted"].append(acc)
        if exc is not None:
            logger.debug("Scope follow-up failed for %s: %s", acc, exc)
            attempted_map[acc] = {
                "record_sha256": row["record_sha256"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary["errors"].append({"accession": acc, "error": str(exc)})
            continue
        try:
            passages = _extract_scope_passages(papers, max_papers=max_papers_per_candidate)
            artifact_payload = {
                "accession": acc,
                "protein_name": row["protein_name"],
                "organism": row["organism"],
                "query": row["query"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "passages": passages,
                "scope": (
                    "Retrieved primary-literature abstract excerpts for substrate-scope review; "
                    "not verified target activity."
                ),
            }
            artifact_path = root / "artifacts" / "evidence" / "scope_followup" / f"{acc}.json"
            atomic_write_json(artifact_path, artifact_payload)
            artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

            if passages:
                evidence_block = {
                    "query": row["query"],
                    "artifact_path": str(artifact_path.relative_to(root)),
                    "artifact_sha256": artifact_sha,
                    "passages": passages,
                    "scope": artifact_payload["scope"],
                }
                registry_updates.append((acc, None, {"scope_followup_evidence": evidence_block}))
                attempted_map[acc] = {
                    "record_sha256": row["record_sha256"],
                    "status": "found",
                    "n_passages": len(passages),
                    "artifact": str(artifact_path.relative_to(root)),
                }
                summary["enriched"].append(acc)
            else:
                attempted_map[acc] = {
                    "record_sha256": row["record_sha256"],
                    "status": "empty",
                    "n_passages": 0,
                    "artifact": str(artifact_path.relative_to(root)),
                }
                summary["empty"].append(acc)
            ledger["total_attempted"] = int(ledger.get("total_attempted", 0)) + 1
        except Exception as exc2:  # noqa: BLE001 - artifact write must never halt discovery
            logger.debug("Scope follow-up failed for %s: %s", acc, exc2)
            attempted_map[acc] = {
                "record_sha256": row["record_sha256"],
                "status": "error",
                "error": f"{type(exc2).__name__}: {exc2}",
            }
            summary["errors"].append({"accession": acc, "error": str(exc2)})

    if registry_updates:
        update_candidate_registry_batch(root, registry_updates)
    atomic_write_json(_ledger_path(root), ledger)
    return summary
