"""Foldseek-based structural-homolog expansion for enzyme discovery.

Structural similarity is an expansion signal, not evidence that a candidate
catalyzes the requested reaction.  Every hit therefore carries its seed and
similarity metrics so later chemistry and structural evaluation can audit it.
"""

from __future__ import annotations

import re
import time

import requests

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool

_FOLDSEEK_API = "https://search.foldseek.com/api"
_ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
_TARGET_ACCESSION = re.compile(r"AF-([A-Z0-9]+)-F\d+", re.I)
_HEADERS = {"User-Agent": getattr(config, "USER_AGENT", "enzyme-discovery-agent/1.0")}


def _fetch_alphafold_structure(accession: str) -> str | None:
    """Fetch the current AlphaFold PDB for an accession without hard-coding a version."""
    try:
        metadata = requests.get(_ALPHAFOLD_API.format(accession=accession), headers=_HEADERS, timeout=30)
        predictions = metadata.json() if metadata.ok else []
        pdb_url = predictions[0].get("pdbUrl") if predictions else None
        if not pdb_url:
            return None
        structure = requests.get(pdb_url, headers=_HEADERS, timeout=30)
    except (requests.RequestException, ValueError, IndexError):
        return None
    return structure.text if structure.ok and "ATOM" in structure.text[:1000] else None


def _parse_foldseek_results(payload: dict, seed_accession: str) -> list[dict]:
    """Extract unique AlphaFold/PDB homolog records from a Foldseek response."""
    homologs: list[dict] = []
    seen = {seed_accession.upper()}
    for database_result in payload.get("results", []):
        database = database_result.get("db", "")
        for group in database_result.get("alignments", []) or []:
            alignments = group if isinstance(group, list) else [group]
            for alignment in alignments:
                header = alignment.get("theader", "") or alignment.get("target", "")
                match = _TARGET_ACCESSION.search(header)
                if not match:
                    continue
                accession = match.group(1).upper()
                if accession in seen:
                    continue
                seen.add(accession)
                try:
                    tm_score = float(alignment.get("eval"))
                    if not 0.0 <= tm_score <= 1.0:
                        tm_score = None
                except (TypeError, ValueError):
                    tm_score = None
                homologs.append({
                    "accession": accession,
                    "source_database": database,
                    "tm_score": tm_score,
                    "sequence_identity": alignment.get("seqId"),
                    "organism": alignment.get("taxName"),
                    "description": re.sub(r"^AF-[A-Z0-9]+-F\d+\S*\s*", "", header, flags=re.I)[:120],
                })
    return sorted(homologs, key=lambda hit: (hit["tm_score"] is None, -(hit["tm_score"] or 0)))


@tool(
    description=(
        "Expand a promising enzyme candidate with Foldseek structural homologs from AlphaFold DB "
        "and PDB. Use only after the seed has mechanistic or substrate-scope support. A structural "
        "hit is not proof of the target reaction: evaluate its catalytic machinery and native substrate "
        "scope before ranking it."
    ),
    category="database",
    parallel_safe=False,
    params={
        "accession": {"type": "string", "description": "UniProt accession of the supported seed enzyme."},
        "max_hits": {"type": "integer", "description": "Maximum homologs to return.", "default": 30},
        "min_tm_score": {"type": "number", "description": "Minimum TM-score (0–1); 0.5 is a same-fold default.", "default": 0.5},
    },
)
def search_structural_homologs(accession: str, max_hits: int = 30, min_tm_score: float = 0.5) -> dict:
    """Submit a Foldseek search and return provenance-rich structural homologs."""
    seed_accession = (accession or "").strip().upper()
    if not seed_accession:
        return {"status": "error", "summary": "A seed UniProt accession is required.", "data": None}
    if max_hits < 1 or not 0.0 <= min_tm_score <= 1.0:
        return {"status": "error", "summary": "max_hits must be positive and min_tm_score must be between 0 and 1.", "data": None}

    structure = _fetch_alphafold_structure(seed_accession)
    if not structure:
        return {"status": "partial", "summary": f"No AlphaFold structure available for {seed_accession}.", "data": {"seed_accession": seed_accession, "homologs": []}, "count": 0}

    try:
        submission = requests.post(
            f"{_FOLDSEEK_API}/ticket",
            files={"q": (f"{seed_accession}.pdb", structure, "text/plain")},
            data=[("mode", "tmalign"), *[("database[]", database) for database in ("afdb50", "afdb-swissprot", "pdb100")]],
            headers=_HEADERS,
            timeout=60,
        )
        ticket_id = submission.json().get("id") if submission.ok else None
    except (requests.RequestException, ValueError) as exc:
        return {"status": "error", "summary": f"Foldseek submission failed: {exc}", "data": None}
    if not ticket_id:
        return {"status": "error", "summary": "Foldseek did not return a search ticket.", "data": None}

    for _ in range(40):
        time.sleep(5)
        try:
            status = requests.get(f"{_FOLDSEEK_API}/ticket/{ticket_id}", headers=_HEADERS, timeout=30).json().get("status")
        except (requests.RequestException, ValueError):
            continue
        if status == "COMPLETE":
            break
        if status == "ERROR":
            return {"status": "error", "summary": "Foldseek reported an error for this search.", "data": None}
    else:
        return {"status": "partial", "summary": "Foldseek timed out after 200 seconds.", "data": {"seed_accession": seed_accession, "homologs": []}, "count": 0}

    try:
        response = requests.get(f"{_FOLDSEEK_API}/result/{ticket_id}/0", headers=_HEADERS, timeout=60)
        payload = response.json() if response.ok else {}
    except (requests.RequestException, ValueError) as exc:
        return {"status": "error", "summary": f"Foldseek result retrieval failed: {exc}", "data": None}

    homologs = [hit for hit in _parse_foldseek_results(payload, seed_accession) if (hit["tm_score"] or 0) >= min_tm_score][:max_hits]
    return {
        "status": "success" if homologs else "partial",
        "summary": f"Found {len(homologs)} structural homologs for {seed_accession} at TM-score ≥ {min_tm_score}.",
        "data": {
            "seed_accession": seed_accession,
            "search_method": "Foldseek TM-align against AlphaFold DB and PDB",
            "minimum_tm_score": min_tm_score,
            "homologs": homologs,
            "interpretation": "Structural similarity is a candidate-expansion signal; validate catalytic residues, substrate scope, and reaction direction before ranking.",
        },
        "count": len(homologs),
    }
