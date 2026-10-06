"""Foldseek-based structural expansion into the ESM Metagenomic Atlas.

Reaches beyond annotated sequence databases into metagenomic "dark matter": a
supported candidate's structure is used to find structural homologs among the ESM
Atlas's predicted metagenomic structures. Hits come back as bare metagenomic
identifiers with a sequence and no functional annotation, so structural similarity
here is only an expansion signal -- every hit carries its seed and similarity
metrics, and its catalytic machinery and substrate scope must be evaluated before it
is ranked.

Route: the ESM Atlas REST API is gated, so queries go through the public Foldseek
server, which hosts the Atlas as the metagenomic target database. It is a shared
public service: query volume is kept modest and polling is polite.
"""

from __future__ import annotations

import re
import time

import requests

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools.structural_homolog_search import _fetch_alphafold_structure

_FOLDSEEK_API = "https://search.foldseek.com/api"
# The Foldseek server hosts the ESM Atlas as the standard (non-motif) metagenomic
# structure index; database[] must carry its path, not a display name.
_ATLAS_DB = getattr(config, "ESM_ATLAS_FOLDSEEK_DB", "mgnify_esm30")
_HEADERS = {"User-Agent": getattr(config, "HTTP_USER_AGENT", "enzyme-discovery-agent/1.0")}
_MGNIFY_ID = re.compile(r"(MGYP\d+)", re.I)


def _parse_atlas_results(payload: dict, seed_accession: str) -> list[dict]:
    """Extract unique metagenomic homolog records from a Foldseek Atlas response."""
    homologs: list[dict] = []
    seen: set[str] = set()
    for database_result in payload.get("results", []):
        database = database_result.get("db", "")
        for group in database_result.get("alignments", []) or []:
            alignments = group if isinstance(group, list) else [group]
            for alignment in alignments:
                target = alignment.get("target", "") or ""
                match = _MGNIFY_ID.search(target)
                if not match:
                    continue
                mgnify_id = match.group(1).upper()
                if mgnify_id in seen:
                    continue
                seen.add(mgnify_id)
                try:
                    probability = float(alignment.get("prob"))
                except (TypeError, ValueError):
                    probability = None
                sequence = (alignment.get("tSeq") or alignment.get("tseq")
                            or (alignment.get("dbAln") or "").replace("-", "")) or None
                homologs.append({
                    "mgnify_id": mgnify_id,
                    "source_database": "ESMAtlas",
                    "probability": probability,
                    "sequence_identity": alignment.get("seqId"),
                    "aln_length": alignment.get("alnLength"),
                    "sequence": sequence,
                })
    return sorted(homologs, key=lambda hit: (hit["probability"] is None,
                                             -(hit["probability"] or 0)))


@tool(
    description=(
        "Expand a promising enzyme candidate with structural homologs from the ESM "
        "metagenomic structure atlas (Foldseek). Use only after the seed has "
        "mechanistic or substrate-scope support. Hits are unannotated metagenomic "
        "sequences: a structural match is not proof of the target reaction -- assign "
        "their family and evaluate catalytic machinery and native substrate scope "
        "before ranking them."
    ),
    category="database",
    parallel_safe=False,
    params={
        "accession": {"type": "string", "description": "UniProt accession of the supported seed enzyme."},
        "max_hits": {"type": "integer", "description": "Maximum homologs to return.", "default": 30},
        "min_probability": {"type": "number", "description": "Minimum Foldseek match probability (0–1).", "default": 0.5},
    },
)
def search_esm_atlas_homologs(accession: str, max_hits: int = 30, min_probability: float = 0.5) -> dict:
    """Submit a Foldseek Atlas search and return provenance-rich metagenomic homologs."""
    seed_accession = (accession or "").strip().upper()
    if not seed_accession:
        return {"status": "error", "summary": "A seed UniProt accession is required.", "data": None}
    if max_hits < 1 or not 0.0 <= min_probability <= 1.0:
        return {"status": "error", "summary": "max_hits must be positive and min_probability must be between 0 and 1.", "data": None}

    structure = _fetch_alphafold_structure(seed_accession)
    if not structure:
        return {"status": "partial", "summary": f"No AlphaFold structure available for {seed_accession}.", "data": {"seed_accession": seed_accession, "homologs": []}, "count": 0}

    try:
        submission = requests.post(
            f"{_FOLDSEEK_API}/ticket",
            files={"q": (f"{seed_accession}.pdb", structure, "text/plain")},
            data=[("mode", "3diaa"), ("database[]", _ATLAS_DB)],
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

    homologs = [hit for hit in _parse_atlas_results(payload, seed_accession)
                if (hit["probability"] or 0) >= min_probability][:max_hits]
    return {
        "status": "success" if homologs else "partial",
        "summary": f"Found {len(homologs)} metagenomic structural homologs for {seed_accession} at probability ≥ {min_probability}.",
        "data": {
            "seed_accession": seed_accession,
            "search_method": "Foldseek 3Di+AA against the ESM Metagenomic Atlas",
            "minimum_probability": min_probability,
            "homologs": homologs,
            "interpretation": "Metagenomic structural similarity is a candidate-expansion signal; assign the family, then validate catalytic residues, substrate scope, and reaction direction before ranking.",
        },
        "count": len(homologs),
    }
