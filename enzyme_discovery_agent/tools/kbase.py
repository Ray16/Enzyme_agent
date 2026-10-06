"""ModelSEED / KBase reaction database tools.

Queries the ModelSEED Solr API (the biochemistry database underlying KBase)
to systematically discover enzyme families by EC class and reaction chemistry.
Provides deterministic reaction-space coverage that complements the LLM's
reasoning-based search.

API endpoints (public, no authentication required):
    Reactions: https://modelseed.org/solr/reactions/select
    Compounds: https://modelseed.org/solr/compounds/select
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools._http import request_with_retry

logger = logging.getLogger("enzyme_agent")

_SOLR_BASE = "https://modelseed.org/solr"
_REACTION_FIELDS = "id,name,ec_numbers,definition,compound_ids,pathways,aliases"
_COMPOUND_FIELDS = "id,name,formula,smiles,aliases"
_MAX_ROWS = 500
_USER_AGENT = "EnzymeDiscoveryAgent/1.0"


def _solr_query(core: str, q: str, fl: str, rows: int = _MAX_ROWS) -> dict | None:
    """Execute a Solr query against ModelSEED and return parsed JSON."""
    params = urllib.parse.urlencode({
        "q": q,
        "rows": str(rows),
        "wt": "json",
        "fl": fl,
    })
    url = f"{_SOLR_BASE}/{core}/select?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        raw = request_with_retry(req, timeout=30)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("ModelSEED Solr query failed: %s", exc)
        return None


@tool(
    description=(
        "Search the ModelSEED reaction database for reactions by EC class or "
        "keyword. Returns reactions with EC numbers, substrates, and compound "
        "IDs that can be resolved to SMILES via get_modelseed_compound. Use "
        "this to systematically enumerate all enzyme families that catalyze a "
        "given reaction class (e.g. EC 6.3 for C-N bond-forming ligases)."
    ),
    category="database",
    parallel_safe=True,
    params={
        "ec_class": {
            "type": "string",
            "description": (
                "A full or partial EC number at any depth — e.g. a top class "
                "'<n>', a subclass '<n>.<n>', or a full '<n>.<n>.<n>.<n>'. "
                "Supports partial ECs."
            ),
        },
        "keyword": {
            "type": "string",
            "description": (
                "Optional text filter on reaction name or definition — a "
                "substring of the reaction name, or a substrate/product keyword. "
                "Applied as AND with EC class."
            ),
            "default": "",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of reactions to return.",
            "default": 50,
        },
    },
)
def search_modelseed_reactions(
    ec_class: str,
    keyword: str = "",
    max_results: int = 50,
) -> dict:
    """Search ModelSEED for reactions by EC class and optional keyword."""
    ec_class = ec_class.strip().rstrip(".")
    if not ec_class:
        return {
            "status": "error",
            "summary": "ec_class is required (e.g. '6.3' or '3.5.1').",
            "data": None,
            "count": 0,
            "suggestions": ["Provide an EC prefix like '6.3' or '3.5.1'."],
        }

    q_parts = [f"ec_numbers:{ec_class}.*"]
    if keyword:
        q_parts.append(f"(definition:*{keyword}* OR name:*{keyword}*)")
    q = " AND ".join(q_parts)

    result = _solr_query("reactions", q, _REACTION_FIELDS, rows=min(max_results, _MAX_ROWS))
    if result is None:
        return {
            "status": "error",
            "summary": "ModelSEED Solr query failed.",
            "data": None,
            "count": 0,
            "suggestions": ["Check network connectivity; retry later."],
        }

    docs = result.get("response", {}).get("docs", [])
    total = result.get("response", {}).get("numFound", 0)

    all_ecs: set[str] = set()
    ec_subclass_counts: dict[str, int] = {}
    reactions = []

    for doc in docs:
        ecs = doc.get("ec_numbers", [])
        for ec in ecs:
            all_ecs.add(ec)
            parts = ec.split(".")
            if len(parts) >= 3:
                sub = f"{parts[0]}.{parts[1]}.{parts[2]}"
                ec_subclass_counts[sub] = ec_subclass_counts.get(sub, 0) + 1

        cpd_str = doc.get("compound_ids", [""])[0] if doc.get("compound_ids") else ""
        cpd_ids = [c.strip() for c in cpd_str.split(";") if c.strip()]

        reactions.append({
            "reaction_id": doc.get("id", ""),
            "name": doc.get("name", ""),
            "ec_numbers": ecs,
            "definition": doc.get("definition", ""),
            "compound_ids": cpd_ids,
            "pathways": doc.get("pathways", []),
        })

    kw_note = f" matching '{keyword}'" if keyword else ""
    summary = (
        f"Found {total} reactions in EC {ec_class}.*{kw_note} "
        f"({len(all_ecs)} unique EC numbers, "
        f"{len(ec_subclass_counts)} subclasses)."
    )

    suggestions = []
    if reactions:
        suggestions.append(
            "Use get_modelseed_compound with compound IDs to fetch substrate "
            "SMILES for fingerprint similarity comparison."
        )
    if total > len(reactions):
        suggestions.append(
            f"Only {len(reactions)} of {total} reactions returned. "
            f"Narrow with a keyword filter or more specific EC prefix."
        )

    return {
        "status": "success",
        "summary": summary,
        "data": reactions,
        "count": len(reactions),
        "ec_summary": dict(sorted(ec_subclass_counts.items())),
        "suggestions": suggestions,
    }


@tool(
    description=(
        "Look up a ModelSEED compound by ID and return its name, formula, "
        "and SMILES structure. Use this to resolve compound IDs from "
        "search_modelseed_reactions into chemical structures for fingerprint "
        "similarity comparison with target substrates."
    ),
    category="database",
    parallel_safe=True,
    params={
        "compound_id": {
            "type": "string",
            "description": (
                "ModelSEED compound ID, e.g. 'cpd00053' for L-Glutamine. "
                "Obtained from the compound_ids field of "
                "search_modelseed_reactions results."
            ),
        },
    },
)
def get_modelseed_compound(compound_id: str) -> dict:
    """Fetch a ModelSEED compound's name, formula, and SMILES."""
    compound_id = compound_id.strip()
    if not compound_id:
        return {
            "status": "error",
            "summary": "compound_id is required (e.g. 'cpd00053').",
            "data": None,
            "count": 0,
            "suggestions": ["Provide a ModelSEED compound ID like 'cpd00053'."],
        }

    result = _solr_query("compounds", f"id:{compound_id}", _COMPOUND_FIELDS, rows=1)
    if result is None:
        return {
            "status": "error",
            "summary": f"ModelSEED compound lookup failed for {compound_id}.",
            "data": None,
            "count": 0,
            "suggestions": ["Check the compound ID format (e.g. 'cpd00053')."],
        }

    docs = result.get("response", {}).get("docs", [])
    if not docs:
        return {
            "status": "error",
            "summary": f"Compound {compound_id} not found in ModelSEED.",
            "data": None,
            "count": 0,
            "suggestions": [
                "Verify the compound ID. Use search_modelseed_reactions to "
                "find valid compound IDs."
            ],
        }

    doc = docs[0]
    aliases_raw = doc.get("aliases", [])
    aliases: dict[str, str] = {}
    for entry in aliases_raw:
        if ":" in entry:
            key, val = entry.split(":", 1)
            aliases[key.strip()] = val.strip()

    data = {
        "id": doc.get("id", ""),
        "name": doc.get("name", ""),
        "formula": doc.get("formula", ""),
        "smiles": doc.get("smiles", ""),
        "aliases": aliases,
    }

    smiles = data["smiles"]
    summary = f"{data['name']} ({data['formula']})"
    if smiles:
        summary += f" | SMILES: {smiles[:60]}"

    return {
        "status": "success",
        "summary": summary,
        "data": data,
        "count": 1,
        "suggestions": [],
    }
