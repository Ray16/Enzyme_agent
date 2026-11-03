"""ModelSEED / KBase reaction database tools.

Queries native reaction chemistry and cross-references from ModelSEED for
evidence-guided reaction-to-enzyme discovery. Retrieved reactions are leads,
not a claim of complete coverage or demonstrated target activity.

API endpoints (public, no authentication required):
    Reactions: https://modelseed.org/solr/reactions/select
    Compounds: https://modelseed.org/solr/compounds/select
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools._http import request_with_retry

logger = logging.getLogger("enzyme_agent")

_SOLR_BASE = "https://modelseed.org/solr"
_REACTION_FIELDS = "id,name,ec_numbers,definition,compound_ids,pathways,aliases,equation,direction,reversibility"
_COMPOUND_FIELDS = "id,name,formula,smiles,aliases"
_MAX_ROWS = 500
_USER_AGENT = "EnzymeDiscoveryAgent/1.0"


def _solr_query(core: str, q: str, fl: str, rows: int = _MAX_ROWS, start: int = 0) -> dict | None:
    """Execute a Solr query against ModelSEED and return parsed JSON."""
    params = urllib.parse.urlencode({
        "q": q,
        "rows": str(rows),
        "wt": "json",
        "fl": fl,
        "start": str(start),
        "sort": "id asc",
    })
    url = f"{_SOLR_BASE}/{core}/select?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        raw = request_with_retry(req, timeout=30)
        result = json.loads(raw)
        result["source_url"] = url
        return result
    except Exception as exc:
        logger.warning("ModelSEED Solr query failed: %s", exc)
        return None


def _values(value):
    """Accept scalar, list and legacy semicolon-separated database fields."""
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    return list(dict.fromkeys(part.strip() for item in values if item is not None
                             for part in str(item).split(';') if part.strip() and part.strip().lower() != 'null'))


def _cross_references(value):
    output = {}
    for entry in value if isinstance(value, list) else [value]:
        if isinstance(entry, str) and ':' in entry:
            source, identifiers = entry.split(':', 1)
            output.setdefault(source.strip(), []).extend(_values(identifiers))
    return {source:list(dict.fromkeys(ids)) for source,ids in output.items()}


def _chemical_term(value):
    """Solr stores reaction text with significant case; match text literally across ASCII case."""
    return ''.join(f'[{char.lower()}{char.upper()}]' if char.isascii() and char.isalpha()
                   else '\\'+char if char in '.?+*|{}[]()"\\#@&<>~/:^-'
                   else char for char in value)


@tool(
    description=(
        "Discover native reactions by chemical terms, participating compound IDs, "
        "reaction ID or an evidence-derived EC prefix. Compare reacting roles, "
        "direction and substrate scope before mapping a small relevant set of "
        "returned EC numbers to search_uniprot(query='ec:<returned EC>'). "
        "Compound structures are available through get_modelseed_compound. "
        "Cross-references and pagination preserve routes beyond the first page. "
        "A database reaction match is a candidate lead, not proof of target activity."
    ),
    category="database",
    parallel_safe=True,
    params={
        "ec_class": {
            "type": "string",
            "default": "",
            "description": (
                "Optional full or partial EC number derived from target chemistry or retrieved evidence. "
                "A full four-component number matches exactly; a prefix searches its descendants."
            ),
        },
        "keyword": {
            "type": "string",
            "description": (
                "Optional chemical terms in the reaction name or equation text. "
                "Can discover reactions before an enzyme name or EC is known."
            ),
            "default": "",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of reactions to return.",
            "default": 50,
        },
        "compound_ids": {"type":"array", "items":{"type":"string"}, "default":[],
                         "description":"Optional retrieved compound IDs; every ID must participate in the reaction. This filter alone does not determine which side or role they occupy."},
        "reaction_id": {"type":"string", "default":"", "description":"Optional retrieved reaction ID to inspect."},
        "offset": {"type":"integer", "default":0, "description":"Zero-based result offset; use next_offset to continue an incomplete search."},
    },
)
def search_modelseed_reactions(
    ec_class: str = "",
    keyword: str = "",
    max_results: int = 50,
    compound_ids: list[str] | None = None,
    reaction_id: str = "",
    offset: int = 0,
) -> dict:
    """Retrieve a bounded, source-linked page of native reaction hypotheses."""
    ec_class = re.sub(r'\.(?:\*|-)$', '', ec_class.strip().rstrip('.'))
    compound_ids = _values(compound_ids)
    invalid = (ec_class and not re.fullmatch(r'\d+(?:\.\d+){0,3}',ec_class)
               or reaction_id and not re.fullmatch(r'rxn\d+',reaction_id)
               or any(not re.fullmatch(r'cpd\d+',value) for value in compound_ids)
               or offset < 0 or max_results <= 0)
    if invalid or not any((ec_class, keyword.strip(), compound_ids, reaction_id)):
        return {
            "status": "error",
            "summary": "Provide a valid reaction, compound, chemical-term or EC filter, positive max_results and nonnegative offset.",
            "data": None,
            "count": 0,
            "suggestions": ["Derive the search from target chemistry or retrieved native-reaction evidence."],
        }

    q_parts = []
    if ec_class:
        suffix = '.*' if len(ec_class.split('.')) < 4 else ''
        q_parts.append(f"ec_numbers:{ec_class}{suffix}")
    if reaction_id:
        q_parts.append(f'id:{reaction_id}')
    q_parts.extend(f'compound_ids:{value}' for value in compound_ids)
    if keyword:
        for token in keyword.split():
            token = _chemical_term(token)
            q_parts.append(f"(definition:/.*{token}.*/ OR name:/.*{token}.*/)")
    q = " AND ".join(q_parts)

    result = _solr_query("reactions", q, _REACTION_FIELDS, rows=min(max_results, _MAX_ROWS), start=offset)
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
        ecs = _values(doc.get("ec_numbers"))
        for ec in ecs:
            all_ecs.add(ec)
            parts = ec.split(".")
            if len(parts) >= 3:
                sub = f"{parts[0]}.{parts[1]}.{parts[2]}"
                ec_subclass_counts[sub] = ec_subclass_counts.get(sub, 0) + 1

        cpd_ids = _values(doc.get("compound_ids"))

        reactions.append({
            "reaction_id": doc.get("id", ""),
            "name": doc.get("name", ""),
            "ec_numbers": ecs,
            "complete_ec_numbers": [ec for ec in ecs if re.fullmatch(r'\d+(?:\.\d+){3}',ec)],
            "definition": doc.get("definition", ""),
            "compound_ids": cpd_ids,
            "pathways": doc.get("pathways", []),
            "equation": doc.get("equation", ""),
            "direction": doc.get("direction"),
            "reversibility": doc.get("reversibility"),
            "cross_references": _cross_references(doc.get("aliases")),
            "scope": "Recorded native reaction; transfer to the target remains a hypothesis.",
        })

    kw_note = f" matching '{keyword}'" if keyword else ""
    summary = (
        f"Found {total} native reactions for the supplied filters{kw_note}; returned {len(reactions)} at offset {offset} "
        f"({len(all_ecs)} unique EC numbers, "
        f"{len(ec_subclass_counts)} subclasses)."
    )

    suggestions = []
    if reactions:
        suggestions.append(
            "Use get_modelseed_compound with compound IDs to fetch substrate "
            "structures, then compare complete reacting roles and substrate context. "
            "For chemically justified reactions, resolve returned EC numbers with "
            "search_uniprot(query='ec:<returned EC>') or follow the recorded reaction cross-references."
        )
    next_offset = offset + len(reactions) if reactions and total > offset + len(reactions) else None
    if next_offset is not None:
        suggestions.append(
            f"More reactions remain; continue with offset={next_offset}. "
            "This page does not establish complete reaction-space coverage."
        )

    return {
        "status": "success",
        "summary": summary,
        "data": reactions,
        "count": len(reactions),
        "total_count": total,
        "offset": offset,
        "next_offset": next_offset,
        "source_url": result.get("source_url", f"{_SOLR_BASE}/reactions/select"),
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
                "A retrieved ModelSEED compound ID. "
                "Obtained from the compound_ids field of "
                "search_modelseed_reactions results."
            ),
        },
    },
)
def get_modelseed_compound(compound_id: str) -> dict:
    """Fetch a ModelSEED compound's name, formula, and SMILES."""
    compound_id = compound_id.strip()
    if not re.fullmatch(r'cpd\d+',compound_id):
        return {
            "status": "error",
            "summary": "Provide a valid retrieved compound_id.",
            "data": None,
            "count": 0,
            "suggestions": ["Provide a compound ID returned by search_modelseed_reactions."],
        }

    result = _solr_query("compounds", f"id:{compound_id}", _COMPOUND_FIELDS, rows=1)
    if result is None:
        return {
            "status": "error",
            "summary": f"ModelSEED compound lookup failed for {compound_id}.",
            "data": None,
            "count": 0,
            "suggestions": ["Check the retrieved compound ID."],
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
