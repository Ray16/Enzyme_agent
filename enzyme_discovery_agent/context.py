"""LLM context compaction: strip bulky fields from tool results."""

from __future__ import annotations

from enzyme_discovery_agent import config

ARTIFACT_CATEGORIES = {
    "search_pubmed": "literature",
    "search_openalex": "literature",
    "search_uniprot": "database",
    "fetch_uniprot_sequence": "database",
    "search_kegg_enzyme": "database",
    "search_interpro": "database",
    "get_protein_domains": "database",
    "evaluate_candidate": "candidates",
    "screen_candidates": "candidates",
}

STRIP_KEYS = frozenset({"sequence", "sequence_preview", "openalex_id", "member_databases"})

TRUNCATE_KEYS = {
    "abstract": config.ABSTRACT_TRUNCATE_CHARS,
    "description": config.DESCRIPTION_TRUNCATE_CHARS,
}

CAP_LIST_KEYS = {
    "genes": 3, "papers": config.PAPER_LIST_CAP,
    "results": config.RESULTS_LIST_CAP, "entries": config.RESULTS_LIST_CAP,
    "hits": config.RESULTS_LIST_CAP, "reasoning": 3, "function_comments": 2,
    "catalytic_activities": 3, "authors": 3,
}


def compact_for_llm(obj):
    """Recursively strip bulky fields from a tool result for LLM context."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in STRIP_KEYS:
                continue
            if k in TRUNCATE_KEYS and isinstance(v, str) and len(v) > TRUNCATE_KEYS[k]:
                out[k] = v[:TRUNCATE_KEYS[k]] + "..."
            elif k in CAP_LIST_KEYS and isinstance(v, list) and len(v) > CAP_LIST_KEYS[k]:
                cap = CAP_LIST_KEYS[k]
                out[k] = [compact_for_llm(i) for i in v[:cap]]
                out[k].append(f"... +{len(v) - cap} more")
            else:
                out[k] = compact_for_llm(v)
        return out
    if isinstance(obj, list):
        return [compact_for_llm(i) for i in obj]
    return obj
