"""Tool registry: @tool decorator, auto OpenAI schema generation, dispatch, and search deduplication.

Usage:
    from enzyme_discovery_agent.tool_registry import tool

    @tool(
        description="Validate a SMILES string and return canonical form",
        category="chemistry",
        parallel_safe=True,
        params={
            "smiles": {"type": "string", "description": "SMILES string to validate"}
        }
    )
    def validate_smiles(smiles: str) -> dict:
        ...
"""

from __future__ import annotations

import difflib
import functools
import json
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger("enzyme_agent")

# Global registries
_TOOLS: dict[str, dict] = {}  # name -> {func, schema, category, parallel_safe}
_SEARCH_HISTORY: list[dict] = []  # for deduplication
_SEARCH_HISTORY_LOCK = threading.Lock()

# Tools whose queries should be deduplicated
_SEARCHABLE_CATEGORIES = frozenset({"database", "literature"})


def tool(
    description: str,
    category: str = "general",
    parallel_safe: bool = True,
    params: dict[str, dict] | None = None,
    injected_params: list[str] | None = None,
) -> Callable:
    """Decorator that registers a function as an agent tool.

    Args:
        description: Tool description for the LLM.
        category: Tool category (chemistry, database, literature, reasoning, substrate).
        parallel_safe: Whether this tool can run concurrently with others.
        params: Parameter schemas {name: {type, description, enum?, ...}}.
            If None, auto-generates from function signature.
        injected_params: Parameter names that the agent loop injects (e.g., "run_dir").
            These are excluded from the OpenAI schema so the LLM doesn't see them.
    """
    injected = set(injected_params or [])

    def decorator(func: Callable) -> Callable:
        # Build OpenAI function-calling schema
        func_params = params or {}
        properties = {}
        required = []

        for pname, pschema in func_params.items():
            if pname in injected:
                continue
            prop = {"type": pschema.get("type", "string")}
            if "description" in pschema:
                prop["description"] = pschema["description"]
            if "enum" in pschema:
                prop["enum"] = pschema["enum"]
            if "items" in pschema:
                prop["items"] = pschema["items"]
            if "default" in pschema:
                prop["default"] = pschema["default"]
            else:
                required.append(pname)
            properties[pname] = prop

        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

        _TOOLS[func.__name__] = {
            "func": func,
            "schema": schema,
            "category": category,
            "parallel_safe": parallel_safe,
            "injected_params": injected,
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tool_name = func.__name__
        return wrapper

    return decorator


def get_all_tools() -> list[dict]:
    """Return list of OpenAI tool definitions for all registered tools."""
    return [info["schema"] for info in _TOOLS.values()]


def get_tool_names() -> list[str]:
    """Return all registered tool names."""
    return list(_TOOLS.keys())


def get_tools_by_category() -> dict[str, list[str]]:
    """Return tool names grouped by category."""
    by_cat: dict[str, list[str]] = {}
    for name, info in _TOOLS.items():
        by_cat.setdefault(info["category"], []).append(name)
    return by_cat


def get_parallel_safe_tools() -> frozenset[str]:
    """Return set of tool names that can safely run concurrently."""
    return frozenset(name for name, info in _TOOLS.items() if info["parallel_safe"])


def get_tool_info(name: str) -> dict | None:
    """Return full info dict for a tool, or None if not found."""
    return _TOOLS.get(name)


def execute_tool(name: str, args: dict, injected_args: dict | None = None) -> dict:
    """Execute a registered tool by name.

    Args:
        name: Tool function name.
        args: Arguments from the LLM.
        injected_args: Arguments injected by the agent loop (e.g., run_dir).

    Returns:
        Standardized result dict with status/summary/data/count/suggestions.
    """
    # Handle args that failed JSON parsing in the agent loop
    if "_parse_error" in args:
        return {
            "status": "error",
            "summary": f"Invalid tool arguments: {args['_parse_error']}",
            "data": None,
            "suggestions": [
                "Your tool call had invalid JSON arguments. Please retry with valid JSON.",
                f"Raw input: {args.get('_raw', 'N/A')}",
            ],
        }

    info = _TOOLS.get(name)
    if info is None:
        return {
            "status": "error",
            "summary": f"Unknown tool: {name}",
            "data": None,
            "suggestions": [f"Available tools: {', '.join(_TOOLS.keys())}"],
        }

    allowed = set(info["schema"]["function"]["parameters"]["properties"]) | set(info["injected_params"])
    unknown = sorted(set(args) - allowed)
    if unknown:
        return {
            "status": "error",
            "summary": f"Invalid arguments for {name}: {', '.join(unknown)}.",
            "data": None,
            "suggestions": [f"Use only these arguments: {', '.join(sorted(allowed))}."],
        }

    required = set(info["schema"]["function"]["parameters"].get("required", []))
    missing = sorted(required - set(args))
    if missing:
        return {
            "status": "error",
            "summary": f"Missing required arguments for {name}: {', '.join(missing)}.",
            "data": None,
            "suggestions": [
                f"Retry {name} with: {', '.join(sorted(required))}."
            ],
        }

    # Merge injected args
    merged_args = dict(args)
    if injected_args:
        for k, v in injected_args.items():
            if k in info["injected_params"]:
                merged_args[k] = v

    try:
        result = info["func"](**merged_args)

        # If tool already returns standardized format, pass through
        if isinstance(result, dict) and "status" in result:
            return result

        # Wrap raw results in standardized format
        return _standardize_result(result, name)
    except Exception as e:
        logger.warning("Tool %s failed: %s", name, e, exc_info=True)
        return {
            "status": "error",
            "summary": f"{name} failed: {e}",
            "data": None,
            "suggestions": [_suggest_recovery(name, str(e))],
        }


def _standardize_result(result: Any, tool_name: str) -> dict:
    """Wrap a raw tool result in standardized format."""
    if isinstance(result, list):
        return {
            "status": "success" if result else "partial",
            "summary": f"{tool_name} returned {len(result)} results",
            "data": result,
            "count": len(result),
        }
    if isinstance(result, dict):
        # Check for error in the result
        if "error" in result:
            return {
                "status": "error",
                "summary": result["error"],
                "data": result,
            }
        return {
            "status": "success",
            "summary": f"{tool_name} completed",
            "data": result,
        }
    if isinstance(result, bool):
        return {
            "status": "success",
            "summary": f"{tool_name}: {result}",
            "data": {"result": result},
        }
    if isinstance(result, (int, float)):
        return {
            "status": "success",
            "summary": f"{tool_name}: {result}",
            "data": {"result": result},
        }
    if isinstance(result, str):
        return {
            "status": "success" if result else "partial",
            "summary": f"{tool_name}: {len(result)} chars",
            "data": {"result": result},
        }
    return {
        "status": "success",
        "summary": f"{tool_name} completed",
        "data": {"result": result},
    }


def _suggest_recovery(tool_name: str, error_msg: str) -> str:
    """Generate a recovery suggestion based on the error."""
    if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
        return "API timed out. Try again or use a different search strategy."
    if "404" in error_msg or "not found" in error_msg.lower():
        return "Resource not found. Check the identifier and try a different query."
    if "429" in error_msg or "rate limit" in error_msg.lower():
        return "Rate limited. Wait a moment before retrying."
    if "invalid smiles" in error_msg.lower():
        return "Invalid SMILES string. Validate the SMILES first with validate_smiles."
    return f"Try a different approach or query. Error: {error_msg}"


# ---------------------------------------------------------------------------
# Search deduplication
# ---------------------------------------------------------------------------

def check_search_duplicate(tool_name: str, args: dict, threshold: float = 0.90) -> dict | None:
    """Check if a semantically similar search was already executed.

    Returns the cached result if a duplicate is found, else None.
    """
    info = _TOOLS.get(tool_name)
    if info is None or info["category"] not in _SEARCHABLE_CATEGORIES:
        return None

    # Build a normalized query string from the args
    query_str = _normalize_query(args)
    if not query_str:
        return None

    with _SEARCH_HISTORY_LOCK:
        for entry in _SEARCH_HISTORY:
            if entry["tool"] != tool_name:
                continue
            similarity = _query_similarity(query_str, entry["query"])
            if similarity >= threshold:
                return {
                    "status": "partial",
                    "summary": (
                        f"Similar search already executed (similarity={similarity:.0%}). "
                        f"Previous query: '{entry['query']}'. Try a different strategy."
                    ),
                    "data": entry.get("result_summary"),
                    "suggestions": [
                        "Use different keywords or a different search axis.",
                        "Try searching by gene name, pathway, or protein fold instead.",
                    ],
                    "deduplicated": True,
                }
    return None


def record_search(tool_name: str, args: dict, result_summary: Any = None) -> None:
    """Record a search execution for future deduplication checks."""
    info = _TOOLS.get(tool_name)
    if info is None or info["category"] not in _SEARCHABLE_CATEGORIES:
        return

    query_str = _normalize_query(args)
    if query_str:
        with _SEARCH_HISTORY_LOCK:
            _SEARCH_HISTORY.append({
                "tool": tool_name,
                "query": query_str,
                "args": args,
                "result_summary": _compact_summary(result_summary),
            })


def clear_search_history() -> None:
    """Reset the search deduplication history."""
    with _SEARCH_HISTORY_LOCK:
        _SEARCH_HISTORY.clear()


def _normalize_query(args: dict) -> str:
    """Extract and normalize the query string from tool arguments."""
    # Try common query parameter names
    for key in ("query", "ec_number", "uniprot_accession", "doi", "substrate_smiles"):
        if key in args and args[key]:
            val = str(args[key]).lower().strip()
            # Remove extra whitespace
            return " ".join(val.split())
    return ""


def _query_similarity(q1: str, q2: str) -> float:
    """Compute similarity between two query strings using SequenceMatcher."""
    if q1 == q2:
        return 1.0
    # Token-level Jaccard as a fast approximation
    tokens1 = set(q1.split())
    tokens2 = set(q2.split())
    if not tokens1 or not tokens2:
        return 0.0
    jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    # Also check character-level similarity for short queries
    seq_sim = difflib.SequenceMatcher(None, q1, q2).ratio()
    return max(jaccard, seq_sim)


def _compact_summary(result: Any) -> Any:
    """Create a compact summary of a result for dedup cache."""
    if result is None:
        return None
    if isinstance(result, dict):
        return {
            "status": result.get("status"),
            "count": result.get("count"),
            "summary": result.get("summary"),
        }
    if isinstance(result, list):
        return {"count": len(result)}
    return None
