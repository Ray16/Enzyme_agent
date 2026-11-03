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
import re
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
            # Preserve nested object contracts and validation constraints. A
            # shallow whitelist silently turned grounded source objects into
            # arbitrary dictionaries before they reached the serving model.
            import copy
            prop = copy.deepcopy(pschema)
            prop.setdefault('type','string')
            if "default" not in pschema:
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


def _coerce_one(value, declared_type):
    """Best-effort coerce a single tool-arg value to its declared JSON-schema type.

    Some models (notably open-weight ones served over OpenAI-compatible endpoints)
    serialize integer/number/boolean tool arguments as STRINGS ("10", "true"); the
    Claude/argo path happened to send real types, so tools assume them.  A value that
    cannot be coerced is returned unchanged for the normal error path.
    """
    try:
        if declared_type == "integer":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
                return int(value.strip())
            if isinstance(value, float) and value.is_integer():
                return int(value)
        elif declared_type == "number":
            if isinstance(value, str):
                return float(value.strip())
        elif declared_type == "boolean":
            if isinstance(value, str):
                s = value.strip().lower()
                if s in ("true", "1", "yes"):
                    return True
                if s in ("false", "0", "no"):
                    return False
        elif declared_type in ("array", "object"):
            if isinstance(value, str):
                return json.loads(value)
    except (ValueError, TypeError, json.JSONDecodeError):
        return value
    return value


def _coerce_arg_types(args: dict, properties: dict) -> dict:
    """Coerce args to the types declared in the tool schema (backend-agnostic), so the
    agent tolerates a backend that stringifies numeric/boolean tool arguments."""
    return {k: _coerce_one(v, (properties.get(k) or {}).get("type")) for k, v in args.items()}


def execute_tool(name: str, args: dict, injected_args: dict | None = None) -> dict:
    """Execute a registered tool by name.

    Args:
        name: Tool function name.
        args: Arguments from the LLM.
        injected_args: Arguments injected by the agent loop (e.g., run_dir).

    Returns:
        Standardized result dict with status/summary/data/count/suggestions.
    """
    if not isinstance(args, dict):
        return {"status": "error", "summary": "Tool arguments must be a JSON object.",
                "data": None, "suggestions": ["Retry with an object of named arguments."]}
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

    # Coerce arg types to the declared schema -- a backend may stringify ints/bools and
    # tools assume real types (e.g. slicing with max_results).  Backend-agnostic.
    properties = info["schema"]["function"]["parameters"].get("properties", {})
    args = _coerce_arg_types(args, properties)

    # Container args that survived coercion as the wrong type must not reach the tool.
    # _coerce_one is best-effort and returns an unparseable value UNCHANGED, which for
    # a declared array/object means a bare string reaches a tool that then iterates it
    # character-by-character ('str' has no attribute 'get').  There is no "normal error
    # path" downstream for these, so reject here with a corrective message instead.
    _EXPECTED = {"array": (list, tuple), "object": (dict,), "string": (str,),
                 "integer": (int,), "number": (int, float), "boolean": (bool,)}
    for key, value in args.items():
        declared = (properties.get(key) or {}).get("type")
        expected = _EXPECTED.get(declared)
        if expected and (not isinstance(value, expected) or
                         (declared in {"integer", "number"} and isinstance(value, bool))):
            return {
                "status": "error",
                "summary": (
                    f"Argument '{key}' of {name} must be a JSON {declared}, "
                    f"got {type(value).__name__}."
                ),
                "data": None,
                "suggestions": [
                    f"Pass '{key}' as a real JSON {declared}, not as a quoted string.",
                    f"Omit '{key}' to let {name} use the data already collected in this run.",
                ],
            }
        enum = (properties.get(key) or {}).get("enum")
        if enum is not None and value not in enum:
            return {"status": "error", "summary": f"Argument '{key}' of {name} must be one of {enum}.",
                    "data": None, "suggestions": ["Retry with a permitted value."]}

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

def _search_key(tool_name: str, args: dict) -> str | None:
    info = _TOOLS.get(tool_name)
    if info is None or info["category"] not in _SEARCHABLE_CATEGORIES or not isinstance(args, dict):
        return None
    props = info["schema"]["function"]["parameters"].get("properties", {})
    canonical = {key: schema["default"] for key, schema in props.items() if "default" in schema}
    canonical.update(_coerce_arg_types(args, props))
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)


def check_search_duplicate(tool_name: str, args: dict, threshold: float = 0.90, *, history=None) -> dict | None:
    """Recognize an identical successful request; ``threshold`` is legacy."""
    key = _search_key(tool_name, args)
    if key is None:
        return None
    with _SEARCH_HISTORY_LOCK:
        for entry in (_SEARCH_HISTORY if history is None else history):
            if entry["tool"] == tool_name and entry.get("key") == key:
                return {
                    "status": "partial",
                    "summary": "Identical completed search already executed in this run.",
                    "data": entry.get("result_summary"),
                    "suggestions": [
                        "Use different keywords or a different search axis.",
                        "Try searching by gene name, pathway, or protein fold instead.",
                    ],
                    "deduplicated": True,
                }
            if tool_name == "search_uniprot" and entry.get("tool") == tool_name:
                prior = entry.get("args") or {}
                prior_result = entry.get("result_summary") or {}
                gene = str(args.get("gene_name") or "").strip().casefold()
                old_gene = str(prior.get("gene_name") or "").strip().casefold()
                def terms(value):
                    value = str(value or "").strip()
                    if not value:
                        return set()
                    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)*", value):
                        return None
                    words = value.casefold().split()
                    return None if any(w in {"and", "or", "not"} for w in words) else set(words)
                old, new = terms(prior.get("function_filter")), terms(args.get("function_filter"))
                if (gene and gene == old_gene and prior_result.get("search_completed") is True
                        and prior_result.get("count") == 0 and old is not None and new is not None
                        and old.issubset(new)):
                    return {"status": "partial", "data": None, "deduplicated": True,
                            "summary": "A completed search for this gene with the same or fewer filter terms returned no records.",
                            "suggestions": ["Broaden the filter or search another source; a zero result does not prove biological absence."]}
    return None


def record_search(tool_name: str, args: dict, result_summary: Any = None, *, history=None) -> None:
    """Record a search execution for future deduplication checks."""
    key = _search_key(tool_name, args)
    completed_empty_uniprot = (tool_name == "search_uniprot" and isinstance(result_summary, dict)
                               and result_summary.get("search_completed") is True
                               and result_summary.get("status") in {"success", "partial"}
                               and result_summary.get("count") == 0)
    if key is None or not isinstance(result_summary, dict) or (result_summary.get("status") != "success" and not completed_empty_uniprot):
        return
    with _SEARCH_HISTORY_LOCK:
        (_SEARCH_HISTORY if history is None else history).append({
            "tool": tool_name,
            "key": key,
            "args": args,
            "result_summary": _compact_summary(result_summary),
        })


def clear_search_history(*, history=None) -> None:
    """Reset the search deduplication history."""
    with _SEARCH_HISTORY_LOCK:
        (_SEARCH_HISTORY if history is None else history).clear()


def _normalize_query(args: dict) -> str:
    """Extract and normalize the query string from tool arguments."""
    # Try common query parameter names
    for key in ("doi", "pmid", "query", "ec_number", "uniprot_accession", "substrate_smiles"):
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
            "search_completed": result.get("search_completed"),
        }
    if isinstance(result, list):
        return {"count": len(result)}
    return None
