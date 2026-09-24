"""Explicit registration of the built-in agent tools.

Tool modules use the :func:`tool` decorator to define their schemas.  Keeping
their imports in this one idempotent function makes registration intentional
and lets callers construct/test the registry without importing ``agent.py``.
"""

from __future__ import annotations

import importlib
import threading

_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAPPED = False

_BUILTIN_TOOL_MODULES = (
    "enzyme_discovery_agent.tools.chemistry",
    "enzyme_discovery_agent.tools.database",
    "enzyme_discovery_agent.tools.kbase",
    "enzyme_discovery_agent.tools.literature",
    "enzyme_discovery_agent.tools.fulltext",
    "enzyme_discovery_agent.tools.reasoning",
    "enzyme_discovery_agent.tools.family_reactions",
    "enzyme_discovery_agent.tools.structural_evidence",
    "enzyme_discovery_agent.tools.structure_io",
    "enzyme_discovery_agent.tools.structural_homolog_search",
    "enzyme_discovery_agent.tools.substrate",
    "enzyme_discovery_agent.tools.measurement_extraction",
    "enzyme_discovery_agent.tools.mechanism_extraction",
    "enzyme_discovery_agent.tools.embedding_map",
    "enzyme_discovery_agent.tools.esm_expansion",
    "enzyme_discovery_agent.models.notebook",
)


def register_builtin_tools() -> None:
    """Register every built-in tool exactly once in a thread-safe way."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return
        for module_name in _BUILTIN_TOOL_MODULES:
            importlib.import_module(module_name)
        _BOOTSTRAPPED = True
