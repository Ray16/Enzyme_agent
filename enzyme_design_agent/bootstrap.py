"""Explicit registration of the design agent's tools.

Mirrors the discovery agent's bootstrap: one idempotent function whose whole
body is a list of module imports, so registration is intentional and a registry
can be built for a test without importing ``agent.py``.

Registration happens at module import via the ``@tool`` decorator, and Python
caches modules, so a tool reachable from both agents' bootstraps is registered
exactly once no matter which runs first.
"""
from __future__ import annotations

import importlib
import threading

_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAPPED = False

#: The design toolkit. Two strategies, deliberately both present in one
#: registry: choosing between them is the agent's job, and it cannot choose
#: what it cannot see.
_DESIGN_TOOL_MODULES = (
    "enzyme_design_agent.tools.alphaprotein_novo",  # generate a scaffold around a catalytic motif
    "enzyme_design_agent.tools.disco_design",       # generate a scaffold around a ligand (motif-free)
    "enzyme_design_agent.tools.genzyme_design",     # generate a pocket and protein from a reaction
    "enzyme_design_agent.tools.rfd4_design",        # all-atom motif/ligand-conditioned co-design
    "enzyme_design_agent.tools.saturation_scan",    # redesign an existing one
    "enzyme_design_agent.tools.saturation_campaign",
)


def register_design_tools() -> None:
    """Register every design tool exactly once, thread-safely."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return
        for module_name in _DESIGN_TOOL_MODULES:
            importlib.import_module(module_name)
        _BOOTSTRAPPED = True


def design_tool_names() -> tuple[str, ...]:
    """Names of the registered design tools.

    Resolved from each tool's defining module rather than from a second
    hand-maintained list, so a tool added to ``_DESIGN_TOOL_MODULES`` reaches
    the agent's schema set without a matching edit here going missing.
    """
    from enzyme_discovery_agent.tool_registry import get_tool_info, get_tool_names

    register_design_tools()
    wanted = set(_DESIGN_TOOL_MODULES)
    names = []
    for name in get_tool_names():
        info = get_tool_info(name) or {}
        func = info.get("func")
        if getattr(func, "__module__", None) in wanted:
            names.append(name)
    return tuple(sorted(names))


def design_tool_schemas() -> list[dict]:
    """OpenAI tool schemas for the design toolkit only.

    The design agent is handed its own tools, not the discovery agent's: a loop
    that can also run literature and database search is a different agent with
    a different failure mode, and mixing them makes "why did it do that"
    unanswerable.
    """
    from enzyme_discovery_agent.tool_registry import get_tool_info

    return [get_tool_info(n)["schema"] for n in design_tool_names()]
