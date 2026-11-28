"""Enzyme Discovery Agent — autonomous literature, database, and mechanism discovery."""
from __future__ import annotations

__all__ = ["EnzymeDiscoveryAgent", "EnzymeAgent", "register_builtin_tools"]


def __getattr__(name: str):
    if name in ("EnzymeDiscoveryAgent", "EnzymeAgent"):
        from .agent import EnzymeAgent

        return EnzymeAgent
    if name == "register_builtin_tools":
        from .bootstrap import register_builtin_tools

        return register_builtin_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
