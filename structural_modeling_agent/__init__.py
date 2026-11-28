"""Independent structural-modeling agent and its scientific tools."""
from __future__ import annotations

__all__ = ["StructuralModelingAgent", "ModelingAgent"]


def __getattr__(name: str):
    if name in ("StructuralModelingAgent", "ModelingAgent"):
        from .agent import ModelingAgent

        return ModelingAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
