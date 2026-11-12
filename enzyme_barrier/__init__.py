"""Compatibility imports for structural_modeling_agent.barrier.

The implementation is owned by the structural-modeling agent.
"""
from structural_modeling_agent.barrier import ReactionSpec, AtomRef, BondChange
from structural_modeling_agent.barrier import __all__


def __getattr__(name):
    from structural_modeling_agent import barrier
    return getattr(barrier, name)
