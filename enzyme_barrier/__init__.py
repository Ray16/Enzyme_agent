"""Reviewed chemical setups, embedded reaction pilots, and gated free energies.

Legacy cluster scans remain exploratory. See README.md for implementation scope,
validation evidence, and the outstanding AS9/GlnA3 method-validation work.
"""
from .spec import ReactionSpec, AtomRef, BondChange
__all__ = ["ReactionSpec", "AtomRef", "BondChange", "carve", "scan_barrier"]

def __getattr__(name):            # lazy: scan pulls in fairchem/ase, keep carve import light
    if name == "carve":
        from .carve import carve
        return carve
    if name == "scan_barrier":
        from .scan import scan_barrier
        return scan_barrier
    raise AttributeError(name)
