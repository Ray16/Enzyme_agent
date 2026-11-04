"""Saturation mutagenesis: redesign an existing scaffold position by position.

The counterpart to de novo generation.  Where the generative tools build a
backbone when no scaffold holds the geometry, this takes a scaffold that already
binds and asks which single substitutions in the pocket change the chemistry.

Three stages, deliberately separable because they fail and evolve at different
rates:

``pocket``   geometry -> which positions line the bound complex
``variants`` positions -> a saturation library with stable, mergeable ids
``results``  predicted structures -> a scored table, metric supplied by
             ``structural_modeling_agent.proxies`` rather than fixed here
"""
from .pocket import Pocket, Position, define_pocket  # noqa: F401
from .variants import AA20, Variant, build_variants, mutate  # noqa: F401

__all__ = [
    "define_pocket", "Pocket", "Position",
    "build_variants", "Variant", "mutate", "AA20",
]
