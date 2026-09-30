"""Rules for mechanism-specific structural-proxy interpretation.

This module holds *rules*, not enzymes. Two general rules do the work:

RULE 1 -- which phosphorus is attacked follows from the reaction's nucleotide
products, because that is what the transfer chemistry *is*:

    ... + ATP = ... + AMP + diphosphate   ->  the nucleophile attacked P-alpha
                                              (adenylate; leaves along Pa--O3A; 1 Mg2+)
    ... + ATP = ... + ADP + phosphate     ->  the nucleophile attacked P-gamma
                                              (acyl-phosphate; leaves along Pg--O3B; 2 Mg2+)

  Read it from the annotated equation (UniProt/Rhea) per enzyme. This is the same
  principle ``cofactor_stoichiometry.py`` applies to metal count: per-enzyme
  annotation, never a mechanism table keyed on family names. Inferring mechanism
  from a family name is unsound -- it previously assigned an adenylate-forming
  ligase to the acyl-phosphate coordinate and scored the wrong phosphorus.

RULE 2 -- an angle is only a calibrated activity signal inside the regime the
assay panel actually covered. Everything else is a mechanistic transfer:
``evidence_scope`` records which, and transfers must never be reported as
validated activity models.

Family-name keywords are a *fallback* for enzymes with no annotated equation.
They are data, not policy, and live in ``knowledge/family_mechanism_fallback.json``
so adding one needs no code change and carries no implied validation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROXY_VERSION = "step1-angle-v2"
MIN_GATED_POSES = 5
COMPATIBLE_ANGLE = 150.0
STRONG_ANGLE = 165.0

_FALLBACK_PATH = Path(__file__).resolve().parent / "knowledge" / "family_mechanism_fallback.json"


@dataclass(frozen=True)
class ProxyMechanism:
    mechanism: str
    phosphate: str
    electrophile: str
    leaving_axis: str
    family_group: str
    evidence_scope: str
    # Catalytic Mg2+ per active site, carried with the mechanism so cofold setup
    # and geometry scoring cannot disagree about stoichiometry. They previously
    # drifted because the count lived only in prose.
    metal_count: int = 0


# RULE 1, as data: the nucleotide-product signature of each transfer coordinate.
# Keyed on chemistry, not on any enzyme, family or pathway.
_COORDINATES = {
    "adenylate": {
        "mechanism": "acyl-adenylate",
        "phosphate": "PA",
        "electrophile": "ATP P-alpha",
        "leaving_axis": "P-alpha--O3A",
        "metal_count": 1,
        # AMP released with diphosphate: the nucleophile displaced PPi at P-alpha.
        "requires": (re.compile(r"\bAMP\b", re.I), re.compile(r"diphosphate", re.I)),
        "excludes": (re.compile(r"\bADP\b", re.I),),
    },
    "acyl_phosphate": {
        "mechanism": "acyl-phosphate",
        "phosphate": "PG",
        "electrophile": "ATP P-gamma",
        "leaving_axis": "P-gamma--O3B",
        "metal_count": 2,
        # ADP released: the nucleophile displaced ADP at P-gamma.
        "requires": (re.compile(r"\bADP\b", re.I),),
        "excludes": (re.compile(r"\bAMP\b", re.I),),
    },
}


def _build(coordinate: str, family_group: str, evidence_scope: str) -> ProxyMechanism:
    spec = _COORDINATES[coordinate]
    return ProxyMechanism(
        spec["mechanism"], spec["phosphate"], spec["electrophile"],
        spec["leaving_axis"], family_group, evidence_scope, spec["metal_count"],
    )


def mechanism_from_reaction(catalytic_activities, *, calibrated: bool = False):
    """RULE 1: resolve the coordinate from an annotated reaction equation.

    ``catalytic_activities`` is a Rhea/UniProt equation string or a list of them.
    Abstains (returns None) when no equation is given, when none matches, or when
    an equation names both nucleotide products -- ambiguity is not a licence to
    guess. ``calibrated`` marks the candidate as inside the assay-validated
    regime; callers that cannot establish that must leave it False.
    """
    if isinstance(catalytic_activities, str):
        catalytic_activities = [catalytic_activities]
    text = " ".join(str(a) for a in (catalytic_activities or []) if a)
    if not text.strip():
        return None
    scope = "panel_validated" if calibrated else "reaction_derived"
    matches = [
        name for name, spec in _COORDINATES.items()
        if all(p.search(text) for p in spec["requires"])
        and not any(p.search(text) for p in spec["excludes"])
    ]
    if len(matches) != 1:
        return None
    return _build(matches[0], f"reaction:{_COORDINATES[matches[0]]['phosphate']}", scope)


@lru_cache(maxsize=1)
def _fallback_entries() -> tuple[dict, ...]:
    """Family-name keyword fallback, loaded from data (absent file -> no fallback)."""
    try:
        with _FALLBACK_PATH.open() as handle:
            return tuple(json.load(handle).get("entries", []))
    except (OSError, ValueError):
        return ()


def proxy_mechanism_for_family(family: str | None) -> ProxyMechanism | None:
    """Fallback coordinate from a family name; abstains for unknown families.

    Weaker than :func:`mechanism_from_reaction` and only for enzymes lacking an
    annotated equation: a family name does not determine transfer chemistry.
    """
    normalized = (family or "").strip().lower()
    if not normalized:
        return None
    for entry in _fallback_entries():
        if any(str(k).lower() in normalized for k in entry.get("keywords", ())):
            return _build(
                entry["coordinate"],
                entry.get("family_group", "unnamed group"),
                "panel_validated" if entry.get("calibrated") else "mechanistically_transferred",
            )
    return None


def resolve_mechanism(family: str | None = None, catalytic_activities=None,
                      *, calibrated: bool = False):
    """Best available coordinate and the basis it rests on.

    Returns ``(ProxyMechanism | None, basis)`` with basis ``"reaction"``,
    ``"family"`` or ``"unresolved"``. The annotated reaction wins; a family name
    is consulted only when no equation resolves.
    """
    from_reaction = mechanism_from_reaction(catalytic_activities, calibrated=calibrated)
    if from_reaction is not None:
        return from_reaction, "reaction"
    from_family = proxy_mechanism_for_family(family)
    if from_family is not None:
        return from_family, "family"
    return None, "unresolved"


def structural_support_tier(
    angle: float | None,
    n_gated: int,
    mechanism_status: str,
) -> str:
    """RULE 2 applied to an eligible mean: abstention is not failure."""
    if mechanism_status != "assigned" or n_gated < MIN_GATED_POSES or angle is None:
        return "inconclusive"
    if angle >= STRONG_ANGLE:
        return "strong geometric support"
    if angle >= COMPATIBLE_ANGLE:
        return "compatible"
    return "weak geometric support"
