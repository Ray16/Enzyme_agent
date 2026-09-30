"""Shared policy for mechanism-specific structural proxy interpretation.

The amide-synthetase assay panel validates the P-alpha angle proxy only for the
AMP-binding/ANL superfamily. Analogous coordinates in other ligase families are
useful mechanistic hypotheses, but remain transferred rather than calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass


PROXY_VERSION = "step1-angle-v2"
MIN_GATED_POSES = 5
COMPATIBLE_ANGLE = 150.0
STRONG_ANGLE = 165.0


@dataclass(frozen=True)
class ProxyMechanism:
    mechanism: str
    phosphate: str
    electrophile: str
    leaving_axis: str
    family_group: str
    evidence_scope: str
    # Catalytic Mg2+ per active site. Carried HERE so the cofold setup and the
    # geometry scoring can never disagree about the mechanism's metal count --
    # they previously drifted, because the stoichiometry lived only in prose.
    # One Mg for adenylate (Pa) transfer, two for two-metal acyl-phosphate (Pg).
    metal_count: int = 0


_ANL_VALIDATED_KEYS = ("amp-binding", "anl")
# Adenylate-forming families recognised by name. Dapdiamide (DdaG) sits here, NOT
# in the acyl-phosphate groups: its UniProt/Rhea reaction is
#   (S)-2,3-diaminopropanoate + fumarate + ATP = N(3)-fumaroyl-Dap + AMP + diphosphate
# i.e. AMP + PPi, which is Pa attack with one Mg. It was previously grouped with
# the ATP-grasp/acyl-phosphate families on a family-name guess, which scored the
# WRONG phosphorus for it. Prefer mechanism_from_catalytic_activities() when the
# reaction is available -- this name map is only the fallback.
_ANL_TRANSFER_KEYS = (
    "iuca", "iucc", "aerobactin",
    "dapdiamide", "fumarate--(s)-2,3- diaminopropanoate",
)
_PG_GROUPS = (
    (("glutamine synthetase", "gamma-glutamylputrescine"), "Glutamine synthetase"),
    (("rimk", "lysx", "tyramine--l-glutamate"), "RimK/bS6"),
    (
        ("phosphoribosylglycinamide synthetase", "formyltransferase"),
        "PurD/GAR",
    ),
    (
        ("atp-grasp", "pyrrolysine", "l-propargylglycine"),
        "ATP-grasp",
    ),
    (
        ("phosphoribosylformylglycinamidine cyclo-ligase",),
        "phosphoribosylformylglycinamidine cyclo-ligase",
    ),
    (("carbamoyl-phosphate synthase",), "carbamoyl-phosphate synthase"),
)


_PA_TRANSFER_GROUPS = (
    (("iuca", "iucc", "aerobactin"), "Aerobactin NIS"),
    (("dapdiamide", "fumarate--(s)-2,3- diaminopropanoate"), "Dapdiamide"),
)


def _adenylate(family_group: str, evidence_scope: str) -> ProxyMechanism:
    """Pa attack along the Pa--O3A axis, one catalytic Mg2+ (AMP + PPi released)."""
    return ProxyMechanism("acyl-adenylate", "PA", "ATP P-alpha", "P-alpha--O3A",
                          family_group, evidence_scope, 1)


def _acyl_phosphate(family_group: str, evidence_scope: str) -> ProxyMechanism:
    """Pg attack along the Pg--O3B axis, two catalytic Mg2+ (ADP + Pi released)."""
    return ProxyMechanism("acyl-phosphate", "PG", "ATP P-gamma", "P-gamma--O3B",
                          family_group, evidence_scope, 2)


# Rhea half-reaction signatures. Which phosphorus is attacked is a *property of the
# reaction*, so read it from the annotated equation instead of guessing from the
# family name: AMP + diphosphate means the carboxylate attacked Pa; ADP + phosphate
# means it attacked Pg. This is the same principle cofactor_stoichiometry.py uses
# for metal count -- per-enzyme annotation, no mechanism table keyed on names.
_AMP_RE = re.compile(r"\bAMP\b", re.I)
_ADP_RE = re.compile(r"\bADP\b", re.I)


def mechanism_from_catalytic_activities(activities) -> ProxyMechanism | None:
    """Resolve the reaction coordinate from UniProt/Rhea catalytic-activity text.

    Authoritative and per-enzyme, so it is preferred over
    :func:`proxy_mechanism_for_family`. Returns None when the annotation is absent
    or names both nucleotide products (abstain rather than guess).
    """
    if isinstance(activities, str):
        activities = [activities]
    text = " ".join(str(a) for a in (activities or []))
    if not text.strip():
        return None
    amp = bool(_AMP_RE.search(text)) and "diphosphate" in text.lower()
    adp = bool(_ADP_RE.search(text))
    if amp and not adp:
        return _adenylate("Rhea:AMP+PPi", "reaction_derived")
    if adp and not amp:
        return _acyl_phosphate("Rhea:ADP+Pi", "reaction_derived")
    return None


def proxy_mechanism_for_family(family: str | None) -> ProxyMechanism | None:
    """Fallback coordinate from the family name; abstain for unknown families.

    Only a fallback: family names are a weak proxy for mechanism (they once put
    the adenylate-forming Dapdiamide ligase in the acyl-phosphate group). Callers
    holding a catalytic-activity annotation should use
    :func:`mechanism_from_catalytic_activities` first.
    """
    normalized = (family or "").strip().lower()
    if any(key in normalized for key in _ANL_VALIDATED_KEYS):
        return _adenylate("ANL/AMP-binding", "panel_validated")
    for keys, group in _PA_TRANSFER_GROUPS:
        if any(key in normalized for key in keys):
            return _adenylate(group, "mechanistically_transferred")
    for keys, group in _PG_GROUPS:
        if any(key in normalized for key in keys):
            return _acyl_phosphate(group, "mechanistically_transferred")
    return None


def resolve_mechanism(family: str | None = None, catalytic_activities=None):
    """Best available coordinate plus the basis it came from.

    Returns ``(ProxyMechanism | None, basis)`` where basis is ``"reaction"``,
    ``"family"`` or ``"unresolved"``. Reaction annotation wins; the family name is
    consulted only when the reaction is unavailable or ambiguous.
    """
    from_reaction = mechanism_from_catalytic_activities(catalytic_activities)
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
    """Interpret an eligible conditional mean without treating abstention as failure."""
    if mechanism_status != "assigned" or n_gated < MIN_GATED_POSES or angle is None:
        return "inconclusive"
    if angle >= STRONG_ANGLE:
        return "strong geometric support"
    if angle >= COMPATIBLE_ANGLE:
        return "compatible"
    return "weak geometric support"
