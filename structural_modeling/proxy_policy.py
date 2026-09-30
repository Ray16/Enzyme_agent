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
_ANL_TRANSFER_KEYS = ("iuca", "iucc", "aerobactin")
_PG_GROUPS = (
    (("glutamine synthetase", "gamma-glutamylputrescine"), "Glutamine synthetase"),
    (("rimk", "lysx", "tyramine--l-glutamate"), "RimK/bS6"),
    (("dapdiamide", "fumarate--(s)-2,3- diaminopropanoate"), "Dapdiamide"),
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


def proxy_mechanism_for_family(family: str | None) -> ProxyMechanism | None:
    """Return the supported reaction coordinate, or abstain for unknown families."""
    normalized = (family or "").strip().lower()
    if any(key in normalized for key in _ANL_VALIDATED_KEYS):
        return ProxyMechanism(
            "acyl-adenylate",
            "PA",
            "ATP P-alpha",
            "P-alpha--O3A",
            "ANL/AMP-binding",
            "panel_validated",
        )
    if any(key in normalized for key in _ANL_TRANSFER_KEYS):
        return ProxyMechanism(
            "acyl-adenylate",
            "PA",
            "ATP P-alpha",
            "P-alpha--O3A",
            "Aerobactin NIS",
            "mechanistically_transferred",
        )
    for keys, group in _PG_GROUPS:
        if any(key in normalized for key in keys):
            return ProxyMechanism(
                "acyl-phosphate",
                "PG",
                "ATP P-gamma",
                "P-gamma--O3B",
                group,
                "mechanistically_transferred",
            )
    return None


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
