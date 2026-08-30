"""Falsifiable family-hypothesis records and their retrieval verdicts.

A family the model proposes is not "plausible because the model named it": it is
a HYPOTHESIS with predictions about what a genuine member's native reaction must
show and what observation would rule the family out.  Retrieval then tests those
predictions against reviewed database records via the atomic claim validator
(``models.claim_validation``), and this module turns the member verdicts into a
family-level verdict.

Crucially, absence of evidence is ``insufficient_family_evidence`` — never
silently "feasible".  Such a family may remain an exploration lead but is never
promoted as an evidence-backed route.
"""

from __future__ import annotations

ROLES = frozenset({"direct_catalysis", "substrate_activation", "adjacent"})

FAMILY_VERDICTS = frozenset({
    "supported_family_hypothesis",
    "contradicted_family_hypothesis",
    "insufficient_family_evidence",
})


def _terms(value) -> list[str]:
    if not isinstance(value, list):
        if value is None:
            return []
        value = [value]
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def normalize_family_hypothesis(fam: dict | None) -> dict:
    """Coerce a derived family dict into a first-class falsifiable record.

    Tolerant of missing fields (a reasoning model may omit some under a token
    cap) so a partial derivation is never dropped; it derives what it safely can
    from the fields that are present.
    """
    fam = fam or {}
    substrates_free = fam.get("substrates_free")
    predicted_topology = fam.get("predicted_substrate_topology")
    if not predicted_topology and substrates_free is True:
        # A family that natively acts on free small molecules predicts exactly
        # that topology for a genuine member.
        predicted_topology = "free_small_molecule"
    role = str(fam.get("role") or "direct_catalysis").strip()
    if role not in ROLES:
        role = "direct_catalysis"
    return {
        "name": str(fam.get("name") or "").strip(),
        "validation_status": fam.get("validation_status", "hypothesis"),
        "evidence_basis": fam.get(
            "evidence_basis", "model_reasoning_from_target_chemistry"),
        # The causal chain the hypothesis asserts.
        "target_constraint_addressed": _terms(fam.get("target_constraint_addressed")),
        "required_catalytic_capability":
            str(fam.get("required_catalytic_capability") or "").strip(),
        "proposed_activation_strategy":
            str(fam.get("proposed_activation_strategy") or "").strip(),
        "search_descriptors": _terms(fam.get("descriptors")),
        # Predictions a genuine member's record must satisfy.
        "predicted_native_reaction": str(fam.get("native_reaction") or "").strip(),
        "predicted_native_direction": str(fam.get("native_direction") or "").strip(),
        "predicted_substrate_topology": predicted_topology or None,
        "substrates_free": substrates_free,
        # What would falsify the hypothesis.
        "disqualifying_observations": _terms(fam.get("disqualifying_observations")),
        "role": role,
    }


def family_verdict(reviewed_member_validations: list[dict] | None) -> str:
    """Aggregate reviewed-member claim validations into a family-level verdict.

    ``supported_family_hypothesis``  – at least one reviewed member of the exact
        family is promotion-eligible (every required claim supported), i.e. the
        predicted reaction pattern is genuinely realized.
    ``contradicted_family_hypothesis`` – no member is eligible AND at least one
        reviewed member's record contradicts a required claim.
    ``insufficient_family_evidence`` – neither support nor contradiction was
        established (including the no-reviewed-members case).  This is NOT
        feasibility.
    """
    validations = [v for v in (reviewed_member_validations or []) if v]
    eligible = sum(v.get("promotion_eligible") is True for v in validations)
    has_contradiction = any(
        any(c.get("verdict") == "contradicted" for c in v.get("claims") or [])
        for v in validations
    )
    if eligible:
        return "supported_family_hypothesis"
    if validations and has_contradiction:
        return "contradicted_family_hypothesis"
    return "insufficient_family_evidence"
