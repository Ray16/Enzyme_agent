"""Deterministic, claim-level validation over retrieved protein records.

Model output and search-result names are hypotheses.  This module converts the
fields of one retrieved UniProt record into atomic verdicts without asking an
LLM to judge another LLM.  Callers can therefore distinguish support,
contradiction, and missing evidence before a family or candidate is promoted.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from enzyme_discovery_agent.chemistry_logic import classify_reaction_chemistry
from enzyme_discovery_agent.models.target_profile import is_opposite_direction

SCHEMA_VERSION = "1.0"
VERDICTS = frozenset({"supported", "contradicted", "insufficient_evidence"})

_CARRIER_RE = re.compile(
    r"\[(?:a|ar)?cp\]|holo-\[|carrier\s+protein|pantetheine|"
    r"peptidyl[- ]carrier|acyl[- ]carrier",
    re.IGNORECASE,
)
_POLYMER_RE = re.compile(r"\(n(?:[+\-]\d)?\)|\bpoly(?:mer|peptide|saccharide)\b", re.IGNORECASE)
_MACROMOLECULE_RE = re.compile(
    r"\b(?:protein|peptide|polypeptide|rna|dna|trna|ribosom|cell wall)\b",
    re.IGNORECASE,
)
_SMALL_MOLECULE_RE = re.compile(
    r"\b(?:atp|adp|amp|gtp|gdp|nadp?h?|fadh?2?|coa|phosphate|pyrophosphate|"
    r"water|ammonia|amine|amino acid|carboxyl(?:ate|ic acid)?|alcohol|aldehyde|"
    r"ketone|sugar|fatty acid)\b|\bCHEBI:\d+\b",
    re.IGNORECASE,
)


def _claim(
    claim_type: str,
    subject: str,
    asserted_value,
    verdict: str,
    *,
    source: str,
    locator: str,
    observed_value=None,
    reason: str,
) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"invalid claim verdict: {verdict}")
    return {
        "claim_type": claim_type,
        "subject": subject,
        "asserted_value": asserted_value,
        "verdict": verdict,
        "evidence": {
            "source": source,
            "locator": locator,
            "observed_value": observed_value,
        },
        "reason": reason,
    }


def uniprot_record_state(entry: dict, expected_accession: str = "") -> dict:
    """Return exact identity and lifecycle facts for a UniProt JSON record."""
    actual = str(entry.get("primaryAccession") or "").strip()
    entry_type = str(entry.get("entryType") or "").strip()
    inactive = entry_type.casefold() == "inactive" or bool(entry.get("inactiveReason"))
    identity_matches = bool(actual) and (
        not expected_accession or actual.casefold() == expected_accession.strip().casefold()
    )
    return {
        "actual_accession": actual,
        "entry_type": entry_type,
        "active": bool(actual) and not inactive,
        "identity_matches": identity_matches,
        "reviewed": "reviewed" in entry_type.casefold() and "unreviewed" not in entry_type.casefold(),
    }


def reaction_topology(reactions: list[str]) -> str | None:
    """Classify topology only when the record text provides a usable signal."""
    text = " ".join(str(r) for r in reactions if str(r).strip())
    if not text:
        return None
    if _CARRIER_RE.search(text):
        return "carrier_protein_tethered"
    if _POLYMER_RE.search(text):
        return "polymer"
    if _MACROMOLECULE_RE.search(text):
        return "macromolecule"
    # Absence of tethering vocabulary is not positive evidence of a free
    # substrate. Require at least one explicit small-molecule identity signal.
    if _SMALL_MOLECULE_RE.search(text):
        return "free_small_molecule"
    return None


def _record_reactions(entry: dict) -> list[str]:
    out = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") != "CATALYTIC ACTIVITY":
            continue
        value = str((comment.get("reaction") or {}).get("name") or "").strip()
        if value:
            out.append(value)
    return out


def _record_ec_numbers(entry: dict) -> list[str]:
    values: list[str] = []
    description = entry.get("proteinDescription") or {}
    names = [description.get("recommendedName") or {}]
    names.extend(description.get("alternativeNames") or [])
    names.extend(description.get("submissionNames") or [])
    for name in names:
        for item in name.get("ecNumbers") or []:
            value = str(item.get("value") or "").strip()
            if value and value not in values:
                values.append(value)
    return values


def _record_interpro_ids(entry: dict) -> list[str]:
    return [
        str(x.get("id")) for x in entry.get("uniProtKBCrossReferences", [])
        if x.get("database") == "InterPro" and x.get("id")
    ]


def validate_uniprot_claims(
    entry: dict,
    *,
    expected_accession: str = "",
    expected_fold_id: str = "",
    target_direction: str | None = None,
    target_bond_types: set[str] | None = None,
    target_topology: str | None = None,
    require_reviewed: bool = False,
    require_native_reaction: bool = True,
) -> dict:
    """Validate promotion-relevant claims against one retrieved UniProt record.

    Required claims fail closed.  A missing field is ``insufficient_evidence``;
    it is never treated as support.  ``promotion_eligible`` means every required
    claim is supported, not merely that no contradiction was noticed.
    """
    subject = expected_accession or str(entry.get("primaryAccession") or "unknown")
    source = "UniProtKB"
    state = uniprot_record_state(entry, expected_accession)
    claims: list[dict] = []

    if not state["actual_accession"]:
        identity_verdict = "insufficient_evidence"
        identity_reason = "record has no primary accession"
    elif not state["identity_matches"]:
        identity_verdict = "contradicted"
        identity_reason = "retrieved primary accession does not match the asserted accession"
    else:
        identity_verdict = "supported"
        identity_reason = "retrieved primary accession matches exactly"
    claims.append(_claim(
        "accession_identity", subject, expected_accession or subject, identity_verdict,
        source=source, locator="primaryAccession", observed_value=state["actual_accession"] or None,
        reason=identity_reason,
    ))

    claims.append(_claim(
        "record_active", subject, True,
        "supported" if state["active"] else "contradicted",
        source=source, locator="entryType/inactiveReason",
        observed_value={"entryType": state["entry_type"],
                        "inactiveReason": entry.get("inactiveReason")},
        reason=("record is active" if state["active"] else "record is inactive or deleted"),
    ))

    claims.append(_claim(
        "reviewed_status", subject, True,
        "supported" if state["reviewed"] else (
            "contradicted" if state["entry_type"] else "insufficient_evidence"),
        source=source, locator="entryType", observed_value=state["entry_type"] or None,
        reason=("record is reviewed" if state["reviewed"] else
                "record is not reviewed" if state["entry_type"] else "record type is absent"),
    ))

    fold_ids = _record_interpro_ids(entry)
    if expected_fold_id:
        fold_verdict = "supported" if expected_fold_id in fold_ids else (
            "contradicted" if fold_ids else "insufficient_evidence")
        claims.append(_claim(
            "family_membership", subject, expected_fold_id, fold_verdict,
            source=source, locator="uniProtKBCrossReferences[database=InterPro]",
            observed_value=fold_ids,
            reason=("record explicitly cross-references the asserted family" if fold_verdict == "supported"
                    else "record cross-references different families" if fold_verdict == "contradicted"
                    else "record has no InterPro family evidence"),
        ))

    reactions = _record_reactions(entry)
    claims.append(_claim(
        "native_reaction", subject, "recorded catalytic activity",
        "supported" if reactions else "insufficient_evidence",
        source=source, locator="comments[CATALYTIC ACTIVITY].reaction.name",
        observed_value=reactions,
        reason=("record contains an explicit catalytic reaction" if reactions
                else "record contains no catalytic-activity reaction"),
    ))

    chemistry = classify_reaction_chemistry(_record_ec_numbers(entry), reactions)
    observed_direction = chemistry.get("reaction_direction")
    if target_direction:
        if observed_direction == target_direction:
            verdict, reason = "supported", "recorded reaction direction matches the target"
        elif is_opposite_direction(observed_direction, target_direction):
            verdict, reason = "contradicted", "recorded reaction direction is opposite the target"
        else:
            verdict, reason = "insufficient_evidence", "record does not establish the target direction"
        claims.append(_claim(
            "reaction_direction", subject, target_direction, verdict,
            source=source, locator="proteinDescription.*.ecNumbers + comments[CATALYTIC ACTIVITY]",
            observed_value=observed_direction, reason=reason,
        ))

    wanted_bonds = set(target_bond_types or set())
    observed_bonds = set(chemistry.get("bond_types") or [])
    if wanted_bonds:
        if observed_bonds & wanted_bonds:
            verdict, reason = "supported", "recorded reaction classification shares a target bond type"
        elif observed_bonds:
            verdict, reason = "contradicted", "recorded reaction classification has a different bond type"
        else:
            verdict, reason = "insufficient_evidence", "record does not establish a bond type"
        claims.append(_claim(
            "bond_type", subject, sorted(wanted_bonds), verdict,
            source=source, locator="proteinDescription.*.ecNumbers + comments[CATALYTIC ACTIVITY]",
            observed_value=sorted(observed_bonds), reason=reason,
        ))

    observed_topology = reaction_topology(reactions)
    if target_topology:
        if observed_topology == target_topology:
            verdict, reason = "supported", "recorded reaction uses the required substrate topology"
        elif observed_topology:
            verdict, reason = "contradicted", "recorded reaction uses an incompatible substrate topology"
        else:
            verdict, reason = "insufficient_evidence", "record does not establish substrate topology"
        claims.append(_claim(
            "substrate_topology", subject, target_topology, verdict,
            source=source, locator="comments[CATALYTIC ACTIVITY].reaction.name",
            observed_value=observed_topology, reason=reason,
        ))

    required = {"accession_identity", "record_active"}
    if require_native_reaction:
        required.add("native_reaction")
    if require_reviewed:
        required.add("reviewed_status")
    if expected_fold_id:
        required.add("family_membership")
    if target_direction:
        required.add("reaction_direction")
    if wanted_bonds:
        required.add("bond_type")
    if target_topology:
        required.add("substrate_topology")
    by_type = {c["claim_type"]: c for c in claims}
    blocking = [by_type[name] for name in sorted(required)
                if name not in by_type or by_type[name]["verdict"] != "supported"]
    return {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "claims": claims,
        "required_claims": sorted(required),
        "promotion_eligible": not blocking,
        "blocking_claims": [c["claim_type"] for c in blocking],
    }
