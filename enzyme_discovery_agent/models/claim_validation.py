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
# Polymer/oligomer signal. Rhea writes a repeating chain both as "(n)" and with a
# bracketed subscript, e.g. "[monomer]n + H2O = [monomer]n-x + [monomer]x", so
# recognize the "]n" / "]n-x" / "]n+1" subscript form as well as "oligomer".
_POLYMER_RE = re.compile(
    r"\(n(?:[+\-]\d)?\)|\]\s*n(?:[+\-]\w+)?|\boligomer\b|"
    r"\bpoly(?:mer|peptide|saccharide)\b",
    re.IGNORECASE,
)
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


def topology_compatible(observed: str | None, target: str | None) -> bool:
    """True if an enzyme with ``observed`` native substrate topology can act on a
    ``target`` of the given topology. Directional, not strict equality:

    - Carrier-tethered is a distinct machinery requirement -> must match exactly.
    - A GENUINE polymer/macromolecule target needs a polymer/macromolecule-native
      enzyme (a monomer-only enzyme cannot reach it) -- this preserves the guard.
    - Any other target (free small molecule, mixture, or a discrete small-molecule
      PROXY standing in for a polymer) is accommodated by any non-tethered enzyme,
      so a polymer/oligomer-native enzyme is never wrongly excluded against a proxy.
    """
    if observed == target:
        return True
    if "carrier_protein_tethered" in (observed, target):
        return False
    if target in ("polymer", "macromolecule"):
        return observed in ("polymer", "macromolecule")
    return True


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


def _quote_reaction_support(evidence: dict, direction: str | None, bonds: set[str]) -> bool:
    """Require an explicit, unqualified candidate-specific reaction statement.

    This deliberately narrow lexical check is not a general entailment model.
    Ambiguous wording remains pending for independent review.
    """
    from enzyme_discovery_agent.chemistry_logic import extract_target_bond_type

    quote = str(evidence.get("evidence_quote") or "")
    anchors = [str(evidence.get(key) or "").strip() for key in
               ("enzyme_name", "deposited_accession", "accession")]
    uncertain = re.compile(r"\b(?:not|no|never|neither|nor|unable|failed|fails|lack(?:s|ed)?|"
                           r"may|might|could|possibly|hypothesized|hypothesis|hypothetical|"
                           r"predicted|prediction|putative|proposed|suggested|simulation|simulated|"
                           r"whether|compared|unlike|whereas|although)\b", re.I)
    if uncertain.search(quote):
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", quote):
        if not any(anchor and re.search(
            r"(?<!\w)" + re.escape(anchor) + r"(?!\w)\s+"
            r"(?:directly\s+)?(?:cataly[sz]\w*|hydroly[sz]\w*|cleav\w*|"
            r"degrad\w*|ligat\w*|form\w*|synthesi[sz]\w*|transfer\w*|"
            r"oxidi[sz]\w*|reduc\w*)\b", sentence, re.I) for anchor in anchors):
            continue
        parsed = extract_target_bond_type(sentence)
        if direction and parsed.get("direction") != direction:
            continue
        if bonds and not (set(parsed.get("bond_types") or []) & bonds):
            continue
        return True
    return False


def _canonical_bond_tokens(values) -> set[str]:
    """Reduce target bond-type descriptors to the canonical token vocabulary.

    Target profiles can describe bond types as phrases while native reactions
    use canonical tokens. Apply the same tokenizer to both representations.
    Preserve an unrecognized descriptor in lowercase so an unknown bond type
    cannot silently match a different reaction.
    """
    from enzyme_discovery_agent.chemistry_logic import canonical_bond_types
    return canonical_bond_types(values)


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
    literature_evidence: dict | None = None,
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
    from enzyme_discovery_agent.models.native_annotation import summarize_native_annotation
    claims[-1]['evidence']['annotation_provenance'] = summarize_native_annotation(entry)

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

    wanted_bonds = _canonical_bond_tokens(target_bond_types)
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
            verdict, reason = "supported", "recorded reaction and target share the same broad topology category"
        elif observed_topology and topology_compatible(observed_topology, target_topology):
            verdict, reason = "insufficient_evidence", (
                "native and target topologies differ; broad compatibility is possible but "
                "native topology alone does not establish target acceptance")
        elif observed_topology:
            verdict, reason = "contradicted", (
                "recorded reaction's substrate topology cannot accommodate the target "
                "(native substrate is simpler, or a carrier/free mismatch)")
        else:
            verdict, reason = "insufficient_evidence", "record does not establish substrate topology"
        claims.append(_claim(
            "substrate_topology", subject, target_topology, verdict,
            source=source, locator="comments[CATALYTIC ACTIVITY].reaction.name",
            observed_value=observed_topology, reason=reason,
        ))

    # Primary literature can fill absent function annotations, but cannot erase
    # a contradictory native reaction in the authoritative database record.
    if not reactions and literature_evidence:
        from enzyme_discovery_agent.models.literature_evidence import verify_literature_evidence
        from enzyme_discovery_agent.chemistry_logic import extract_target_bond_type
        if verify_literature_evidence(literature_evidence, subject, entry):
            mechanism = literature_evidence.get("mechanism") or {}
            quote = literature_evidence["evidence_quote"]
            extracted = extract_target_bond_type(quote)
            raw_direction = str(mechanism.get("direction") or "")
            direction = (raw_direction if raw_direction in {"formation", "cleavage", "transfer", "redox"}
                         else extract_target_bond_type(raw_direction).get("direction"))
            raw_bonds = mechanism.get("bond_types") or [mechanism.get("bond_formed")]
            bonds = set()
            for value in raw_bonds if isinstance(raw_bonds, list) else [raw_bonds]:
                if value:
                    bonds.update(extract_target_bond_type(str(value)).get("bond_types") or [])
            # A quote contradicting the extraction invalidates that extraction.
            quote_direction = extracted.get("direction")
            quote_bonds = set(extracted.get("bond_types") or [])
            consistent = not (quote_direction and direction and quote_direction != direction)
            consistent = consistent and not (quote_bonds and bonds and not quote_bonds & bonds)
            consistent = consistent and _quote_reaction_support(literature_evidence, direction, bonds)
            if consistent:
                native_topology = reaction_topology([quote])
                replacements = {
                    "native_reaction": ("supported", quote),
                    "reaction_direction": (
                        "supported" if direction == target_direction else
                        "contradicted" if is_opposite_direction(direction, target_direction) else
                        "insufficient_evidence", direction),
                    "bond_type": (
                        "supported" if bonds & wanted_bonds else
                        "contradicted" if bonds else "insufficient_evidence", sorted(bonds)),
                }
                if native_topology and target_topology:
                    replacements["substrate_topology"] = (
                        "supported" if native_topology == target_topology else
                        "insufficient_evidence" if topology_compatible(native_topology, target_topology)
                        else "contradicted", native_topology)
                for index, claim in enumerate(claims):
                    kind = claim["claim_type"]
                    if kind in replacements and claim["verdict"] == "insufficient_evidence":
                        verdict, observed = replacements[kind]
                        claims[index] = _claim(
                            kind, subject, claim["asserted_value"], verdict,
                            source="primary_literature", locator="evidence_quote",
                            observed_value=observed,
                            reason="candidate-linked source passed a narrow lexical check; expert entailment is not established",
                        )
                        claims[index]["evidence"].update({
                            "paper": literature_evidence["paper"],
                            "text_sha256": literature_evidence["text_sha256"],
                            "quote": quote,
                        })

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
    by_type = {c["claim_type"]: c for c in claims}
    blocking = [by_type[name] for name in sorted(required)
                if name not in by_type or by_type[name]["verdict"] != "supported"]
    # substrate_topology is NOT a "must be supported" required claim: it blocks only on
    # a DEFINITE contradiction (native substrate topology known to be incompatible with
    # the target -- e.g. a monomer-only enzyme for a genuine polymer target, or a
    # carrier/free mismatch). It never blocks on insufficient_evidence, because a native
    # reaction whose topology cannot be parsed (Rhea notation varies) must not be rejected
    # on that basis -- and the target is often a small-molecule PROXY for a polymer with no
    # reliable topology ground truth. The claim is still recorded as advisory evidence.
    topo_claim = by_type.get("substrate_topology")
    if topo_claim is not None and topo_claim["verdict"] == "contradicted":
        blocking.append(topo_claim)
    # Record WHY each blocking claim blocks. A required claim can fail either
    # because the record contradicts the target (a real mismatch) or because the
    # record is simply unannotated (insufficient_evidence) -- the uncharacterised
    # and metagenomic enzymes whose function lives in papers, not UniProt. These
    # need different handling: a contradiction is a genuine negative; an absence is
    # a lead to resolve. promotion_eligible is unchanged here (both still block),
    # but the distinction is now visible so absence can be routed, not buried.
    blocked_by_absence = sorted(c["claim_type"] for c in blocking
                                if c.get("verdict") == "insufficient_evidence")
    blocked_by_contradiction = sorted(c["claim_type"] for c in blocking
                                      if c.get("verdict") == "contradicted")
    return {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "claims": claims,
        "required_claims": sorted(required),
        "promotion_eligible": not blocking,
        "blocking_claims": [c["claim_type"] for c in blocking],
        "blocked_by_absence": blocked_by_absence,
        "blocked_by_contradiction": blocked_by_contradiction,
        # True when nothing is contradicted and the only barrier is missing
        # annotation: a candidate to resolve (literature, homology) rather than reject.
        "blocked_only_by_absence": bool(blocking) and not blocked_by_contradiction,
    }
