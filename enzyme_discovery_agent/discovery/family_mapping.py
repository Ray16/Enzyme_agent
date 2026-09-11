"""Resolve mechanism-family hypotheses to surveyable InterPro entries.

Text search proposes database mappings; it does not by itself prove that an
InterPro entry represents the hypothesized catalytic family. Returned mappings
therefore carry explicit lexical evidence and a separate validation flag.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypedDict

from enzyme_discovery_agent.tools.family_reactions import (
    _LIST_PAGE_MAX,
    _reviewed_member_count,
)


class FamilyFoldMapping(TypedDict, total=False):
    """A ranked, auditable family-hypothesis to InterPro mapping."""

    fold_id: str
    lexical_match: int
    matched_term: str
    mapping_validated: bool
    source: str
    mapping_rank: int
    is_primary: bool
    mapping_score: int
    entry_name: str
    entry_type: str
    compatibility_evidence: list[str]
    incompatibility_evidence: list[str]


InterProSearch = Callable[..., dict[str, Any] | list[dict[str, Any]]]

_SURVEYABLE_ENTRY_TYPES = frozenset({"family", "domain"})
_FALLBACK_ENTRY_TYPES = frozenset({"homologous_superfamily"})
_GENERIC_FAMILY_TOKENS = frozenset({
    "atp", "dependent", "fold", "domain", "family", "superfamily",
    "amide", "enzyme", "like",
    "independent", "type", "carboxylate", "amine",
})

_ACTION_EQUIVALENCE = {
    "synthetase": "bond_formation",
    "synthase": "bond_formation",
    "ligase": "bond_formation",
    "transferase": "group_transfer",
    "amidotransferase": "group_transfer",
    "reductase": "redox",
    "dehydrogenase": "redox",
    "oxidase": "redox",
    "hydrolase": "hydrolysis",
    "peptidase": "hydrolysis",
    "couple": "bond_formation",
    "couples": "bond_formation",
    "coupling": "bond_formation",
    "form": "bond_formation",
    "forms": "bond_formation",
    "formation": "bond_formation",
    "aminolysis": "bond_formation",
    "reduction": "redox",
    "oxidation": "redox",
    "hydrolysis": "hydrolysis",
}


def _action_classes(text: str) -> set[str]:
    """Extract broad catalytic actions without campaign-specific vocabulary."""
    tokens = set(re.findall(r"[a-z][a-z0-9]{2,}", text.casefold()))
    actions = {
        action for token in tokens
        if (action := _ACTION_EQUIVALENCE.get(token)) is not None
    }
    suffix_actions = (
        ("transferase", "group_transfer"),
        ("reductase", "redox"),
        ("dehydrogenase", "redox"),
        ("oxidase", "redox"),
        ("hydrolase", "hydrolysis"),
        ("peptidase", "hydrolysis"),
    )
    actions.update(
        action for token in tokens for suffix, action in suffix_actions
        if token.endswith(suffix)
    )
    return actions

def _distinctive_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z][a-z0-9]{2,}", text.casefold())
        if token not in _GENERIC_FAMILY_TOKENS
    }


def _token_matches(tokens: set[str], text: str) -> int:
    normalized = text.casefold()
    return sum(
        bool(re.search(rf"\b{re.escape(token)}\b", normalized))
        for token in tokens
    )


def _mapping_evidence(
    family: dict[str, Any], entry_text: str, entry_type: str,
) -> tuple[int, list[str], list[str]]:
    """Score an entry using only the current run's hypothesis predictions."""
    normalized_entry = " ".join(entry_text.casefold().replace("-", " ").split())
    name = str(family.get("name") or "")
    descriptors = [
        str(value) for value in (
            family.get("descriptors") or family.get("search_descriptors") or []
        )
    ]
    identity_tokens = _distinctive_tokens(" ".join([name, *descriptors]))
    mechanism_text = " ".join([
        str(family.get("required_catalytic_capability") or ""),
        str(family.get("proposed_activation_strategy") or ""),
        str(family.get("native_reaction") or family.get("predicted_native_reaction") or ""),
    ])
    mechanism_tokens = _distinctive_tokens(mechanism_text)
    hypothesis_actions = _action_classes(mechanism_text)
    if not hypothesis_actions:
        hypothesis_actions = _action_classes(" ".join([name, *descriptors]))
    entry_actions = _action_classes(normalized_entry)

    identity_matches = _token_matches(identity_tokens, normalized_entry)
    mechanism_matches = _token_matches(mechanism_tokens, normalized_entry)
    evidence: list[str] = []
    contradictions: list[str] = []
    score = 4 * identity_matches + min(mechanism_matches, 4)

    normalized_name = " ".join(name.casefold().replace("-", " ").split())
    if normalized_name and normalized_name in normalized_entry:
        score += 8
        evidence.append("family_name_phrase")
    for descriptor in descriptors:
        phrase = " ".join(descriptor.casefold().replace("-", " ").split())
        if len(phrase.split()) >= 2 and phrase in normalized_entry:
            score += 4
            evidence.append("descriptor_phrase")
            break
    if identity_matches:
        evidence.append(f"identity_tokens:{identity_matches}")
    if mechanism_matches:
        evidence.append(f"mechanism_tokens:{mechanism_matches}")
    if entry_type == "family":
        evidence.append("functional_family_entry")

    compatible_actions = hypothesis_actions & entry_actions
    conflicting_actions = entry_actions - hypothesis_actions
    if compatible_actions:
        score += 5
        evidence.append("compatible_catalytic_action")
    if hypothesis_actions and conflicting_actions:
        score -= 10
        contradictions.append("conflicting_catalytic_action")

    disqualifying_text = " ".join(
        str(value) for value in family.get("disqualifying_observations") or []
    )
    disqualifying_tokens = _distinctive_tokens(disqualifying_text)
    # Disqualifying observations usually restate the desired reaction (for
    # example, "lacks <required capability>"). Remove all positive hypothesis
    # vocabulary so only contradiction-specific annotation terms count against
    # a mapping.
    disqualifying_tokens -= identity_tokens | mechanism_tokens
    disqualifying_matches = _token_matches(disqualifying_tokens, normalized_entry)
    if disqualifying_matches:
        score -= min(12, 4 * disqualifying_matches)
        contradictions.append(
            f"hypothesis_disqualifier_tokens:{disqualifying_matches}"
        )

    return score, evidence, contradictions


def resolve_family_fold_candidates(
    family: dict[str, Any],
    count_cache: dict[str, int],
    search_interpro: InterProSearch,
    max_folds: int = 3,
) -> list[FamilyFoldMapping]:
    """Return ranked InterPro mappings for one family hypothesis.

    Functional family/domain entries are preferred. A broad homologous
    superfamily is retained only when no coherent entry can be resolved.
    Multiple mappings remain visible because text resolution is ambiguous and a
    failed primary mapping must not erase a plausible alternate.
    """
    name = str(family.get("name") or "")
    descriptors = [
        str(value) for value in (
            family.get("descriptors") or family.get("search_descriptors") or []
        )
    ]
    name_tokens = _distinctive_tokens(name)
    hypothesis_tokens = _distinctive_tokens(" ".join([name, *descriptors]))

    def surveyable_entries(
        term: str,
        *,
        allow_broad: bool = False,
    ) -> list[dict[str, Any]]:
        term = term.strip()
        if not term:
            return []
        try:
            result = search_interpro(term, max_results=20)
        except Exception:  # noqa: BLE001 - one search term must not sink resolution
            return []

        entries = result.get("data") if isinstance(result, dict) else result
        allowed_types = set(_SURVEYABLE_ENTRY_TYPES)
        if allow_broad:
            allowed_types.update(_FALLBACK_ENTRY_TYPES)

        matches: list[dict[str, Any]] = []
        for entry in entries or []:
            accession = str(entry.get("accession") or entry.get("id") or "")
            entry_type = str(entry.get("type") or "").lower()
            if not accession.upper().startswith("IPR") or entry_type not in allowed_types:
                continue

            member_count = count_cache.get(accession)
            if member_count is None:
                try:
                    member_count = _reviewed_member_count(accession)
                except Exception:  # noqa: BLE001 - unavailable counts are not fatal
                    member_count = 0
                count_cache[accession] = member_count

            # A family/domain entry whose reviewed membership saturates the cheap
            # count probe pools a whole superfamily of unrelated functions; it is
            # a broad structural bin, not the coherent catalytic family a
            # hypothesis names. Treat such saturated entries like an explicit
            # homologous_superfamily: usable only as a last-resort fallback when
            # nothing narrower resolves, never the primary survey target. This is
            # what keeps a specific hypothesis from mapping onto a generic fold
            # that lexically shares an action word but pools off-target members.
            is_saturated = member_count >= _LIST_PAGE_MAX
            is_broad_fallback = entry_type in _FALLBACK_ENTRY_TYPES or is_saturated
            # A coherent, bounded family/domain mapping remains auditable even
            # when it has few reviewed proteins. Reviewed count controls evidence
            # strength, not whether its complete exact membership may be
            # inventoried; only the saturated/superfamily bins are held back.
            if is_broad_fallback and not allow_broad:
                continue
            entry_text = " ".join([
                str(entry.get("name") or ""),
                str(entry.get("description") or ""),
            ])
            lexical_match = _token_matches(hypothesis_tokens, entry_text)
            name_match = _token_matches(name_tokens, entry_text)
            score, evidence, contradictions = _mapping_evidence(
                family, entry_text, entry_type
            )
            if member_count > 0:
                score += 3
                evidence.append("reviewed_members_available")
            else:
                score -= 3
                contradictions.append("no_reviewed_members")
            matches.append({
                "accession": accession,
                "member_count": member_count,
                "entry_type": entry_type,
                "entry_name": str(entry.get("name") or ""),
                "lexical_match": lexical_match,
                "name_match": name_match,
                "mapping_score": score,
                "compatibility_evidence": evidence,
                "incompatibility_evidence": contradictions,
            })

        return sorted(
            matches,
            key=lambda item: (
                item["entry_type"] in _FALLBACK_ENTRY_TYPES,
                -item["mapping_score"],
                -item["lexical_match"],
                -int(item["member_count"] > 0),
                item["accession"],
            ),
        )

    ranked: dict[str, dict[str, Any]] = {}

    def add_term(term: str, term_rank: int, *, allow_broad: bool = False) -> None:
        for match in surveyable_entries(term, allow_broad=allow_broad):
            accession = match["accession"]
            lexical_match = match["lexical_match"]
            name_match = match["name_match"]
            score = match["mapping_score"]
            rank = (
                score,
                int(match["member_count"] > 0),
                lexical_match,
                -term_rank,
                match["member_count"],
            )
            if accession in ranked and rank <= ranked[accession]["rank"]:
                continue
            ranked[accession] = {
                "fold_id": accession,
                "rank": rank,
                "lexical_match": lexical_match,
                "matched_term": term,
                "mapping_validated": (
                    (
                        bool({"family_name_phrase", "descriptor_phrase"} & set(
                            match["compatibility_evidence"]
                        ))
                        or bool(name_tokens)
                        and name_tokens <= _distinctive_tokens(match["entry_name"])
                    )
                    and not match["incompatibility_evidence"]
                ),
                "mapping_score": score,
                "member_count": match["member_count"],
                "entry_name": match["entry_name"],
                "entry_type": match["entry_type"],
                "compatibility_evidence": match["compatibility_evidence"],
                "incompatibility_evidence": match["incompatibility_evidence"],
            }

    def has_sufficient_mappings() -> bool:
        return sum(
            item["mapping_validated"]
            and item["mapping_score"] >= 6
            and item["member_count"] > 0
            and not item["incompatibility_evidence"]
            for item in ranked.values()
        ) >= max_folds

    # Resolve from the authoritative family name first. Descriptor searches are
    # fallbacks that fill missing mapping slots, not mandatory fan-out after the
    # result is already sufficient. This preserves recall while avoiding dozens
    # of redundant InterPro and UniProt count requests for verbose hypotheses.
    add_term(name, 0)
    for term_rank, term in enumerate(descriptors, start=1):
        if has_sufficient_mappings():
            break
        add_term(term, term_rank)

    if not ranked:
        add_term(name, 0, allow_broad=True)
        for term_rank, term in enumerate(descriptors, start=1):
            if has_sufficient_mappings():
                break
            add_term(term, term_rank, allow_broad=True)

    ordered = sorted(
        ranked.values(),
        key=lambda item: (
            -item["rank"][0],
            -item["rank"][1],
            -item["rank"][2],
            -item["rank"][3],
            -item["rank"][4],
            item["fold_id"],
        ),
    )[:max_folds]
    mappings: list[FamilyFoldMapping] = []
    for mapping_rank, item in enumerate(ordered, start=1):
        mappings.append({
            "fold_id": item["fold_id"],
            "lexical_match": item["lexical_match"],
            "matched_term": item["matched_term"],
            "mapping_validated": item["mapping_validated"],
            "source": "interpro_text_resolution",
            "mapping_rank": mapping_rank,
            "is_primary": mapping_rank == 1,
            "mapping_score": item["mapping_score"],
            "entry_name": item["entry_name"],
            "entry_type": item["entry_type"],
            "compatibility_evidence": item["compatibility_evidence"],
            "incompatibility_evidence": item["incompatibility_evidence"],
        })
    return mappings


def resolve_family_folds(
    family: dict[str, Any],
    count_cache: dict[str, int],
    search_interpro: InterProSearch,
    max_folds: int = 3,
) -> list[str]:
    """Return only fold identifiers for callers using the legacy interface."""
    return [
        mapping["fold_id"]
        for mapping in resolve_family_fold_candidates(
            family,
            count_cache,
            search_interpro,
            max_folds=max_folds,
        )
    ]
