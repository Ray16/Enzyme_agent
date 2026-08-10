"""Answer-agnostic derivation of the enzyme mechanism families for a reaction.

Different structural folds can catalyse the SAME bond-forming chemistry.  If
discovery anchors on whichever family the first good papers happen to name, it
collapses coverage onto that one fold and silently misses the others.  This
module derives — from the reaction TYPE alone (the bond formed, the reacting
groups, the cofactor), never from any specific target enzyme — the distinct
mechanism families capable of the reaction, so the agent can search each one
from the literature before it evaluates.

Firewall: the model lists FAMILIES / FOLDS and mechanism DESCRIPTORS, plus
reaction-anchored literature terms.  It must not name specific example enzymes
or genes — those are the answers the search is meant to find, not seed with.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("enzyme_agent")

_SYSTEM = (
    "You are an enzymology strategist. Given a target reaction, enumerate the "
    "DISTINCT enzyme MECHANISM families / structural folds known to catalyse "
    "that BOND-FORMING chemistry. Reason from the reaction TYPE — the bond "
    "formed, the reacting chemical groups, and the cofactor — NOT from any "
    "specific target molecule or product.\n"
    "Cover the space broadly: different folds that make the same bond each count "
    "as a SEPARATE family, and mechanistically adjacent families (e.g. distinct "
    "ligase/synthetase superfamilies that reach the same product) must each be "
    "listed. Aim for the complete set of plausible families, not just the most "
    "obvious one.\n"
    "For EACH family give:\n"
    "  - name: a short family / fold name (a mechanism class, not an enzyme);\n"
    "  - descriptors: annotation/domain/fold words that identify its members in "
    "a sequence database (e.g. a catalytic-domain name, a fold family, a "
    "cofactor-binding motif) — generic mechanism terms for THIS reaction's "
    "families, whatever they are;\n"
    "  - literature_terms: search phrases that CROSS the family/fold with the "
    "TARGET SUBSTRATE CLASS — the reactive groups, size, and type of the "
    "reaction inputs in the query (e.g. a fold paired with the target's "
    "substrate class). This steers the search to the family MEMBER whose "
    "substrate resembles the target, not just the family's textbook archetype; "
    "include for each family at least one such fold-plus-target-substrate term.\n"
    "STRICT: do NOT name any specific enzyme, gene, protein, natural product, or "
    "organism anywhere — only mechanism/fold classes, generic descriptors, and "
    "the target substrate's own chemical class from the query. "
    'Return ONLY JSON: {"families":[{"name":"..","descriptors":["..",".."],'
    '"literature_terms":["..",".."]}]}'
)

# Guard the firewall: drop any descriptor/term that looks like a specific
# gene/enzyme token (a short mixed-case symbol) rather than a mechanism word.
_ENZYME_TOKEN_RE = re.compile(r"^[A-Z][a-z]{1,3}[A-Z0-9]$")


def _clean_terms(terms) -> list[str]:
    out = []
    for t in terms or []:
        s = str(t).strip()
        if not s:
            continue
        # Reject bare gene-symbol-like tokens (e.g. "AbcD"); keep phrases.
        if len(s.split()) == 1 and _ENZYME_TOKEN_RE.match(s):
            continue
        out.append(s)
    return out


def derive_mechanism_families(
    query: str, substrate_smiles: str, llm_client, max_families: int = 8,
) -> list[dict]:
    """Derive the distinct mechanism families that can catalyse the reaction.

    Returns a list of ``{"name", "descriptors", "literature_terms"}`` (possibly
    empty on failure — the caller degrades gracefully to the generic
    distinct-family count).
    """
    if llm_client is None:
        return []
    from enzyme_discovery_agent import config

    user = f"Target reaction / query: {query}"
    if substrate_smiles:
        user += f"\nSubstrate SMILES: {substrate_smiles}"
    try:
        response = llm_client.call(
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            temperature=config.LLM_TEMPERATURE,
            timeout=max(config.LLM_TIMEOUT, 120),
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("mechanism-family derivation failed: %s", exc)
        return []

    raw = []
    try:
        raw = (json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
               .get("families") or [])
    except Exception:  # noqa: BLE001
        # A long response can truncate mid-list and break whole-blob parsing.
        # Family objects contain no nested braces, so salvage each complete one
        # independently (the truncated tail simply won't match).
        for blob in re.findall(r"\{[^{}]*\}", content):
            try:
                item = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("name"):
                raw.append(item)
    if not raw:
        logger.warning("mechanism-family derivation returned no parseable families")
        return []

    families = []
    seen = set()
    for fam in raw[:max_families]:
        name = str(fam.get("name") or "").strip()
        descriptors = _clean_terms(fam.get("descriptors"))
        lit_terms = _clean_terms(fam.get("literature_terms"))
        key = name.casefold()
        if not name or not descriptors or key in seen:
            continue
        seen.add(key)
        families.append({
            "name": name,
            "descriptors": descriptors,
            "literature_terms": lit_terms,
        })
    return families


_RANK_SYSTEM = (
    "You are an enzymology strategist. You are given the TARGET reaction's "
    "substrates and a list of candidate enzyme MECHANISM families (folds). Rank "
    "the families by how closely each family's ARCHETYPAL native substrates "
    "resemble the TARGET substrates — judged ONLY on chemistry: the reactive "
    "functional groups involved in the bond, how many of each are present, the "
    "carbon-chain size, and saturation/aromaticity. A family whose members "
    "natively act on substrates bearing the same reactive groups, in the same "
    "count and a similar size, ranks HIGHER. Reason from substrate shape, not "
    "from any known enzyme.\n"
    "STRICT: do NOT name any specific enzyme, gene, protein, natural product, or "
    "organism. Refer to families only by the names given to you, and describe "
    "substrates only by their chemical class/groups/size.\n"
    'Return ONLY JSON: {"ranking":[{"name":"<exact family name from the list>",'
    '"resemblance":"high|medium|low","reason":"<one short clause on substrate '
    'shape>"}, ...]} covering every family, best match first.'
)


def rank_families_by_substrate_resemblance(
    families: list[dict], query: str, substrate_smiles: str, llm_client,
) -> list[dict]:
    """Order derived families by how well their native substrates resemble the
    target substrates (the substrate axis — leakage-safe).

    Returns the same family dicts, reordered best-match-first, each annotated
    with ``promise`` (high/medium/low) and ``promise_reason``.  Degrades to the
    input order (unannotated) on any failure so the caller never depends on it.
    """
    if llm_client is None or len(families) <= 1:
        return families
    from enzyme_discovery_agent import config

    listing = "\n".join(
        f"- {f['name']}: {'; '.join(f.get('descriptors') or [])}" for f in families
    )
    user = f"Target reaction / query: {query}"
    if substrate_smiles:
        user += f"\nTarget substrate SMILES: {substrate_smiles}"
    user += f"\n\nFamilies to rank:\n{listing}"
    try:
        response = llm_client.call(
            messages=[{"role": "system", "content": _RANK_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=config.LLM_TEMPERATURE,
            timeout=max(config.LLM_TIMEOUT, 120),
        )
        content = response.choices[0].message.content or ""
        ranking = (json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
                   .get("ranking") or [])
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("mechanism-family ranking failed: %s", exc)
        return families

    by_key = {f["name"].casefold(): f for f in families}
    ordered = []
    for row in ranking:
        fam = by_key.pop(str(row.get("name") or "").strip().casefold(), None)
        if fam is None:
            continue
        resemblance = str(row.get("resemblance") or "").strip().lower()
        fam["promise"] = resemblance if resemblance in {"high", "medium", "low"} else "medium"
        # Strip any stray gene-symbol-like token from the free-text reason.
        reason_tokens = [
            tok for tok in str(row.get("reason") or "").split()
            if not _ENZYME_TOKEN_RE.match(tok.strip(".,;:()"))
        ]
        fam["promise_reason"] = " ".join(reason_tokens).strip()
        ordered.append(fam)
    # Any family the ranker omitted keeps its original relative order at the end.
    ordered.extend(by_key.values())
    return ordered
