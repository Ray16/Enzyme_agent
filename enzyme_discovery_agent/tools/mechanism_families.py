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
    "a sequence database (e.g. the adenylation domain, an ATP-grasp fold, a "
    "carrier-protein ligase, a condensation domain) — generic mechanism terms;\n"
    "  - literature_terms: reaction- and mechanism-anchored search phrases.\n"
    "STRICT: do NOT name any specific enzyme, gene, protein, natural product, or "
    "organism anywhere — only mechanism/fold classes and generic descriptors. "
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
        obj = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("mechanism-family derivation failed: %s", exc)
        return []

    families = []
    seen = set()
    for fam in (obj.get("families") or [])[:max_families]:
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
