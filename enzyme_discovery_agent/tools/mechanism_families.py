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

from enzyme_discovery_agent.chemistry_logic import extract_target_bond_type

logger = logging.getLogger("enzyme_agent")

# Genuine opposites: a cleavage family cannot serve a formation target (or vice
# versa).  A grounded chemical fact, not a tuned knob.
_OPPOSITE_DIRECTIONS = {("formation", "cleavage"), ("cleavage", "formation")}
_HYDROLYSIS_TERMS = re.compile(r"\bh2o\b|\bwater\b")
_ARROW = re.compile(r"->|=>|→|<=>|=")


def _family_native_direction(fam: dict) -> str | None:
    """Best-effort native direction ('formation'|'cleavage'|None) for a family.

    CODE-derived, not trusted to the model's prose: (1) a native reaction that
    CONSUMES water (H2O left of the arrow) is a hydrolysis => cleavage, whatever
    the model labelled it; (2) else the model's stated ``native_direction``; (3)
    else a keyword parse of the reaction text.  None when nothing resolves."""
    rxn = str(fam.get("native_reaction") or "")
    left = _ARROW.split(rxn, maxsplit=1)[0].lower() if rxn else ""
    if _HYDROLYSIS_TERMS.search(left):
        return "cleavage"
    stated = str(fam.get("native_direction") or "").strip().lower()
    if stated in ("formation", "synthesis", "ligation", "condensation"):
        return "formation"
    if stated in ("cleavage", "hydrolysis", "degradation", "lysis"):
        return "cleavage"
    return extract_target_bond_type(rxn).get("direction")


def family_passes_constraints(fam: dict, target_direction: str | None) -> tuple[bool, str]:
    """Deterministic hard-constraint filter for a derived family: return
    ``(ok, reason_if_dropped)``.

    Enforces in CODE the two constraints the prompt states but the model does not
    reliably self-apply: (a) native direction must not be the OPPOSITE of the
    target's (drops native-cleavage families for a formation target — proteases,
    hydrolases, esterases — however the model labelled them); (b) native donor and
    acceptor must be FREE substrates (drops carrier-/protein-bound families).
    ``target_direction`` None => the direction check is skipped (unknown target),
    so an undetermined target never silently empties the list."""
    if fam.get("substrates_free") is False:
        return False, "native donor/acceptor is not a free small molecule"
    fam_dir = _family_native_direction(fam)
    if target_direction and fam_dir and (fam_dir, target_direction) in _OPPOSITE_DIRECTIONS:
        return False, f"native direction '{fam_dir}' is opposite the target '{target_direction}'"
    return True, ""

_SYSTEM = (
    "You are an enzymology strategist. Given a target reaction, enumerate the "
    "DISTINCT enzyme MECHANISM families / structural folds known to catalyse "
    "that BOND-FORMING chemistry. Reason from the reaction TYPE — the bond "
    "formed, the reacting chemical groups, and the cofactor — NOT from any "
    "specific target molecule or product.\n"
    "List the DISTINCT valid mechanism families — different structural folds or "
    "activation chemistries (a different activated intermediate, cofactor, or "
    "catalytic-domain fold) that EACH, judged on their OWN native reaction, "
    "satisfy EVERY constraint below. Genuinely distinct QUALIFYING folds are "
    "separate families and must each be listed; do not collapse distinct "
    "qualifying folds into one umbrella entry.\n"
    "PRECISION OVER BREADTH: there is NO target count and NO minimum. List ONLY "
    "families that actually qualify, however few — a SHORT, precise list is far "
    "better than a padded one. Do NOT add a marginal, wrong-direction, or "
    "wrong-substrate family to reach a number: a wrong family wastes the search "
    "and is worse than omitting it. When unsure whether a family qualifies, leave "
    "it out.\n"
    "DIRECTION CONSTRAINT (strict): include ONLY families whose NATIVE reaction "
    "proceeds in the SAME direction as the target reaction (the direction named "
    "or implied by the query). EXCLUDE any family whose native reaction runs the "
    "OPPOSITE direction to the target — do NOT list a family that natively "
    "catalyses the REVERSE transformation, even if it is theoretically reversible "
    "or could be pushed 'in reverse' under engineered conditions. Native "
    "direction must match the target direction; a family that natively undoes the "
    "target transformation is not a valid family and must be omitted.\n"
    "CHEMISTRY-MATCH CONSTRAINT (strict): every family must natively make the "
    "SAME bond/linkage as the target, acting DIRECTLY on FREE substrates of the "
    "same chemical class as the target inputs. EXCLUDE a family if (a) its native "
    "product is a DIFFERENT bond/linkage than the target's, or (b) it requires "
    "its donor to be pre-activated as a carrier-bound or otherwise chemically "
    "activated form instead of the free substrate class the target starts from, "
    "or (c) its native product is only an intermediate en route to — not — the "
    "target's finished bond. Mechanistic adjacency ('a step on the way', 'could "
    "be diverted to the target', 'shares an activation step') is NOT sufficient: "
    "the family's OWN native transformation must itself be the target's bond on "
    "the target's kind of free substrates, or omit it.\n"
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
    "include for each family at least one such fold-plus-target-substrate term;\n"
    "  - native_reaction: the family's representative NATIVE reaction in its true "
    "PHYSIOLOGICAL direction, written generically as 'reactants -> products' with "
    "chemical CLASSES only (no enzyme/gene/product/organism name). Include the "
    "cofactor/co-substrate the native reaction uses, and put H2O on the correct "
    "side — a family that natively HYDROLYSES/cleaves consumes water as a "
    "REACTANT (left of the arrow);\n"
    "  - native_direction: exactly 'formation' or 'cleavage' — whether the "
    "family's NATIVE physiological reaction MAKES or BREAKS the bond. Judge the "
    "native role, NOT whether it could be run in reverse in vitro;\n"
    "  - substrates_free: true ONLY if BOTH the native donor AND the native "
    "acceptor are FREE small molecules of the target's substrate class; false if "
    "either is carrier-bound, protein-tethered, or a fixed cofactor;\n"
    "  - archetypal_native_substrates: the SPECIFIC representative compounds the "
    "family's ARCHETYPE natively transforms — name the actual donor AND acceptor "
    "as concrete, resolvable compound names (each naming ONE molecule with a "
    "definite structure), NOT a generic class word. When a native partner is "
    "genuinely a single small molecule, name that exact molecule. These are the "
    "family's OWN native substrates, used only to judge how far the family's "
    "chemistry sits from the target — do NOT substitute the target's molecules.\n"
    "STRICT: do NOT name any specific enzyme, gene, protein, natural product, or "
    "organism anywhere — only mechanism/fold classes, generic descriptors, the "
    "family's own native substrate compounds, and the target substrate's own "
    "chemical class from the query. "
    'Return ONLY JSON: {"families":[{"name":"..","descriptors":["..",".."],'
    '"literature_terms":["..",".."],"native_reaction":"reactants -> products",'
    '"native_direction":"formation|cleavage","substrates_free":true,'
    '"archetypal_native_substrates":["..",".."]}]}'
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


def _derive_once(query: str, substrate_smiles: str, llm_client, max_families: int,
                 exclude: list[str] | None = None) -> list[dict]:
    """A single derivation call -> parsed family dicts (may be empty).

    ``exclude`` names families already found on prior attempts; the model is
    asked for ADDITIONAL distinct folds beyond them, so a retry broadens the set
    instead of repeating the first (often single, umbrella) answer.
    """
    from enzyme_discovery_agent import config

    user = f"Target reaction / query: {query}"
    if substrate_smiles:
        user += f"\nSubstrate SMILES: {substrate_smiles}"
    if exclude:
        user += (
            "\n\nYou have ALREADY listed these families: " + "; ".join(exclude) +
            ".\nList ONLY ADDITIONAL, genuinely DISTINCT folds/activation "
            "chemistries NOT already covered above — different structural "
            "superfamilies or a different activated intermediate/cofactor. Do "
            "not restate or rename the ones above."
        )
    try:
        response = llm_client.call(
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            temperature=config.LLM_TEMPERATURE,
            timeout=max(config.LLM_TIMEOUT, 120),
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("mechanism-family derivation call failed: %s", exc)
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

    out = []
    for fam in raw[:max_families]:
        name = str(fam.get("name") or "").strip()
        descriptors = _clean_terms(fam.get("descriptors"))
        lit_terms = _clean_terms(fam.get("literature_terms"))
        if not name or not descriptors:
            continue
        out.append({
            "name": name,
            "descriptors": descriptors,
            "literature_terms": lit_terms,
            # Grounding the deterministic direction/substrate filter reasons over.
            "native_reaction": str(fam.get("native_reaction") or "").strip(),
            "native_direction": str(fam.get("native_direction") or "").strip(),
            "substrates_free": fam.get("substrates_free"),
        })
    return out


def derive_mechanism_families(
    query: str, substrate_smiles: str, llm_client, max_families: int = 10,
    min_families: int = 4, max_attempts: int = 6,
) -> list[dict]:
    """Derive the distinct mechanism families that can catalyse the reaction.

    A single LLM call sees only a slice of the fold space, so it silently drops
    families — including, on some draws, the fold whose native chemistry is
    CLOSEST to the target.  So don't stop at a low floor: sample repeatedly and
    UNION the families by name, each attempt told what is already covered so it
    returns ADDITIONAL distinct folds, and keep going until the model CONVERGES
    (an attempt beyond the first adds nothing new) or the hard ``max_families`` /
    ``max_attempts`` caps are hit.  Convergence — not a fixed count — is the
    exhaustiveness signal: the loop broadens across several samples instead of
    trusting one.  ``min_families`` only guards against declaring convergence on a
    single thin first sample.  Returns [] only if every attempt yields nothing.
    """
    if llm_client is None:
        return []

    # Target reaction DIRECTION, derived once from the query, drives the
    # deterministic drop of inverse-direction families.  Substrate helps only the
    # bond keywords; direction comes from the reaction verb in the query.
    target_direction = extract_target_bond_type(
        f"{query} {substrate_smiles}".strip()).get("direction")

    families: list[dict] = []
    seen: set[str] = set()   # every NAME proposed (valid or dropped), so retries broaden
    dropped = 0
    for attempt in range(max_attempts):
        # After the first attempt, tell the model what it already produced (valid
        # OR dropped) so a retry ADDS distinct folds rather than repeating a single
        # umbrella family, or re-proposing a family already ruled out.
        exclude = sorted(seen) or None
        added = 0
        for fam in _derive_once(query, substrate_smiles, llm_client, max_families,
                                exclude=exclude):
            key = fam["name"].casefold()
            if key in seen:
                continue
            seen.add(key)
            # Deterministic hard filter: drop inverse-direction / non-free-substrate
            # families in CODE (the model states these constraints but does not
            # reliably self-apply them), so the survey/coverage budget is never
            # spent on a family whose own native reaction disqualifies it.
            ok, reason = family_passes_constraints(fam, target_direction)
            if not ok:
                dropped += 1
                logger.info("mechanism family dropped: %s (%s)", fam["name"], reason)
                continue
            families.append(fam)
            added += 1
        if len(families) >= max_families:
            break
        # Converged: a broadening attempt (at least the second sample) surfaced no
        # new fold, so the model has no more distinct families to offer.  Stop —
        # but only once past ``min_families``, so one thin first sample followed by
        # a dry retry does not end derivation prematurely.
        if added == 0 and attempt >= 1 and len(families) >= min_families:
            break
        # Give up rather than spin if the model stays dry well below the floor.
        if added == 0 and attempt >= 2:
            break
    if not families:
        logger.warning("mechanism-family derivation returned no parseable families")
    elif dropped:
        logger.info(
            "mechanism-family derivation: kept %d families, dropped %d "
            "(inverse-direction / non-free-substrate)", len(families), dropped)
    return families[:max_families]


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


# Generic tokens shared across bond-forming mechanism families — useless for
# deciding whether a SPECIFIC family was actually searched, because they appear
# in almost every family's description.  Answer-agnostic: these are reaction-
# shape words, not any enzyme or answer name.
_GENERIC_FAMILY_TOKENS = frozenset({
    "atp", "adp", "amp", "gtp", "dependent", "fold", "domain", "superfamily",
    "family", "enzyme", "ligase", "synthetase", "synthase", "transferase",
    "bond", "acid", "amine", "amino", "free", "carboxylate", "carboxylic",
    "intermediate", "substrate", "primary", "secondary", "group", "type",
    "like", "variant", "binding", "active", "site", "metal", "cofactor",
    "reaction", "formation", "class", "catalytic", "mechanism", "nucleotide",
    "and", "the", "of", "with", "via", "for", "an", "a",
})


def _family_search_tokens(fam: dict) -> set[str]:
    """Distinctive lowercase tokens for a derived family (name + descriptors),
    minus the generic reaction vocabulary shared by all families."""
    text = " ".join([
        str(fam.get("name", "")),
        " ".join(fam.get("descriptors") or []),
    ]).lower()
    # Split on hyphens too, so a compound like "atp-dependent" reduces to its
    # component words and both fall to the generic stopword filter.
    toks = re.findall(r"[a-z][a-z0-9]{2,}", text)
    return {t for t in toks if t not in _GENERIC_FAMILY_TOKENS}


def unexplored_promising_families(
    families: list[dict],
    search_queries: list[str],
    promise_levels: tuple[str, ...] = ("high", "medium"),
) -> list[dict]:
    """Promising derived families NOT yet touched by any targeted search.

    A family counts as explored once ANY recorded search query contains one of
    its DISTINCTIVE tokens (family name / descriptor words, minus generic
    reaction vocabulary).  Used to keep a run from finalizing while a family it
    derived as promising was never actually searched — the observed failure where
    the pool saturated on one fold and other promising families the agent had
    itself named were left undrilled.  Pure and deterministic (offline-testable);
    the caller bounds how many times it forces a targeted search.
    """
    haystack = " ".join(search_queries).lower()
    unexplored = []
    for fam in families or []:
        if str(fam.get("promise", "")).lower() not in promise_levels:
            continue
        toks = _family_search_tokens(fam)
        if toks and not any(t in haystack for t in toks):
            unexplored.append(fam)
    return unexplored
