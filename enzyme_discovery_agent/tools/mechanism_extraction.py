"""Extract structured REACTION-MECHANISM FACTS for an enzyme from the literature.

Scope: this is a FACT extractor, not a reasoner, and it is CLASS-AGNOSTIC — it works for any
enzyme chemistry (substitution, hydrolysis, redox / hydride or electron transfer, radical,
addition-elimination, isomerization, lyase, ligase, metal-catalyzed, ...). It finds a mechanism
reference (or uses a supplied one), reads the full text, and returns the mechanism as a typed
record of grounded facts: a general layer that fits every class — mechanism_class, the chemical
step, the bonds broken/formed, the reaction participants (with roles), the reactive atom pair,
catalytic residues, cofactors, metals, reference structures — plus a substitution
specialization (substitution center, nucleophile / electrophilic center / leaving group,
is_clean_inline_substitution) that is filled only when the step is a nucleophilic substitution
or addition. It deliberately does NOT decide the geometry proxy (which atoms make an attack
distance/angle, which selector to use): that judgment varies per enzyme and is left to the main
agent to reason about from these facts.

Reaction-agnostic by design: for a redox or lyase enzyme the substitution fields are null and
the general fields carry the mechanism, so the agent still gets the context it needs to decide
how to model. The is_clean_inline_substitution flag tells the agent whether an attack-
distance+angle proxy is even meaningful (True for a clean SN2, False/null otherwise).

Robust paper selection: a mechanism query often returns papers that only *mention* the enzyme
(physiology, expression, remediation) with no mechanistic content. Reading the first hit and
trusting it is how a generic (e.g. "water") nucleophile gets emitted with low confidence and
still returned as ``ok``. So this module discovers candidates through the agent's OWN federated
literature stack — PubMed + OpenAlex + Europe PMC + Semantic Scholar searched in parallel, then
``consolidate_pool``'s cross-source dedup/merge + semantic (RRF) rerank (the merge back-fills a
PMCID onto broad-source hits so they stay routable to full text) — then extracts from the
ranked candidates in order, SCORES each extraction, and returns the best, flagging the result
when even the best is weakly grounded instead of passing it off as authoritative. A single-
source Europe PMC keyword search remains as a fallback when the stack is unavailable.

Public entry point: the ``extract_reaction_mechanism`` @tool.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import urllib.parse

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools.fulltext import _get
from enzyme_discovery_agent.tools.mechanism_schema import (
    MECHANISM_FACTS_SCHEMA_VERSION, validate_mechanism_facts,
)
from enzyme_discovery_agent.tools.measurement_extraction import (
    resolve_paper, fetch_jats_or_text, _jats_to_text, _pdf_to_text, _llm_json,
)
from enzyme_discovery_agent.tools.literature_extraction import select_relevant_sections

logger = logging.getLogger("enzyme_agent")

_MECH_SYS = (
    "You are an enzyme-mechanism analyst. From the paper text, extract the catalytic mechanism "
    "of the named enzyme as GROUNDED FACTS in STRICT JSON (no prose, no code fences). This must "
    "work for ANY enzyme class (substitution, hydrolysis, redox / hydride or electron transfer, "
    "radical, addition-elimination, isomerization, lyase, ligase, metal-catalyzed, ...). Report "
    "only what the text/chemistry supports; never invent residue numbers or atom names — use "
    "null when unstated. If the paper does not actually describe this enzyme's catalytic "
    "mechanism, say so via confidence:\"low\" and leave unstated fields null rather than guessing. "
    "Do NOT recommend any modeling geometry; only report facts the caller can reason over. Shape:\n"
    "{\n"
    '  "reaction": "<one line, reactants -> products>",\n'
    '  "mechanism_class": "SN2|SN1|SNAr|addition_elimination|acyl_transfer|hydrolysis|redox_hydride|redox_electron|radical|pericyclic|isomerization|elimination|carboxylation|aldol|metal_catalyzed|acid_base|other|null",\n'
    '  "chemical_step": "<one line: which bonds break and form in the catalyzed step>",\n'
    "  // GENERAL fields below — fill for EVERY class:\n"
    '  "bonds_broken": [{"atoms":"<e.g. C2-Cl, Calpha-H, C-O ester>", "on":"substrate|cofactor|protein|null"}],\n'
    '  "bonds_formed": [{"atoms":"<e.g. Oasp-C2, C-Nnad>", "on":"substrate|cofactor|protein|null"}],\n'
    '  "participants": [{"role":"nucleophile|electrophile|leaving_group|hydride_donor|hydride_acceptor|proton_donor|proton_acceptor|electron_donor|electron_acceptor|radical_initiator|metal_center|electrophile_activator|oxyanion_stabilizer|stabilizer|base|acid|other", "identity":"<text>", "atom":"<atom name/element or null>", "on":"substrate|cofactor|protein|water|metal|null"}],\n'
    '  "reactive_pair": {"atom_a":"<donor/attacking atom or null>", "atom_b":"<accepting/attacked atom or null>", "note":"<the two atoms whose approach defines the chemical step, if one applies; else nulls>"},\n'
    '  "substrate": {"identity":"<the transformed substrate>", "reactive_atom":"<atom on the substrate that is attacked/transformed, e.g. C2 / carbonyl C>", "smiles":"<SMILES if the text gives it, else null>"},\n'
    '  "cofactors": [{"name":"<e.g. NAD/SAM/ATP/FAD/PAPS>", "count":<int>, "ccd":"<3-letter PDB/CCD ligand code if known, e.g. NAD/SAM/ATP, else null>", "role":"<text>"}],\n'
    '  "metals":    [{"name":"<e.g. Mg/Zn/Fe>", "count":<int>, "ccd":"<CCD code, e.g. ZN/MG/FE, else null>", "role":"<text>"}],\n'
    '  "catalytic_residues": [{"residue":"<e.g. His289>", "role":"base|acid|nucleophile|hydride_relay|metal_ligand|stabilizer|other"}],\n'
    '  "stereochemistry": "inversion|retention|racemization|none|null",   // configurational outcome at the reaction center\n'
    '  "covalent_intermediate": {"forms": true|false|null, "residue":"<e.g. Ser160 / catalytic Asp / null>", "description":"<e.g. acyl-enzyme / aspartyl-ester / none>"},\n'
    '  "michaelis_structure": "<single PDB ID of the pre-reactive / substrate-bound (Michaelis) complex best for seeding a model, or null>",\n'
    '  "reference_structures": ["<PDB IDs mentioned as the mechanistic/Michaelis structure>"],\n'
    "  // SUBSTITUTION SPECIALIZATION below — fill ONLY when the step is a nucleophilic\n"
    "  // substitution or addition; otherwise set substitution_center null, the three groups\n"
    "  // null, and is_clean_inline_substitution null:\n"
    '  "substitution_center": "carbon|phosphorus|sulfur|other|null",\n'
    '  "is_clean_inline_substitution": true|false|null,   // one-step backside displacement?\n'
    '  "nucleophile":   {"source":"protein_residue|substrate|cofactor|water|null", "identity":"<text or null>", "atom":"<atom name/element or null>", "residue":"<e.g. Asp124 or null>"},\n'
    '  "electrophile":  {"identity":"<text or null>", "atom":"<atom name/element or null>", "on":"substrate|cofactor|null"},\n'
    '  "leaving_group": {"identity":"<text or null>", "atom":"<atom name/element or null>", "on":"substrate|cofactor|null"},\n'
    '  "evidence": "<short supporting quote or reasoning>",\n'
    '  "confidence": "high|medium|low"\n'
    "}\n"
    "Always fill the general fields (mechanism_class, chemical_step, bonds_broken/formed, "
    "participants, catalytic_residues, cofactors, metals, reactive_pair when identifiable) — "
    "these carry the mechanism for non-substitution enzymes too. Also fill the MODELING fields "
    "when the text supports them: substrate (identity + the reactive atom + SMILES if given), "
    "cofactor/metal ccd codes and counts, stereochemistry (the configurational outcome), "
    "covalent_intermediate (whether a covalent enzyme intermediate forms and on which residue), "
    "and michaelis_structure (the single best pre-reactive/substrate-bound PDB). If the step is "
    "not a clean one-step in-line substitution, set is_clean_inline_substitution false (or null "
    "when the substitution framing does not apply at all) and leave the three substitution "
    "groups null."
)

# Title words that mark a paper as likely to CARRY a catalytic mechanism. Deliberately
# MECHANISM-CLASS-AGNOSTIC: every enzyme has a "mechanism", "active site", "structure" and
# "catalytic residues" regardless of chemistry, but only SOME have a "nucleophile" /
# "inversion" / "stereochemistry". Seeding those substitution-specific terms would bias
# retrieval and ranking toward substitution enzymes and hurt recall for redox / lyase /
# metalloenzyme mechanisms — which contradicts this tool's reaction-agnostic purpose
# (see CLAUDE.md). So the cue list stays generic.
_MECH_TITLE_CUES = (
    "mechanism", "mechanistic", "structure", "structural", "active site", "catalytic",
    "catalysis", "catalytic residues", "crystal", "transition state", "cofactor",
    "reaction", "kinetic", "biochem", "enzymolog",
)
# Title words that mark a paper as likely NOT about the catalysis (it merely mentions the
# enzyme) — down-weighted so a physiology/omics hit never outranks a mechanism paper.
_MECH_TITLE_NEG = (
    "physiolog", "expression", "tolerance", "heterolog", "microbiome", "microbial community",
    "remediation", "bioremediation", "phylogen", "genome", "metagenom", "review of",
    "clinical", "epidemiol", "abundance", "diversity", "ecolog",
)
# Phrases in the extractor's own `evidence` that admit the paper lacked mechanistic content.
_FACT_DISCLAIMERS = (
    "no specific", "not provided", "not specified", "no mechanistic", "no catalytic",
    "does not provide", "not described", "no details", "limited to", "no structural",
    "no pdb", "unclear", "no information",
)
# Mechanism vocabulary used to CENTRE the paper excerpt on the sections that carry the
# catalytic mechanism (Results/Discussion, active-site/structure) rather than the intro +
# methods that fill a naive head-truncation. Class-BALANCED on purpose: it spans
# substitution, redox, radical, metal and pericyclic chemistry so excerpt selection does not
# bias toward one mechanism class (the same reason the retrieval query stays class-agnostic).
_MECH_EXCERPT_CUES = (
    "mechanism", "mechanistic", "catalytic", "catalysis", "active site", "catalytic residue",
    "transition state", "nucleophil", "electrophil", "leaving group", "proton transfer",
    "general base", "general acid", "oxyanion", "hydride", "electron transfer", "radical",
    "cofactor", "metal ion", "coordinat", "stereochem", "inversion", "retention",
    "intermediate", "covalent", "crystal structure", "substrate binding", "michaelis",
    # PDB-accession context, so the Michaelis / reference-structure IDs (often in a
    # data-availability line or figure caption) land in the excerpt.
    "pdb", "protein data bank", "accession", "deposited", "data availability",
)


def _paper_text(paper: dict) -> str:
    content, kind = fetch_jats_or_text(paper)
    if not content:
        return ""
    if kind == "xml":
        return _jats_to_text(content)
    if kind == "pdf":
        return _pdf_to_text(content)
    return ""


def _ref_brief(ref: dict) -> dict:
    return {"pmcid": ref.get("pmcid"), "doi": ref.get("doi"), "title": ref.get("title")}


def _search_candidates(query: str, n: int = 6) -> list[dict]:
    """Return up to `n` Europe PMC core-search hits as paper records (not just the top one)."""
    url = (f"{config.EUROPEPMC_API}/search?query={urllib.parse.quote(query)}"
           f"&resultType=core&format=json&pageSize={n}")
    raw = _get(url, timeout=30)
    if not raw:
        return []
    try:
        results = json.loads(raw).get("resultList", {}).get("result", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [{
        "pmcid": (r.get("pmcid") or "").upper(),
        "pmid": r.get("pmid") or r.get("id") or "",
        "doi": r.get("doi") or "",
        "title": r.get("title") or "",
        "is_oa": (r.get("isOpenAccess") == "Y"),
        "source": "europepmc",
    } for r in results]


def _rank_mechanism_papers(papers: list[dict]) -> list[dict]:
    """Order candidates so mechanism/structure papers (and OA ones we can actually read)
    come first, and enzyme-mentioning physiology/omics papers sink."""
    def score(p: dict) -> float:
        t = (p.get("title") or "").lower()
        s = float(sum(1 for c in _MECH_TITLE_CUES if c in t))
        s -= 2.0 * sum(1 for c in _MECH_TITLE_NEG if c in t)
        if p.get("is_oa"):
            s += 2.0          # we can only extract from full text we can fetch
        if p.get("pmcid"):
            s += 1.0
        return s
    return sorted(papers, key=score, reverse=True)


def _to_ref(rec: dict) -> dict | None:
    """Map a consolidated literature record to a fetchable paper ref (needs a PMCID or DOI)."""
    pmcid = (rec.get("pmcid") or "").upper()
    doi = rec.get("doi") or ""
    if not pmcid and not doi:
        return None
    return {"pmcid": pmcid, "pmid": rec.get("pmid") or "", "doi": doi,
            "title": rec.get("title") or "", "is_oa": rec.get("is_oa"),
            "source": "+".join(rec.get("sources", []) or []) or "consolidated"}


def _discover_candidates(enzyme: str, max_results: int = 6) -> list[dict]:
    """Federated candidate discovery via the agent's OWN multi-source literature stack:
    PubMed + OpenAlex + Europe PMC + Semantic Scholar searched in parallel, then
    ``consolidate_pool`` dedups/merges across sources and applies the semantic (RRF) rerank.

    The cross-source merge back-fills a PMCID (from the Europe PMC record) onto a hit that a
    broader source surfaced without one, so a paper found ANYWHERE becomes routable to full
    text — which is why we no longer need to restrict the search to Europe PMC. Returns
    fetchable refs best-ranked first; [] if the stack is unavailable (caller then uses the
    Europe PMC keyword fallback)."""
    try:
        from enzyme_discovery_agent.tools import literature as _lit
    except Exception as e:  # noqa: BLE001 - optional/circular safety
        logger.debug("literature stack unavailable: %s", e)
        return []
    # Mechanism-class-agnostic query: terms shared by every enzyme's mechanism/structure
    # literature. NOT "nucleophile" — that presumes a substitution mechanism and would bias
    # retrieval away from redox / lyase / metalloenzyme candidates.
    query = f"{enzyme} catalytic mechanism active site structure catalytic residues"
    fns = (_lit.search_pubmed, _lit.search_openalex, _lit.search_europepmc,
           _lit.search_semantic_scholar)
    pool: list[dict] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(fns)) as ex:
            futs = [ex.submit(fn, query, max_results=12, rerank_query=query) for fn in fns]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:  # noqa: BLE001 - isolate one source's failure
                    logger.debug("mechanism source search failed: %s", e)
                    continue
                if isinstance(r, dict) and r.get("status") != "error":
                    pool += r.get("data", []) or []
    except Exception as e:  # noqa: BLE001
        logger.debug("federated candidate search failed: %s", e)
        return []
    if not pool:
        return []
    try:
        ranked = _lit.consolidate_pool(query, pool, max_results=max_results).get("ranked", [])
    except Exception as e:  # noqa: BLE001 - rerank/model hiccup: keep raw pool order
        logger.debug("consolidate_pool failed (%s); using raw pool order", e)
        ranked = pool[:max_results]
    return [r for r in (_to_ref(p) for p in ranked) if r]


def _score_facts(facts: dict) -> tuple[float, bool]:
    """Score an extraction's groundedness — CLASS-AGNOSTIC. Returns (score, is_grounded).

    A grounded result carries real mechanistic content for whatever chemistry the enzyme does:
    named reaction participants (with roles/identities), the bonds broken/formed, catalytic
    residues, cofactors/metals, and/or reference structures — not a low-confidence flag or an
    "no detail in this text" disclaimer. A named protein/cofactor/substrate nucleophile is a
    bonus for substitution cases, but its ABSENCE is not penalized: many classes (redox,
    isomerase, lyase) have no nucleophile, and forcing one is the bug we are fixing. A bare
    water fallback or empty facts scores low, which drives the caller to try another paper."""
    if not isinstance(facts, dict):
        return -99.0, False
    conf = (facts.get("confidence") or "").lower()
    s = {"high": 3.0, "medium": 1.0}.get(conf, -1.0)
    # General mechanistic content (applies to any enzyme class).
    parts = [p for p in (facts.get("participants") or [])
             if isinstance(p, dict) and (p.get("identity") or "").strip()]
    if parts:
        s += 2.0
    if facts.get("bonds_broken") or facts.get("bonds_formed"):
        s += 1.0
    if facts.get("catalytic_residues"):
        s += 1.0
    if facts.get("cofactors") or facts.get("metals"):
        s += 0.5
    if facts.get("reference_structures"):
        s += 1.0
    # Substitution specialization: reward a named non-water nucleophile; do NOT penalize a
    # legitimately null one (redox / lyase / isomerase have none).
    nu = facts.get("nucleophile") or {}
    if (nu.get("source") or "").lower() in ("protein_residue", "cofactor", "substrate") \
            and (nu.get("identity") or "").strip():
        s += 1.0
    ev = (facts.get("evidence") or "").lower()
    if any(d in ev for d in _FACT_DISCLAIMERS):
        s -= 2.0
    grounded = conf in ("high", "medium") and s >= 3.0
    return s, grounded


def _extract_from(ref: dict, enzyme: str, llm_client) -> dict | None:
    text = _paper_text(ref)
    if not text:
        return None
    # Section-aware excerpt centred on the mechanism-bearing parts of the paper, anchored on
    # the enzyme's own name tokens. A naive text[:14000] head-truncation keeps the intro +
    # methods and often drops the Results/Discussion where the catalytic mechanism lives.
    identifiers = [w for w in enzyme.replace("/", " ").replace("-", " ").split() if len(w) >= 3]
    excerpt = select_relevant_sections(
        text, identifiers=identifiers, max_chars=14000, extra_cues=_MECH_EXCERPT_CUES)
    user = [{"type": "text",
             "text": f"Enzyme: {enzyme}\n\nPaper full text (excerpt):\n{excerpt}"}]
    facts = _llm_json(llm_client, _MECH_SYS, user)
    # Accept any structured mechanism record (class-agnostic): a substitution enzyme carries a
    # nucleophile, but a redox/lyase/isomerase carries participants / bonds / a mechanism_class
    # instead. Require only that SOME core mechanistic field is present.
    if not isinstance(facts, dict) or not any(
            facts.get(k) for k in ("participants", "nucleophile", "chemical_step",
                                   "mechanism_class", "bonds_broken", "bonds_formed")):
        return None
    return facts


@tool(
    description=(
        "Extract structured REACTION-MECHANISM FACTS for a candidate enzyme from the literature: "
        "finds (or uses a supplied) mechanism reference, reads the full text, and returns the "
        "substitution center, the nucleophile / electrophilic center / leaving group (with atom "
        "names when stated), catalytic residues, cofactors, and metals, plus whether the step "
        "reads as a clean one-step in-line substitution. Ranks several candidate papers toward "
        "mechanism/structure sources and gates on extraction confidence, so a paper that only "
        "mentions the enzyme does not yield a guessed mechanism. Returns FACTS only — it does not "
        "decide any modeling geometry; the caller reasons about that."
    ),
    category="literature",
    parallel_safe=True,
    injected_params=["llm_client"],
    params={
        "enzyme": {"type": "string",
                   "description": "Candidate enzyme name / accession / EC to characterize."},
        "paper": {"type": "string",
                  "description": "Optional DOI/PMID/PMCID of a mechanism paper; if omitted, several are searched and ranked.",
                  "default": ""},
    },
)
def extract_reaction_mechanism(enzyme: str, paper: str = "", llm_client=None) -> dict:
    """Find a mechanism reference for `enzyme` and return its mechanism as structured facts.

    Without an explicit `paper`, searches several mechanism/structure queries, ranks the
    candidates, and extracts from them in order — keeping the best-scored (most grounded)
    result and stopping early once one is confidently grounded. The response carries
    ``papers_tried`` (what was attempted and why) and, when even the best result is weak, a
    ``warning`` plus ``status: "low_confidence"`` so the caller does not treat a guess as fact.
    """
    if paper:
        candidates = [resolve_paper(paper)]
    else:
        # Primary: the agent's federated multi-source search + semantic rerank.
        candidates = _discover_candidates(enzyme, max_results=6)
        if not candidates:
            # Fallback: single-source Europe PMC keyword search + keyword ranking.
            pool: list[dict] = []
            for q in (f"{enzyme} catalytic mechanism active site",
                      f"{enzyme} crystal structure catalytic residues",
                      f"{enzyme} reaction mechanism"):
                pool += _search_candidates(q, n=6)
            seen, uniq = set(), []
            for p in pool:
                key = p.get("pmcid") or p.get("doi") or p.get("title")
                if key and key not in seen:
                    seen.add(key)
                    uniq.append(p)
            candidates = _rank_mechanism_papers(uniq)[:5]
        if not candidates:
            candidates = [resolve_paper(f"{enzyme} mechanism structure active site")]

    tried: list[dict] = []
    best, best_score, best_ref = None, -1e9, None
    for ref in candidates:
        facts = _extract_from(ref, enzyme, llm_client)
        if facts is None:
            tried.append({"paper": _ref_brief(ref), "status": "no_text_or_facts"})
            continue
        sc, grounded = _score_facts(facts)
        tried.append({"paper": _ref_brief(ref),
                      "confidence": facts.get("confidence"),
                      "nucleophile": (facts.get("nucleophile") or {}).get("identity"),
                      "score": sc, "status": "grounded" if grounded else "weak"})
        if sc > best_score:
            best, best_score, best_ref = facts, sc, ref
        if grounded and ((facts.get("confidence") or "").lower() == "high" or sc >= 5.0):
            break  # confidently grounded — no need to burn more candidates

    if best is None:
        return {"status": "partial",
                "summary": f"No open-access mechanism text with usable facts for '{enzyme}'.",
                "data": None,
                "paper": candidates[0] if candidates else {},
                "papers_tried": tried,
                "suggestions": ["Supply a specific mechanism/structure paper DOI/PMCID."]}

    best.setdefault("provenance", {}).update(
        {"pmcid": best_ref.get("pmcid"), "doi": best_ref.get("doi"), "title": best_ref.get("title")})
    _sc, grounded = _score_facts(best)
    status = "ok" if grounded else "low_confidence"
    # Class-agnostic summary: prefer the general chemical_step; fall back to Nu->E..LG for
    # substitution enzymes where that framing is the informative one.
    step = (best.get("chemical_step") or "").strip()
    if not step:
        step = (f"Nu={(best.get('nucleophile') or {}).get('identity')} -> "
                f"E={(best.get('electrophile') or {}).get('identity')} .. "
                f"LG={(best.get('leaving_group') or {}).get('identity')}")
    summary = (f"{enzyme}: {best.get('mechanism_class', '?')} | {step}"
               f" | clean_inline={best.get('is_clean_inline_substitution')}"
               f" | confidence={best.get('confidence')}")
    result = {"status": status, "summary": summary, "data": best, "paper": best_ref,
              "papers_tried": tried, "schema_version": MECHANISM_FACTS_SCHEMA_VERSION}
    violations = validate_mechanism_facts(best)
    if violations:
        result["schema_violations"] = violations
        logger.debug("mechanism facts schema violations for %s: %s", enzyme, violations)
    if not grounded:
        result["warning"] = (
            "Best available extraction is low-confidence / weakly grounded — the resolved "
            "papers may not describe this enzyme's catalytic mechanism. Treat the nucleophile "
            "and residues as tentative and, if it matters, supply a specific mechanism/structure "
            "paper via `paper=`.")
    return result
