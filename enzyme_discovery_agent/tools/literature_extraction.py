"""Extract native substrates and reaction mechanism from a candidate's paper.

Curated substrate/mechanism annotation is missing for a large share of real
catalysts (TrEMBL / under-annotated entries whose identity is tied to a paper
only through a deposited sequence accession).  Their substrates and mechanism DO
exist — in the characterization paper's full text, which the run already
downloaded during full-text mining.  This module recovers them:

  1. link a candidate to its paper deterministically, via the deposited GenBank
     accession recorded in ``source_tool`` (``fulltext_genbank:<ACC>``);
  2. have the LLM extract, for that enzyme, its native/accepted substrates and
     reaction mechanism from the paper text (a small, focused call that returns
     reliably — unlike the whole-pool ranking call);
  3. resolve the extracted substrate names to SMILES so they feed the same
     accommodation-facts machinery as curated substrates.

The extracted mechanism is reused three ways: as ranking evidence (is this the
target reaction?), as input to downstream structural modelling (which atoms/
groups react), and as a paper-relevance check (a matching mechanism confirms the
paper is on-topic; no extractable mechanism means it is not).
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
from pathlib import Path

from enzyme_discovery_agent.chemistry_logic import resolve_substrate_names_to_smiles

logger = logging.getLogger("enzyme_agent")

# ``source_tool`` marks full-text-mined candidates as ``fulltext_genbank:<ACC>``
# (or ``fulltext_uniprot``); the GenBank accession is the deterministic link to
# the paper whose full text contains it.
_GENBANK_SOURCE_RE = re.compile(r"fulltext_genbank:([A-Za-z0-9_.]+)")


def _load_fulltext_manifest(run_dir: Path) -> list[dict]:
    path = run_dir / "artifacts" / "fulltext" / "manifest.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def find_candidate_paper(run_dir: Path, source_tool: str) -> dict | None:
    """Locate the paper a full-text-mined candidate came from.

    Returns ``{"deposited_accession", "text_path", "doi", "title"}`` or None when
    the candidate was not recovered from a deposited accession (nothing to link).
    """
    m = _GENBANK_SOURCE_RE.search(source_tool or "")
    if not m:
        return None
    acc = m.group(1)
    ft_dir = run_dir / "artifacts" / "fulltext"
    manifest = _load_fulltext_manifest(run_dir)
    # Read via the manifest's relative text paths (text lives under txt/ in the
    # current layout), rather than globbing the top-level folder.
    for rec in manifest:
        tf = rec.get("text_file")
        if not tf:
            continue
        txt = ft_dir / tf
        try:
            if acc in txt.read_text(errors="ignore"):
                return {
                    "deposited_accession": acc,
                    "text_path": str(txt),
                    "doi": rec.get("doi", ""),
                    "pmcid": rec.get("pmcid", ""),
                    "pmid": rec.get("pmid", ""),
                    "title": rec.get("title", ""),
                }
        except OSError:
            continue
    return None


# Cues that mark the extraction-relevant content, so we send the model the
# high-signal parts of a paper instead of all 48k chars.  Starter set — easy to
# extend.  Windows around these (plus around the enzyme's own identifiers) cover
# the priority sections (Results/Methods/tables/kinetics) even when PDF-extracted
# text has lost its section headings.
_SECTION_HEADINGS = (
    "abstract", "results", "materials and methods", "methods", "experimental",
    "substrate scope", "enzyme characterization", "kinetic",
)
_DATA_CUES = (
    "kcat", "k cat", "km", "k m", "kcat/km", "turnover", "specific activity",
    "conversion", "yield", "titer", "titre", "growth rate", "u/mg", "µmol",
    "umol", "mm ", "mM", "°c", " ph ", "buffer", "incubat", "expressed in",
    "purified", "mutant", "variant", "active site", "catalytic residue",
)


def select_relevant_sections(
    text: str, identifiers: list[str], max_chars: int = 12000, window: int = 700,
    extra_cues: tuple = (),
) -> str:
    """Return a high-signal excerpt: the abstract head plus windows around section
    headings, quantitative/condition cues, and the enzyme's own identifiers.

    Cuts a ~48k-char paper to ~10-12k while keeping what the extraction schema
    needs (substrates, mechanism, catalytic residues, kinetics + conditions,
    cited enzymes).  Section-agnostic so it works on PDF-extracted flat text as
    well as XML-derived text.

    ``extra_cues`` adds caller-specific anchor terms (e.g. mechanism vocabulary
    for the mechanism extractor) so the excerpt centres on the sections that
    caller needs, rather than the default substrate/kinetics-oriented cues.
    """
    if len(text) <= max_chars:
        return text
    low = text.lower()
    spans: list[tuple[int, int]] = [(0, 1500)]  # abstract / opening always kept
    cues = list(_SECTION_HEADINGS) + list(_DATA_CUES) + [c.lower() for c in extra_cues] + [
        i.lower() for i in identifiers if i and len(i) >= 3
    ]
    for cue in cues:
        start = 0
        while True:
            idx = low.find(cue, start)
            if idx == -1:
                break
            spans.append((max(0, idx - window), idx + window))
            start = idx + window  # stride so one dense region is not re-scanned forever
    # Merge overlapping spans, then accumulate up to the budget.
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, total = [], 0
    for s, e in merged:
        chunk = text[s:e]
        if total + len(chunk) > max_chars:
            chunk = chunk[: max_chars - total]
        out.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return "\n[...]\n".join(out)


_EXTRACT_SYSTEM = (
    "You read an enzymology paper (full text) and extract grounded, structured "
    "facts. The reaction of interest is in the query. Report ONLY what the text "
    "states — never guess; omit a field rather than invent it. Return ONLY JSON:\n"
    "{\n"
    '  "paper_relevant": true|false,\n'
    '  "relevance_reason": "<=20 words: does it characterise/discuss an ENZYME for '
    'the query reaction TYPE (right chemistry AND direction)?",\n'
    '  "is_characterization_paper": true|false,\n'
    '  "enzymes": [{\n'
    '     "name":"..", "genbank":"..", "organism":"source organism if stated",\n'
    '     "native_substrates":["exact chemical names resolvable to a structure"],\n'
    '     "evidence_quote":"verbatim passage naming this enzyme and stating its experimentally characterized reaction",\n'
    '     "reaction":{"bond_formed":"bond type explicitly supported by the paper",'
    '"direction":"formation|cleavage|transfer|redox|other","cofactor":"cofactor identity or none",'
    '"reactive_groups":"group on each substrate that forms the bond","summary":"<=25 words"},\n'
    '     "catalytic_residues":[{"residue":"<one-letter/three-letter + position>","role":"<=8 words"}],\n'
    '     "characterization":[{"metric":"kcat|Km|kcat/Km|conversion|yield|'
    'specific_activity|titer|growth_rate|Tm|optimum_pH|optimum_temperature|other",'
    '"value":"..","units":"..","substrate":"which substrate/condition it was '
    'measured on","conditions":{"temperature":"..","pH":"..","buffer":"..",'
    '"substrate_concentration":"..","cofactor":"..","host_or_system":"..","time":".."}}]\n'
    '  }],\n'
    '  "homologs_or_related_enzymes":[{"name":"..","organism":"..",'
    '"relationship":"homolog|same_family|cited_analogue","identity":"% if stated",'
    '"why_relevant":"<=15 words"}]\n'
    "}\n"
    "Rules: native_substrates = the ACTUAL input substrates, each named as a "
    "specific compound (not a generic class), so it resolves to one structure. "
    "catalytic_residues = active-site residues the paper identifies (mutagenesis / "
    "structure), for downstream structural modelling. characterization = MEASURED "
    "results; ALWAYS attach the experimental conditions they were obtained under "
    "(a number without its conditions is not useful). Set is_characterization_paper "
    "false if no kinetics/activity/conversion are reported. "
    "homologs_or_related_enzymes = OTHER query-type enzymes named ANYWHERE in the "
    "body (not just the abstract) worth following up — omit incidental mentions. "
    "Set paper_relevant=false and enzymes=[] when the paper does not concern a "
    "query-type enzyme reaction (purely chemical process, or opposite direction)."
)


_DISCOVERY_EXTRACT_SYSTEM = (
    "Extract discovery leads from the supplied text for the user's target reaction. "
    "Return only JSON with paper_relevant (true/false/null), relevance_reason, "
    "is_characterization_paper, enzymes, and homologs_or_related_enzymes. "
    "For each enzyme include its name, organism, deposited genbank identifier if "
    "explicitly linked, native_substrates, reaction (direction, bond_formed, summary), "
    "and an exact evidence_quote linking that enzyme to its experimentally studied reaction. "
    "For related enzymes include name, organism, relationship, and why_relevant. "
    "Separate characterized enzymes from cited leads. Report only what the text "
    "supports; do not invent identifiers or infer target activity from a name. "
    "Discovery relevance includes the same reaction chemistry on different substrates, "
    "experimentally demonstrated promiscuity, and informative related catalysts. "
    "Lack of a test on the exact target substrates or product does not establish irrelevance. "
    "Record exact-target activity separately as target_activity_demonstrated (true/false/null). "
    "Keep one enzyme per record; do not combine multiple gene names into a single name. "
    "For an unstated organism use an empty string, never a placeholder organism name. "
    "If the text establishes irrelevance, set paper_relevant=false and explain why. "
    "If it is inconclusive, use null. Detailed kinetics and residues will be examined later."
)


def extract_from_paper(
    paper_text: str, deposited_accession: str, protein_name: str, llm_client,
    query: str = "", organism: str = "", max_chars: int = 48000,
    discovery_only: bool = False,
) -> dict:
    """LLM-extract from a paper's full text, in one focused pass:

      * ``paper_relevant`` + ``relevance_reason`` — does it characterise a
        query-type enzyme reaction (right chemistry AND direction)?  This is the
        proper relevance filter: a lexical/semantic score cannot tell a synthesis
        paper from a degradation one on the same material, but the mechanism can;
      * ``enzymes`` — the characterised enzyme(s): native substrates + mechanism;
      * ``homologs_or_related_enzymes`` — other query-type enzymes named in the
        BODY (which abstract-level extraction misses) → discovery leads.

    Returns ``{}`` on an empty/unparseable response.
    """
    if not paper_text:
        return {}
    target = (
        f"This paper deposited sequence accession {deposited_accession} "
        f"(UniProt annotation: {protein_name or 'uncharacterized'}"
        + (f", organism: {organism}" if organism else "") + "). "
        "List EVERY characterised enzyme in the paper (there may be several); for "
        "each give its name, organism, and — where the paper links it — its "
        "deposited accession, so the target enzyme above can be identified by its "
        "accession/organism/name rather than assumed."
    )
    if discovery_only:
        target = "Identify characterized enzymes and related leads explicitly supported by this passage."
    excerpt = select_relevant_sections(paper_text, [deposited_accession, protein_name], max_chars=min(max_chars, 12000)) if discovery_only else paper_text[:max_chars]
    user = (
        f"Query reaction of interest: {query or 'enzymatic reaction'}\n\n"
        f"{target}\n\nPAPER:\n{excerpt}"
    )
    from enzyme_discovery_agent import config
    response = llm_client.call(
        messages=[{"role": "system", "content": _DISCOVERY_EXTRACT_SYSTEM if discovery_only else _EXTRACT_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=config.LLM_TEMPERATURE,
        timeout=config.DISCOVERY_EXTRACTION_TIMEOUT if discovery_only else max(config.LLM_TIMEOUT, 500),
        **({"max_retries": 1, "max_tokens": 2048} if discovery_only else {}),
    )
    if response is None:
        return {}
    content = response.choices[0].message.content or ""
    try:
        return json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
    except (AttributeError, json.JSONDecodeError):
        logger.debug("literature extraction: unparseable response for %s", deposited_accession)
        return {}


_RELEVANCE_SYSTEM = (
    "You gate whether a paper is worth reading during the EXPLORATION stage of an "
    "enzyme-discovery query. Given its title and abstract, keep it only if it is "
    "SUBSTANTIVELY ABOUT AN ENZYME that catalyses the query reaction TYPE — its "
    "catalytic MECHANISM, characterisation, structure/active site, biosynthetic "
    "role, or substrate scope — with the same TRANSFORMATION CHEMISTRY and the same "
    "DIRECTION (forming vs cleaving the bond).\n"
    "Judge by the REACTION TYPE and enzyme MECHANISM, NOT by the specific "
    "substrates named in the query. An enzyme that catalyses the same transformation on "
    "ANALOGOUS substrates (a different but comparable pair of reactants, a model "
    "substrate) IS relevant — substrate scope and specificity are assessed later, "
    "not here. Mechanistically related enzyme families are relevant even without "
    "the exact substrates.\n"
    "REJECT a paper where the enzyme/reaction is only an INCIDENTAL KEYWORD in an "
    "unrelated context — disease pathogenesis, epidemiology, vaccine development, "
    "ecology, or clinical studies that merely mention a shared keyword in passing "
    "without characterising a query-type enzyme; and reject a PURELY CHEMICAL "
    "(non-enzymatic) process or an unrelated bond chemistry / opposite direction.\n"
    "IMPORTANT: an ENZYMATIC or BIOCATALYTIC synthesis paper IS relevant even when "
    "its GOAL is an applied product, polymer, or material (or its monomers/"
    "precursors) — the APPLICATION framing does not make it a materials study; "
    "keep it. Only a materials/chemistry paper with NO enzymatic transformation "
    "is off-topic.\n"
    "When a genuine enzyme paper is borderline on substrate, keep it. "
    'Return ONLY JSON: {"relevant": true|false, "reason": "<=15 words"}.'
)


# Titles that are NOT research articles — conference posters/proceedings, front
# matter, indexes, errata.  Dropped even when abstract-less, so the fail-open
# below keeps real abstract-less primary papers without keeping this junk.
_NON_ARTICLE_RE = re.compile(
    r'^\s*(?:review for|decision letter for|author response for)\b|'
    r"\bposters?\b|poster session|poster abstract|\bproceedings\b|"
    r"table of contents|^\s*contents\s*$|author index|subject index|"
    r"^\s*index\s*$|errat(?:um|a)\b|corrigend|front ?matter|back ?matter|"
    r"supplementary (?:material|information|data)\b|meeting abstracts?|"
    r"conference abstracts?|abstracts? of the\b",
    re.IGNORECASE,
)


def _is_non_article(title: str) -> bool:
    return bool(_NON_ARTICLE_RE.search((title or "").strip()))


def mechanism_gate_query(run_dir, fallback_query: str) -> str:
    """A substrate-free, mechanism-only description of the target for the
    relevance gate: ``<bond> bond <direction>`` derived from the run's reaction
    analysis (e.g. "<bond-type> bond formation (ligation / synthesis)").

    The gate must judge a paper by REACTION TYPE, not by the specific substrates
    or applied product named in the query — but handing it the raw query lets the
    literal product framing dominate, so it drops mechanistically-relevant papers
    that never mention that product.  Distilling to bond + direction removes that
    fixation.  Falls back to the raw query when the bond type is unknown.  Uses
    only the grounded bond-type / direction (no specific family), so it carries no
    answer hint.
    """
    try:
        from enzyme_discovery_agent.tools.family_reactions import _target_profile_for_run
        direction, bonds = _target_profile_for_run(Path(run_dir))
    except Exception:  # noqa: BLE001 - gate must never crash on target parsing
        return fallback_query
    if not bonds:
        return fallback_query
    bond_str = " / ".join(sorted(bonds))
    dir_phrase = {
        "formation": "formation (ligation / synthesis)",
        "cleavage": "cleavage (hydrolysis / degradation)",
    }.get(direction or "", direction or "")
    target = f"an enzyme that catalyses {bond_str} bond {dir_phrase}".strip()
    return target


def is_paper_relevant(title: str, abstract: str, query: str, llm_client) -> tuple[bool, str]:
    """Cheap title+abstract relevance gate (no full-text download).

    Replaces the semantic-score threshold, which cannot tell a synthesis paper
    from a degradation one on the same material.  On any error, returns
    ``(True, ...)`` — fail OPEN so a gate failure never silently drops papers.
    """
    # Drop obvious non-articles (posters, proceedings, front matter, errata) by
    # title first — even abstract-less ones, so the fail-open below does not let
    # this junk through.
    if _is_non_article(title):
        return False, "not a research article (poster/proceedings/front matter)"
    # The abstract is the only content this gate can judge on: a bare title does
    # not reveal whether a paper CHARACTERISES a query-type enzyme, and the model
    # defaults to "not relevant" when handed a title alone — silently dropping
    # abstract-less primary papers (peer-review components, deposit records)
    # that are exactly the ones worth reading.  With no substantive abstract,
    # fail OPEN: keep it for full-text screening, which reads the body anyway.
    if len((abstract or "").strip()) < 40:
        return True, "no abstract to judge — kept for full-text screening"
    from enzyme_discovery_agent import config
    user = (f"Query: {query}\n\nTitle: {title}\n\nAbstract: {abstract[:1500]}")
    try:
        response = llm_client.call(
            messages=[{"role": "system", "content": _RELEVANCE_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=config.LLM_TEMPERATURE,
            timeout=max(config.LLM_TIMEOUT, 120),
        )
        content = response.choices[0].message.content or ""
        obj = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
        return bool(obj.get("relevant", True)), str(obj.get("reason", ""))
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.debug("relevance gate error (failing open): %s", exc)
        return True, "relevance gate error — kept"


_SELECT_SYSTEM = (
    "You choose which scientific papers are most worth reading IN FULL to discover "
    "ENZYMES that catalyse a stated target reaction. You receive the target reaction "
    "described by its REACTION TYPE (not any specific answer) and a numbered list of "
    "candidate papers (title + abstract).\n\n"
    "Select the papers whose FULL TEXT would best help identify or characterise "
    "enzymes that catalyse the target reaction — prefer PRIMARY studies that "
    "discover, characterise, engineer, or resolve the mechanism or structure of such "
    "enzymes. Do NOT select: purely chemical, organometallic, or catalyst-free "
    "synthesis papers; materials-science papers; reviews or perspectives; papers on "
    "unrelated reactions; or papers only about inhibitors. Judge a recent preprint "
    "on its content and never penalise it for being a preprint.\n\n"
    "You are given both the target REACTION TYPE and the APPLICATION CONTEXT from the "
    "request. Rank primarily by relevance to the target reaction, but among otherwise "
    "comparable papers give ADDITIONAL weight to those relevant to the stated "
    "application or end-use — a paper that both concerns the right enzymatic "
    "transformation and targets the application is the most valuable to read. Select "
    "FEWER papers rather than padding the list with weakly-relevant ones.\n\n"
    "Respond with STRICT JSON only: {\"selected\": [<paper indices, most relevant "
    "first>]}."
)


def select_papers_to_read(run_dir, papers, k, llm_client, query, fallback,
                          prefilter=60):
    """LLM-reasoned selection of which candidate papers to read in full.

    The agent (not a keyword scorer) decides which retrieved papers are most
    relevant to the DERIVED target reaction, so it can recognise on-topic primary
    biology and reject chemical-synthesis / materials / review / off-topic papers
    that merely share surface vocabulary.  Answer-agnostic: the target is described
    only by its reaction type (``mechanism_gate_query``), never by any answer entity.

    Robust by construction: pre-trims the pool to ``prefilter`` by the cheap
    enzymology ranker (to bound tokens), then asks for a priority-ordered pick.  On a
    missing client, an empty pool, or any error it returns ``fallback()`` — so it can
    never do worse than the prior selection.  Returns a priority-ordered list of
    papers; the caller downloads down the list until its read budget is met, so a
    paywalled pick is backfilled by the next-most-relevant one.
    """
    if not papers:
        return []
    if llm_client is None:
        return fallback()
    from enzyme_discovery_agent.state_store import atomic_write_json
    target = mechanism_gate_query(run_dir, query)
    context = {"target": target, "query": query, "prompt": _SELECT_SYSTEM,
               "model": str(getattr(llm_client, "active_label", None)
                            or getattr(llm_client, "model_name", None)
                            or type(llm_client).__name__), "k": k, "prefilter": prefilter}
    signature = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    serving_model = [context["model"]]
    def paper_signature(paper):
        return hashlib.sha256(json.dumps(paper, sort_keys=True, default=str).encode()).hexdigest()
    memo_path = Path(run_dir) / "agent_state" / "literature_paper_priority.json"
    try:
        memo = json.loads(memo_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        memo = {}
    order = {key: rank for rank, key in enumerate(memo.get("order", []))}
    keys = [paper_signature(p) for p in papers]
    if memo.get("context_signature") == signature and set(keys) <= set(order):
        # A changing download queue does not invalidate a recorded relevance
        # judgement. New or changed evidence and target/model changes do.
        return [p for _, p in sorted(zip(keys, papers), key=lambda row: order[row[0]])]

    def remember(ranked):
        ranked_keys = [paper_signature(p) for p in ranked]
        if set(keys) <= set(ranked_keys) and not getattr(llm_client, "budget_exhausted", False):
            actual_signature = hashlib.sha256(json.dumps(
                {**context, "model": serving_model[0]}, sort_keys=True).encode()).hexdigest()
            atomic_write_json(memo_path, {"context_signature": actual_signature, "order": ranked_keys})
        return ranked

    from enzyme_discovery_agent.tools.paper_relevance import rank_papers_by_enzymology
    pre = rank_papers_by_enzymology(run_dir, papers, fallback=lambda: papers)
    pre = pre[:max(prefilter, k)]
    listing = []
    for i, p in enumerate(pre):
        title = (p.get("title") or "").strip()[:160]
        abstract = (p.get("abstract") or "").strip()[:200]
        listing.append(f"[{i}] {title}\n{abstract}")
    want = min(len(pre), max(k * 2, k))
    # Hand the model BOTH the transformation (primary) and the application context
    # from the user's own request (secondary) — the raw query is task input, so this
    # carries no hardcoded answer hint, and it lets the model prefer papers that match
    # both the reaction and the intended use (which pure mechanism framing misses).
    user = (
        f"Target reaction (what the enzyme must catalyse): {target}\n"
        f"Application context (from the request; SECONDARY signal): {query}\n\n"
        "Candidate papers:\n" + "\n\n".join(listing) +
        f"\n\nSelect up to {want} papers most useful for discovering enzymes for the "
        "target reaction; among strong matches, give additional weight to papers "
        "relevant to the application context. Fewer if few are relevant; most "
        "relevant first, as their indices."
    )
    from enzyme_discovery_agent import config
    try:
        response = llm_client.call(
            messages=[{"role": "system", "content": _SELECT_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=config.LLM_TEMPERATURE,
            timeout=max(config.LLM_TIMEOUT, 120),
        )
        serving_model[0] = getattr(response, "_enzyme_serving_model", None) or serving_model[0]
        content = response.choices[0].message.content or ""
        obj = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
        idxs = obj.get("selected") or []
        chosen, seen = [], set()
        for i in idxs:
            if isinstance(i, bool):
                continue
            if isinstance(i, int) and 0 <= i < len(pre) and i not in seen:
                seen.add(i)
                chosen.append(pre[i])
        if chosen:
            # Model choices affect order, not recall. A weak model returning one
            # favorite must not hide every other unread primary paper.
            pre_ids = {id(p) for p in pre}
            return remember(chosen + [p for i, p in enumerate(pre) if i not in seen]
                            + [p for p in papers if id(p) not in pre_ids])
    except Exception as exc:  # noqa: BLE001 - fail safe to the keyword order
        logger.debug("LLM paper selection error (falling back): %s", exc)
    # Empty model selections still retain the full evidence pool. Cache the
    # fallback order for a valid response; transient call failures remain retryable.
    ranked = fallback()
    return remember(ranked) if 'response' in locals() and response is not None else ranked


def _match_enzyme_record(
    records: list[dict], deposited_accession: str,
    organism: str = "", protein_name: str = "",
) -> dict | None:
    """Pick the extracted enzyme record for THIS candidate.

    A paper often characterises several enzymes, and the candidate's deposited
    accession is frequently absent from the body text — so matching on the
    accession alone silently returns the wrong enzyme.  Match, in order, on:
    (1) the deposited GenBank accession, (2) the source ORGANISM, (3) the
    protein name.  Only fall back to the sole record when there is exactly one.
    """
    base = deposited_accession.split(".")[0].lower()
    for rec in records:
        if str(rec.get("genbank", "")).split(".")[0].lower() == base and base:
            return rec
    org = (organism or "").lower().strip()
    if org:
        # match on the distinctive genus+species (first two tokens)
        org_key = " ".join(org.split()[:2])
        for rec in records:
            if org_key and org_key in str(rec.get("organism", "")).lower():
                return rec
    name = (protein_name or "").lower().strip()
    _generic = ("uncharacterized protein", "hypothetical protein",
                "putative protein", "")
    if name and name not in _generic:
        for rec in records:
            rn = str(rec.get("name", "")).lower()
            if rn and (rn in name or name in rn):
                return rec
    return records[0] if len(records) == 1 else None


def extract_substrates_and_mechanism(
    run_dir: Path, accession: str, source_tool: str, protein_name: str, llm_client,
    query: str = "", organism: str = "",
) -> dict | None:
    """Full recovery for one candidate: link paper → extract → resolve SMILES.

    Returns ``{native_substrates:[{name,smiles}], mechanism:{...}, paper:{...},
    paper_relevant, relevance_reason, related_enzymes:[...]}`` or None when no
    paper link exists.
    """
    paper = find_candidate_paper(Path(run_dir), source_tool)
    if not paper:
        return None
    try:
        text = Path(paper["text_path"]).read_text(errors="ignore")
    except OSError:
        return None
    # Anchor the excerpt on signals that ARE in the body — the accession, the
    # organism (genus+species), and the protein-name tokens — so the trimmed
    # text centres on THIS enzyme rather than a co-characterised one.
    identifiers = [paper["deposited_accession"]]
    identifiers += [w for w in (protein_name or "").split() if len(w) >= 3]
    identifiers += (organism or "").split()[:2]
    text = select_relevant_sections(text, identifiers=identifiers)
    obj = extract_from_paper(text, paper["deposited_accession"], protein_name,
                             llm_client, query=query, organism=organism)
    related = obj.get("homologs_or_related_enzymes") or []
    base = {
        "paper": paper,
        "relevance_reason": obj.get("relevance_reason", ""),
        "is_characterization_paper": obj.get("is_characterization_paper", False),
        "homologs_or_related_enzymes": related,
    }
    if not obj.get("paper_relevant", True):
        return {**base, "paper_relevant": False, "native_substrates": [],
                "mechanism": None, "catalytic_residues": [], "characterization": []}
    rec = _match_enzyme_record(obj.get("enzymes") or [], paper["deposited_accession"],
                               organism=organism, protein_name=protein_name)
    if not rec:
        return {**base, "paper_relevant": False, "native_substrates": [],
                "mechanism": None, "catalytic_residues": [], "characterization": [],
                "relevance_reason": base["relevance_reason"] or "no characterised query-type enzyme found"}
    substrates = [{"name": n, "smiles": resolve_substrate_names_to_smiles(n)}
                  for n in (rec.get("native_substrates") or [])]
    from enzyme_discovery_agent.models.literature_evidence import build_literature_evidence
    evidence = build_literature_evidence(
        accession, rec, paper, paper["text_path"], identity_basis="deposited_accession")
    return {
        **base,
        "paper_relevant": True,
        "native_substrates": substrates,
        "mechanism": rec.get("reaction"),
        "catalytic_residues": rec.get("catalytic_residues") or [],
        "characterization": rec.get("characterization") or [],
        "enzyme_name_in_paper": rec.get("name"),
        "literature_evidence": evidence,
    }
