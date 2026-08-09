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
    for txt in ft_dir.glob("*.txt"):
        try:
            if acc in txt.read_text(errors="ignore"):
                rec = next((r for r in manifest if r.get("text_file") == txt.name), {})
                return {
                    "deposited_accession": acc,
                    "text_path": str(txt),
                    "doi": rec.get("doi", ""),
                    "title": rec.get("title", ""),
                }
        except OSError:
            continue
    return None


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
    '     "name":"..", "genbank":"..",\n'
    '     "native_substrates":["exact chemical names resolvable to a structure"],\n'
    '     "reaction":{"bond_formed":"amide|ester|thioester|peptide|other",'
    '"direction":"formation|cleavage","cofactor":"ATP|none|..",'
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
    "Rules: native_substrates = the ACTUAL acids/amines/acceptors, each named as a "
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


def extract_from_paper(
    paper_text: str, deposited_accession: str, protein_name: str, llm_client,
    query: str = "", max_chars: int = 48000,
) -> dict:
    """LLM-extract from a paper's full text, in one focused pass:

      * ``paper_relevant`` + ``relevance_reason`` — does it characterise a
        query-type enzyme reaction (right chemistry AND direction)?  This is the
        proper relevance filter: a semantic score cannot tell an amide-*synthetase*
        paper from a nylon-*depolymerisation* one, but the mechanism can;
      * ``enzymes`` — the characterised enzyme(s): native substrates + mechanism;
      * ``related_enzymes`` — other query-type enzymes named in the BODY (which
        abstract-level extraction misses) → discovery leads.

    Returns ``{}`` on an empty/unparseable response.
    """
    if not paper_text:
        return {}
    user = (
        f"Query reaction of interest: {query or 'enzymatic reaction'}\n\n"
        f"This paper deposited sequence accession {deposited_accession} "
        f"(UniProt annotation: {protein_name or 'uncharacterized'}). Extract per "
        f"the schema.\n\nPAPER:\n{paper_text[:max_chars]}"
    )
    from enzyme_discovery_agent import config
    response = llm_client.call(
        messages=[{"role": "system", "content": _EXTRACT_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=config.LLM_TEMPERATURE,
        timeout=max(config.LLM_TIMEOUT, 500),
    )
    if response is None:
        return {}
    content = response.choices[0].message.content or ""
    try:
        return json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
    except (AttributeError, json.JSONDecodeError):
        logger.debug("literature extraction: unparseable response for %s", deposited_accession)
        return {}


def _match_enzyme_record(records: list[dict], deposited_accession: str) -> dict | None:
    """Pick the extracted enzyme record for this candidate's deposited accession.

    Matches on the GenBank accession (version-insensitive); if only one enzyme
    was characterized, use it.
    """
    base = deposited_accession.split(".")[0].lower()
    for rec in records:
        gb = str(rec.get("genbank", "")).split(".")[0].lower()
        if gb and gb == base:
            return rec
    return records[0] if len(records) == 1 else None


def extract_substrates_and_mechanism(
    run_dir: Path, accession: str, source_tool: str, protein_name: str, llm_client,
    query: str = "",
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
    obj = extract_from_paper(text, paper["deposited_accession"], protein_name,
                             llm_client, query=query)
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
    rec = _match_enzyme_record(obj.get("enzymes") or [], paper["deposited_accession"])
    if not rec:
        return {**base, "paper_relevant": False, "native_substrates": [],
                "mechanism": None, "catalytic_residues": [], "characterization": [],
                "relevance_reason": base["relevance_reason"] or "no characterised query-type enzyme found"}
    substrates = [{"name": n, "smiles": resolve_substrate_names_to_smiles(n)}
                  for n in (rec.get("native_substrates") or [])]
    return {
        **base,
        "paper_relevant": True,
        "native_substrates": substrates,
        "mechanism": rec.get("reaction"),
        "catalytic_residues": rec.get("catalytic_residues") or [],
        "characterization": rec.get("characterization") or [],
        "enzyme_name_in_paper": rec.get("name"),
    }
