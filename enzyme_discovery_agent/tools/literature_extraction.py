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
    "You extract grounded facts about a characterized enzyme from its paper. "
    "For the enzyme identified by the given deposited sequence accession (and any "
    "co-characterized enzymes deposited alongside it), return ONLY JSON:\n"
    '{"enzymes":[{"name":"..","genbank":"..",'
    '"native_substrates":["exact chemical names resolvable to a structure"],'
    '"reaction":{"bond_formed":"amide|ester|thioester|peptide|other",'
    '"direction":"formation|cleavage","cofactor":"ATP|none|..",'
    '"reactive_groups":"which group on each substrate forms the bond",'
    '"summary":"<=25 words"}}]}\n'
    "native_substrates are the ACTUAL acids/amines/acceptors the enzyme acts on, "
    "named precisely (e.g. 'cadaverine', '5-chlorovaleric acid'), not generic "
    "classes. If the paper does not characterize the enzyme's reaction, return "
    '{"enzymes":[]} — do not guess.'
)


def extract_from_paper(
    paper_text: str, deposited_accession: str, protein_name: str, llm_client,
    max_chars: int = 48000,
) -> list[dict]:
    """LLM-extract native substrates + mechanism for the paper's enzyme(s).

    Returns a list of ``{name, genbank, native_substrates, reaction}`` dicts
    (empty if the paper does not characterize the reaction — which also flags the
    paper as irrelevant for this candidate).
    """
    if not paper_text:
        return []
    user = (
        f"This paper deposited sequence accession {deposited_accession} "
        f"(UniProt annotation: {protein_name or 'uncharacterized'}). Extract each "
        f"characterized enzyme's native/accepted substrates and reaction "
        f"mechanism.\n\nPAPER:\n{paper_text[:max_chars]}"
    )
    from enzyme_discovery_agent import config
    response = llm_client.call(
        messages=[{"role": "system", "content": _EXTRACT_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=config.LLM_TEMPERATURE,
        timeout=max(config.LLM_TIMEOUT, 500),
    )
    if response is None:
        return []
    content = response.choices[0].message.content or ""
    try:
        obj = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
    except (AttributeError, json.JSONDecodeError):
        logger.debug("literature extraction: unparseable response for %s", deposited_accession)
        return []
    return obj.get("enzymes", []) or []


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
) -> dict | None:
    """Full recovery for one candidate: link paper → extract → resolve SMILES.

    Returns ``{native_substrates:[{name,smiles}], mechanism:{...}, paper:{...},
    paper_relevant: bool}`` or None when no paper link exists.
    """
    paper = find_candidate_paper(Path(run_dir), source_tool)
    if not paper:
        return None
    try:
        text = Path(paper["text_path"]).read_text(errors="ignore")
    except OSError:
        return None
    records = extract_from_paper(text, paper["deposited_accession"], protein_name, llm_client)
    rec = _match_enzyme_record(records, paper["deposited_accession"])
    if not rec:
        # No extractable characterized reaction → the paper is not a relevant
        # characterization of this enzyme.
        return {"native_substrates": [], "mechanism": None, "paper": paper,
                "paper_relevant": False}
    substrates = []
    for name in (rec.get("native_substrates") or []):
        smiles = resolve_substrate_names_to_smiles(name)
        substrates.append({"name": name, "smiles": smiles})
    return {
        "native_substrates": substrates,
        "mechanism": rec.get("reaction"),
        "enzyme_name_in_paper": rec.get("name"),
        "paper": paper,
        "paper_relevant": True,
    }
