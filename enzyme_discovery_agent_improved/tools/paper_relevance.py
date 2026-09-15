"""Enzymology-profile paper ranking for full-text harvest selection.

The full-text harvest can only deep-read a small number of papers, so WHICH ones
it reads decides what the run learns.  Previously the harvest walked papers in
native accumulation order (the semantic reranker is off by default / its optional
model is usually absent, so it returned the input order unchanged), spending the
budget on whatever happened to arrive first.

This module ranks papers by how well they inform the SEARCH — i.e. how closely a
paper's title+abstract matches the target transformation's ENZYMOLOGY, taken
entirely from the run's own derived reasoning (``target_reasoning.json`` +
``mechanism_families.json``).  It is:

  * answer-agnostic  — terms come from the target chemistry and the agent's own
    mechanism hypotheses, never from any known answer enzyme/accession;
  * direction-aware  — a formation target down-ranks hydrolysis papers and a
    cleavage target down-ranks synthesis papers, from ``reaction_direction``;
  * cheap + always-on — title/abstract lexical scoring, no embedding model, no LLM;
  * safe            — degrades to the previous ordering when the contract is
    missing or yields no signal, so it can only help.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("enzyme_agent")

# Generic, target-INDEPENDENT signals (no leakage: not derived from any answer).
_ENZYME_GATE = (
    "enzym", "synthetase", "synthase", "ligase", "hydrolase", "transferase",
    "oxidoreductase", "lyase", "isomerase", "biocataly", "catalyz", "catalytic",
    "biosynthe", "adenylat", "active site", "crystal structure",
)
# Off-domain noise any enzyme-discovery run wants to avoid reading deeply.
_OFFDOMAIN = (
    "poly(amide", "amide-imide", "polyimide", "copolyamide", "membrane",
    "nanofiber", "electrospun", "elastomer", "dielectric", "resin", "coating",
    "thermal propert", "cancer", "tumor", "antiviral", "clinical trial",
    "alzheimer", "imaging", "drug delivery",
)
# Mechanism tokens worth recognizing inside free-text activation descriptions.
_MECH_TOKENS = (
    "acyl-adenylate", "acyl adenylate", "acyl-amp", "acyl-phosphate",
    "adenylate", "adenylation", "thioester", "anhydride", "phosphoryl", "acyl-coa",
)
_STOP = {"fold", "type", "like", "amine", "amide", "bond", "acid", "free",
         "small", "molecule", "the", "and", "with", "via", "for"}


def build_enzymology_profile(target_reasoning: dict, mechanism_families=None) -> dict:
    """Build the scoring profile from the derived contract (see module docstring)."""
    tr = target_reasoning or {}
    direction = (tr.get("reaction_direction") or "").lower()
    is_formation = "form" in direction or "synth" in direction or "ligat" in direction
    is_cleavage = ("cleav" in direction or "hydrol" in direction
                   or "degrad" in direction or "depoly" in direction)

    strong: set[str] = set()
    weak: set[str] = set()

    # Bond-type phrases, direction-aware.
    for bond in (tr.get("bond_types") or []):
        b = str(bond).lower().strip()
        if not b:
            continue
        weak.add(b)
        if is_cleavage:
            strong.update({f"{b} hydrolase", f"{b} bond cleavage",
                           f"{b} hydrolysis", f"{b} bond hydrolysis"})
        else:  # default to formation
            strong.update({f"{b} bond synthetase", f"{b} synthetase",
                           f"{b} ligase", f"{b} bond formation",
                           f"{b} bond synthesis", f"{b} bond forming"})

    # Direction verbs.
    if is_cleavage:
        weak.update(["hydrolysis", "hydrolase", "hydrolyz", "cleavage",
                     "degradation", "depolymeriz"])
    else:
        weak.update(["synthesis", "synthetase", "synthase", "ligase", "ligation",
                     "forming", "formation", "biosynthesis", "condensation",
                     "conjugat", "amidation"])

    # Activation-mechanism tokens, from the contract's free-text categories.
    cat_text = " ".join(str(c).lower() for c in (tr.get("activation_categories") or []))
    cof_text = " ".join(str(c).lower() for c in (tr.get("cofactor_requirements") or []))
    blob = cat_text + " " + cof_text
    for tok in _MECH_TOKENS:
        if tok in blob:
            strong.add(tok)
    for cof in ("atp", "gtp", "adenylate", "adenosine triphosphate", "coa",
                "nucleoside triphosphate", "magnesium"):
        if cof in blob:
            weak.add(cof)

    # Reactive-group tokens.
    rg = {str(g).lower() for g in (tr.get("reactive_groups") or [])}
    if "carboxyl" in rg:
        weak.update(["carboxyl", "carboxylic acid", "carboxylate"])
    if any("amine" in g for g in rg):
        weak.update(["amine", "amino"])
    if "carbonyl" in rg:
        weak.add("carbonyl")

    # Enzyme-class tokens the AGENT hypothesized (answer-agnostic mechanism families).
    fams = mechanism_families if isinstance(mechanism_families, list) else \
        (mechanism_families or {}).get("families", []) if mechanism_families else []
    for fam in fams or []:
        name = (fam.get("name") or "").lower() if isinstance(fam, dict) else ""
        for tok in re.findall(r"[a-z][a-z\-]{3,}", name):
            if tok not in _STOP:
                weak.add(tok)

    # Opposite-direction terms become disqualifiers.
    if is_cleavage:
        negative = ["synthetase", "synthesis", "ligase", "ligation", "forming"]
    elif is_formation:
        negative = ["hydrolase", "hydrolysis", "hydrolyz", "depolymeriz",
                    "degradation", "deconstruction"]
    else:
        negative = []

    return {
        "strong": sorted(strong),
        "weak": sorted(weak - set(negative)),
        "negative_dir": negative,
        "offdomain": list(_OFFDOMAIN),
        "gate": list(_ENZYME_GATE),
        "usable": bool(strong or weak),
    }


def score_paper(paper: dict, profile: dict) -> float:
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    if not (title or abstract):
        return float("-inf")
    # Must show some enzyme/catalysis signal to be worth a deep read.
    if not any(g in title or g in abstract for g in profile["gate"]):
        return float("-inf")
    s = 0.0
    for ph in profile["strong"]:
        if ph in title:
            s += 5
        elif ph in abstract:
            s += 2
    for tok in profile["weak"]:
        if tok in title:
            s += 3
        elif tok in abstract:
            s += 1
    for tok in profile["negative_dir"]:
        if tok in title:
            s -= 4
        elif tok in abstract:
            s -= 2
    for tok in profile["offdomain"]:
        if tok in title:
            s -= 4
        elif tok in abstract:
            s -= 2
    return s


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def rank_papers_by_enzymology(run_dir, papers: list[dict], fallback):
    """Return ``papers`` ordered most-search-guiding first.

    ``fallback`` is a zero-arg callable returning the previous ordering; it is
    used verbatim when the contract is absent or the profile has no usable signal,
    so this can never do worse than the prior behaviour.
    """
    if not papers:
        return []
    tr = _load_json(Path(run_dir) / "target_reasoning.json")
    if not tr:
        return fallback()
    mf = _load_json(Path(run_dir) / "mechanism_families.json")
    profile = build_enzymology_profile(tr, mf)
    if not profile["usable"]:
        return fallback()
    scored = [(score_paper(p, profile), i, p) for i, p in enumerate(papers)]
    # Papers with no enzyme signal (-inf) sink to the end but are retained.
    scored.sort(key=lambda t: (t[0] != float("-inf"), t[0], -t[1]), reverse=True)
    logger.info("Harvest ranking: enzymology-profile order (%d strong / %d weak terms)",
                len(profile["strong"]), len(profile["weak"]))
    return [p for _, _, p in scored]
