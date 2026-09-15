"""Standalone validation: profile-guided selection of papers for full-text reading.

Problem found in run 20260911_121301: the full-text harvest reads the first ~6-8
DOWNLOADABLE papers in native accumulation order — there is NO relevance ranking
(semantic reranker is off by default and sentence-transformers is absent, so
_semantic_rerank returns native order). So the deep-read budget is spent on
whatever accumulated first, not on the most search-guiding papers.

This prototype scores each paper (TITLE + ABSTRACT only — cheap, no full-text
reads) against the target's DERIVED ENZYMOLOGY PROFILE (from target_reasoning.json
+ mechanism_families.json), never against any known answer. Objective: put the
papers that best GUIDE the search (relevant enzyme class / mechanism / reaction)
at the top of the same small read budget.

Answer-agnostic by construction: the profile terms come from the target chemistry
and the agent's own mechanism hypotheses. SfaB/answer terms are NOT used; where
the SfaB paper lands is reported only as an incidental byproduct, never scored for.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

RUN = Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_discovery_agent_improved/"
           "runs/20260911_121301_find_amide_synthetases_that_ligate_adipic_acid_and_"
           "hexamethylenediamine_for_nylo")


def recover_pool() -> list[dict]:
    """The paper pool in native accumulation order (what the harvest walks)."""
    seen, pool = set(), []
    for path in sorted(glob.glob(str(RUN / "artifacts/literature/*.json"))):
        try:
            art = json.load(open(path))
        except Exception:
            continue
        data = art.get("result", {}).get("data", [])
        for p in data if isinstance(data, list) else []:
            if not isinstance(p, dict):
                continue
            k = (p.get("pmid"), p.get("doi"), (p.get("title") or "").lower())
            if k in seen or not p.get("title"):
                continue
            seen.add(k)
            pool.append(p)
    return pool


def build_profile() -> dict:
    """Enzymology profile from the agent's own derived artifacts (firewall-clean).

    Returns weighted positive phrases, negative/disqualifier phrases, and the
    enzyme-signal gate terms. No answer (SfaB/accession/organism) terms."""
    tr = json.loads((RUN / "target_reasoning.json").read_text())
    mf = json.loads((RUN / "mechanism_families.json").read_text())

    # Strong multi-word enzymology signals (the transformation + its mechanism).
    strong = [
        "amide bond synthetase", "amide synthetase", "amido synthetase",
        "amide bond formation", "amide bond forming", "amide ligase",
        "acyl-adenylate", "acyl adenylate", "acyl-amp", "acyl-phosphate",
        "adenylate-forming", "adenylate forming", "atp-grasp", "atp-dependent",
        "amide bond synthesis", "amidation", "peptide bond formation",
    ]
    # Weaker single-token signals derived from the contract fields.
    weak = set()
    for b in tr.get("bond_types", []):
        weak.add(b.lower())                       # amide
    weak.update(["synthetase", "ligase", "ligation", "adenyl", "synthetase",
                 "atp", "condensation", "conjugat", "biocataly", "carboxyl",
                 "diamine", "dicarboxyl", "amine"])
    # reaction direction words
    weak.update(["synthesis", "forming", "formation", "biosynthesis"])
    # enzyme-class words the AGENT hypothesized (answer-agnostic mechanism families)
    for fam in (mf if isinstance(mf, list) else mf.get("families", []) or []):
        name = (fam.get("name") or "").lower() if isinstance(fam, dict) else ""
        for tok in re.findall(r"[a-z][a-z\-]{3,}", name):
            if tok not in ("fold", "type", "like", "amine", "amide"):
                weak.add(tok)

    # Must show SOME enzyme/catalysis signal to be worth a deep read at all.
    gate = ["enzym", "synthetase", "ligase", "biocataly", "catalyz", "catalytic",
            "biosynthe", "adenyl"]

    # Disqualifiers: wrong direction + off-domain (materials / medical) noise.
    negative = [
        # wrong direction (contract disqualifier: hydrolysis of amides)
        ("hydrolase", 4), ("hydrolysis", 4), ("hydrolyz", 4), ("depolymeriz", 4),
        ("degradation", 3), ("deconstruction", 3), ("recycling", 2),
        # materials-science polyamide chemistry (amide bond, but no enzyme)
        ("poly(amide", 5), ("amide-imide", 5), ("polyimide", 5), ("copolyamide", 5),
        ("membrane", 4), ("nanofiber", 4), ("electrospun", 4), ("elastomer", 4),
        ("dielectric", 4), ("resin", 3), ("coating", 3), ("thermal propert", 4),
        # medical / off-domain
        ("cancer", 4), ("tumor", 4), ("antiviral", 4), ("clinical", 3),
        ("alzheimer", 4), ("imaging", 3), ("inhibitor", 2), ("drug", 2),
    ]
    return {"strong": strong, "weak": sorted(weak), "gate": gate, "negative": negative,
            "raw_query": tr.get("query", "")}


def score(paper: dict, prof: dict) -> float:
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    if not any(g in title or g in abstract for g in prof["gate"]):
        return float("-inf")                       # no enzyme signal -> not a read
    s = 0.0
    for ph in prof["strong"]:
        if ph in title:
            s += 5
        elif ph in abstract:
            s += 2
    for tok in prof["weak"]:
        if tok in title:
            s += 3
        elif tok in abstract:
            s += 1
    for tok, w in prof["negative"]:
        if tok in title:
            s -= w
        elif tok in abstract:
            s -= w * 0.5
    return s


def main() -> None:
    pool = recover_pool()
    prof = build_profile()
    print(f"pool: {len(pool)} papers (native accumulation order)\n")

    print("=== CURRENT selection (native order, first 8 that are the mined set) ===")
    mined = {r.get("title", "")[:50] for r in
             json.load(open(RUN / "artifacts/fulltext/papers_enzymes.json"))}
    for i, p in enumerate(pool):
        if p.get("title", "")[:50] in mined:
            print(f"  #{i:<3} {p['title'][:78]}")

    ranked = sorted(pool, key=lambda p: score(p, prof), reverse=True)
    print("\n=== PROPOSED selection: top 8 by enzymology-profile score ===")
    for p in ranked[:8]:
        print(f"  {score(p, prof):5.1f}  {p['title'][:78]}")

    # Incidental byproduct check ONLY (not part of the objective):
    for rank, p in enumerate(ranked):
        if "streptoalloteichus" in (p.get("title") or "").lower():
            print(f"\n[byproduct check] SfaB paper moves native #? -> profile rank "
                  f"#{rank} ('{p['title'][:55]}')")
    # McbA papers, likewise incidental
    mcba = [(r, p) for r, p in enumerate(ranked) if "mcba" in (p.get("title") or "").lower()]
    for r, p in mcba[:3]:
        print(f"[byproduct check] McbA paper at profile rank #{r}: {p['title'][:55]}")


if __name__ == "__main__":
    main()
