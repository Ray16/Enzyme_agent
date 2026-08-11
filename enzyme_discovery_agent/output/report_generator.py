"""Unified report generator for confidence-weighted and explorative reports."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.notebook import get_notebook_text_for_report
from enzyme_discovery_agent.prompts.report_prompts import (
    EXPLORATIVE_REPORT_SYSTEM,
    build_confidence_report_prompt,
    build_explorative_report_prompt,
    build_mechanistic_ranking_prompt,
    build_pass2_profile_prompt,
    build_scoring_prompt,
    _pass1_compact,
)

logger = logging.getLogger("enzyme_agent")


def strip_preamble(text: str) -> str:
    """Strip LLM preamble before the first markdown header."""
    idx = text.find("\n#")
    if idx == -1:
        idx = text.find("# ")
    if idx > 0:
        return text[idx:].lstrip("\n")
    return text


def is_error_text(text: str) -> bool:
    """Check if text is an error/timeout message rather than a real report."""
    t = text.strip()
    return (
        t.startswith("LLM timed out")
        or t.startswith("LLM failed")
        or len(t) < 100
    )


def _looks_like_report(text: str, ranking: list[dict], min_hits: int = 4) -> bool:
    """A real candidate report covers several ranked candidates, not just a header.

    A thinking model at temperature intermittently returns a truncated stub
    (e.g. just the title + one line) that clears ``is_error_text``'s 100-char
    floor but is not a usable report.  Require the prose to actually mention
    several of the ranked accessions/genes before accepting it; otherwise the
    caller falls through to a deterministic render built directly from data.
    """
    if is_error_text(text):
        return False
    hits = 0
    for entry in ranking[: max(min_hits * 3, 12)]:
        acc = entry.get("accession") or ""
        gene = entry.get("gene") or ""
        if (acc and acc in text) or (gene and len(gene) > 2 and gene in text):
            hits += 1
            if hits >= min_hits:
                return True
    return hits >= min_hits


def _write_deterministic_report(
    run_dir: Path,
    filename: str,
    query: str,
    all_evals: list[dict],
    ranking: list[dict] | None = None,
    ranking_filename: str = "pass1_ranking.json",
) -> None:
    """Write a factual report when a provider returns an empty completion.

    A ranked candidate artifact is more useful than an empty Markdown file and
    avoids silently treating a zero-token provider response as successful
    scientific reporting.  This renderer deliberately preserves individual
    evidence fields rather than manufacturing a composite score.

    ``ranking_filename`` names the machine-readable ranking to (re)write so the
    explorative fallback persists its own chemistry ranking without clobbering
    the confidence report's ``pass1_ranking.json``.
    """
    by_accession = {
        item.get("data", item).get("accession", ""): item.get("data", item)
        for item in all_evals
    }
    if ranking is None:
        ranking = _deterministic_evidence_ranking(query, all_evals)
    # Keep the machine-readable ranking synchronized with the visible report.
    import json as _json
    (run_dir / ranking_filename).write_text(_json.dumps(ranking, indent=2) + "\n")

    lines = [
        "# Candidate evaluation report",
        "",
        "## Target",
        "",
        query,
        "",
        "## Ranked candidates",
        "",
        "This report was rendered directly from the completed evaluation artifacts because "
        "the report-model response was empty. Rankings and evidence are preserved; no "
        "composite score has been introduced.",
        "",
        "| Rank | Accession | Enzyme | Gene | Organism | Tier | EC | Native activity | Rationale |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    for item in ranking:
        data = by_accession.get(item.get("accession", ""), {})
        if not data:
            continue
        gene = _first_gene_label(data)
        ec = ", ".join(data.get("ec_numbers") or []) or "—"
        activity = (data.get("catalytic_activities") or ["—"])[0].replace("|", "\\|")
        rationale = str(item.get("rationale") or "—").replace("|", "\\|")
        lines.append(
            f"| {item.get('rank', '—')} | {data.get('accession', '—')} | "
            f"{data.get('protein_name', '—')} | {gene} | {data.get('organism', '—')} | "
            f"{item.get('tier', '—')} | {ec} | {activity} | {rationale} |"
        )

    lines.extend(["", "## Candidate evidence", ""])
    ordered_accessions = [r.get("accession", "") for r in ranking]
    for accession in ordered_accessions:
        data = by_accession.get(accession)
        if not data:
            continue
        name = data.get("protein_name") or accession
        lines.extend([f"### {accession} — {name}", ""])
        if data.get("catalytic_activities"):
            lines.append(f"- Native activity: {data['catalytic_activities'][0]}")
        if data.get("critical_warnings"):
            lines.append("- Critical warnings: " + "; ".join(data["critical_warnings"]))
        if data.get("weaknesses"):
            lines.append("- Limitations: " + "; ".join(data["weaknesses"]))
        structures = (data.get("structural_data") or {}).get("pdb_ids") or []
        if structures:
            lines.append("- Experimental structures: " + ", ".join(structures))
        lines.append("")

    (run_dir / filename).write_text("\n".join(lines) + "\n")


def _deterministic_evidence_ranking(query: str, all_evals: list[dict]) -> list[dict]:
    """Rank fallback reports by categorical evidence precedence, never a score.

    In a polymer/oligomer target, demonstrated cleavage of a *linear oligomer*
    is stronger evidence than activity on a dimer, which is in turn stronger
    than activity restricted to a cyclic dimer.  This fixes the misleading
    file-order fallback where a cyclic-dimer enzyme could appear first.
    """
    polymer_target = any(word in query.lower() for word in ("polymer", "oligomer"))

    def classify(data: dict) -> tuple[int, str, str]:
        text = " ".join(
            [data.get("protein_name", "")]
            + (data.get("catalytic_activities") or [])
            + [item.get("name", "") for item in data.get("native_substrates", [])]
        ).lower()
        bond_types = set((data.get("reaction_chemistry") or {}).get("bond_types") or [])
        amide_cleavage = "amide" in bond_types and (
            (data.get("reaction_chemistry") or {}).get("reaction_direction") == "cleavage"
        )
        protein_name = data.get("protein_name", "").lower()
        cyclic = "cyclic" in text or "lactam" in text
        # An equation may use [n] for a chain while the curated enzyme name
        # still identifies a dimer-specific hydrolase.  The latter is the
        # more specific substrate-scope annotation.
        named_oligomer_hydrolase = "oligomer" in protein_name
        dimer_specific = "dimer" in protein_name
        oligomer = ("oligomer" in text or "[n" in text) and not dimer_specific
        dimer = "dimer" in text
        if polymer_target and amide_cleavage and (named_oligomer_hydrolase or oligomer) and not cyclic:
            return 0, "1", "Direct linear-oligomer amide-hydrolysis precedent."
        if polymer_target and amide_cleavage and dimer and not cyclic:
            return 1, "1", "Direct linear-dimer/short-oligomer amide-hydrolysis precedent."
        if polymer_target and amide_cleavage and cyclic and dimer:
            return 2, "2", "Cyclic-dimer amide-hydrolysis precedent; weaker for a linear polymer backbone."
        if amide_cleavage:
            return 3, "2", "Amide-cleavage chemistry, but without demonstrated linear oligomer scope."
        return 4, "3", "Mechanistic analogue without native linear-amide-oligomer evidence."

    ranked = []
    for data in (item.get("data", item) for item in all_evals):
        precedence, tier, rationale = classify(data)
        structures = (data.get("structural_data") or {}).get("pdb_ids") or []
        # Categorical precedence is decisive.  Curation and experimental
        # structure only resolve ties within the same evidence category.
        ranked.append((
            precedence,
            0 if data.get("is_swissprot") else 1,
            -len(structures),
            data.get("accession", ""),
            tier,
            rationale,
        ))
    ranked.sort()
    return [
        {"rank": i, "accession": accession, "tier": tier, "rationale": rationale}
        for i, (_, _, _, accession, tier, rationale) in enumerate(ranked, start=1)
    ]


def _ortholog_key(data: dict) -> str | None:
    """Group true orthologs — the SAME enzyme in different organisms — together.

    Uses the normalised gene symbol plus the primary EC number.  Suffix noise
    that only distinguishes copies of one enzyme (e.g. ``geneX``, ``geneX-1``,
    ``geneX-II``, ``geneX_2``) is stripped so they collapse, while
    genuinely different enzymes that merely share a gene letter or an EC do NOT
    (different gene stem or different EC → different key).  Only i/v/x are
    treated as roman suffixes so a gene legitimately ending in 'c'/'d'/'m'
    (murC, iucC) is left intact.
    """
    genes = data.get("gene_names") or []
    gene = (genes[0] if genes else "").lower().strip()
    gene = re.sub(r"[\s_\-]*(?:[ivx]+|\d+)$", "", gene)  # drop paralog/copy suffix
    ec = (data.get("ec_numbers") or [""])[0]
    if gene and ec:
        return f"{gene}|{ec}"
    if gene:
        return f"gene:{gene}"
    name = re.sub(r"\b\d+\b", "", (data.get("protein_name") or "").lower()).strip()
    name = re.sub(r"\s+", " ", name)
    return f"name:{name}" if name else None


def _collapse_orthologs(
    pass1_ranking: list[dict], acc_to_eval: dict[str, dict],
) -> list[dict]:
    """Collapse same-enzyme orthologs into one representative entry.

    Experimentalists want DISTINCT enzyme families at the top, not the same
    enzyme repeated across a dozen organisms (one enzyme was filling 7 of the
    top 10 slots).  The best-ranked member of each ortholog group stays as the
    entry; the others are attached as ``orthologs`` (accession + organism) so
    the enzyme is represented once but its organism spread is preserved, not
    lost.  Rank order is otherwise unchanged.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for entry in pass1_ranking:
        acc = entry.get("accession", "")
        ev = acc_to_eval.get(acc, {})
        d = ev.get("data", ev)
        key = _ortholog_key(d) or f"acc:{acc}"
        if key not in groups:
            entry.setdefault("orthologs", [])
            groups[key] = entry
            order.append(key)
        else:
            rep = groups[key]
            rep.setdefault("orthologs", []).append({
                "accession": acc,
                "organism": d.get("organism", ""),
                "gene": entry.get("gene", ""),
            })
    collapsed = [groups[k] for k in order]
    for i, entry in enumerate(collapsed, start=1):
        entry["rank"] = i
        if entry.get("orthologs"):
            entry["n_orthologs"] = len(entry["orthologs"])
    return collapsed


def _candidate_is_gated(data: dict) -> bool:
    """A hard mechanistic contradiction (direction / bond-type / severe-size
    mismatch) disqualifies a candidate from the top, whatever else it has."""
    return bool(data.get("critical_warnings"))


def _accommodation_score(data: dict) -> float | None:
    """Substrate-accommodation fit in [0,1], or None when not computed.

    The enzyme must bind EVERY target substrate, and a pocket accommodates a
    substrate only if it fits on ALL physical axes — so:

      * the fit of one (target-substrate → best-native) pair is the WORST of its
        grounded similarities (functional groups, aliphatic chain length, size),
        NOT just functional-group overlap; and
      * the overall score is the WORST across the target substrates.

    A native that is aromatic / cyclic where the target is aliphatic is further
    capped, because a ring system the pocket evolved around will not admit a
    linear aliphatic chain.  This bottleneck view (grounded facts, no tuned
    weights) demotes an enzyme whose native merely shares a functional-group
    TYPE with the target but is otherwise much larger, aromatic, or rigid — a
    bulky native scored against a small flexible target, which functional-group
    overlap alone wrongly rewarded.  Returns None when
    substrate SMILES were unavailable, so callers treat it as neutral.
    """
    sim = data.get("substrate_similarity") or {}
    comps = sim.get("property_comparisons") or []
    if not comps:
        return None
    per_substrate = []
    for c in comps:
        dims = [c.get(k) for k in (
            "functional_group_similarity", "chain_length_similarity", "mw_similarity",
        )]
        dims = [d for d in dims if isinstance(d, (int, float))]
        if not dims:
            continue
        fit = min(dims)
        if c.get("aromaticity_match") is False:   # aromatic native, aliphatic target
            fit = min(fit, 0.3)
        if c.get("cyclicity_match") is False:     # ring native, acyclic target
            fit = min(fit, 0.5)
        per_substrate.append(fit)
    return min(per_substrate) if per_substrate else None


def _accommodation_facts_payload(data: dict) -> list[dict]:
    """Substrate-accommodation evidence for the LLM to REASON over (not a score).

    Per target substrate (matched to its closest native): functional-group TYPES
    and COUNTS (target vs native), aliphatic chain length, molecular weight, and
    whether the native carries aromatic/ring character the target lacks.  The LLM
    decides accommodation from these; the code never collapses them to a number.
    """
    sim = data.get("substrate_similarity") or {}
    comps = sim.get("property_comparisons") or []
    subs = []
    for c in comps:
        subs.append({
            "target_fg_counts": {k: v for k, v in (c.get("functional_group_counts_target") or {}).items() if v},
            "native_fg_counts": {k: v for k, v in (c.get("functional_group_counts_native") or {}).items() if v},
            "target_aliphatic_chain": c.get("chain_length_target"),
            "native_aliphatic_chain": c.get("chain_length_native"),
            "target_mw": c.get("mw_target"),
            "native_mw": c.get("mw_native"),
            "native_has_aromatic_target_lacks": c.get("aromaticity_match") is False,
            "native_has_ring_target_lacks": c.get("cyclicity_match") is False,
        })
    return subs


def _candidate_evidence_payload(data: dict) -> dict:
    """The full grounded evidence the LLM ranks a promising candidate on.

    Four groups, so promise = accommodates the substrate AND is the right
    reaction AND is actually usable: substrate-accommodation facts (primary),
    reaction/mechanism, experimental provenance, and structure/expression
    feasibility.  All are facts pulled from the evaluation; no scoring here.
    """
    lit = data.get("literature") or {}
    struct = data.get("structural_data") or {}
    mut = data.get("mutagenesis") or {}
    expr = data.get("expression_data") or {}
    chem = data.get("reaction_chemistry") or {}
    cats = data.get("catalytic_activities") or []
    return {
        "accession": data.get("accession"),
        "gene": (data.get("gene_names") or [""])[0],
        "protein_name": data.get("protein_name", ""),
        "organism": data.get("organism", ""),
        "ec": data.get("ec_numbers") or [],
        "family": data.get("interpro_family", ""),
        "native_substrates": [n.get("name") for n in data.get("native_substrates", []) if n.get("name")][:6],
        "reaction": {
            "native_reaction": (cats[0][:160] if cats else ""),
            "bond_types": chem.get("bond_types"),
            "direction": chem.get("reaction_direction"),
        },
        "substrate_accommodation": _accommodation_facts_payload(data),
        "provenance": {
            "swissprot_reviewed": bool(data.get("is_swissprot")),
            "characterization_papers": lit.get("characterization_papers", 0),
            "engineering_papers": lit.get("engineering_papers", 0),
            "n_mutations_studied": mut.get("n_mutations", 0),
        },
        "feasibility": {
            "has_pdb": bool(struct.get("pdb_ids")),
            "has_alphafold": bool(struct.get("has_alphafold")),
            "ecoli_expression": bool(expr.get("ecoli_expression")),
            "bacterial_expression": bool(expr.get("bacterial_expression")),
            "eukaryotic_origin": bool(expr.get("is_eukaryote")),
        },
    }


def _llm_focus_rank(
    shortlist: list[dict], query: str, llm_client, token_usage, log_fn=None,
) -> list[dict] | None:
    """Have the LLM RANK the shortlisted promising candidates by reasoning over
    grounded substrate-accommodation facts.  Returns parsed rows or None.

    Kept small (this model returns empty output above ~10-12 items when asked to
    reason and emit a structured list), which is why we focus on the most
    promising candidates rather than ranking the whole pool.
    """
    payload = [_candidate_evidence_payload(ev.get("data", ev)) for ev in shortlist]
    system = (
        "You are ranking a SHORTLIST of the most promising enzyme candidates for a "
        "target reaction. Reason over the grounded EVIDENCE given per candidate — "
        "never fame or name recognition. Weigh, in this order:\n"
        "1. SUBSTRATE ACCOMMODATION (primary): can the pocket physically fit BOTH "
        "target substrates? Compare functional-group types AND counts, aliphatic "
        "chain length, molecular weight, and aromatic/ring character. A pocket "
        "evolved for a large, aromatic, cyclic, or phosphorylated native substrate "
        "CANNOT admit a small linear aliphatic molecule even when one functional-"
        "group type overlaps. A native matching on group counts AND size AND "
        "aliphatic character is a genuine lead.\n"
        "2. REACTION/MECHANISM: it should form (not cleave) the target bond type via "
        "the target chemistry; a cleavage/hydrolysis direction is a strong negative.\n"
        "3. PROVENANCE: Swiss-Prot review, characterization and engineering papers, "
        "and studied mutations increase confidence and engineerability.\n"
        "4. FEASIBILITY: available structure (PDB/AlphaFold) and bacterial/E. coli "
        "expression ease downstream work.\n\n"
        "Rank the most promising first. Return ONLY a JSON array, one object per "
        "line:\n"
        '[{"accession":"..","rank":1,"tier":1,"reason":"<=20 words citing the '
        'specific evidence"}]\n'
        "tier 1 = strong lead, 2 = partial, 3 = weak."
    )
    user = (f"Target reaction / substrates: {query}\n\n"
            f"Shortlist ({len(payload)} candidates) with evidence:\n"
            + json.dumps(payload, default=str))
    response = llm_client.call(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=config.LLM_TEMPERATURE,
        timeout=max(config.LLM_TIMEOUT, 600),
    )
    if response is None:
        return None
    token_usage.record("focus_rank", response)
    content = response.choices[0].message.content or ""
    parsed = _parse_scoring_response(content)
    if not parsed and log_fn:
        log_fn("  Focus-rank: LLM returned no parseable ranking; using fallback order")
    return parsed


def _deterministic_mechanistic_ranking(
    rankable_evals: list[dict], tier_score: dict[int, float],
    evidence_tiebreak: bool = True,
    family_rank_by_acc: dict[str, int] | None = None,
) -> list[dict]:
    """Rank by mechanistic compatibility deterministically — the reliable spine.

    A single LLM call cannot reliably rank 100+ candidates (a thinking model
    exhausts the completion budget on reasoning tokens and returns empty), so
    the LLM ranking is best-effort only; THIS is the fallback that must never
    mis-order.  It applies the P0–P5 framework as a strict lexicographic
    priority over GROUNDED FACTS (no weighted composite):

      P0 hard gate  — any critical_warning → bottom, Tier 3.
      PF family     — the agent's grounded fold decision: an on-target fold
                      (expanded / direction == same) ranks above a neutral fold,
                      which ranks above a stopped (off-target) fold.  This makes
                      the run's own family navigation authoritative over ranking,
                      ahead of the fame signals below (neutral when no family
                      landscape exists, so a plain run is unaffected).
      P1 substrate  — higher functional-group similarity ranks higher
                      (neutral when substrate SMILES were unavailable).
      P2 curation   — Swiss-Prot reviewed ranks higher.
      P3 depth      — more characterisation papers ranks higher.
      P4 structure  — experimental PDB > AlphaFold > none.
      P5 engineering— more engineering papers ranks higher.

    ``evidence_tiebreak`` controls P2–P5: when False the order is decided by
    the hard gate, family relevance, and substrate chemistry alone (evidence
    ignored), which is the suitability-only spine the explorative report ranks
    on — a curation/citation-agnostic view of which native chemistry is closest
    to the target.  ``family_rank_by_acc`` maps accession → {0 on-target,
    1 neutral, 2 stopped}; absent accessions default to neutral.
    """
    family_rank_by_acc = family_rank_by_acc or {}
    scored = []
    for ev in rankable_evals:
        d = ev.get("data", ev)
        acc = d.get("accession", "")
        if not acc:
            continue
        gated = _candidate_is_gated(d)
        fam_rank = family_rank_by_acc.get(acc, 1)
        fit = _accommodation_score(d)
        fit_key = -(fit if fit is not None else 0.5)  # None → neutral midpoint
        lit = d.get("literature") or {}
        struct = d.get("structural_data") or {}
        struct_rank = 0 if (struct.get("pdb_ids")) else (1 if struct.get("has_alphafold") else 2)
        if evidence_tiebreak:
            key = (
                1 if gated else 0,
                fam_rank,
                fit_key,
                0 if d.get("is_swissprot") else 1,
                -int(lit.get("characterization_papers") or 0),
                struct_rank,
                -int(lit.get("engineering_papers") or 0),
                acc,
            )
        else:
            # Suitability-only: hard gate, family relevance, then substrate
            # chemistry, then a stable accession tiebreak.  No curation signal.
            key = (1 if gated else 0, fam_rank, fit_key, acc)
        scored.append((key, acc, d, gated))
    scored.sort(key=lambda x: x[0])

    clean = [s for s in scored if not s[3]]
    n_clean = len(clean)
    t1 = max(1, int(round(n_clean * 0.2)))
    t2 = max(t1, int(round(n_clean * 0.7)))

    ranking = []
    clean_seen = 0
    for i, (_, acc, d, gated) in enumerate(scored, start=1):
        if gated:
            tier = 3
        else:
            tier = 1 if clean_seen < t1 else (2 if clean_seen < t2 else 3)
            clean_seen += 1
        fit = _accommodation_score(d)
        bits = []
        if gated:
            bits.append("HARD-GATED: " + "; ".join(d.get("critical_warnings", []))[:120])
        if fit is not None:
            bits.append(f"accommodation={fit:.2f}")
        if d.get("is_swissprot"):
            bits.append("Swiss-Prot")
        cp = (d.get("literature") or {}).get("characterization_papers")
        if cp:
            bits.append(f"{cp} char papers")
        ranking.append({
            "rank": i,
            "accession": acc,
            "gene": (d.get("gene_names") or [""])[0],
            "score": tier_score.get(tier, 5.0),
            "tier": tier,
            "rationale": " | ".join(bits) or "mechanistic analogue",
        })
    return ranking


def _retry_compact_llm_report(
    llm_client: LLMClient,
    query: str,
    all_evals: list[dict],
    ranking: list[dict] | None = None,
    suitability_first: bool = False,
) -> str:
    """Request a concise LLM report after an empty long-form completion.

    ``suitability_first`` switches the compact prompt to the explorative
    philosophy: the supplied ranking is only the candidate POOL and the model
    RE-RANKS by biochemical suitability alone (bond/direction/substrate fit),
    ignoring curation/citation depth for ordering — so the compact fallback,
    which is what actually ships when the long-form call returns empty, still
    produces a chemistry-first view distinct from the confidence report.
    """
    import json as _json

    ranking = ranking or _deterministic_evidence_ranking(query, all_evals)
    facts = []
    for item in all_evals:
        data = item.get("data", item)
        fact = {
            "accession": data.get("accession"),
            "protein_name": data.get("protein_name"),
            "gene_names": data.get("gene_names"),
            "organism": data.get("organism"),
            "ec_numbers": data.get("ec_numbers"),
            "native_activity": (data.get("catalytic_activities") or [""])[0],
            "critical_warnings": data.get("critical_warnings") or [],
        }
        if suitability_first:
            # Give the model the substrate signal it needs to re-rank by chemistry.
            subs = data.get("native_substrates") or []
            fact["native_substrates"] = "; ".join(
                s.get("name", "") for s in subs if s.get("name")
            )
            fit = _accommodation_score(data)
            if fit is not None:
                fact["substrate_accommodation"] = round(fit, 2)
        facts.append(fact)
    if suitability_first:
        instruction = (
            "Write a concise Markdown SUITABILITY-FOCUSED (explorative) enzyme-candidate "
            "report using only the evidence below. Return Markdown text, beginning with '#'. "
            "Do not return JSON, tool calls, or an empty response. Treat the ranking below as "
            "the candidate POOL only: RE-RANK candidates by biochemical suitability for the "
            "target reaction ALONE — how closely each native chemistry matches the target bond "
            "formed, reaction direction, and substrate size/topology/functional groups. Do NOT "
            "let curation status or citation depth change a rank position; note those only as a "
            "confidence label. Push any candidate with a bond-type/direction/size critical "
            "warning to the bottom. Always name each enzyme, never a bare accession. "
            "Distinguish direct evidence from inference.\n\n"
        )
    else:
        instruction = (
            "Write a concise Markdown enzyme-candidate report using only the evidence below. "
            "Return Markdown text, beginning with '#'. Do not return JSON, tool calls, or an empty response. "
            "Preserve the supplied ranking and distinguish direct evidence from inference.\n\n"
        )
    prompt = (
        instruction +
        f"Target:\n{query}\n\nRanking:\n{_json.dumps(ranking, indent=2)}\n\n"
        f"Candidate facts:\n{_json.dumps(facts, indent=2)}"
    )
    response = llm_client.call(
        messages=[{"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE,
        timeout=max(config.LLM_TIMEOUT, 300),
    )
    if response is None:
        return ""
    return strip_preamble(response.choices[0].message.content or "")


def _first_gene_label(candidate: dict) -> str:
    """Return a safe display label when UniProt has no gene annotation."""
    genes = candidate.get("gene_names") or []
    return str(genes[0]) if genes else "?"


def _select_calibration_anchors(
    all_compact: list[dict],
    n_anchors: int = 5,
) -> list[dict]:
    """Select calibration anchor candidates spanning the quality range.

    Selects programmatically based on data signals — no hardcoded names:
    - 1 best: highest tanimoto AND no critical warnings AND has_pdb
    - 1 worst: has critical warnings
    - 1 high-tanimoto no-PDB: good substrate match, limited structural data
    - 1 low-tanimoto swissprot: well-characterised but poor substrate match
    - 1 middle: median tanimoto, some characterisation

    Returns at most n_anchors unique candidates.
    """
    import statistics as _stats

    def tanimoto(c):
        return c.get("tanimoto") or 0.0

    with_warnings = [c for c in all_compact if c.get("critical_warnings")]
    without_warnings = [c for c in all_compact if not c.get("critical_warnings")]

    anchors = []
    used_accs = set()

    def add(candidate):
        if candidate and candidate["accession"] not in used_accs:
            anchors.append(candidate)
            used_accs.add(candidate["accession"])

    # Anchor 1: best overall (high tanimoto, swissprot, has_pdb, no warnings)
    best = sorted(
        without_warnings,
        key=lambda c: (
            tanimoto(c),
            c.get("is_swissprot", False),
            c.get("pdb_count", 0),
            c.get("char_papers", 0),
        ),
        reverse=True,
    )
    add(best[0] if best else None)

    # Anchor 2: worst (critical warning — bottom tier)
    if with_warnings:
        add(with_warnings[0])

    # Anchor 3: high tanimoto but no structural data (promising but uncharacterised)
    no_struct = sorted(
        [c for c in without_warnings if c.get("pdb_count", 0) == 0
         and not c.get("has_alphafold")],
        key=lambda c: tanimoto(c),
        reverse=True,
    )
    add(no_struct[0] if no_struct else None)

    # Anchor 4: swissprot, well-characterised, but low tanimoto (known but mismatch)
    well_char = sorted(
        [c for c in without_warnings if c.get("is_swissprot") and
         c.get("char_papers", 0) >= 2],
        key=lambda c: tanimoto(c),
    )
    add(well_char[0] if well_char else None)

    # Anchor 5: middle tanimoto (median)
    sorted_by_tan = sorted(without_warnings, key=tanimoto)
    if sorted_by_tan:
        mid_idx = len(sorted_by_tan) // 2
        add(sorted_by_tan[mid_idx])

    return anchors[:n_anchors]


def _parse_scoring_response(text: str) -> list[dict] | None:
    """Parse a JSON array from an LLM scoring response.

    Tolerant of truncation: a whole-list ranking can be cut off by the
    completion-token cap mid-array, leaving no closing ``]``.  Rather than
    discarding the entire ranking (and dropping to a worse ungated path), we
    recover every complete ``{...}`` object that precedes the truncation point.
    Returns None only when no complete object can be recovered.
    """
    import re as _re
    import json as _json

    start = text.find("[")
    if start == -1:
        return None
    # Fast path: a well-formed, complete array.
    json_match = _re.search(r'\[\s*\{.*\}\s*\]', text, _re.DOTALL)
    if json_match:
        try:
            return _json.loads(json_match.group())
        except Exception:
            pass
    # Recovery path: scan object-by-object from the first '[', keeping every
    # top-level {...} that parses, and stop at the first incomplete one.
    recovered: list[dict] = []
    depth = 0
    obj_start = -1
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start != -1:
                try:
                    recovered.append(_json.loads(text[obj_start:i + 1]))
                except Exception:
                    break
                obj_start = -1
    return recovered or None


def _check_anchor_drift(
    batch_scores: list[dict],
    batch_index: int,
    anchor_accs: set[str],
    reference_scores: dict[str, float],
    log_fn=None,
    drift_threshold: float = 2.0,
) -> None:
    """Compare anchor scores in this batch against reference scores. Log if drift > threshold."""
    for entry in batch_scores:
        acc = entry.get("accession")
        if acc not in anchor_accs:
            continue
        score = entry.get("score")
        if score is None or acc not in reference_scores:
            continue
        drift = abs(score - reference_scores[acc])
        if drift > drift_threshold and log_fn:
            log_fn(
                f"  WARNING: calibration drift batch {batch_index}: "
                f"{acc} scored {score:.1f} vs reference {reference_scores[acc]:.1f} "
                f"(drift={drift:.1f} > {drift_threshold})"
            )


# ---------------------------------------------------------------------------
# Family-relevance binding: the agent's grounded stop/expand decisions over the
# fold landscape become AUTHORITATIVE over ranking.  A member of a fold the agent
# STOPPED (native chemistry unlike the target) is hard-gated to the bottom like a
# mechanistic contradiction; a member of an on-target fold (agent-expanded, or
# direction == "same" in the grounded landscape) is floated into Tier-1
# eligibility, so a from-scratch LLM re-rank can no longer promote a stopped-
# family enzyme on characterisation depth alone.  Answer-agnostic by
# construction: the folds and their relevance come from the run's OWN discovered
# candidates measured against its OWN target — no hardcoded family.
# ---------------------------------------------------------------------------

def _registry_fold_map(run_dir: Path) -> dict[str, set[str]]:
    """accession -> {InterPro fold ids}, from screened-candidate registry
    metadata (the light screen call persists ``interpro_ids`` there for exactly
    the evaluated set).  Empty dict when the registry is unavailable."""
    try:
        from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
        reg = load_candidate_registry(run_dir).get("candidates", {})
    except Exception:  # noqa: BLE001 - ranking must never crash on a registry read
        return {}
    out: dict[str, set[str]] = {}
    for v in reg.values():
        if not isinstance(v, dict):
            continue
        acc = v.get("accession")
        ids = (v.get("metadata") or {}).get("interpro_ids")
        if acc and ids:
            out.setdefault(acc, set()).update(i for i in ids if i)
    return out


def _family_relevance_partition(
    run_dir: Path, all_evals: list[dict],
) -> tuple[set[str], set[str], dict[str, str]]:
    """Partition candidate accessions by the agent's grounded family decisions.

    Returns ``(stopped_accs, ontarget_accs, stop_reason_by_acc)``:
      * ``stopped_accs`` — members of a fold the agent marked ``stop`` (off-target
        native chemistry); hard-gated to the bottom of the ranking.
      * ``ontarget_accs`` — members of an agent-``expand``ed fold, or of a fold the
        grounded landscape scored direction ``same`` vs the target; Tier-1 eligible.
    A stop takes precedence over on-target (the agent's explicit judgement wins).
    Degrades to empty sets when the landscape / decisions are absent, so a run
    without family navigation ranks exactly as it did before.
    """
    try:
        from enzyme_discovery_agent.models.family_decisions import load_family_decisions
        decisions = load_family_decisions(run_dir)
    except Exception:  # noqa: BLE001
        decisions = {}
    stopped_folds, stop_reason_by_fold, ontarget_folds = set(), {}, set()
    for fid, v in decisions.items():
        if not isinstance(v, dict):
            continue
        if v.get("decision") == "stop":
            stopped_folds.add(fid)
            stop_reason_by_fold[fid] = v.get("reason") or ""
        elif v.get("decision") == "expand":
            ontarget_folds.add(fid)
    # The grounded landscape adds folds whose members natively run the SAME
    # direction as the target even where the agent issued no explicit expand.
    try:
        budget = json.loads((run_dir / "family_budget.json").read_text())
        for f in budget if isinstance(budget, list) else []:
            fid = f.get("fold_id")
            if (fid and f.get("direction_vs_target") == "same"
                    and not f.get("agent_stopped")):
                ontarget_folds.add(fid)
    except (OSError, ValueError, AttributeError):
        pass
    ontarget_folds -= stopped_folds

    fold_map = _registry_fold_map(run_dir)
    stopped_accs: set[str] = set()
    ontarget_accs: set[str] = set()
    stop_reason_by_acc: dict[str, str] = {}
    for ev in all_evals:
        acc = ev.get("data", ev).get("accession")
        if not acc:
            continue
        ids = fold_map.get(acc)
        if not ids:
            continue
        hit = ids & stopped_folds
        if hit:
            stopped_accs.add(acc)
            stop_reason_by_acc[acc] = stop_reason_by_fold.get(sorted(hit)[0], "")
        elif ids & ontarget_folds:
            ontarget_accs.add(acc)
    return stopped_accs, ontarget_accs, stop_reason_by_acc


def _family_rank_map(stopped_accs: set[str], ontarget_accs: set[str]) -> dict[str, int]:
    """{accession -> 0 on-target | 2 stopped} for the deterministic spine
    (neutral accessions are simply absent → default 1).  Stop dominates."""
    fr = {acc: 0 for acc in ontarget_accs}
    fr.update({acc: 2 for acc in stopped_accs})
    return fr


def generate_two_pass_report(
    run_dir: Path,
    all_evals: list[dict],
    query: str,
    llm_client: LLMClient,
    token_usage: TokenUsage,
    log_fn=None,
    top_n: int = 30,
    batch_size: int = 100,  # unused, kept for API compatibility
    n_anchors: int = 5,    # unused, kept for API compatibility
) -> None:
    """Two-pass confidence report: mechanistic ranking then profile writing.

    Pass 1 (mechanistic ranking):
        - Hard filter: polymer-substrate candidates are flagged and pushed to bottom.
        - Single LLM call over all candidates with a richer compact format:
          catalytic activity text, bond type, per-substrate FG comparison.
        - LLM ranks by mechanistic compatibility, not by re-reading derived scores.

    Pass 2 (report writing):
        - Top `top_n` candidates by LLM rank get full profiles.
        - LLM ranking from Pass 1 is authoritative.

    Falls back to single-pass generation if ranking fails.
    """
    import json as _json

    from enzyme_discovery_agent.models.notebook import get_notebook_text_for_report
    notebook_text = get_notebook_text_for_report(run_dir)

    # Drop failed evaluations (data is None, e.g. an entry that could not be
    # fetched): they are not candidates, and the code below assumes dict data.
    all_evals = [e for e in (all_evals or [])
                 if isinstance(e, dict) and isinstance(e.get("data", e), dict)]

    # ── Hard filter: separate polymer candidates ──────────────────────────
    polymer_evals = [
        ev for ev in all_evals
        if ev.get("data", ev).get("requires_polymer_substrate", False)
    ]
    rankable_evals = [
        ev for ev in all_evals
        if not ev.get("data", ev).get("requires_polymer_substrate", False)
    ]
    if polymer_evals and log_fn:
        polymer_genes = [
            _first_gene_label(ev.get("data", ev))
            for ev in polymer_evals
        ]
        log_fn(f"  Hard filter: {len(polymer_evals)} polymer-substrate candidates "
               f"pushed to bottom ({', '.join(polymer_genes)})")

    # ── Pass 1: FOCUS ranking (code narrows, LLM reasons over evidence) ───
    # A single LLM call cannot rank the whole pool — this thinking model returns
    # empty output above ~10-12 items asked to reason and emit a structured list.
    # So: (1) a deterministic, gated baseline orders everything reliably;
    # (2) we shortlist the most promising fact-bearing, non-contradictory
    # candidates; (3) the LLM REASONS over their full grounded evidence
    # (substrate accommodation, mechanism, provenance, feasibility) to rank just
    # those.  Code narrows; the LLM judges the leads.
    tier_score = {1: 8.0, 2: 5.0, 3: 2.0}
    # Bind the agent's grounded family stop/expand decisions to ranking BEFORE
    # any tiering: on-target-fold members become Tier-1 eligible and stopped-fold
    # members are hard-gated below, so the from-scratch LLM re-rank can no longer
    # promote a stopped-family enzyme (e.g. a heavily-characterised off-target
    # ligase) over an on-target one on fame alone.
    stopped_accs, ontarget_accs, stop_reasons = _family_relevance_partition(
        run_dir, all_evals
    )
    family_rank_by_acc = _family_rank_map(stopped_accs, ontarget_accs)
    if log_fn and (stopped_accs or ontarget_accs):
        log_fn(f"  Family binding: floated {len(ontarget_accs)} on-target-fold "
               f"candidate(s), gated {len(stopped_accs)} stopped-fold candidate(s)")
    baseline = _deterministic_mechanistic_ranking(
        rankable_evals, tier_score, family_rank_by_acc=family_rank_by_acc
    )
    acc_to_eval = {ev.get("data", ev).get("accession", ""): ev for ev in all_evals}

    shortlist = []
    for entry in baseline:
        if len(shortlist) >= config.RANKING_FOCUS_SHORTLIST:
            break
        ev = acc_to_eval.get(entry.get("accession", ""))
        if not ev:
            continue
        d = ev.get("data", ev)
        if _candidate_is_gated(d):
            continue
        if not (d.get("substrate_similarity") or {}).get("property_comparisons"):
            continue
        shortlist.append(ev)

    if log_fn:
        log_fn(f"  Focus ranking: LLM reasoning over the {len(shortlist)} most "
               f"promising candidates (of {len(rankable_evals)})")
    llm_rows = (_llm_focus_rank(shortlist, query, llm_client, token_usage, log_fn)
                if shortlist else None)

    pass1_ranking = []
    if llm_rows:
        short_accs = {ev.get("data", ev).get("accession", "") for ev in shortlist}
        placed = set()
        for r in sorted(llm_rows, key=lambda x: x.get("rank", 9999)):
            acc = r.get("accession", "")
            if acc not in short_accs or acc in placed:
                continue
            placed.add(acc)
            d = acc_to_eval.get(acc, {}).get("data", {})
            tier = r.get("tier", 1)
            pass1_ranking.append({
                "rank": len(pass1_ranking) + 1,
                "accession": acc,
                "gene": (d.get("gene_names") or [""])[0],
                "score": tier_score.get(tier, 5.0),
                "tier": tier,
                "rationale": r.get("reason", "") or "LLM-reasoned lead",
            })
        for ev in shortlist:  # keep any shortlist member the LLM omitted
            acc = ev.get("data", ev).get("accession", "")
            if acc not in placed:
                d = ev.get("data", ev)
                pass1_ranking.append({
                    "rank": len(pass1_ranking) + 1, "accession": acc,
                    "gene": (d.get("gene_names") or [""])[0],
                    "score": tier_score.get(2, 5.0), "tier": 2,
                    "rationale": "Shortlisted; not ordered by the model.",
                })
        if log_fn:
            log_fn(f"  Focus ranking: LLM ranked {len(placed)} promising candidates")
    elif log_fn:
        log_fn("  Focus ranking unavailable; using deterministic gated baseline")

    # Append the rest of the pool (below the focused leads) in deterministic
    # order, so the report still covers everything.
    focus_accs = {e["accession"] for e in pass1_ranking}
    for entry in baseline:
        if entry.get("accession") not in focus_accs:
            entry["rank"] = len(pass1_ranking) + 1
            pass1_ranking.append(entry)

    # Deduplicate (LLM may hallucinate polymer candidates it knows from training)
    seen_accs = set()
    deduped = []
    for entry in pass1_ranking:
        acc = entry.get("accession", "")
        if acc and acc not in seen_accs:
            seen_accs.add(acc)
            deduped.append(entry)
    pass1_ranking = deduped

    # Re-assign ranks after deduplication
    for i, entry in enumerate(pass1_ranking, start=1):
        entry["rank"] = i

    # Recover any candidate the ranker omitted (truncated output, or the LLM
    # simply skipped it): append it rather than silently dropping it from the
    # report.  Missing rows land mid-list at Tier 2 so the hard gate below can
    # still push a mechanistically-inconsistent one to the bottom on evidence.
    ranked_accs = {e.get("accession", "") for e in pass1_ranking}
    rank_cursor = len(pass1_ranking)
    for ev in rankable_evals:
        d = ev.get("data", ev)
        acc = d.get("accession", "")
        if not acc or acc in ranked_accs:
            continue
        rank_cursor += 1
        ranked_accs.add(acc)
        pass1_ranking.append({
            "rank": rank_cursor,
            "accession": acc,
            "gene": (d.get("gene_names") or [""])[0],
            "score": tier_score.get(2, 5.0),
            "tier": 2,
            "rationale": "Not ranked by the model; recovered so it is not dropped.",
        })
    n_recovered = rank_cursor - len(deduped)
    if n_recovered > 0 and log_fn:
        log_fn(f"  Recovered {n_recovered} candidate(s) the ranker omitted "
               f"(truncated or skipped output)")

    # Append polymer candidates at the bottom
    next_rank = len(pass1_ranking) + 1
    for ev in polymer_evals:
        d = ev.get("data", ev)
        acc = d.get("accession", "")
        if acc in seen_accs:
            continue  # already appears in LLM output; skip duplicate
        pass1_ranking.append({
            "rank": next_rank,
            "accession": acc,
            "gene": (d.get("gene_names") or [""])[0],
            "score": 1.0,
            "tier": 3,
            "rationale": "Polymer substrate enzyme — requires primer, cannot ligate two free molecules.",
        })
        next_rank += 1

    # ── Deterministic hard gate: enforce P0 in code, not on the LLM ────────
    # A reaction-direction, bond-type, or severe-size contradiction is a hard
    # mechanistic disqualifier — an amide *hydrolase* cannot be a top candidate
    # for an amide-*formation* target, however deep its characterisation.  The
    # ranking prompt asks the LLM to bottom these (P0), but a generative ranker
    # applies that softly and keeps floating "direct amide-chemistry" cleavage
    # enzymes to the top.  ``critical_warnings`` holds ONLY these three hard
    # contradictions (bond-type / direction / severe-size mismatch), so enforce
    # it deterministically: pin every warned candidate to Tier 3 and sort it
    # below all mechanistically-consistent candidates, preserving the LLM's
    # relative order within each group.
    gated_accs = {
        ev.get("data", ev).get("accession", "")
        for ev in all_evals
        if ev.get("data", ev).get("critical_warnings")
    }
    gated_accs.discard("")
    if gated_accs:
        n_gated = 0
        for entry in pass1_ranking:
            if entry.get("accession") in gated_accs:
                n_gated += 1
                entry["tier"] = 3
                entry["score"] = min(entry.get("score", 2.0), 1.5)
                if "hard-gated" not in entry.get("rationale", "").lower():
                    entry["rationale"] = (
                        "HARD-GATED (mechanistic contradiction — "
                        "reaction-direction / bond-type / size mismatch); ranked "
                        "below all mechanistically consistent candidates. "
                        + entry.get("rationale", "")
                    ).strip()
        pass1_ranking.sort(
            key=lambda e: (
                1 if e.get("accession") in gated_accs else 0,
                e.get("rank", 9999),
            )
        )
        for i, entry in enumerate(pass1_ranking, start=1):
            entry["rank"] = i
        if log_fn:
            log_fn(f"  Hard gate: demoted {n_gated} candidate(s) with a "
                   f"mechanistic contradiction below all consistent candidates")

    # ── Collapse orthologs: one enzyme, one entry ─────────────────────────
    # The same enzyme in many organisms was occupying many top slots (most of
    # the top slots were copies of one enzyme).  Experimentalists want distinct enzyme families,
    # so fold same-enzyme orthologs into a single representative that carries the
    # list of organisms it appears in.  Runs AFTER the gate so a representative
    # inherits the correct tier.
    _acc_to_eval = {ev.get("data", ev).get("accession", ""): ev for ev in all_evals}
    n_before_collapse = len(pass1_ranking)
    pass1_ranking = _collapse_orthologs(pass1_ranking, _acc_to_eval)
    n_folded = n_before_collapse - len(pass1_ranking)
    if n_folded > 0 and log_fn:
        log_fn(f"  Ortholog collapse: folded {n_folded} same-enzyme ortholog(s) "
               f"into representatives ({len(pass1_ranking)} distinct entries)")

    # Coverage is measured over accessions actually represented (each entry
    # plus its folded orthologs), NOT the number of visible entries — ortholog
    # collapse legitimately shrinks the entry count and must not read as
    # truncation.
    n_covered = sum(1 + len(e.get("orthologs", [])) for e in pass1_ranking)
    n_ranked = len(pass1_ranking)
    n_total = len(all_evals)
    if log_fn:
        log_fn(f"  Pass 1 complete: {n_ranked} distinct entries covering "
               f"{n_covered}/{n_total} candidates")

    if n_covered < n_total * 0.8:
        if log_fn:
            log_fn(f"  WARNING: only {n_covered}/{n_total} candidates covered — "
                   f"LLM may have truncated")

    # Persist the durable ranking before asking the provider to write prose.
    # This also makes a report-generation retry/recovery independent of the
    # second provider call.
    with open(run_dir / "pass1_ranking.json", "w") as f:
        _json.dump(pass1_ranking, f, indent=2)

    # ── Pass 2: full profiles for top candidates ──────────────────────────
    acc_to_eval = {
        ev.get("data", ev).get("accession", ""): ev
        for ev in all_evals
    }

    top_accs = [r["accession"] for r in pass1_ranking[:top_n] if r.get("accession")]
    top_evals = [acc_to_eval[acc] for acc in top_accs if acc in acc_to_eval]

    if log_fn:
        log_fn(f"  Pass 2: writing report for top {len(top_evals)} candidates...")

    p2_messages = build_pass2_profile_prompt(
        query, top_evals, pass1_ranking, notebook_text
    )
    p2_response = llm_client.call(
        messages=p2_messages,
        temperature=config.LLM_TEMPERATURE,
        timeout=max(config.LLM_TIMEOUT, 900),
    )

    if p2_response is None:
        report_text = ""
    else:
        token_usage.record("pass2_report", p2_response)
        report_text = strip_preamble(p2_response.choices[0].message.content or "")

    if is_error_text(report_text):
        if log_fn:
            log_fn("  Pass 2 returned an empty/invalid report; retrying with compact LLM prompt")
        report_text = _retry_compact_llm_report(llm_client, query, all_evals, pass1_ranking)
        if is_error_text(report_text):
            raise RuntimeError("LLM returned no substantive report after compact retry")

    with open(run_dir / "report_confidence.md", "w") as f:
        f.write(report_text)

    if log_fn:
        log_fn(f"  Two-pass report: {len(report_text)} chars, "
               f"{n_ranked} candidates ranked, top {len(top_evals)} profiled")


def generate_two_pass_explorative(
    run_dir: Path,
    all_evals: list[dict],
    query: str,
    llm_client: LLMClient,
    token_usage: TokenUsage,
    log_fn=None,
    top_n: int = 30,
) -> None:
    """Suitability-only (explorative) report, rendered by the robust two-pass path.

    Uses the SAME robust top-N render as the confidence report instead of the
    old single giant call that ranked+formatted every candidate at once — that
    call reliably returned empty on a thinking model and dropped the explorative
    report into a generic-evidence fallback (bare accessions, everything Tier 3).

    The candidate POOL is the confidence report's ``pass1_ranking.json`` (an
    LLM-focus-ranked, ortholog-collapsed, robust spine) when present; the model
    then RE-RANKS it by biochemical suitability alone via the explorative system
    prompt (see ``authoritative_ranking=False``).  We do NOT rank on a pure
    chemistry spine in code because the substrate-accommodation signal is often
    too sparse to order candidates (most sit at a neutral default), so a code
    sort would degenerate to alphabetical; the model does the suitability
    ordering by reasoning over the full per-substrate data.
    """
    import json as _json

    notebook_text = get_notebook_text_for_report(run_dir)
    all_evals = [e for e in (all_evals or [])
                 if isinstance(e, dict) and isinstance(e.get("data", e), dict)]
    if not all_evals:
        if log_fn:
            log_fn("  Skipping explorative report -- no evaluations available")
        return

    acc_to_eval = {ev.get("data", ev).get("accession", ""): ev for ev in all_evals}
    tier_score = {1: 8.0, 2: 5.0, 3: 2.0}

    # Prefer the confidence report's durable ranking as the candidate pool; it is
    # robust and ortholog-collapsed.  Fall back to a deterministic mechanistic
    # spine (evidence tiebreak KEPT, so it never degenerates to alphabetical) if
    # the explorative report is generated standalone before the confidence one.
    ranking: list[dict] = []
    p1_path = run_dir / "pass1_ranking.json"
    if p1_path.exists():
        try:
            ranking = _json.loads(p1_path.read_text())
        except (OSError, ValueError):
            ranking = []
    if not ranking:
        polymer_evals = [ev for ev in all_evals
                         if ev.get("data", ev).get("requires_polymer_substrate", False)]
        rankable_evals = [ev for ev in all_evals
                          if not ev.get("data", ev).get("requires_polymer_substrate", False)]
        ranking = _deterministic_mechanistic_ranking(rankable_evals, tier_score)
        next_rank = len(ranking) + 1
        for ev in polymer_evals:
            d = ev.get("data", ev)
            acc = d.get("accession", "")
            if not acc:
                continue
            ranking.append({
                "rank": next_rank, "accession": acc,
                "gene": (d.get("gene_names") or [""])[0],
                "score": 1.0, "tier": 3,
                "rationale": "Polymer substrate enzyme — requires primer, cannot ligate two free molecules.",
            })
            next_rank += 1
        ranking = _collapse_orthologs(ranking, acc_to_eval)
        for i, entry in enumerate(ranking, start=1):
            entry["rank"] = i

    top_accs = [r["accession"] for r in ranking[:top_n] if r.get("accession")]
    top_evals = [acc_to_eval[acc] for acc in top_accs if acc in acc_to_eval]

    if log_fn:
        log_fn(f"  Explorative report: {len(ranking)} candidate pool; model "
               f"re-ranks top {len(top_evals)} by suitability")

    messages = build_pass2_profile_prompt(
        query, top_evals, ranking, notebook_text,
        system_prompt=EXPLORATIVE_REPORT_SYSTEM,
        report_label="suitability-focused (explorative)",
        authoritative_ranking=False,
    )
    temperature = config.EXPLORATIVE_REPORT_TEMPERATURE
    if temperature is None:
        temperature = config.LLM_TEMPERATURE

    report_text = ""
    try:
        response = llm_client.call(
            messages=messages, temperature=temperature,
            timeout=max(config.LLM_TIMEOUT, 900),
        )
        if response is not None:
            token_usage.record("explorative_report", response)
            report_text = strip_preamble(response.choices[0].message.content or "")
    except Exception as exc:
        logger.warning("Explorative pass-2 call failed: %s", exc)

    # The long-form call reliably returns empty on this thinking model, and the
    # compact retry is itself flaky (intermittently a truncated header stub), so
    # retry it a couple of times and accept the first response that actually
    # covers several ranked candidates — not merely clears the 100-char floor.
    if not _looks_like_report(report_text, ranking):
        for attempt in range(2):
            if log_fn:
                log_fn(f"  Explorative report thin; compact retry {attempt + 1}/2")
            report_text = _retry_compact_llm_report(
                llm_client, query, all_evals, ranking, suitability_first=True
            )
            if _looks_like_report(report_text, ranking):
                break

    if not _looks_like_report(report_text, ranking):
        # Never emit a stub: render the chemistry ranking deterministically.
        if log_fn:
            log_fn("  Explorative report still thin; writing deterministic ranking")
        _write_deterministic_report(
            run_dir, "report_explorative.md", query, all_evals, ranking=ranking,
            ranking_filename="explorative_ranking.json",
        )
        return

    (run_dir / "report_explorative.md").write_text(report_text)
    if log_fn:
        log_fn(f"  Explorative report: {len(report_text)} chars")


def generate_report(
    report_type: str,
    run_dir: Path,
    all_evals: list[dict],
    query: str,
    llm_client: LLMClient,
    token_usage: TokenUsage,
    log_fn=None,
) -> None:
    """Generate a report (confidence or explorative) via LLM call.

    Args:
        report_type: "confidence" or "explorative"
        run_dir: Run output directory
        all_evals: List of evaluation result dicts
        query: Original user query
        llm_client: LLM client instance
        token_usage: Token usage accumulator
        log_fn: Optional logging function (msg: str) -> None
    """
    # Drop failed evaluations (data is None): not candidates, and the prompt
    # builders below assume dict data.
    all_evals = [e for e in (all_evals or [])
                 if isinstance(e, dict) and isinstance(e.get("data", e), dict)]

    if not all_evals:
        if log_fn:
            log_fn(f"  Skipping {report_type} report -- no evaluations available")
        return

    if log_fn:
        log_fn(f"  Generating {report_type} report...")

    notebook_text = get_notebook_text_for_report(run_dir)

    if report_type == "explorative":
        messages = build_explorative_report_prompt(query, all_evals, notebook_context=notebook_text)
        temperature = config.EXPLORATIVE_REPORT_TEMPERATURE
        filename = "report_explorative.md"
    else:
        messages = build_confidence_report_prompt(query, all_evals, notebook_context=notebook_text)
        temperature = config.LLM_TEMPERATURE
        filename = "report_confidence.md"

    try:
        response = llm_client.call(
            messages=messages,
            temperature=temperature,
            timeout=max(config.LLM_TIMEOUT, 300),
        )
        if response is None:
            if log_fn:
                log_fn(f"  {report_type.title()} report generation failed (no response)")
            return

        token_usage.record(f"{report_type}_report", response)

        report_text = response.choices[0].message.content or ""
        report_text = strip_preamble(report_text)

        if is_error_text(report_text):
            if log_fn:
                log_fn(f"  {report_type.title()} report was empty; retrying with compact LLM prompt")
            report_text = _retry_compact_llm_report(llm_client, query, all_evals)
            if is_error_text(report_text):
                raise RuntimeError(f"LLM returned no substantive {report_type} report after compact retry")


        with open(run_dir / filename, "w") as f:
            f.write(report_text)

        if log_fn:
            log_fn(f"  {report_type.title()} report: {len(report_text)} chars")

    except Exception as e:
        logger.warning("Failed to generate %s report: %s", report_type, e)
        if log_fn:
            log_fn(f"  {report_type.title()} report generation failed: {e}")
