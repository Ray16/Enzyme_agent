"""Unified report generator for confidence-weighted and explorative reports."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.notebook import get_notebook_text_for_report
from enzyme_discovery_agent.prompts.report_prompts import (
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


def _write_deterministic_report(
    run_dir: Path,
    filename: str,
    query: str,
    all_evals: list[dict],
    ranking: list[dict] | None = None,
) -> None:
    """Write a factual report when a provider returns an empty completion.

    A ranked candidate artifact is more useful than an empty Markdown file and
    avoids silently treating a zero-token provider response as successful
    scientific reporting.  This renderer deliberately preserves individual
    evidence fields rather than manufacturing a composite score.
    """
    by_accession = {
        item.get("data", item).get("accession", ""): item.get("data", item)
        for item in all_evals
    }
    if ranking is None:
        ranking = _deterministic_evidence_ranking(query, all_evals)
    # Keep the machine-readable ranking synchronized with the visible report.
    import json as _json
    (run_dir / "pass1_ranking.json").write_text(_json.dumps(ranking, indent=2) + "\n")

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
    that only distinguishes copies of one enzyme (``puuA``, ``puuA-I``,
    ``puuA-II``, ``puuA_1``, ``puuA_2``) is stripped so they collapse, while
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


def _best_fg_similarity(data: dict) -> float | None:
    """Max functional-group similarity across the target substrates, if computed.

    This is the substrate-accommodation signal: a pocket evolved for a very
    different functional-group profile will not accept the target even with the
    right chemistry.  Returns None when substrate SMILES were unavailable (the
    axis is simply inactive), so callers treat it as neutral rather than bad.
    """
    sim = data.get("substrate_similarity") or {}
    comps = sim.get("property_comparisons") or []
    vals = [c.get("functional_group_similarity") for c in comps
            if isinstance(c.get("functional_group_similarity"), (int, float))]
    return max(vals) if vals else None


def _deterministic_mechanistic_ranking(
    rankable_evals: list[dict], tier_score: dict[int, float],
) -> list[dict]:
    """Rank by mechanistic compatibility deterministically — the reliable spine.

    A single LLM call cannot reliably rank 100+ candidates (a thinking model
    exhausts the completion budget on reasoning tokens and returns empty), so
    the LLM ranking is best-effort only; THIS is the fallback that must never
    mis-order.  It applies the P0–P5 framework as a strict lexicographic
    priority over GROUNDED FACTS (no weighted composite):

      P0 hard gate  — any critical_warning → bottom, Tier 3.
      P1 substrate  — higher functional-group similarity ranks higher
                      (neutral when substrate SMILES were unavailable).
      P2 curation   — Swiss-Prot reviewed ranks higher.
      P3 depth      — more characterisation papers ranks higher.
      P4 structure  — experimental PDB > AlphaFold > none.
      P5 engineering— more engineering papers ranks higher.
    """
    scored = []
    for ev in rankable_evals:
        d = ev.get("data", ev)
        acc = d.get("accession", "")
        if not acc:
            continue
        gated = _candidate_is_gated(d)
        fg = _best_fg_similarity(d)
        fg_key = -(fg if fg is not None else 0.5)  # None → neutral midpoint
        lit = d.get("literature") or {}
        struct = d.get("structural_data") or {}
        struct_rank = 0 if (struct.get("pdb_ids")) else (1 if struct.get("has_alphafold") else 2)
        key = (
            1 if gated else 0,
            fg_key,
            0 if d.get("is_swissprot") else 1,
            -int(lit.get("characterization_papers") or 0),
            struct_rank,
            -int(lit.get("engineering_papers") or 0),
            acc,
        )
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
        fg = _best_fg_similarity(d)
        bits = []
        if gated:
            bits.append("HARD-GATED: " + "; ".join(d.get("critical_warnings", []))[:120])
        if fg is not None:
            bits.append(f"FG-sim={fg:.2f}")
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
) -> str:
    """Request a concise LLM report after an empty long-form completion."""
    import json as _json

    ranking = ranking or _deterministic_evidence_ranking(query, all_evals)
    facts = []
    for item in all_evals:
        data = item.get("data", item)
        facts.append({
            "accession": data.get("accession"),
            "protein_name": data.get("protein_name"),
            "gene_names": data.get("gene_names"),
            "organism": data.get("organism"),
            "ec_numbers": data.get("ec_numbers"),
            "native_activity": (data.get("catalytic_activities") or [""])[0],
            "critical_warnings": data.get("critical_warnings") or [],
        })
    prompt = (
        "Write a concise Markdown enzyme-candidate report using only the evidence below. "
        "Return Markdown text, beginning with '#'. Do not return JSON, tool calls, or an empty response. "
        "Preserve the supplied ranking and distinguish direct evidence from inference.\n\n"
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

    # ── Pass 1: single mechanistic ranking call ───────────────────────────
    if log_fn:
        log_fn(f"  Pass 1: mechanistic ranking of {len(rankable_evals)} candidates...")

    messages = build_mechanistic_ranking_prompt(
        query=query,
        all_evals=rankable_evals,
        notebook_context=notebook_text,
    )

    response = llm_client.call(
        messages=messages,
        # Sonnet 5 rejects the legacy OpenAI temperature parameter.  The
        # ranking prompt is deterministic in structure; provider defaults are
        # preferable to an incompatible request parameter.
        temperature=config.LLM_TEMPERATURE,
        timeout=max(config.LLM_TIMEOUT, 900),
        # One JSON row per candidate: the whole ranked list must fit in a single
        # completion, or it truncates and the two-pass gate is lost.  Scale the
        # ceiling to the candidate count (~40 tokens/row) with a generous floor.
        max_tokens=max(8192, len(rankable_evals) * 60),
    )

    tier_score = {1: 8.0, 2: 5.0, 3: 2.0}
    parsed = None
    if response is not None:
        token_usage.record("pass1_ranking", response)
        content = response.choices[0].message.content or ""
        parsed = _parse_scoring_response(content)

    # The single whole-list LLM ranking is BEST-EFFORT only: a thinking model
    # can spend the entire completion budget on reasoning tokens and return
    # empty content (finish_reason=length), and a long list can truncate.  When
    # it does not yield a usable ranking for most candidates, fall back to the
    # DETERMINISTIC GATED ranking — never to an ungated single-pass prose
    # report, which is what previously let direction-mismatch hydrolases (an
    # amide *cleavage* enzyme for an amide *formation* target) float to the top.
    usable = bool(parsed) and len(parsed) >= 0.6 * max(1, len(rankable_evals))
    if usable:
        if log_fn:
            log_fn(f"  Pass 1: LLM ranked {len(parsed)} candidates")
        pass1_ranking = []
        for entry in parsed:
            tier = entry.get("tier", 2)
            pass1_ranking.append({
                "rank": entry.get("rank", 0),
                "accession": entry.get("accession", ""),
                "gene": entry.get("gene", ""),
                "score": tier_score.get(tier, 5.0),
                "tier": tier,
                "rationale": entry.get("rationale", ""),
            })
        pass1_ranking.sort(key=lambda e: e.get("rank", 9999))
    else:
        if log_fn:
            got = 0 if not parsed else len(parsed)
            log_fn(f"  Pass 1 LLM ranking unusable ({got}/{len(rankable_evals)} rows); "
                   f"using deterministic gated ranking")
        pass1_ranking = _deterministic_mechanistic_ranking(rankable_evals, tier_score)

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
    # The same enzyme in many organisms was occupying many top slots (7 of the
    # top 10 were PuuA copies).  Experimentalists want distinct enzyme families,
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
