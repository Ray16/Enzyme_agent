"""Deterministic mechanistic ranking, ortholog collapsing, and evidence projections.

Extracted from ``output/report_generator.py`` so runtime models, context
projection, CSV export, and report rendering share one domain-layer source of
truth without depending on report formatting.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from enzyme_discovery_agent.models.ortholog import ortholog_key as _ortholog_key
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")

_TIER_LABEL = {
    1: "prioritized hypothesis",
    2: "candidate to investigate",
    3: "deprioritized analogue",
}


def _expression_pref(data: dict) -> int:
    """Prefer documented host expression within a group; origin stays neutral."""
    from enzyme_discovery_agent.models.expression import expression_preference
    return expression_preference(data)


def _collapse_orthologs(
    pass1_ranking: list[dict], acc_to_eval: dict[str, dict],
) -> list[dict]:
    """Collapse verified exact sequences into one representative entry.

    A matching name or gene alone does not establish interchangeable proteins.
    The legacy ``orthologs`` field preserves exact-sequence alternatives and
    their source accessions; it is not a phylogenetic inference. Preserve the
    best group position and the existing representative-selection policy.
    """
    groups: dict[str, list[dict]] = {}
    order: list[str] = []

    def _data(entry: dict) -> dict:
        ev = acc_to_eval.get(entry.get("accession", ""), {})
        return ev.get("data", ev)

    for entry in pass1_ranking:
        key = _ortholog_key(_data(entry)) or f"acc:{entry.get('accession', '')}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(entry)

    collapsed: list[dict] = []
    for key in order:
        members = groups[key]
        # Representative: most expressible first, then best (lowest) original rank.
        rep_src = max(
            members,
            key=lambda e: (_expression_pref(_data(e)), -(e.get("rank") or 1_000_000)),
        )
        best_ranked = min(members, key=lambda e: (e.get("rank") or 1_000_000))
        rep = dict(rep_src)
        rep["orthologs"] = [
            {
                "accession": m.get("accession", ""),
                "organism": _data(m).get("organism", ""),
                "gene": m.get("gene", ""),
            }
            for m in members
            if m is not rep_src
        ]
        # Record why this member represents the group, so the choice is auditable.
        if len(members) > 1:
            overridden = rep_src is not best_ranked
            rep["representative_selection"] = {
                "reason": "e_coli_expressibility" if overridden else "top_ranked",
                "chosen": rep_src.get("accession", ""),
                "expression_pref": _expression_pref(_data(rep_src)),
                "members": [m.get("accession", "") for m in members],
            }
            if overridden:
                logger.info(
                    "Ortholog collapse [%s]: kept %s as representative for E. coli "
                    "expressibility (pref=%d) over higher-ranked %s (rank %s)",
                    key, rep_src.get("accession", ""),
                    _expression_pref(_data(rep_src)),
                    best_ranked.get("accession", ""), best_ranked.get("rank"),
                )
        # Keep the group at the best rank held by any of its members.
        rep["_group_rank"] = min((m.get("rank") or 1_000_000) for m in members)
        collapsed.append(rep)

    collapsed.sort(key=lambda e: e["_group_rank"])
    for i, entry in enumerate(collapsed, start=1):
        entry["rank"] = i
        entry.pop("_group_rank", None)
        if entry.get("orthologs"):
            entry["n_orthologs"] = len(entry["orthologs"])
    return collapsed


def _candidate_is_gated(data: dict) -> bool:
    """A hard mechanistic contradiction (direction / bond-type / severe-size
    mismatch) disqualifies a candidate from the top, whatever else it has."""
    return bool(data.get("critical_warnings"))


def _accommodation_score(data: dict) -> float | None:
    """Substrate-accommodation fit in [0,1], or None when not computed."""
    from enzyme_discovery_agent.chemistry.fit import accommodation_score
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
    if role_transfer_summary(data)['review_required']:
        return None
    return accommodation_score(data.get("substrate_similarity") or {})


def _accommodation_facts_payload(data: dict) -> list[dict]:
    """Substrate-accommodation evidence for the LLM to reason over (not a score)."""
    sim = data.get("substrate_similarity") or {}
    comps = sim.get("property_comparisons") or []
    subs = []
    for c in comps:
        mw_t = c.get("mw_target")
        mw_n = c.get("mw_native")
        mw_ratio = (
            round(mw_n / mw_t, 2)
            if isinstance(mw_t, (int, float)) and isinstance(mw_n, (int, float)) and mw_t > 0
            else None
        )
        pocket_rel = (
            "larger_native_envelope" if mw_ratio is not None and mw_ratio >= 1.35 else
            "smaller_native_envelope" if mw_ratio is not None and mw_ratio <= 0.70 else
            "comparable_envelope" if mw_ratio is not None else None
        )
        subs.append({
            "target_smiles": c.get('target_smiles'),
            "native_smiles": c.get('native_smiles'),
            "target_fg_counts": {k: v for k, v in (c.get("functional_group_counts_target") or {}).items() if v},
            "native_fg_counts": {k: v for k, v in (c.get("functional_group_counts_native") or {}).items() if v},
            "target_aliphatic_chain": c.get("chain_length_target"),
            "native_aliphatic_chain": c.get("chain_length_native"),
            "target_mw": mw_t,
            "native_mw": mw_n,
            **({"native_to_target_mw_ratio": mw_ratio, "pocket_size_relation": pocket_rel} if mw_ratio is not None else {}),
            "native_has_aromatic_target_lacks": c.get("aromaticity_match") is False,
            "native_has_ring_target_lacks": c.get("cyclicity_match") is False,
        })
    return subs


def _candidate_evidence_payload(data: dict) -> dict:
    """The full grounded evidence the LLM ranks a promising candidate on."""
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
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
        "full_evidence_source": data.get("full_evidence_source"),
        "assessment_receipt": data.get("assessment_receipt"),
        "organism": data.get("organism", ""),
        "ec": data.get("ec_numbers") or [],
        "family": data.get("interpro_family", ""),
        "native_claim_validation": {
            'is_enzyme': data.get('is_enzyme'),
            'promotion_eligible': (data.get('claim_validation') or {}).get('promotion_eligible'),
            'missing_claims': (data.get('claim_validation') or {}).get('blocking_claims') or [],
            'critical_warnings': data.get('critical_warnings') or [],
        },
        "reaction": {
            "native_reaction": (data.get('native_substrate_reaction_scope') or {}).get('reaction') or (cats[0] if cats else ''),
            "bond_types": chem.get("bond_types"),
            "direction": chem.get("reaction_direction"),
        },
        "function_comments": data.get('function_comments') or [],
        "reactant_occurrences": [
            {key: item.get(key) for key in ('name', 'smiles', 'occurrence_id', 'stoichiometry_status')}
            for item in data.get('native_substrates') or []
        ],
        "native_substrates": [n.get("name") for n in (data.get("native_substrates") or []) if n.get("name")],
        "substrate_accommodation": _accommodation_facts_payload(data),
        "priority_assessment": data.get("priority_assessment"),
        "literature_evidence": data.get("literature_evidence"),
        "source_function_evidence": data.get("source_function_evidence"),
        "participant_roles": role_transfer_summary(data),
        "other_recorded_activities": [
            activity for activity in cats
            if activity != (data.get('native_substrate_reaction_scope') or {}).get('reaction')
        ],
        "mechanism_evidence": data.get("mechanism_evidence"),
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


def _mechanism_evidence_level(data: dict) -> str:
    """Classify mechanism support without equating missing evidence with failure."""
    evidence = data.get("mechanism_evidence") or {}
    facts = evidence.get("facts") or {}
    confidence = str(facts.get("confidence") or "").casefold()
    provenance = evidence.get("paper") or facts.get("provenance") or {}
    cited = bool(provenance.get("doi") or provenance.get("pmcid") or provenance.get("pmid"))
    if evidence.get("status") == "ok" and confidence == "high" and cited:
        return "experimental"
    if evidence.get("status") == "ok" and confidence in {"high", "medium"}:
        return "supported"
    if evidence and (evidence.get("status") == "low_confidence" or confidence == "low"):
        return "tentative"
    return "unknown"


def _mechanism_evidence_rank(data: dict) -> int:
    return {"experimental": 0, "supported": 1, "tentative": 2, "unknown": 3}[
        _mechanism_evidence_level(data)
    ]


def _priority_axes(data, *, evidence_tiebreak=True, fallback=None):
    """Observed context only; missing dimensions receive no invented value."""
    axes = {}
    for index, pair in enumerate((data.get('substrate_similarity') or {}).get('property_comparisons') or []):
        target = str(pair.get('target_smiles') or index)
        for metric in ('mw_similarity', 'volume_similarity', 'chain_length_similarity',
                       'logP_similarity', 'tpsa_similarity'):
            value = pair.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
                key = (target, metric)
                axes[key] = min(axes.get(key, value), value)
    if not axes and isinstance(fallback, (int, float)) and math.isfinite(fallback) and 0 <= fallback <= 1:
        axes[('legacy', 'resemblance')] = fallback
    if evidence_tiebreak and _mechanism_evidence_level(data) != 'unknown':
        axes[('mechanism', 'source_support')] = 3 - _mechanism_evidence_rank(data)
    return axes


def _priority_fronts(rows):
    """Noncompensatory fronts; unequal evidence coverage stays incomparable."""
    remaining = list(rows)
    front = 0
    while remaining:
        def dominates(left, right):
            first, second = left.get('bottleneck_fit'), right.get('bottleneck_fit')
            if (first is None) != (second is None):
                return False
            if first is not None and second is not None and first != second:
                return first > second
            a, b = left['axes'], right['axes']
            return bool(a) and a.keys() == b.keys() and all(a[k] >= b[k] for k in a) and any(a[k] > b[k] for k in a)

        current = [row for row in remaining if not any(dominates(other, row) for other in remaining)]
        for row in current:
            row['front'] = front
        members = {row['accession'] for row in current}
        remaining = [row for row in remaining if row['accession'] not in members]
        front += 1


def _evidence_level(d: dict) -> str:
    """Coarse, explainable curation/characterisation level (a fact, not a rank key)."""
    lit = d.get("literature") or {}
    struct = d.get("structural_data") or {}
    cp = int(lit.get("characterization_papers") or 0)
    sw = bool(d.get("is_swissprot"))
    has_struct = bool(struct.get("pdb_ids") or struct.get("has_alphafold"))
    if sw and cp >= 3 and has_struct:
        return "high"
    if sw and cp >= 1:
        return "medium"
    if sw or cp >= 1:
        return "low"
    return "none"


def _reaction_direction(d: dict) -> str:
    chem = d.get("reaction_chemistry") or {}
    return str(chem.get("reaction_direction") or "-")


def _deterministic_mechanistic_ranking(
    rankable_evals: list[dict], tier_score: dict[int, float],
    evidence_tiebreak: bool = True,
    family_rank_by_acc: dict[str, int] | None = None,
    fallback_substrate_fit_by_acc: dict[str, float] | None = None,
    excluded_out: list[dict] | None = None,
) -> list[dict]:
    """Known conflicts first, then source-backed priorities and observed tradeoffs."""
    from enzyme_discovery_agent.models.evaluations import priority_concerns, prioritized_hypothesis, shortlist_eligible
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
    from enzyme_discovery_agent.chemistry.fit import comparison_coverage
    from enzyme_discovery_agent.chemistry.reactions import evaluate_partner_gate_for_candidate, partner_gate_reason

    family_rank_by_acc = family_rank_by_acc or {}
    groups = {}
    for evaluation in rankable_evals:
        data = evaluation.get('data', evaluation)
        accession = data.get('accession')
        if not accession or not shortlist_eligible(data):
            continue
        gate = evaluate_partner_gate_for_candidate(data)
        if gate['action'] == 'exclude':
            if excluded_out is not None:
                excluded_out.append({
                    'accession': accession, 'protein_name': data.get('protein_name', ''),
                    'ec_numbers': '; '.join(data.get('ec_numbers') or []),
                    'partner_class': gate['partner_class'],
                    'partner_class_basis': gate['basis'], 'evidence': gate['evidence'],
                    'reason': partner_gate_reason(gate),
                })
            continue
        assessment = data.get('priority_assessment') or {}
        concerns = priority_concerns(data)
        weak = bool(concerns) or assessment.get('decision') == 'deprioritize' or gate['action'] == 'review_demote'
        strong = not weak and prioritized_hypothesis(data)
        tier = 3 if weak else 1 if strong else 2
        roles = role_transfer_summary(data)
        fallback = None if roles['review_required'] else (fallback_substrate_fit_by_acc or {}).get(accession)
        row = {
            'accession': accession, 'data': data, 'tier': tier, 'concerns': concerns, 'gate': gate,
            'bottleneck_fit': _accommodation_score(data),
            'axes': (
                {} if roles['review_required'] or comparison_coverage(data.get('substrate_similarity') or {})['complete'] is False
                else _priority_axes(data, evidence_tiebreak=False, fallback=fallback)
            ),
        }
        key = (tier == 3, tier != 1)
        groups.setdefault(key, []).append(row)

    ordered = []
    for key, rows in sorted(groups.items()):
        _priority_fronts(rows)
        from enzyme_discovery_agent.models.candidate_registry import family_label
        strata = {}
        for row in sorted(rows, key=lambda row: row['accession']):
            evidence_order = (
                row['front'], row['tier'] != 1,
                family_rank_by_acc.get(row['accession'], 1),
                _mechanism_evidence_rank(row['data']) if evidence_tiebreak else 0,
            )
            family = family_label(row['data'], row['accession'])
            strata.setdefault(evidence_order, {}).setdefault(family, []).append(row)
        for _, families in sorted(strata.items()):
            for index in range(max(map(len, families.values()))):
                ordered.extend(members[index] for members in families.values() if index < len(members))

    ranking = []
    for rank, row in enumerate(ordered, 1):
        data = row['data']
        tier = row['tier']
        assessment = data.get('priority_assessment') or {}
        bits = []
        if row['gate']['action'] == 'review_demote':
            bits.append('Nucleophile review: ' + partner_gate_reason(row['gate']))
        if row['concerns']:
            bits.append('Known concern: ' + row['concerns'][0])
        if assessment.get('reason'):
            bits.append(assessment['reason'])
        fit = _accommodation_score(data)
        if fit is not None:
            bits.append(f'accommodation context={fit:.2f}')
        mechanism = _mechanism_evidence_level(data)
        if mechanism != 'unknown':
            bits.append('mechanism=' + mechanism)
        if not assessment and not prioritized_hypothesis(data):
            bits.append('Provisional analogue; holistic priority assessment is pending')
        if role_transfer_summary(data)['review_required']:
            bits.append('Participant-role transfer remains unresolved')
        ranking.append({
            'rank': rank, 'accession': row['accession'],
            'gene': (data.get('gene_names') or [''])[0],
            'tier': tier, 'tier_label': _TIER_LABEL[tier],
            'score': tier_score.get(tier, 5.0), 'rationale': ' | '.join(bits),
            'substrate_fit': fit, 'evidence_level': _evidence_level(data),
            'family_relevance': {0: 'on-target', 1: 'neutral', 2: 'off-target(stopped)'}.get(
                family_rank_by_acc.get(row['accession'], 1), 'neutral'
            ),
            'direction': _reaction_direction(data), 'priority_assessment': assessment or None,
            'partner_class': row['gate']['partner_class'], 'partner_class_basis': row['gate']['basis'],
        })
    return ranking


def _registry_fold_map(run_dir: Path) -> dict[str, set[str]]:
    """accession -> {InterPro fold ids}, from screened-candidate registry metadata."""
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
    """Partition candidate accessions by the agent's grounded family decisions."""
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
    try:
        budget = json.loads((run_file(run_dir, 'family_budget.json')).read_text())
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
    """{accession -> 0 on-target} for the deterministic spine."""
    return {acc: 0 for acc in ontarget_accs}


def _fold_fit_map(run_dir: Path) -> dict[str, float]:
    """accession -> its fold's holistic substrate fit (0-1), from the survey landscape."""
    try:
        folds = json.loads((run_file(run_dir, 'family_landscape.json')).read_text())
    except (OSError, ValueError):
        return {}
    out: dict[str, float] = {}
    for f in folds if isinstance(folds, list) else []:
        hf = (f.get("holistic_fit") or {}).get("max")
        if not isinstance(hf, (int, float)):
            continue
        for m in f.get("top_members") or []:
            acc = m.get("accession")
            if acc:
                out.setdefault(acc, float(hf))
    return out


def _literature_named_accs(run_dir: Path) -> set[str]:
    """Accessions the reading step surfaced BY NAME (registry ``literature_name``)."""
    try:
        from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
        reg = load_candidate_registry(run_dir).get("candidates", {})
    except Exception:  # noqa: BLE001 - ranking must never crash on a registry read
        return set()
    named: set[str] = set()
    for v in reg.values():
        if not isinstance(v, dict):
            continue
        acc = v.get("accession")
        if acc and str((v.get("metadata") or {}).get("literature_name") or "").strip():
            named.add(acc)
    return named
