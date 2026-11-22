"""Cross-family comparative chemical synthesis for admitted discovery candidates.

Isolated per-batch target reviews cannot compare mechanistic trade-offs across
families or collapse ortholog duplicates across batches. This module groups all
currently admitted (``prioritize`` / ``investigate``) candidates by resolved
family, collapses exact-sequence and same-gene orthologs while preserving
alternatives, and produces a structured family-comparison artifact
(``artifacts/comparative_synthesis.json``) plus a concise Markdown summary.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from enzyme_discovery_agent.models.candidate_evidence import candidate_record
from enzyme_discovery_agent.models.candidate_registry import (
    discovery_status,
    family_label,
    load_candidate_registry,
)
from enzyme_discovery_agent.models.ortholog import filter_highly_similar_orthologs
from enzyme_discovery_agent.models.target_review import eligible
from enzyme_discovery_agent.state_store import atomic_write_json


def _load_structural_geometry(run_dir: Path) -> dict[str, dict]:
    """Read completed structural geometry observations (multi-step or single-step) if present.

    Structural geometry is reported strictly as a structural observation, never
    as demonstrated catalytic activity.
    """
    def _flt(row, key):
        val = row.get(key)
        try:
            return round(float(val), 3) if val not in (None, "") else None
        except (TypeError, ValueError):
            return None

    for arm in ("cofolded", "family_cofold"):
        # Multi-step paired summary (when a mechanism has >=2 steps)
        summary_path = run_dir / arm / "analysis" / "both_steps_summary.csv"
        if summary_path.is_file():
            out: dict[str, dict] = {}
            try:
                with summary_path.open(encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        acc = (row.get("accession") or "").strip()
                        if not acc:
                            continue
                        out[acc] = {
                            "arm": arm,
                            "step_1_harmonic_distance_A": _flt(row, "step_1_harmonic_distance_A"),
                            "step_1_near_attack_fraction": _flt(row, "step_1_near_attack_fraction"),
                            "step_2_harmonic_distance_A": _flt(row, "step_2_harmonic_distance_A"),
                            "step_2_near_attack_fraction": _flt(row, "step_2_near_attack_fraction"),
                            "larger_step_distance_A": _flt(row, "larger_step_distance_A"),
                            "scope": "Step-specific structural geometry only; not demonstrated activity.",
                        }
            except OSError:
                continue
            if out:
                return out
        # Fallback for single-step mechanisms (e.g. hydrolases, lyases, isomerases, oxidoreductases)
        ensemble_path = run_dir / arm / "analysis" / "ensemble_summary.csv"
        if ensemble_path.is_file():
            out = {}
            try:
                with ensemble_path.open(encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        acc = (row.get("accession") or "").strip()
                        if not acc:
                            continue
                        dist = _flt(row, "harmonic_distance_A")
                        frac = _flt(row, "near_attack_fraction")
                        prev = out.get(acc)
                        if prev is None or (dist is not None and (prev["larger_step_distance_A"] is None or dist > prev["larger_step_distance_A"])):
                            out[acc] = {
                                "arm": arm,
                                "step_1_harmonic_distance_A": dist,
                                "step_1_near_attack_fraction": frac,
                                "step_2_harmonic_distance_A": None,
                                "step_2_near_attack_fraction": None,
                                "larger_step_distance_A": dist,
                                "scope": "Step-specific structural geometry only; not demonstrated activity.",
                            }
            except OSError:
                continue
            if out:
                return out
    return {}


def _decision_rank(decision: str) -> int:
    return {"prioritize": 0, "investigate": 1}.get(decision, 2)


def _evidence_rank(level: str) -> int:
    return {"direct_target_observation": 0, "mechanistic_transfer": 1, "speculative": 2}.get(level, 3)


def _dedupe_orthologs(members: list[dict], identity_threshold: float = 0.85) -> tuple[list[dict], list[dict]]:
    """Collapse only highly similar sequences (>= identity_threshold); retain divergent orthologs."""
    return filter_highly_similar_orthologs(
        members,
        identity_threshold=identity_threshold,
        seq_getter=lambda m: m.get("_sequence"),
        family_getter=lambda m: "same_family",
    )


def build_comparative_brief(
    run_dir: str | Path,
    *,
    max_per_family: int = 5,
    max_families: int | None = None,
) -> dict:
    """Build a deterministic, family-grouped chemical trade-off comparison."""
    root = Path(run_dir)
    registry = load_candidate_registry(root, strict=True).get("candidates", {})
    structural_by_acc = _load_structural_geometry(root)

    by_family: dict[str, list[dict]] = {}
    for acc, entry in sorted(registry.items()):
        if discovery_status(entry) == "rejected":
            continue
        meta = entry.get("metadata") or {}
        if not eligible(root, acc, meta, purpose="ranking"):
            continue
        assessment = meta.get("target_assessment") or {}
        data = candidate_record(meta)
        fam = family_label(meta, acc)
        mech = data.get("mechanism_evidence") or {}
        activation = (mech.get("activation") or {}).get("activation_mode")
        rc = data.get("reaction_center_analysis") or {}
        constraints = assessment.get("constraints") or []
        major_constraints = [
            c.get("finding") for c in constraints
            if isinstance(c, dict) and c.get("severity") == "major" and c.get("finding")
        ]
        by_family.setdefault(fam, []).append({
            "accession": acc,
            "protein_name": data.get("protein_name") or "Unresolved protein name",
            "organism": data.get("organism") or "",
            "decision": assessment.get("decision"),
            "evidence_level": assessment.get("evidence_level"),
            "activation_mode": activation,
            "reaction_center": {
                k: rc.get(k) for k in ("molecularity", "consumed_groups", "formed_groups", "unchanged_spectator_groups")
                if rc.get(k)
            } if rc else None,
            "mechanism_and_step": assessment.get("mechanism_and_step"),
            "target_roles": assessment.get("target_roles") or [],
            "major_constraints": major_constraints,
            "unknowns": (assessment.get("unknowns") or [])[:4],
            "revisit_when": assessment.get("revisit_when"),
            "structural_geometry": structural_by_acc.get(acc),
            "_sequence": data.get("sequence") or "",
        })

    family_rows = []
    modeled_all: list[dict] = []
    for fam, members in by_family.items():
        members.sort(key=lambda m: (
            _decision_rank(m["decision"]),
            _evidence_rank(m["evidence_level"]),
            len(m["major_constraints"]),
            m["accession"],
        ))
        reps, collapsed = _dedupe_orthologs(members)
        clean_reps = [
            {k: v for k, v in rep.items() if not k.startswith("_") and v not in (None, "", [])}
            for rep in reps[:max_per_family]
        ]
        activation_modes = sorted({r["activation_mode"] for r in reps if r.get("activation_mode")})
        molecularities = sorted({
            (r.get("reaction_center") or {}).get("molecularity")
            for r in reps if (r.get("reaction_center") or {}).get("molecularity")
        })
        fam_modeled = [m for m in members if m.get("structural_geometry")]
        best_dist = min(
            (m["structural_geometry"]["larger_step_distance_A"] for m in fam_modeled
             if m["structural_geometry"].get("larger_step_distance_A") is not None),
            default=None,
        )
        for m in fam_modeled:
            sg = m["structural_geometry"]
            modeled_all.append({
                "accession": m["accession"],
                "protein_name": m["protein_name"],
                "family": fam,
                "decision": m["decision"],
                **sg,
            })
        family_rows.append({
            "family": fam,
            "n_admitted": len(members),
            "n_prioritized": sum(1 for m in members if m["decision"] == "prioritize"),
            "n_investigate": sum(1 for m in members if m["decision"] == "investigate"),
            "n_distinct_representatives": len(reps),
            "n_modeled": len(fam_modeled),
            **({"best_two_step_distance_A": best_dist} if best_dist is not None else {}),
            "activation_modes": activation_modes,
            "molecularities": molecularities,
            "representatives": clean_reps,
            "collapsed_orthologs": collapsed,
        })

    family_rows.sort(key=lambda f: (-f["n_prioritized"], -f["n_distinct_representatives"], f["family"]))
    modeled_all.sort(key=lambda r: (
        r.get("larger_step_distance_A") if r.get("larger_step_distance_A") is not None else 9999.0,
        r["accession"],
    ))
    brief = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_admitted_candidates": sum(f["n_admitted"] for f in family_rows),
        "total_families": len(family_rows),
        "families": family_rows[:max_families] if max_families is not None else family_rows,
        "scope": (
            "Cross-family comparative summary of source-bound target assessments. "
            "Collapses exact-sequence and shared-annotation orthologs for comparison; "
            "does not establish target activity or replace candidate-level evidence."
        ),
    }
    if modeled_all:
        brief["structural_modeling_summary"] = {
            "total_modeled_admitted": len(modeled_all),
            "top_by_two_step_distance": modeled_all[:10],
            "scope": "Completed Boltz-2 step-specific structural geometry; distinct from experimental activity.",
        }
    return brief


def render_comparative_markdown(brief: dict) -> str:
    """Render a compact Markdown comparison across admitted mechanism families."""
    families = brief.get("families") or []
    if not families:
        return ""
    lines = [
        "## Cross-Family Comparative Chemical Synthesis",
        "",
        f"Admitted candidates: **{brief.get('total_admitted_candidates', 0)}** across "
        f"**{brief.get('total_families', 0)}** family group(s) (orthologs collapsed per family).",
        "",
        "| Family | Prioritized / Investigate | Activation / Molecularity | Lead Representative(s) | Key Limiting Constraint / Uncertainty |",
        "| --- | :---: | --- | --- | --- |",
    ]
    for fam in families:
        acts = ", ".join(fam.get("activation_modes") or ["unspecified"])
        mols = ", ".join(fam.get("molecularities") or [])
        chem = f"{acts} ({mols})" if mols else acts
        reps = "; ".join(
            f"{r['protein_name']} (`{r['accession']}`, {r['decision']})"
            for r in (fam.get("representatives") or [])[:2]
        )
        first = (fam.get("representatives") or [{}])[0]
        limiter = (
            (first.get("major_constraints") or [None])[0]
            or (first.get("unknowns") or [None])[0]
            or "See candidate assessment"
        )
        lines.append(
            f"| {fam['family']} | {fam['n_prioritized']} / {fam['n_investigate']} | "
            f"{chem} | {reps} | {str(limiter)[:120]} |"
        )
    lines.append("")
    sm = brief.get("structural_modeling_summary") or {}
    top_sm = sm.get("top_by_two_step_distance") or []
    if top_sm:
        lines.extend([
            "### Completed Two-Step Structural Geometry (Not Demonstrated Activity)",
            "",
            f"Modeled admitted candidates: **{sm.get('total_modeled_admitted', len(top_sm))}**.",
            "",
            "| Enzyme | Family | Step 1 Dist (Å) | Step 2 Dist (Å) | Worst-Step (Å) |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for item in top_sm[:8]:
            d1 = item.get("step_1_harmonic_distance_A")
            d2 = item.get("step_2_harmonic_distance_A")
            dm = item.get("larger_step_distance_A")
            lines.append(
                f"| {item['protein_name']} (`{item['accession']}`) | {item['family']} | "
                f"{f'{d1:.2f}' if d1 is not None else '-'} | "
                f"{f'{d2:.2f}' if d2 is not None else '-'} | "
                f"{f'{dm:.2f}' if dm is not None else '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_comparative_synthesis(run_dir: str | Path) -> dict:
    """Generate and persist ``artifacts/comparative_synthesis.json``."""
    root = Path(run_dir)
    brief = build_comparative_brief(root)
    brief["markdown"] = render_comparative_markdown(brief)
    atomic_write_json(root / "artifacts" / "comparative_synthesis.json", brief)
    return brief
