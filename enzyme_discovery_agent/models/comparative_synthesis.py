"""Cross-family comparative chemical synthesis for admitted discovery candidates.

Isolated per-batch target reviews cannot compare mechanistic trade-offs across
families or collapse ortholog duplicates across batches. This module groups all
currently admitted (``prioritize`` / ``investigate``) candidates by resolved
family, collapses exact-sequence and same-gene orthologs while preserving
alternatives, and produces a structured family-comparison artifact
(``artifacts/comparative_synthesis.json``) plus a concise Markdown summary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from enzyme_discovery_agent.models.candidate_evidence import candidate_record
from enzyme_discovery_agent.models.candidate_registry import (
    discovery_status,
    family_label,
    load_candidate_registry,
)
from enzyme_discovery_agent.models.ortholog import tentative_label_key
from enzyme_discovery_agent.models.target_review import eligible
from enzyme_discovery_agent.state_store import atomic_write_json


def _decision_rank(decision: str) -> int:
    return {"prioritize": 0, "investigate": 1}.get(decision, 2)


def _evidence_rank(level: str) -> int:
    return {"direct_target_observation": 0, "mechanistic_transfer": 1, "speculative": 2}.get(level, 3)


def _dedupe_orthologs(members: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the strongest representative per exact sequence or shared tentative gene key."""
    representatives: list[dict] = []
    collapsed: list[dict] = []
    seen_seq: set[str] = set()
    seen_gene: set[str] = set()

    for member in members:
        seq = member.get("_sequence") or ""
        gkey = member.get("_gene_key") or ""
        is_dup = (seq and seq in seen_seq) or (gkey and gkey in seen_gene)
        if is_dup:
            collapsed.append({
                "accession": member["accession"],
                "protein_name": member["protein_name"],
                "decision": member["decision"],
                "reason": "exact_sequence" if (seq and seq in seen_seq) else "shared_annotation_group",
            })
            continue
        if seq:
            seen_seq.add(seq)
        if gkey:
            seen_gene.add(gkey)
        representatives.append(member)
    return representatives, collapsed


def build_comparative_brief(
    run_dir: str | Path,
    *,
    max_per_family: int = 3,
    max_families: int = 12,
) -> dict:
    """Build a deterministic, family-grouped chemical trade-off comparison."""
    root = Path(run_dir)
    registry = load_candidate_registry(root, strict=True).get("candidates", {})

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
            "_sequence": data.get("sequence") or "",
            "_gene_key": tentative_label_key(data) or "",
        })

    family_rows = []
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
        family_rows.append({
            "family": fam,
            "n_admitted": len(members),
            "n_prioritized": sum(1 for m in members if m["decision"] == "prioritize"),
            "n_investigate": sum(1 for m in members if m["decision"] == "investigate"),
            "n_distinct_representatives": len(reps),
            "activation_modes": activation_modes,
            "molecularities": molecularities,
            "representatives": clean_reps,
            "collapsed_orthologs": collapsed,
        })

    family_rows.sort(key=lambda f: (-f["n_prioritized"], -f["n_distinct_representatives"], f["family"]))
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_admitted_candidates": sum(f["n_admitted"] for f in family_rows),
        "total_families": len(family_rows),
        "families": family_rows[:max_families],
        "scope": (
            "Cross-family comparative summary of source-bound target assessments. "
            "Collapses exact-sequence and shared-annotation orthologs for comparison; "
            "does not establish target activity or replace candidate-level evidence."
        ),
    }


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
    return "\n".join(lines)


def write_comparative_synthesis(run_dir: str | Path) -> dict:
    """Generate and persist ``artifacts/comparative_synthesis.json``."""
    root = Path(run_dir)
    brief = build_comparative_brief(root)
    brief["markdown"] = render_comparative_markdown(brief)
    atomic_write_json(root / "artifacts" / "comparative_synthesis.json", brief)
    return brief
