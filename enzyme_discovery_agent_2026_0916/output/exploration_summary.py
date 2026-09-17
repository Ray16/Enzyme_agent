"""Write exploration summary from candidate registry."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.exploration_log import append_exploration_log
from enzyme_discovery_agent.output.csv_generator import _load_ranking
from enzyme_discovery_agent.tools.mechanism_family_assignment import (
    assign_mechanism_family,
    load_mechanism_families,
    summarize_assignments,
)

logger = logging.getLogger("enzyme_agent")


def write_exploration_summary(run_dir: Path, log_fn=None) -> None:
    """Load the candidate registry and write an exploration summary."""
    registry = load_candidate_registry(run_dir)
    candidates = registry.get("candidates", {})
    if not candidates:
        return

    # Mechanism-family assignment: tie each evaluated candidate back to the run's
    # own derived mechanism families (all labels come from mechanism_families.json,
    # never from code) so the summary can report the fold/superfamily composition
    # of the actual pool, not just the prior.
    mech_families = load_mechanism_families(run_dir)
    ranking = _load_ranking(run_dir)

    status_counts: dict[str, int] = {}
    evaluated_list: list[dict] = []
    screened_list: list[dict] = []
    family_groups: dict[str, list[dict]] = {}

    for acc, info in candidates.items():
        statuses = info.get("statuses", [])
        latest = statuses[-1]["status"] if statuses else "unknown"
        status_counts[latest] = status_counts.get(latest, 0) + 1
        meta = info.get("metadata", {})

        if latest in ("evaluated", "ranked", "rejected"):
            family = meta.get("interpro_family", "Unknown")
            family_id = meta.get("interpro_family_id", "")
            mech = assign_mechanism_family(meta, mech_families)
            entry = {
                "accession": acc,
                "status": latest,
                "protein_name": meta.get("protein_name", ""),
                "organism": meta.get("organism", ""),
                "interpro_family": family,
                "interpro_family_id": family_id,
                "mechanism_family": mech["name"] if mech else None,
                "mechanism_family_level": mech["level"] if mech else None,
                "rank": (ranking.get(acc) or {}).get("rank"),
            }
            evaluated_list.append(entry)
            family_key = f"{family} ({family_id})" if family_id else family
            family_groups.setdefault(family_key, []).append(entry)
        elif latest == "screened":
            screened_list.append({
                "accession": acc,
                "protein_name": meta.get("protein_name", ""),
                "organism": meta.get("organism", ""),
                "ec_numbers": meta.get("ec_numbers", []),
                "catalytic_activities": meta.get("catalytic_activities", []),
                "function": meta.get("function", "")[:200],
                "gene_names": meta.get("gene_names", []),
            })

    mechanism_rollup = summarize_assignments(evaluated_list, mech_families)

    summary_lines = ["**Exploration Summary**\n"]
    for status, count in sorted(status_counts.items()):
        summary_lines.append(f"- {status}: {count}")
    if mechanism_rollup:
        summary_lines.append(
            f"\n**Mechanism-family composition ({len(mechanism_rollup)} classes):**"
        )
        for row in mechanism_rollup:
            bits = []
            if row.get("level"):
                bits.append(row["level"])
            if row.get("promise"):
                bits.append(f"promise={row['promise']}")
            if row.get("best_rank") is not None:
                bits.append(f"best rank #{row['best_rank']}")
            suffix = f" ({', '.join(bits)})" if bits else ""
            summary_lines.append(
                f"  - {row['name']}: {row['n_candidates']} candidates{suffix}"
            )
    if family_groups:
        summary_lines.append(f"\n**Evaluated by family ({len(family_groups)} families):**")
        for family, members in sorted(family_groups.items()):
            summary_lines.append(f"\n*{family}* -- {len(members)} candidates:")
            for e in members:
                summary_lines.append(
                    f"  - {e['accession']} ({e['protein_name']}, {e['organism']}): {e['status']}"
                )
    if screened_list:
        summary_lines.append(f"\n**Screened but not evaluated ({len(screened_list)}):**")
        for s in screened_list:
            ec_str = ", ".join(s["ec_numbers"]) if s["ec_numbers"] else "no EC"
            summary_lines.append(
                f"  - {s['accession']} ({s['protein_name']}, {s['organism']}): {ec_str}"
            )

    append_exploration_log(
        run_dir,
        entry_type="DECISION",
        title="Run complete -- exploration summary",
        details="\n".join(summary_lines),
    )

    summary_data = {
        "total_candidates": len(candidates),
        "status_counts": status_counts,
        "n_families": len(family_groups),
        "families": {k: len(v) for k, v in family_groups.items()},
        "n_mechanism_families": len([r for r in mechanism_rollup
                                     if r["name"] != "Unassigned"]),
        "mechanism_family_rollup": mechanism_rollup,
        "evaluated_candidates": evaluated_list,
        "screened_candidates": screened_list,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    try:
        with open(run_dir / "exploration_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2, default=str)
    except Exception as e:
        logger.warning("Failed to write exploration summary: %s", e)
