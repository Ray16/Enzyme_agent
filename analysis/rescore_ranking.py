#!/usr/bin/env python3
"""Re-score a completed run's candidate ranking with the CURRENT substrate-fit
logic (enzyme_discovery_agent.substrate_accommodation), without re-running the
agent.  A candidate whose target substrates are now UNACCOMMODATED (e.g. a
functional-group match that rests on a structurally non-comparable counterpart)
drops to fit 0.0; every other candidate keeps its recorded substrate_fit.  The
list is then re-sorted by (-fit, original rank) so equal-fit candidates keep
their original relative order.  Writes ranked_candidates_rescored.csv beside the
run's ranked_candidates.csv and prints the new order.

Usage: PYTHONPATH=<shim> python analysis/rescore_ranking.py RUN_DIR TARGET_SMILES[,TARGET_SMILES...]
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
from enzyme_discovery_agent.substrate_accommodation import accommodation_facts


def rescore(run_dir: str, targets: list[str]) -> list[dict]:
    src = Path(run_dir) / "ranked_candidates.csv"
    rows = list(csv.DictReader(open(src)))

    def new_fit(r: dict) -> float:
        nat = [s.strip() for s in (r.get("native_substrate_smiles") or "").split(";") if s.strip()]
        f = accommodation_facts(targets, nat)
        if f and f["unaccommodated_targets"]:
            return 0.0
        try:
            return float(r["substrate_fit"])
        except (TypeError, ValueError):
            return 0.0

    for r in rows:
        r["_old_rank"] = int(r["rank"])
        r["_new_fit"] = new_fit(r)
    ranked = sorted(rows, key=lambda r: (-r["_new_fit"], r["_old_rank"]))
    for i, r in enumerate(ranked, 1):
        r["_new_rank"] = i
    return ranked


def main() -> None:
    run_dir, targets = sys.argv[1], sys.argv[2].split(",")
    ranked = rescore(run_dir, targets)
    out = Path(run_dir) / "ranked_candidates_rescored.csv"
    cols = ["_new_rank", "_old_rank", "gene", "accession", "interpro_family",
            "_new_fit", "substrate_fit", "evidence_level"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c.lstrip("_") for c in cols])
        for r in ranked:
            w.writerow([r.get(c, "") for c in cols])
    print("wrote", out, f"({len(ranked)} rows)")
    for r in ranked:
        moved = r["_new_rank"] - r["_old_rank"]
        tag = "  <-- demoted (now unaccommodated)" if r["_new_fit"] == 0.0 and r["_old_rank"] <= 20 else ""
        print(f"  {r['_new_rank']:>2} (was {r['_old_rank']:>2}) {r['gene'] or '--':10} {r['accession']:12} "
              f"fit {r['_new_fit']:<4} {r['interpro_family'][:34]:34}{tag}")


if __name__ == "__main__":
    main()
