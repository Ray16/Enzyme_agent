#!/usr/bin/env python3
"""
Run P2Rank pocket prediction on all AlphaFold structures and save results.

For each protein, identifies the top-ranked binding pocket and saves
residue IDs + probability scores. Results are written to p2rank_pockets.json
in each project directory.

Usage:
    python run_p2rank.py amide_synthetase
    python run_p2rank.py amide_hydrolase
    python run_p2rank.py amide_synthetase amide_hydrolase
"""

import csv
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

PRANK_BIN = os.environ.get("PRANK_BIN", os.path.expanduser("~/tools/p2rank/prank"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = min(32, os.cpu_count() or 4)


def run_p2rank(pdb_path, output_dir):
    """Run P2Rank on a single PDB file. Returns parsed predictions."""
    result = subprocess.run(
        [PRANK_BIN, "predict", "-f", pdb_path, "-o", output_dir],
        capture_output=True, text=True, timeout=120,
    )

    # Parse predictions CSV
    basename = os.path.basename(pdb_path)
    csv_path = os.path.join(output_dir, f"{basename}_predictions.csv")
    if not os.path.exists(csv_path):
        return []

    pockets = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("name") or not line:
                continue
            # The CSV uses ", " as delimiter but residue_ids and surf_atom_ids
            # fields contain spaces. Split on ", " pattern for the first 10 fields,
            # then take the rest as residue data.
            # Format: name, rank, score, prob, sas, surf, cx, cy, cz, residue_ids, surf_atom_ids
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10:
                continue

            name = parts[0]
            rank = int(parts[1])
            score = float(parts[2])
            probability = float(parts[3])

            # Residue IDs are in parts[9] (0-indexed), but may span multiple
            # comma-split segments if the field is long. Rejoin from index 9
            # and split on the surf_atom_ids boundary (numeric-only tokens).
            residue_str = ",".join(parts[9:])
            # residue_ids are "A_113 A_114 ..." and surf_atom_ids are "824 827 ..."
            # Split on the first pure-numeric space-separated token sequence
            residues = []
            for token in residue_str.split():
                match = re.match(r"([A-Z])_(\d+)", token)
                if match:
                    residues.append(int(match.group(2)))

            pockets.append({
                "rank": rank,
                "score": score,
                "probability": probability,
                "residues": sorted(residues),
                "n_residues": len(residues),
            })

    return pockets


def process_project(project_dir):
    """Run P2Rank on all structures in a project."""
    struct_dir = os.path.join(project_dir, "alphafold_structures")
    if not os.path.isdir(struct_dir):
        print(f"  No alphafold_structures directory in {project_dir}")
        return {}

    # Get candidate UIDs from CSV
    csv_path = os.path.join(project_dir, "structure_sequence_summary.csv")
    if not os.path.exists(csv_path):
        print(f"  No structure_sequence_summary.csv in {project_dir}")
        return {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        candidates = [r["uniprot_id"] for r in reader]

    print(f"  Processing {len(candidates)} candidates ({MAX_WORKERS} workers)...")

    p2rank_output_dir = os.path.join(project_dir, "p2rank_output")
    os.makedirs(p2rank_output_dir, exist_ok=True)

    # Build work items
    work = []
    for uid in candidates:
        pdb_path = os.path.join(struct_dir, f"AF-{uid}-F1-model_v4.pdb")
        if not os.path.exists(pdb_path):
            print(f"    {uid}: PDB not found, skipping")
            continue
        uid_output = os.path.join(p2rank_output_dir, uid)
        work.append((uid, pdb_path, uid_output))

    # Run in parallel
    results = {}

    def _run_one(args):
        uid, pdb_path, uid_output = args
        try:
            pockets = run_p2rank(pdb_path, uid_output)
            return uid, pockets, None
        except Exception as e:
            return uid, None, str(e)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_one, w): w[0] for w in work}
        for future in as_completed(futures):
            uid, pockets, error = future.result()
            if error:
                print(f"    {uid}: P2Rank failed: {error}")
                continue
            if pockets:
                top = pockets[0]
                results[uid] = {
                    "top_pocket": top,
                    "n_pockets": len(pockets),
                    "all_pockets": pockets[:3],
                }
                print(f"    {uid}: {top['n_residues']} res, "
                      f"prob={top['probability']:.3f}, "
                      f"{len(pockets)} pockets", flush=True)
            else:
                results[uid] = {
                    "top_pocket": None,
                    "n_pockets": 0,
                    "all_pockets": [],
                }
                print(f"    {uid}: no pockets found", flush=True)

    # Save results
    out_path = os.path.join(project_dir, "p2rank_pockets.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved P2Rank results to {out_path}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_p2rank.py <project_dir> [project_dir2 ...]")
        sys.exit(1)

    for project_name in sys.argv[1:]:
        project_dir = os.path.join(BASE_DIR, project_name)
        if not os.path.isdir(project_dir):
            print(f"Project directory not found: {project_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"P2Rank pocket prediction: {project_name}")
        print(f"{'='*60}")
        results = process_project(project_dir)

        # Summary
        if results:
            n_with_pocket = sum(1 for v in results.values() if v["top_pocket"])
            probs = [v["top_pocket"]["probability"] for v in results.values()
                     if v["top_pocket"]]
            print(f"\n  Summary: {n_with_pocket}/{len(results)} have predicted pockets")
            if probs:
                import numpy as np
                print(f"  Probability: mean={np.mean(probs):.3f}, "
                      f"median={np.median(probs):.3f}, "
                      f"range={min(probs):.3f}-{max(probs):.3f}")
                print(f"  High confidence (>0.5): "
                      f"{sum(1 for p in probs if p > 0.5)}")


if __name__ == "__main__":
    main()
