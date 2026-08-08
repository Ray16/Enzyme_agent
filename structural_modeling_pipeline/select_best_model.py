#!/usr/bin/env python3
"""
Select the best model from multi-seed Boltz-2 cofolding.

When --diffusion_samples > 1, Boltz-2 outputs multiple models per candidate
(model_0.pdb, model_1.pdb, ...). This script picks the best one using:

1. For cofactor-dependent enzymes (synthetases): minimize substrate-cofactor
   centroid distance (substrates should be near ATP in the active site).
2. For cofactor-independent enzymes (hydrolases): minimize substrate distance
   to predicted catalytic/pocket residues from the co-folding constraints.

Usage:
    python select_best_model.py <project_dir>
    python select_best_model.py <project_dir> --results-dir cofold_output
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


def parse_pdb_coords(pdb_path):
    """Parse chain coordinates from a PDB file."""
    chain_coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if chain not in chain_coords:
                    chain_coords[chain] = []
                chain_coords[chain].append([x, y, z])
    return chain_coords


def parse_protein_ca_coords(pdb_path):
    """Parse CA atom coordinates for protein chain A, keyed by residue number."""
    ca_coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "A":
                resnum = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_coords[resnum] = np.array([x, y, z])
    return ca_coords


def load_pocket_residues(cofold_input_dir, uid):
    """Load pocket constraint residues from the co-folding YAML input."""
    if yaml is None:
        return []
    yaml_path = os.path.join(cofold_input_dir, f"{uid}.yaml")
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    residues = []
    for constraint in data.get("constraints", []):
        pocket = constraint.get("pocket", {})
        for contact in pocket.get("contacts", []):
            if isinstance(contact, list) and len(contact) == 2 and contact[0] == "A":
                residues.append(contact[1])
    return residues


def score_by_cofactor(chain_coords):
    """Score by substrate-cofactor proximity (for synthetases with ATP)."""
    if "B" not in chain_coords:
        return float("inf"), {}, "cofactor"

    atp_centroid = np.array(chain_coords["B"]).mean(axis=0)
    distances = {}
    total = 0.0
    for chain_id in sorted(chain_coords.keys()):
        if chain_id in ("A", "B", "C"):
            continue
        centroid = np.array(chain_coords[chain_id]).mean(axis=0)
        d = float(np.linalg.norm(centroid - atp_centroid))
        distances[chain_id] = d
        total += d
    return total, distances, "cofactor"


def score_by_pocket_residues(pdb_path, pocket_residues):
    """Score by substrate proximity to predicted catalytic/pocket residues."""
    if not pocket_residues:
        return float("inf"), {}, "pocket"

    chain_coords = parse_pdb_coords(pdb_path)
    ca_coords = parse_protein_ca_coords(pdb_path)

    pocket_positions = [ca_coords[r] for r in pocket_residues if r in ca_coords]
    if not pocket_positions:
        return float("inf"), {}, "pocket"

    pocket_centroid = np.mean(pocket_positions, axis=0)

    distances = {}
    total = 0.0
    for chain_id in sorted(chain_coords.keys()):
        if chain_id == "A":
            continue
        centroid = np.array(chain_coords[chain_id]).mean(axis=0)
        d = float(np.linalg.norm(centroid - pocket_centroid))
        distances[chain_id] = d
        total += d
    return total, distances, "pocket"


def select_best_models(project_dir, results_dir_name="cofold_output"):
    """Select best model for each candidate with multiple samples."""
    results_dir = os.path.join(project_dir, results_dir_name)
    cofold_input_dir = os.path.join(project_dir, "cofold_input")
    if not os.path.isdir(results_dir):
        return

    n_selected = 0
    n_skipped = 0
    n_cofactor = 0
    n_pocket = 0

    for uid_dir in sorted(os.listdir(results_dir)):
        uid_path = os.path.join(results_dir, uid_dir)
        if not os.path.isdir(uid_path):
            continue

        pdb_pattern = os.path.join(uid_path, "**", "*_model_*.pdb")
        pdb_files = sorted(glob.glob(pdb_pattern, recursive=True))

        if len(pdb_files) <= 1:
            n_skipped += 1
            continue

        # Try cofactor-based scoring first
        first_coords = parse_pdb_coords(pdb_files[0])
        has_cofactor = "B" in first_coords and any(
            k not in ("A", "B", "C") for k in first_coords
        )

        if has_cofactor:
            scores = []
            for pdb_path in pdb_files:
                coords = parse_pdb_coords(pdb_path)
                total_dist, per_chain, method = score_by_cofactor(coords)
                model_name = os.path.basename(pdb_path)
                scores.append((total_dist, per_chain, pdb_path, model_name, method))
            n_cofactor += 1
        else:
            pocket_residues = load_pocket_residues(cofold_input_dir, uid_dir)
            scores = []
            for pdb_path in pdb_files:
                total_dist, per_chain, method = score_by_pocket_residues(
                    pdb_path, pocket_residues)
                model_name = os.path.basename(pdb_path)
                scores.append((total_dist, per_chain, pdb_path, model_name, method))
            n_pocket += 1

        scores.sort(key=lambda x: x[0])
        best_dist, best_chains, best_path, best_name, best_method = scores[0]
        worst_dist = scores[-1][0]

        chain_str = ", ".join(f"{k}={v:.1f}A" for k, v in best_chains.items())
        print(f"  {uid_dir}: best={best_name} ({best_method}, "
              f"total={best_dist:.1f}A: {chain_str}) "
              f"from {len(scores)} models (worst={worst_dist:.1f}A)")

        best_dir = os.path.dirname(best_path)
        best_link = os.path.join(best_dir, f"{uid_dir}_best_model.pdb")
        if os.path.exists(best_link):
            os.remove(best_link)
        shutil.copy2(best_path, best_link)

        meta = {
            "uid": uid_dir,
            "n_samples": len(scores),
            "selection_method": best_method,
            "best_model": best_name,
            "best_total_distance": round(best_dist, 2),
            "best_per_chain": {k: round(v, 2) for k, v in best_chains.items()},
            "all_scores": [
                {"model": s[3], "total_distance": round(s[0], 2),
                 "per_chain": {k: round(v, 2) for k, v in s[1].items()},
                 "method": s[4]}
                for s in scores
            ],
        }
        meta_path = os.path.join(best_dir, "model_selection.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        n_selected += 1

    print(f"\n  Selected best model for {n_selected} candidates "
          f"({n_skipped} had single model)")
    print(f"  Methods: {n_cofactor} cofactor-based, {n_pocket} pocket-based")


def main():
    parser = argparse.ArgumentParser(
        description="Select best model from multi-seed Boltz-2 cofolding")
    parser.add_argument("project_dir", help="Project directory")
    parser.add_argument("--results-dir", default="cofold_output",
                        help="Results directory name (default: cofold_output)")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    select_best_models(project_dir, args.results_dir)


if __name__ == "__main__":
    main()
