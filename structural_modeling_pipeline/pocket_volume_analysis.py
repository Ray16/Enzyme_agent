#!/usr/bin/env python3
"""
Pocket Volume Analysis
======================
Compares pre-cofolding pocket geometry to substrate molecular volume,
classifies pocket shape (enclosed vs open cleft), and flags potential
forced-fit artifacts from pocket-constrained cofolding.

Inputs:
    - pocket_analysis.json (both projects): pocket volume_A3, area_A2
    - config.yaml (both projects): substrate SMILES
    - cofolding_metrics.csv: SASA burial, strategy, contacts

Outputs:
    - analysis_results/pocket_volume_analysis.csv
    - analysis_results/pocket_volume_summary.txt

Usage:
    python pocket_volume_analysis.py
    python pocket_volume_analysis.py --output-dir analysis_results
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_projects(config_path=None):
    """Load project directories from pipeline_config.yaml."""
    if config_path is None:
        config_path = os.path.join(BASE_DIR, "pipeline_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    projects = []
    for p in config.get("projects", []):
        path = p["path"]
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        name = p.get("name", os.path.basename(path))
        projects.append((path, name))
    return projects


def compute_substrate_volume(smiles, n_conformers=10):
    """Compute van der Waals volume of a molecule from SMILES.

    Generates multiple 3D conformers and returns the median volume
    to reduce dependence on a single conformer geometry.

    Args:
        smiles: SMILES string (supports charged forms)
        n_conformers: number of conformers to average over

    Returns:
        volume in cubic Angstroms, or None if computation fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  WARNING: Could not parse SMILES: {smiles}", file=sys.stderr)
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    # Generate multiple conformers for robust volume estimate
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(cids) == 0:
        # Fallback: single conformer with random coords
        AllChem.EmbedMolecule(mol, params)
        if mol.GetNumConformers() == 0:
            return None
        cids = [0]

    volumes = []
    for cid in cids:
        try:
            vol = AllChem.ComputeMolVolume(mol, confId=cid)
            volumes.append(vol)
        except Exception:
            continue

    return float(np.median(volumes)) if volumes else None


def load_pocket_data(project_dir):
    """Load pocket analysis data. Returns dict: uid -> pocket info."""
    path = os.path.join(project_dir, "pocket_analysis.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_cofolding_metrics(csv_path):
    """Load cofolding metrics CSV into dict: (project, uid) -> row."""
    if not os.path.exists(csv_path):
        return {}
    metrics = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["project"], row["uid"])
            metrics[key] = row
    return metrics


def load_substrates(project_dir):
    """Load substrate SMILES from config.yaml."""
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.exists(config_path):
        return []
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("target_reaction", {}).get("substrates", [])


def load_protein_names(project_dir):
    """Parse FASTA headers to get gene names."""
    fasta_path = os.path.join(project_dir, "all_sequences.fasta")
    if not os.path.exists(fasta_path):
        return {}
    names = {}
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                parts = header.split()
                if not parts:
                    continue
                uid = parts[0]
                # Extract UID from sp|UID|NAME format
                uid_parts = uid.split("|")
                uid = uid_parts[1] if len(uid_parts) >= 2 else uid_parts[0]
                gn_match = re.search(r"GN=([^\s]+)", header)
                if gn_match:
                    names[uid] = gn_match.group(1)
                elif len(parts) > 1 and len(parts[1]) < 10:
                    names[uid] = parts[1]
    return names


def classify_pocket_shape(sa_v_ratio):
    """Classify pocket geometry based on surface-area-to-volume ratio."""
    if sa_v_ratio < 0.6:
        return "enclosed"
    elif sa_v_ratio < 1.0:
        return "partially_open"
    else:
        return "open_cleft"


def check_flags(volume_ratio, sa_v_ratio, sasa_burial, strategy, pocket_shape):
    """Check for potential structural issues."""
    flags = []

    if volume_ratio is not None:
        if volume_ratio < 1.0:
            flags.append("pocket_too_small")
        elif volume_ratio < 1.5 and sasa_burial is not None and sasa_burial > 0.7:
            flags.append("potential_forced_fit")

        if volume_ratio > 4.0:
            if sasa_burial is not None and sasa_burial < 0.3:
                flags.append("non_specific_surface")
            else:
                flags.append("oversized_pocket")

    if pocket_shape == "open_cleft" and strategy == "pocket_constrained":
        flags.append("constrained_to_groove")

    return flags


def main():
    parser = argparse.ArgumentParser(description="Pocket Volume Analysis")
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "analysis_results"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Pocket Volume Analysis")
    print("=" * 60)

    # Load cofolding metrics for cross-referencing
    metrics_csv = os.path.join(args.output_dir, "cofolding_metrics.csv")
    cofold_metrics = load_cofolding_metrics(metrics_csv)

    all_results = []

    for project_dir, project_name in load_projects():
        print(f"\n--- {project_name.upper()} ---")

        # Compute substrate volumes
        substrates = load_substrates(project_dir)
        if not substrates:
            print(f"  No substrates found in config.yaml, skipping")
            continue

        substrate_volumes = {}
        total_substrate_vol = 0.0
        for s in substrates:
            vol = compute_substrate_volume(s["smiles"])
            substrate_volumes[s["name"]] = vol
            if vol:
                total_substrate_vol += vol
            print(f"  Substrate {s['name']}: {vol:.1f} A3" if vol else
                  f"  Substrate {s['name']}: FAILED")

        if total_substrate_vol == 0:
            print(f"  WARNING: Could not compute any substrate volumes, skipping")
            continue

        print(f"  Total substrate volume: {total_substrate_vol:.1f} A3")

        # Load pocket data and protein names
        pocket_data = load_pocket_data(project_dir)
        protein_names = load_protein_names(project_dir)

        n_no_pocket = 0
        n_flagged = 0

        for uid, entry in sorted(pocket_data.items()):
            bp = entry.get("binding_pocket")
            cofold_key = (project_name, uid)
            cofold_row = cofold_metrics.get(cofold_key, {})

            # Get cofolding metrics
            sasa_burial = None
            strategy = cofold_row.get("strategy", "unknown")
            n_contacts = 0
            try:
                sasa_burial = float(cofold_row["substrate_sasa_burial"])
            except (KeyError, ValueError, TypeError):
                pass
            try:
                n_contacts = int(float(cofold_row.get("substrate_n_contact_residues", 0)))
            except (ValueError, TypeError):
                pass

            result = {
                "project": project_name,
                "uid": uid,
                "enzyme_name": protein_names.get(uid, ""),
                "strategy": strategy,
                "pocket_detected": bp is not None,
                "pocket_volume_A3": None,
                "pocket_area_A2": None,
                "pocket_n_residues": None,
                "sa_v_ratio": None,
                "pocket_shape": "none",
                "total_substrate_volume_A3": total_substrate_vol,
                "volume_ratio": None,
                "sasa_burial": sasa_burial,
                "n_contact_residues": n_contacts,
                "flags": "",
            }

            if bp is None:
                n_no_pocket += 1
                result["flags"] = "no_pocket_detected"
                all_results.append(result)
                continue

            pocket_vol = bp["volume_A3"]
            pocket_area = bp["area_A2"]
            pocket_n_res = bp["n_residues"]
            sa_v = pocket_area / pocket_vol if pocket_vol > 0 else 999.0
            pocket_shape = classify_pocket_shape(sa_v)
            volume_ratio = pocket_vol / total_substrate_vol if total_substrate_vol > 0 else None

            result.update({
                "pocket_volume_A3": pocket_vol,
                "pocket_area_A2": pocket_area,
                "pocket_n_residues": pocket_n_res,
                "sa_v_ratio": round(sa_v, 3),
                "pocket_shape": pocket_shape,
                "volume_ratio": round(volume_ratio, 2) if volume_ratio else None,
            })

            flags = check_flags(volume_ratio, sa_v, sasa_burial, strategy, pocket_shape)
            if flags:
                n_flagged += 1
            result["flags"] = "; ".join(flags)

            all_results.append(result)

        print(f"  {len(pocket_data)} candidates analyzed")
        print(f"  {n_no_pocket} without detected pockets")
        print(f"  {n_flagged} flagged for potential issues")

    # Write CSV
    if not all_results:
        print("\nNo results to write.")
        return

    csv_path = os.path.join(args.output_dir, "pocket_volume_analysis.csv")
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved CSV: {csv_path}")

    # Write summary
    summary_path = os.path.join(args.output_dir, "pocket_volume_summary.txt")
    lines = [
        "=" * 100,
        "POCKET VOLUME ANALYSIS",
        "=" * 100,
        "",
    ]

    # Substrate volumes
    lines.append("SUBSTRATE VOLUMES")
    lines.append("-" * 40)
    for r in all_results:
        proj = r["project"]
        break
    # Report per-project substrate volumes
    for project_dir, project_name in load_projects():
        substrates = load_substrates(project_dir)
        lines.append(f"\n  {project_name.upper()}:")
        for s in substrates:
            vol = compute_substrate_volume(s["smiles"])
            lines.append(f"    {s['name']}: {vol:.1f} A3" if vol else
                         f"    {s['name']}: N/A")
        total = sum(compute_substrate_volume(s["smiles"]) or 0 for s in substrates)
        lines.append(f"    Total: {total:.1f} A3")

    # Summary statistics per project
    for proj_name in ["synthetase", "hydrolase"]:
        proj_results = [r for r in all_results if r["project"] == proj_name]
        if not proj_results:
            continue

        with_pocket = [r for r in proj_results if r["pocket_detected"]]
        vols = [r["pocket_volume_A3"] for r in with_pocket if r["pocket_volume_A3"]]
        ratios = [r["volume_ratio"] for r in with_pocket if r["volume_ratio"]]

        lines.append(f"\n\n{proj_name.upper()} SUMMARY")
        lines.append("-" * 60)
        lines.append(f"  Candidates analyzed: {len(proj_results)}")
        lines.append(f"  With detected pocket: {len(with_pocket)}")
        lines.append(f"  Without pocket: {len(proj_results) - len(with_pocket)}")

        if vols:
            lines.append(f"  Pocket volume: mean={np.mean(vols):.0f} A3, "
                         f"median={np.median(vols):.0f} A3, "
                         f"range=[{min(vols):.0f}, {max(vols):.0f}] A3")
        if ratios:
            lines.append(f"  Volume ratio (pocket/substrate): mean={np.mean(ratios):.1f}x, "
                         f"median={np.median(ratios):.1f}x, "
                         f"range=[{min(ratios):.1f}x, {max(ratios):.1f}x]")

        # Shape distribution
        shapes = [r["pocket_shape"] for r in with_pocket]
        for shape in ["enclosed", "partially_open", "open_cleft"]:
            n = shapes.count(shape)
            if n > 0:
                lines.append(f"  {shape}: {n} ({n/len(shapes)*100:.0f}%)")

        # Flag summary
        flagged = [r for r in proj_results if r["flags"] and r["flags"] != "no_pocket_detected"]
        if flagged:
            lines.append(f"\n  FLAGGED CANDIDATES ({len(flagged)}):")
            for r in flagged:
                name = r["enzyme_name"] or r["uid"]
                sasa_str = f"{r['sasa_burial']:.1%}" if r["sasa_burial"] is not None else "N/A"
                lines.append(f"    {r['uid']:<15} {name:<20} vol_ratio={r['volume_ratio']}x  "
                             f"shape={r['pocket_shape']:<15} SASA={sasa_str}  "
                             f"flags: {r['flags']}")

        # Full candidate table
        lines.append(f"\n  ALL CANDIDATES:")
        lines.append(f"  {'UID':<15} {'Name':<20} {'Vol(A3)':>8} {'Ratio':>6} {'SA/V':>6} "
                     f"{'Shape':<16} {'SASA%':>6} {'Strategy':<20} {'Flags'}")
        lines.append(f"  {'-'*130}")

        for r in sorted(with_pocket, key=lambda x: x["volume_ratio"] or 0):
            name = (r["enzyme_name"][:17] + "...") if len(r["enzyme_name"]) > 20 else r["enzyme_name"]
            sasa_str = f"{r['sasa_burial']:.1%}" if r["sasa_burial"] is not None else "N/A"
            vol_str = f"{r['pocket_volume_A3']:.0f}" if r["pocket_volume_A3"] else "N/A"
            ratio_str = f"{r['volume_ratio']:.1f}x" if r["volume_ratio"] else "N/A"
            sav_str = f"{r['sa_v_ratio']:.2f}" if r["sa_v_ratio"] else "N/A"
            lines.append(f"  {r['uid']:<15} {name:<20} {vol_str:>8} {ratio_str:>6} {sav_str:>6} "
                         f"{r['pocket_shape']:<16} {sasa_str:>6} {r['strategy']:<20} {r['flags']}")

        # No-pocket candidates
        no_pocket = [r for r in proj_results if not r["pocket_detected"]]
        if no_pocket:
            lines.append(f"\n  NO POCKET DETECTED ({len(no_pocket)}):")
            for r in no_pocket:
                name = r["enzyme_name"] or r["uid"]
                sasa_str = f"{r['sasa_burial']:.1%}" if r["sasa_burial"] is not None else "N/A"
                lines.append(f"    {r['uid']:<15} {name:<20} SASA={sasa_str}  strategy={r['strategy']}")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
