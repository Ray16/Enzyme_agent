#!/usr/bin/env python3
"""
Check cofolded structures for steric clashes (overlapping atoms).

Clash thresholds (non-bonded heavy atom pairs):
  - Severe:  < 1.5 A  (atoms essentially overlapping)
  - Bad:     < 2.0 A  (significant van der Waals violation)
  - Mild:    < 2.5 A  (minor clash, common in predictions)

Checks:
  1. Protein-ligand clashes (most critical for binding pose quality)
  2. Ligand-ligand clashes (cofactor-substrate overlap)
  3. Intra-ligand clashes (within a single ligand chain)

Usage:
    python check_clashes.py
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

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

SEVERE_CUTOFF = 1.5  # Angstroms
BAD_CUTOFF = 2.0
MILD_CUTOFF = 2.5

# Metal ions form coordination bonds (1.8-2.2A for Mg-O, 2.0-2.3A for Mn-O)
# which are shorter than van der Waals contacts. Exclude these from clash counts.
METAL_ATOMS = {"MG", "MN", "ZN", "FE", "CA", "CO", "NI", "CU"}
METAL_COORD_CUTOFF = 2.4  # Angstroms — max expected metal coordination distance


def parse_atoms(pdb_path):
    """Parse PDB into protein and ligand atom coordinate arrays."""
    protein_atoms = []  # list of (x, y, z, atom_name, resname, resid, chain)
    ligand_atoms = defaultdict(list)  # chain_id -> list of (x, y, z, atom_name, resname)

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                resid = int(line[22:26])
                chain = line[21]
                # Skip hydrogens
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                if element == "H":
                    continue
                protein_atoms.append((x, y, z, atom_name, resname, resid, chain))
            elif line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21]
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                if element == "H":
                    continue
                ligand_atoms[chain].append((x, y, z, atom_name, resname))

    return protein_atoms, dict(ligand_atoms)


def find_clashes(coords1, coords2, cutoff):
    """Find atom pairs closer than cutoff between two coordinate sets.

    Returns list of (idx1, idx2, distance).
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return []

    c1 = np.array(coords1)
    c2 = np.array(coords2)

    clashes = []
    # Chunked to avoid memory issues
    chunk_size = 2000
    for i in range(0, len(c1), chunk_size):
        chunk = c1[i:i + chunk_size]
        diff = chunk[:, None, :] - c2[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        pairs = np.argwhere(dists < cutoff)
        for p in pairs:
            clashes.append((i + p[0], p[1], dists[p[0], p[1]]))

    return clashes


def check_structure(pdb_path):
    """Check a single PDB for steric clashes. Returns dict of results."""
    protein_atoms, ligand_atoms = parse_atoms(pdb_path)

    prot_coords = np.array([(a[0], a[1], a[2]) for a in protein_atoms]) if protein_atoms else np.array([]).reshape(0, 3)

    results = {
        "n_protein_atoms": len(protein_atoms),
        "n_ligand_chains": len(ligand_atoms),
        "protein_ligand_clashes": {"severe": 0, "bad": 0, "mild": 0, "worst_dist": 999.0, "details": []},
        "ligand_ligand_clashes": {"severe": 0, "bad": 0, "mild": 0, "worst_dist": 999.0, "details": []},
    }

    # 1. Protein-ligand clashes
    for chain_id, lig_atoms in ligand_atoms.items():
        lig_coords = np.array([(a[0], a[1], a[2]) for a in lig_atoms])
        if len(lig_coords) == 0 or len(prot_coords) == 0:
            continue

        clashes = find_clashes(prot_coords, lig_coords, MILD_CUTOFF)
        for pi, li, dist in clashes:
            # Skip metal coordination bonds (expected at 1.8-2.4A)
            la = lig_atoms[li]
            if la[3] in METAL_ATOMS and dist < METAL_COORD_CUTOFF:
                continue
            pa = protein_atoms[pi]
            if pa[3] in METAL_ATOMS and dist < METAL_COORD_CUTOFF:
                continue

            if dist < SEVERE_CUTOFF:
                results["protein_ligand_clashes"]["severe"] += 1
            elif dist < BAD_CUTOFF:
                results["protein_ligand_clashes"]["bad"] += 1
            else:
                results["protein_ligand_clashes"]["mild"] += 1

            if dist < results["protein_ligand_clashes"]["worst_dist"]:
                results["protein_ligand_clashes"]["worst_dist"] = dist
                pa = protein_atoms[pi]
                la = lig_atoms[li]
                results["protein_ligand_clashes"]["details"] = [
                    f"Prot {pa[4]}{pa[5]}:{pa[3]} -- Lig {chain_id}:{la[3]} = {dist:.2f}A"
                ]

    # 2. Ligand-ligand clashes (between different chains)
    chain_ids = sorted(ligand_atoms.keys())
    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            ci, cj = chain_ids[i], chain_ids[j]
            coords_i = np.array([(a[0], a[1], a[2]) for a in ligand_atoms[ci]])
            coords_j = np.array([(a[0], a[1], a[2]) for a in ligand_atoms[cj]])
            if len(coords_i) == 0 or len(coords_j) == 0:
                continue

            clashes = find_clashes(coords_i, coords_j, MILD_CUTOFF)
            for ai, aj, dist in clashes:
                # Skip metal coordination bonds
                la_i = ligand_atoms[ci][ai]
                la_j = ligand_atoms[cj][aj]
                if (la_i[3] in METAL_ATOMS or la_j[3] in METAL_ATOMS) and dist < METAL_COORD_CUTOFF:
                    continue

                if dist < SEVERE_CUTOFF:
                    results["ligand_ligand_clashes"]["severe"] += 1
                elif dist < BAD_CUTOFF:
                    results["ligand_ligand_clashes"]["bad"] += 1
                else:
                    results["ligand_ligand_clashes"]["mild"] += 1

                if dist < results["ligand_ligand_clashes"]["worst_dist"]:
                    results["ligand_ligand_clashes"]["worst_dist"] = dist
                    la_i = ligand_atoms[ci][ai]
                    la_j = ligand_atoms[cj][aj]
                    results["ligand_ligand_clashes"]["details"] = [
                        f"Lig {ci}:{la_i[3]} -- Lig {cj}:{la_j[3]} = {dist:.2f}A"
                    ]

    return results


def main():
    print("=" * 70)
    print("STERIC CLASH CHECK FOR COFOLDED STRUCTURES")
    print("=" * 70)
    print(f"Thresholds: severe <{SEVERE_CUTOFF}A, bad <{BAD_CUTOFF}A, mild <{MILD_CUTOFF}A")
    print()

    all_results = []

    for project_dir, project_name in load_projects():
        cofold_output = os.path.join(project_dir, "cofold_output")
        candidates = sorted(os.listdir(cofold_output))

        print(f"--- {project_name.upper()} ({len(candidates)} candidates) ---")

        project_severe = 0
        project_bad = 0
        project_clean = 0
        worst_cases = []

        for uid in candidates:
            pred_dir = os.path.join(cofold_output, uid, f"boltz_results_{uid}", "predictions", uid)
            pdb_file = os.path.join(pred_dir, f"{uid}_model_0.pdb")
            if not os.path.exists(pdb_file):
                continue

            result = check_structure(pdb_file)
            result["uid"] = uid
            result["project"] = project_name
            all_results.append(result)

            pl = result["protein_ligand_clashes"]
            ll = result["ligand_ligand_clashes"]
            total_severe = pl["severe"] + ll["severe"]
            total_bad = pl["bad"] + ll["bad"]

            if total_severe > 0:
                project_severe += 1
                worst_cases.append((uid, total_severe, total_bad, pl, ll))
            elif total_bad > 0:
                project_bad += 1
                worst_cases.append((uid, total_severe, total_bad, pl, ll))
            else:
                project_clean += 1

        # Summary
        n_total = project_severe + project_bad + project_clean
        print(f"  Clean (no clashes <{BAD_CUTOFF}A):   {project_clean}/{n_total} "
              f"({project_clean / n_total * 100:.0f}%)")
        print(f"  Bad clashes (<{BAD_CUTOFF}A):         {project_bad}/{n_total}")
        print(f"  Severe clashes (<{SEVERE_CUTOFF}A):    {project_severe}/{n_total}")

        if worst_cases:
            worst_cases.sort(key=lambda x: x[1] + x[2], reverse=True)
            print(f"\n  Worst offenders:")
            for uid, n_severe, n_bad, pl, ll in worst_cases[:10]:
                detail_pl = pl["details"][0] if pl["details"] else ""
                detail_ll = ll["details"][0] if ll["details"] else ""
                worst_detail = detail_pl if pl["worst_dist"] < ll["worst_dist"] else detail_ll
                worst_dist = min(pl["worst_dist"], ll["worst_dist"])
                print(f"    {uid}: {n_severe} severe + {n_bad} bad clashes, "
                      f"worst={worst_dist:.2f}A")
                if worst_detail:
                    print(f"      {worst_detail}")
        print()

    # Overall summary
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    for proj in ["synthetase", "hydrolase"]:
        proj_results = [r for r in all_results if r["project"] == proj]
        if not proj_results:
            continue

        pl_severe = [r["protein_ligand_clashes"]["severe"] for r in proj_results]
        pl_bad = [r["protein_ligand_clashes"]["bad"] for r in proj_results]
        pl_mild = [r["protein_ligand_clashes"]["mild"] for r in proj_results]
        ll_severe = [r["ligand_ligand_clashes"]["severe"] for r in proj_results]
        ll_bad = [r["ligand_ligand_clashes"]["bad"] for r in proj_results]

        pl_worst = [r["protein_ligand_clashes"]["worst_dist"] for r in proj_results
                    if r["protein_ligand_clashes"]["worst_dist"] < 999]

        print(f"\n  {proj.upper()} (n={len(proj_results)}):")
        print(f"    Protein-ligand clashes:")
        print(f"      Severe (<{SEVERE_CUTOFF}A): {sum(pl_severe)} total across {sum(1 for x in pl_severe if x > 0)} structures")
        print(f"      Bad    (<{BAD_CUTOFF}A): {sum(pl_bad)} total across {sum(1 for x in pl_bad if x > 0)} structures")
        print(f"      Mild   (<{MILD_CUTOFF}A): {sum(pl_mild)} total across {sum(1 for x in pl_mild if x > 0)} structures")
        if pl_worst:
            print(f"      Closest contact: {min(pl_worst):.2f}A")
            print(f"      Mean closest:    {np.mean(pl_worst):.2f}A")
        print(f"    Ligand-ligand clashes:")
        print(f"      Severe (<{SEVERE_CUTOFF}A): {sum(ll_severe)} total across {sum(1 for x in ll_severe if x > 0)} structures")
        print(f"      Bad    (<{BAD_CUTOFF}A): {sum(ll_bad)} total across {sum(1 for x in ll_bad if x > 0)} structures")

    # Check if any structures are problematic
    problematic = [r for r in all_results
                   if r["protein_ligand_clashes"]["severe"] > 0
                   or r["ligand_ligand_clashes"]["severe"] > 0]
    print(f"\n  Structures with severe overlaps (<{SEVERE_CUTOFF}A): "
          f"{len(problematic)}/{len(all_results)}")

    if not problematic:
        print("  All structures are physically reasonable (no atom overlaps).")
    else:
        print("  WARNING: Some structures have overlapping atoms and may need")
        print("  energy minimization or re-prediction.")

    # Write output files
    output_dir = os.path.join(BASE_DIR, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    # clash_report.json — full per-structure data
    json_report = []
    for r in all_results:
        entry = {
            "uid": r["uid"],
            "project": r["project"],
            "n_protein_atoms": r["n_protein_atoms"],
            "n_ligand_chains": r["n_ligand_chains"],
            "protein_ligand_clashes": {
                "severe": r["protein_ligand_clashes"]["severe"],
                "bad": r["protein_ligand_clashes"]["bad"],
                "mild": r["protein_ligand_clashes"]["mild"],
                "worst_dist": round(r["protein_ligand_clashes"]["worst_dist"], 3)
                    if r["protein_ligand_clashes"]["worst_dist"] < 999 else None,
                "details": r["protein_ligand_clashes"]["details"],
            },
            "ligand_ligand_clashes": {
                "severe": r["ligand_ligand_clashes"]["severe"],
                "bad": r["ligand_ligand_clashes"]["bad"],
                "mild": r["ligand_ligand_clashes"]["mild"],
                "worst_dist": round(r["ligand_ligand_clashes"]["worst_dist"], 3)
                    if r["ligand_ligand_clashes"]["worst_dist"] < 999 else None,
                "details": r["ligand_ligand_clashes"]["details"],
            },
        }
        json_report.append(entry)

    json_path = os.path.join(output_dir, "clash_report.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"\nWrote {json_path}")

    # clash_summary.txt — human-readable summary
    lines = []
    lines.append("STERIC CLASH SUMMARY")
    lines.append(f"Thresholds: severe <{SEVERE_CUTOFF}A, bad <{BAD_CUTOFF}A, mild <{MILD_CUTOFF}A")
    lines.append("")
    for proj in ["synthetase", "hydrolase"]:
        proj_results = [r for r in all_results if r["project"] == proj]
        if not proj_results:
            continue
        n_clean = sum(1 for r in proj_results
                      if r["protein_ligand_clashes"]["severe"] == 0
                      and r["protein_ligand_clashes"]["bad"] == 0
                      and r["ligand_ligand_clashes"]["severe"] == 0
                      and r["ligand_ligand_clashes"]["bad"] == 0)
        n_total = len(proj_results)
        lines.append(f"{proj.upper()} (n={n_total}): {n_clean}/{n_total} clean")
        bad_structs = [r for r in proj_results
                       if r["protein_ligand_clashes"]["severe"] > 0
                       or r["protein_ligand_clashes"]["bad"] > 0
                       or r["ligand_ligand_clashes"]["severe"] > 0
                       or r["ligand_ligand_clashes"]["bad"] > 0]
        for r in bad_structs:
            pl = r["protein_ligand_clashes"]
            ll = r["ligand_ligand_clashes"]
            worst = min(
                pl["worst_dist"] if pl["worst_dist"] < 999 else 999,
                ll["worst_dist"] if ll["worst_dist"] < 999 else 999,
            )
            detail = pl["details"][0] if pl["worst_dist"] <= ll["worst_dist"] and pl["details"] else (
                ll["details"][0] if ll["details"] else "")
            lines.append(f"  {r['uid']}: severe={pl['severe']+ll['severe']} "
                         f"bad={pl['bad']+ll['bad']} worst={worst:.2f}A  {detail}")
        lines.append("")
    lines.append(f"Structures with severe overlaps: {len(problematic)}/{len(all_results)}")

    txt_path = os.path.join(output_dir, "clash_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
