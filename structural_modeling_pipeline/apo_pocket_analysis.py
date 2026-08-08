#!/usr/bin/env python3
"""
Apo Pocket Analysis
====================
Analyze the chemistry of apo (pre-cofolding) catalytic pockets from Phase 1/2
(pocket_analysis.json) vs Boltz-2 substrate contact pockets.

Computes residue chemistry fractions (charged/hydrophobic/polar) for the
predicted catalytic pocket and compares with where Boltz-2 actually placed
the substrate (contact residues from cofolding_metrics.csv).

Output:
    - analysis_results/apo_pocket_chemistry.csv  (per-candidate apo pocket data)

Usage:
    python apo_pocket_analysis.py
"""

import json
import csv
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_projects(config_path=None):
    """Load project directories from pipeline_config.yaml."""
    import yaml
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

CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "GLY"}


def classify_residues(resnames):
    """Compute charged/hydrophobic/polar fractions from a list of 3-letter codes."""
    n = len(resnames)
    if n == 0:
        return {"frac_charged": 0, "frac_hydrophobic": 0, "frac_polar": 0, "n_residues": 0}
    return {
        "frac_charged": sum(1 for r in resnames if r in CHARGED) / n,
        "frac_hydrophobic": sum(1 for r in resnames if r in HYDROPHOBIC) / n,
        "frac_polar": sum(1 for r in resnames if r in POLAR) / n,
        "n_residues": n,
    }


def load_boltz2_contact_chemistry(csv_path):
    """Load Boltz-2 contact pocket chemistry from cofolding_metrics.csv."""
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["project"] != "hydrolase":
                continue
            results[row["uid"]] = {
                "boltz2_frac_charged": float(row["pocket_frac_charged"]),
                "boltz2_frac_hydrophobic": float(row["pocket_frac_hydrophobic"]),
                "boltz2_frac_polar": float(row["pocket_frac_polar"]),
                "boltz2_n_contact": int(row["substrate_n_contact_residues"]),
            }
    return results


def main():
    projects = load_projects()

    # Load Boltz-2 contact pocket chemistry (all projects)
    csv_path = os.path.join(BASE_DIR, "analysis_results", "cofolding_metrics.csv")
    boltz2_all = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                boltz2_all[row["uid"]] = {
                    "boltz2_frac_charged": float(row["pocket_frac_charged"]),
                    "boltz2_frac_hydrophobic": float(row["pocket_frac_hydrophobic"]),
                    "boltz2_frac_polar": float(row["pocket_frac_polar"]),
                    "boltz2_n_contact": int(row["substrate_n_contact_residues"]),
                }

    all_results = []
    for project_dir, project_name in projects:
        pa_path = os.path.join(project_dir, "pocket_analysis.json")
        if not os.path.exists(pa_path):
            continue

        with open(pa_path) as f:
            pocket_data = json.load(f)

        cv_path = os.path.join(project_dir, "catalytic_site_validation.json")
        cat_data = {}
        if os.path.exists(cv_path):
            with open(cv_path) as f:
                cat_data = json.load(f)

        results = []
        for uid in sorted(pocket_data.keys()):
            entry = pocket_data[uid]
            if not entry or not entry.get("binding_pocket"):
                continue
            bp = entry["binding_pocket"]
            lining = bp.get("lining_residues", [])
            if not lining:
                continue

            resnames = [r[0] for r in lining]
            chem = classify_residues(resnames)
            evidence_type = entry.get("evidence", {}).get("type", "none")

            cat_entry = cat_data.get(uid, {})
            core_residues = cat_entry.get("core_residues", [])
            cat_chem = None
            if core_residues:
                resid_to_name = {r[1]: r[0] for r in lining}
                missing = [r for r in core_residues if r not in resid_to_name]
                if missing:
                    pdb_path = os.path.join(project_dir, "alphafold_structures",
                                            f"AF-{uid}-F1-model_v4.pdb")
                    if os.path.exists(pdb_path):
                        with open(pdb_path) as f:
                            for line in f:
                                if not line.startswith("ATOM"):
                                    continue
                                resid = int(line[22:26])
                                if resid in missing:
                                    resid_to_name[resid] = line[17:20].strip()
                cat_names = [resid_to_name[r] for r in core_residues if r in resid_to_name]
                if cat_names:
                    cat_chem = classify_residues(cat_names)

            b2 = boltz2_all.get(uid, {})

            row = {
                "uid": uid,
                "project": project_name,
                "evidence_type": evidence_type,
                "pocket_n_residues": chem["n_residues"],
                "pocket_frac_charged": chem["frac_charged"],
                "pocket_frac_hydrophobic": chem["frac_hydrophobic"],
                "pocket_frac_polar": chem["frac_polar"],
                "boltz2_n_contact": b2.get("boltz2_n_contact", ""),
                "boltz2_frac_charged": b2.get("boltz2_frac_charged", ""),
                "boltz2_frac_hydrophobic": b2.get("boltz2_frac_hydrophobic", ""),
                "boltz2_frac_polar": b2.get("boltz2_frac_polar", ""),
                "n_core_residues": len(core_residues),
                "cat_frac_charged": cat_chem["frac_charged"] if cat_chem else "",
                "cat_frac_hydrophobic": cat_chem["frac_hydrophobic"] if cat_chem else "",
                "cat_frac_polar": cat_chem["frac_polar"] if cat_chem else "",
            }
            results.append(row)

        all_results.extend(results)
        print(f"  {project_name}: {len(results)} candidates with pocket data")

    if not all_results:
        print("No results to write.")
        return
    results = all_results

    # Write per-candidate CSV
    out_csv = os.path.join(BASE_DIR, "analysis_results", "apo_pocket_chemistry.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} entries to {out_csv}\n")

    # Aggregate statistics
    pocket_charged = [r["pocket_frac_charged"] for r in results]
    pocket_hydro = [r["pocket_frac_hydrophobic"] for r in results]
    pocket_polar = [r["pocket_frac_polar"] for r in results]

    boltz2_charged = [r["boltz2_frac_charged"] for r in results if r["boltz2_frac_charged"] != ""]
    boltz2_hydro = [r["boltz2_frac_hydrophobic"] for r in results if r["boltz2_frac_hydrophobic"] != ""]

    print("=" * 70)
    print("CATALYTIC POCKET (Phase 2) vs BOLTZ-2 SUBSTRATE CONTACT POCKET")
    print("=" * 70)
    print(f"\nCandidates with pocket data: {len(results)}")
    print(f"Candidates with Boltz-2 data: {len(boltz2_charged)}")
    print()
    print(f"{'Metric':<35} {'Catalytic Pocket':>17} {'Boltz-2 Contact':>17}")
    print("-" * 70)
    print(f"{'Mean % charged':<35} {np.mean(pocket_charged)*100:>16.1f}% {np.mean(boltz2_charged)*100:>16.1f}%")
    print(f"{'Median % charged':<35} {np.median(pocket_charged)*100:>16.1f}% {np.median(boltz2_charged)*100:>16.1f}%")
    print(f"{'Mean % hydrophobic':<35} {np.mean(pocket_hydro)*100:>16.1f}% {np.mean(boltz2_hydro)*100:>16.1f}%")
    print(f"{'Mean % polar (uncharged)':<35} {np.mean(pocket_polar)*100:>16.1f}%")
    print()

    # Breakdown by evidence type
    print("CATALYTIC POCKET CHEMISTRY BY EVIDENCE TYPE")
    print("-" * 70)
    from collections import defaultdict
    by_ev = defaultdict(list)
    for r in results:
        by_ev[r["evidence_type"]].append(r)

    for ev in sorted(by_ev.keys()):
        entries = by_ev[ev]
        ch = [e["pocket_frac_charged"] for e in entries]
        hy = [e["pocket_frac_hydrophobic"] for e in entries]
        print(f"  {ev} (n={len(entries)}): charged={np.mean(ch)*100:.1f}%, hydrophobic={np.mean(hy)*100:.1f}%")

    # Candidates with known catalytic residues
    cat_results = [r for r in results if r["cat_frac_charged"] != ""]
    if cat_results:
        print(f"\nCANDIDATES WITH KNOWN CATALYTIC RESIDUES (n={len(cat_results)})")
        print("-" * 70)
        for r in cat_results:
            print(f"  {r['uid']}: pocket charged={r['pocket_frac_charged']*100:.1f}%, "
                  f"catalytic residues charged={r['cat_frac_charged']*100:.1f}%, "
                  f"Boltz-2 contact charged={r['boltz2_frac_charged']*100:.1f}%" if r['boltz2_frac_charged'] != "" else
                  f"  {r['uid']}: pocket charged={r['pocket_frac_charged']*100:.1f}%, "
                  f"catalytic residues charged={r['cat_frac_charged']*100:.1f}%")

    # PDB reference structures (if any exist in the results)
    pdb_entries = [r for r in results if len(r["uid"]) == 4 and r["uid"][0].isdigit()]
    if pdb_entries:
        print(f"\nPDB REFERENCE STRUCTURES")
        print("-" * 70)
        for r in pdb_entries:
            print(f"  {r['uid']}: pocket charged={r['pocket_frac_charged']*100:.1f}%, "
                  f"hydrophobic={r['pocket_frac_hydrophobic']*100:.1f}%, "
                  f"n_residues={r['pocket_n_residues']}")


if __name__ == "__main__":
    main()
