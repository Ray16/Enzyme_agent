#!/usr/bin/env python3
"""Pocket analysis on Boltz-2 apo structures.

Reuses active site evidence from the existing pocket_analysis.json files
(generated from AlphaFold structures). For each enzyme:
  1. Loads Boltz-2 predicted PDB
  2. Runs pyKVFinder cavity detection (same parameters as pocket_analysis.py)
  3. Selects pocket with best overlap to existing active site residues
  4. Computes pocket chemistry (hydrophobic/charged/polar fractions)

Output:
    analysis_results/boltz_apo_pocket_analysis.json
"""

import glob
import json
import os

import pyKVFinder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "GLY"}


def find_boltz_pdb(boltz_output_dir, uid):
    patterns = [
        os.path.join(boltz_output_dir, uid, "**", f"{uid}_model_0.pdb"),
        os.path.join(boltz_output_dir, uid, "**", f"{uid}_best_model.pdb"),
        os.path.join(boltz_output_dir, uid, "**", "*.pdb"),
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def detect_pockets(pdb_path, top_n=10):
    try:
        atomic = pyKVFinder.read_pdb(pdb_path)
    except Exception as e:
        print(f"  pyKVFinder read error: {e}")
        return []

    n_atoms = len(atomic)
    est_residues = n_atoms / 8
    if est_residues < 400:
        probe_out, removal_distance = 8.0, 1.0
    else:
        probe_out, removal_distance = 4.0, 2.4

    try:
        vertices = pyKVFinder.get_vertices(atomic, step=0.6, probe_out=probe_out)
        ncav, cavities = pyKVFinder.detect(
            atomic, vertices, step=0.6,
            probe_in=1.4, probe_out=probe_out,
            removal_distance=removal_distance,
            volume_cutoff=5.0, verbose=False,
        )
        if ncav == 0:
            return []

        _, volumes, areas = pyKVFinder.spatial(cavities, step=0.6)
        residues_dict = pyKVFinder.constitutional(
            cavities, atomic, vertices, step=0.6, ignore_backbone=False
        )

        pockets = []
        for name in volumes.keys():
            raw = residues_dict.get(name, [])
            lining = [(r[2], int(r[0]), r[1]) for r in raw]
            if len(lining) < 2:
                continue
            pockets.append({
                "name": name,
                "volume_A3": round(volumes[name], 1),
                "area_A2": round(areas[name], 1),
                "n_residues": len(lining),
                "lining_residues": lining,
            })

        pockets.sort(key=lambda p: -p["volume_A3"])
        return pockets[:top_n]
    except Exception as e:
        print(f"  pyKVFinder detection error: {e}")
        return []


def select_binding_pocket(pockets, active_site_residues):
    if not pockets:
        return None, 0, 0.0
    if not active_site_residues:
        return pockets[0], 0, 0.0

    active_set = set(active_site_residues)
    best_pocket, best_overlap, best_frac = None, 0, 0.0

    for p in pockets:
        pocket_resnums = {r[1] for r in p["lining_residues"]}
        overlap = len(pocket_resnums & active_set)
        frac = overlap / len(active_set)
        if overlap > best_overlap or (
            overlap == best_overlap
            and p["volume_A3"] > (best_pocket or {}).get("volume_A3", 0)
        ):
            best_pocket, best_overlap, best_frac = p, overlap, frac

    if best_overlap == 0:
        return pockets[0], 0, 0.0
    return best_pocket, best_overlap, round(best_frac, 3)


def pocket_chemistry(lining_residues):
    names = [r[0] for r in lining_residues]
    n = len(names)
    if n == 0:
        return {}
    return {
        "frac_hydrophobic": round(sum(1 for r in names if r in HYDROPHOBIC) / n * 100, 1),
        "frac_charged": round(sum(1 for r in names if r in CHARGED) / n * 100, 1),
        "frac_polar": round(sum(1 for r in names if r in POLAR) / n * 100, 1),
        "n_residues": n,
    }


def process_project(project_name, project_dir):
    pa_path = os.path.join(project_dir, "pocket_analysis.json")
    boltz_out = os.path.join(project_dir, "boltz_output")

    if not os.path.exists(pa_path):
        print(f"  No pocket_analysis.json in {project_dir}")
        return {}
    if not os.path.isdir(boltz_out):
        print(f"  No boltz_output/ in {project_dir}")
        return {}

    with open(pa_path) as f:
        pa_data = json.load(f)

    results = {}
    for uid, entry in pa_data.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("role") == "reference":
            continue
        # Skip PDB-code entries (4-char codes starting with digit)
        if len(uid) == 4 and uid[0].isdigit():
            continue

        evidence = entry.get("evidence", {})
        active_site = evidence.get("active_site_residues", [])

        pdb_path = find_boltz_pdb(boltz_out, uid)
        if pdb_path is None:
            continue

        pockets = detect_pockets(pdb_path)
        if not pockets:
            print(f"  {uid}: no cavities detected")
            continue

        pocket, overlap, frac = select_binding_pocket(pockets, active_site)
        if pocket is None:
            continue

        chem = pocket_chemistry(pocket["lining_residues"])
        results[uid] = {
            "uid": uid,
            "project": project_name,
            "boltz_pdb": pdb_path,
            "evidence_type": evidence.get("type", "none"),
            "n_active_site_residues": len(active_site),
            "pocket_overlap": overlap,
            "pocket_overlap_fraction": frac,
            "binding_pocket": pocket,
            "chemistry": chem,
        }
        print(f"  {uid}: {chem.get('n_residues', 0)} lining residues, "
              f"hydrophobic={chem.get('frac_hydrophobic', 0):.1f}%, "
              f"charged={chem.get('frac_charged', 0):.1f}%, "
              f"overlap={overlap}/{len(active_site)}")

    return results


def main():
    all_results = {}

    # Synthetases
    print("Synthetases:")
    syn_dir = os.path.join(BASE_DIR, "amide_synthetase")
    syn_results = process_project("synthetase", syn_dir)
    all_results.update(syn_results)
    print(f"  Total: {len(syn_results)} pockets detected\n")

    # Hydrolases
    print("Hydrolases:")
    hyd_dir = os.path.join(BASE_DIR, "amide_hydrolase")
    hyd_results = process_project("hydrolase", hyd_dir)
    all_results.update(hyd_results)
    print(f"  Total: {len(hyd_results)} pockets detected\n")

    # NylC reference (special case — not in pocket_analysis.json as candidate)
    nylc_uid = "Q1EPR4"
    nylc_pdb = find_boltz_pdb(os.path.join(hyd_dir, "boltz_output"), nylc_uid)
    if nylc_pdb:
        print("NylC reference:")
        pockets = detect_pockets(nylc_pdb)
        if pockets:
            # NylC has known active site from existing pocket_analysis
            pa_path = os.path.join(hyd_dir, "pocket_analysis.json")
            nylc_active = []
            if os.path.exists(pa_path):
                with open(pa_path) as f:
                    pa = json.load(f)
                nylc_entry = pa.get(nylc_uid, {})
                if isinstance(nylc_entry, dict):
                    nylc_active = nylc_entry.get("evidence", {}).get("active_site_residues", [])

            pocket, overlap, frac = select_binding_pocket(pockets, nylc_active)
            if pocket:
                chem = pocket_chemistry(pocket["lining_residues"])
                all_results[nylc_uid] = {
                    "uid": nylc_uid,
                    "project": "hydrolase_reference",
                    "boltz_pdb": nylc_pdb,
                    "evidence_type": "reference",
                    "n_active_site_residues": len(nylc_active),
                    "pocket_overlap": overlap,
                    "pocket_overlap_fraction": frac,
                    "binding_pocket": pocket,
                    "chemistry": chem,
                }
                print(f"  NylC: {chem['n_residues']} lining residues, "
                      f"hydrophobic={chem['frac_hydrophobic']:.1f}%, "
                      f"charged={chem['frac_charged']:.1f}%")
        else:
            print("  NylC: no cavities detected")
    else:
        print("NylC: Boltz-2 PDB not found")

    # Save results
    out_path = os.path.join(BASE_DIR, "analysis_results", "boltz_apo_pocket_analysis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} entries to {out_path}")


if __name__ == "__main__":
    main()
