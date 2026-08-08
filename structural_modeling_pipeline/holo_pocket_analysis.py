#!/usr/bin/env python3
"""
Holo Pocket Analysis (Geometry + Chemistry)
============================================
Runs pyKVFinder pocket detection on cofolded (holo) structures and analyzes
both pocket geometry and residue chemistry, comparing with apo (AlphaFold)
pocket analysis results.

For each candidate enzyme:
1. Extract protein-only chain from cofolded structure
2. Detect cavities with pyKVFinder (same parameters as apo analysis)
3. Identify the pocket closest to the substrate/cofactor ligands
4. Compute geometry (volume, area, shape) and chemistry (charged/hydrophobic/polar)
5. Compare with apo pocket geometry and chemistry

Outputs:
    - analysis_results/holo_pocket_analysis.csv           (per-enzyme holo pocket geometry)
    - analysis_results/holo_vs_apo_comparison.csv         (geometry comparison)
    - analysis_results/holo_pocket_residues.json          (lining residues for visualization)
    - analysis_results/holo_vs_apo_summary.txt            (statistical summary)
    - analysis_results/holo_pocket_chemistry.csv          (per-enzyme holo pocket chemistry)
    - analysis_results/holo_vs_apo_pocket_chemistry.csv   (chemistry comparison)

Usage:
    python holo_pocket_analysis.py
"""

import argparse
import csv
import glob
import json
import os
import sys
import tempfile

import numpy as np
import yaml

try:
    import pyKVFinder
    HAS_PYKVFINDER = True
except ImportError:
    HAS_PYKVFINDER = False

try:
    import prody
    prody.confProDy(verbosity="none")
    HAS_PRODY = True
except ImportError:
    HAS_PRODY = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Residue classification (same as apo_pocket_analysis.py)
CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "GLY"}

# Kyte-Doolittle hydropathy scale
KD_SCALE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}


def compute_kd_hydropathy(resnames):
    """Compute mean Kyte-Doolittle hydropathy from a list of 3-letter codes."""
    vals = [KD_SCALE[r] for r in resnames if r in KD_SCALE]
    if not vals:
        return None
    return round(np.mean(vals), 3)


def classify_residues(resnames):
    """Compute charged/hydrophobic/polar fractions from a list of 3-letter codes."""
    n = len(resnames)
    if n == 0:
        return {"frac_charged": 0, "frac_hydrophobic": 0, "frac_polar": 0, "n_residues": 0}
    return {
        "frac_charged": round(sum(1 for r in resnames if r in CHARGED) / n, 3),
        "frac_hydrophobic": round(sum(1 for r in resnames if r in HYDROPHOBIC) / n, 3),
        "frac_polar": round(sum(1 for r in resnames if r in POLAR) / n, 3),
        "n_residues": n,
    }


def load_projects():
    """Load project directories from pipeline_config.yaml."""
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


def find_cofold_structures(project_dir):
    """Find all cofolded best_model.pdb files in cofold_output/."""
    cofold_dir = os.path.join(project_dir, "cofold_output")
    if not os.path.isdir(cofold_dir):
        return {}

    structures = {}
    for uid_dir in sorted(os.listdir(cofold_dir)):
        uid_path = os.path.join(cofold_dir, uid_dir)
        if not os.path.isdir(uid_path):
            continue
        # Skip special dirs like native_products, native_substrates
        if "_native_" in uid_dir:
            continue

        best_pdb = glob.glob(os.path.join(uid_path, "**", "*_best_model.pdb"), recursive=True)
        if best_pdb:
            structures[uid_dir] = best_pdb[0]

    return structures


def extract_protein_only(pdb_path):
    """Extract protein chain(s) from cofolded PDB. Returns temp file path."""
    struct = prody.parsePDB(pdb_path)
    if struct is None:
        return None
    protein = struct.select("protein")
    if protein is None or protein.numAtoms() < 50:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    prody.writePDB(tmp.name, protein)
    return tmp.name


def get_ligand_coords(pdb_path):
    """Extract ligand (HETATM) coordinates from cofolded structure.

    Returns dict: {chain: numpy array of coords}
    """
    struct = prody.parsePDB(pdb_path)
    if struct is None:
        return {}

    hetero = struct.select("not protein and not water")
    if hetero is None:
        return {}

    chains = {}
    for chain_id in set(hetero.getChids()):
        chain_sel = hetero.select(f"chain {chain_id}")
        if chain_sel is not None:
            chains[chain_id] = chain_sel.getCoords()

    return chains


def detect_pockets(pdb_path, top_n=10):
    """Detect cavities using pyKVFinder. Same parameters as pocket_analysis.py."""
    if not HAS_PYKVFINDER:
        return []

    try:
        atomic = pyKVFinder.read_pdb(pdb_path)
    except Exception as e:
        print(f"  pyKVFinder read error: {e}", flush=True)
        return []

    # Adaptive parameters (same as pocket_analysis.py)
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
            raw_residues = residues_dict.get(name, [])
            lining_residues = [(r[2], int(r[0]), r[1]) for r in raw_residues]
            if len(lining_residues) < 2:
                continue
            # Compute centroid from residue positions (approximate)
            # Compute Kyte-Doolittle hydropathy from residue names
            resnames = [r[0] for r in lining_residues]
            avg_hydropathy = compute_kd_hydropathy(resnames)

            pockets.append({
                "name": name,
                "volume_A3": round(volumes[name], 1),
                "area_A2": round(areas[name], 1),
                "n_residues": len(lining_residues),
                "lining_residues": lining_residues,
                "avg_hydropathy": avg_hydropathy,
            })

        pockets.sort(key=lambda p: -p["volume_A3"])
        return pockets[:top_n]

    except Exception as e:
        print(f"  pyKVFinder detection error: {e}", flush=True)
        return []


def select_pocket_by_ligand_proximity(pockets, ligand_coords, protein_pdb_path):
    """Select pocket closest to ligand atoms.

    For each pocket, compute how many of its lining residues are within
    contact distance of any ligand atom. Pick the pocket with highest overlap.

    Args:
        pockets: list of pocket dicts from detect_pockets
        ligand_coords: dict of {chain: coords array} from get_ligand_coords
        protein_pdb_path: path to protein-only PDB for residue coordinate lookup

    Returns (selected_pocket, n_contact_residues, contact_fraction)
    """
    if not pockets or not ligand_coords:
        return (pockets[0] if pockets else None), 0, 0.0

    # Combine all ligand coordinates
    all_lig_coords = np.vstack(list(ligand_coords.values()))

    # Load protein structure for residue coordinate lookup
    struct = prody.parsePDB(protein_pdb_path)
    if struct is None:
        return pockets[0], 0, 0.0

    protein_ca = struct.select("protein and calpha")
    if protein_ca is None:
        return pockets[0], 0, 0.0

    # Build resnum -> CA coordinate map
    ca_coords = {}
    for atom in protein_ca:
        ca_coords[int(atom.getResnum())] = atom.getCoords()

    best_pocket = None
    best_contacts = 0
    best_fraction = 0.0

    contact_cutoff = 8.0  # Angstroms from CA to any ligand atom

    for p in pockets:
        pocket_resnums = {r[1] for r in p["lining_residues"]}
        n_contacts = 0
        for resnum in pocket_resnums:
            if resnum in ca_coords:
                ca = ca_coords[resnum].reshape(1, -1)
                dists = np.sqrt(((all_lig_coords - ca) ** 2).sum(axis=1))
                if dists.min() <= contact_cutoff:
                    n_contacts += 1

        fraction = n_contacts / len(pocket_resnums) if pocket_resnums else 0

        if n_contacts > best_contacts or (
            n_contacts == best_contacts and
            p["volume_A3"] > (best_pocket or {}).get("volume_A3", 0)
        ):
            best_pocket = p
            best_contacts = n_contacts
            best_fraction = fraction

    if best_contacts == 0:
        return pockets[0], 0, 0.0

    return best_pocket, best_contacts, round(best_fraction, 3)


def get_confidence_scores(cofold_path):
    """Extract confidence scores from the cofolding output."""
    # Find confidence JSON for best model
    pred_dir = os.path.dirname(cofold_path)
    uid = os.path.basename(pred_dir)

    # Try model_0 confidence first (usually the best)
    conf_files = glob.glob(os.path.join(pred_dir, "confidence_*_model_0.json"))
    if not conf_files:
        conf_files = glob.glob(os.path.join(pred_dir, "confidence_*.json"))
    if not conf_files:
        return {}

    with open(conf_files[0]) as f:
        data = json.load(f)

    return {
        "confidence_score": round(data.get("confidence_score", 0), 3),
        "iptm": round(data.get("iptm", 0), 3),
        "ptm": round(data.get("ptm", 0), 3),
        "ligand_iptm": round(data.get("ligand_iptm", 0), 3),
        "complex_plddt": round(data.get("complex_plddt", 0), 3),
    }


def classify_pocket_shape(volume, area):
    """Classify pocket shape by surface-area-to-volume ratio."""
    if volume <= 0:
        return "unknown"
    sa_v = area / volume
    if sa_v > 1.0:
        return "open_cleft"
    elif sa_v > 0.7:
        return "partially_open"
    else:
        return "enclosed"


def load_apo_results(project_dir):
    """Load apo pocket analysis results from pocket_analysis.json."""
    json_path = os.path.join(project_dir, "pocket_analysis.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        data = json.load(f)

    # Only keep AlphaFold entries (apo structures)
    apo = {}
    for uid, entry in data.items():
        if entry.get("source") == "alphafold" and entry.get("binding_pocket"):
            bp = entry["binding_pocket"]
            apo[uid] = {
                "volume_A3": bp["volume_A3"],
                "area_A2": bp["area_A2"],
                "n_residues": bp["n_residues"],
                "lining_residues": set(r[1] for r in bp["lining_residues"]),
            }
    return apo


def compute_residue_overlap(apo_residues, holo_residues):
    """Compute Jaccard similarity between apo and holo pocket residue sets."""
    if not apo_residues or not holo_residues:
        return 0.0, 0, 0, 0
    intersection = len(apo_residues & holo_residues)
    union = len(apo_residues | holo_residues)
    jaccard = intersection / union if union > 0 else 0.0
    return round(jaccard, 3), intersection, len(apo_residues), len(holo_residues)


def load_apo_chemistry(csv_path):
    """Load apo pocket chemistry from apo_pocket_chemistry.csv."""
    apo = {}
    if not os.path.exists(csv_path):
        return apo
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            apo[row["uid"]] = {
                "apo_project": row["project"],
                "apo_n_residues": int(row["pocket_n_residues"]),
                "apo_frac_charged": float(row["pocket_frac_charged"]),
                "apo_frac_hydrophobic": float(row["pocket_frac_hydrophobic"]),
                "apo_frac_polar": float(row["pocket_frac_polar"]),
                "apo_evidence_type": row["evidence_type"],
            }
    return apo


def main():
    parser = argparse.ArgumentParser(description="Holo vs Apo Pocket Analysis")
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "analysis_results"),
                        help="Output directory")
    args = parser.parse_args()

    if not HAS_PYKVFINDER:
        sys.exit("ERROR: pyKVFinder not installed")
    if not HAS_PRODY:
        sys.exit("ERROR: ProDy not installed")

    os.makedirs(args.output_dir, exist_ok=True)

    projects = load_projects()
    print(f"Holo Pocket Analysis (Geometry + Chemistry)")
    print(f"{'='*60}\n")

    # Load apo pocket chemistry for chemistry comparison
    apo_chem = load_apo_chemistry(os.path.join(args.output_dir, "apo_pocket_chemistry.csv"))
    print(f"Loaded {len(apo_chem)} apo pocket chemistry entries\n")

    all_holo_rows = []
    all_comparison_rows = []
    all_chem_rows = []
    all_chem_comparison_rows = []
    holo_pocket_json = {}  # uid -> {lining_residues, volume, ...} for HTML viz

    for project_dir, project_name in projects:
        print(f"\nProject: {project_name} ({project_dir})")
        print(f"{'-'*60}")

        # Load apo results
        apo_data = load_apo_results(project_dir)
        print(f"  Loaded {len(apo_data)} apo pocket results")

        # Find cofolded structures
        cofold_structs = find_cofold_structures(project_dir)
        print(f"  Found {len(cofold_structs)} cofolded structures\n")

        for uid, cofold_path in sorted(cofold_structs.items()):
            print(f"  {uid}...", end="", flush=True)

            # Extract protein-only PDB
            protein_pdb = extract_protein_only(cofold_path)
            if protein_pdb is None:
                print(" SKIP (no protein chain)")
                continue

            try:
                # Get ligand coordinates
                lig_coords = get_ligand_coords(cofold_path)

                # Detect pockets on protein-only structure
                pockets = detect_pockets(protein_pdb)

                if not pockets:
                    print(" no pockets detected")
                    continue

                # Select pocket by ligand proximity
                best_pocket, n_contacts, contact_frac = select_pocket_by_ligand_proximity(
                    pockets, lig_coords, protein_pdb
                )

                if best_pocket is None:
                    print(" no pocket selected")
                    continue

                # Get confidence scores
                conf = get_confidence_scores(cofold_path)

                # Classify shape
                sa_v = round(best_pocket["area_A2"] / best_pocket["volume_A3"], 3) if best_pocket["volume_A3"] > 0 else 0
                shape = classify_pocket_shape(best_pocket["volume_A3"], best_pocket["area_A2"])

                holo_residues = set(r[1] for r in best_pocket["lining_residues"])

                holo_row = {
                    "project": project_name,
                    "uid": uid,
                    "holo_volume_A3": best_pocket["volume_A3"],
                    "holo_area_A2": best_pocket["area_A2"],
                    "holo_n_residues": best_pocket["n_residues"],
                    "holo_sa_v_ratio": sa_v,
                    "holo_shape": shape,
                    "holo_n_cavities": len(pockets),
                    "holo_ligand_contacts": n_contacts,
                    "holo_contact_fraction": contact_frac,
                    "n_ligand_chains": len(lig_coords),
                    **conf,
                }
                all_holo_rows.append(holo_row)

                # Save lining residues for visualization
                holo_pocket_json[uid] = {
                    "project": project_name,
                    "volume_A3": best_pocket["volume_A3"],
                    "area_A2": best_pocket["area_A2"],
                    "n_residues": best_pocket["n_residues"],
                    "lining_residues": best_pocket["lining_residues"],
                }

                # Pocket chemistry
                resnames = [r[0] for r in best_pocket["lining_residues"]]
                chem = classify_residues(resnames)

                chem_row = {
                    "project": project_name,
                    "uid": uid,
                    "holo_volume_A3": best_pocket["volume_A3"],
                    "holo_area_A2": best_pocket["area_A2"],
                    "holo_sa_v_ratio": sa_v,
                    "holo_n_residues": chem["n_residues"],
                    "holo_frac_charged": chem["frac_charged"],
                    "holo_frac_hydrophobic": chem["frac_hydrophobic"],
                    "holo_frac_polar": chem["frac_polar"],
                    "holo_avg_hydropathy": best_pocket.get("avg_hydropathy", ""),
                    "holo_ligand_contacts": n_contacts,
                    "holo_contact_fraction": contact_frac,
                    **conf,
                }
                all_chem_rows.append(chem_row)

                # Compare chemistry with apo
                if uid in apo_chem:
                    apo_c = apo_chem[uid]
                    chem_comparison_row = {
                        "project": project_name,
                        "uid": uid,
                        "apo_n_residues": apo_c["apo_n_residues"],
                        "apo_frac_charged": apo_c["apo_frac_charged"],
                        "apo_frac_hydrophobic": apo_c["apo_frac_hydrophobic"],
                        "apo_frac_polar": apo_c["apo_frac_polar"],
                        "holo_volume_A3": best_pocket["volume_A3"],
                        "holo_n_residues": chem["n_residues"],
                        "holo_frac_charged": chem["frac_charged"],
                        "holo_frac_hydrophobic": chem["frac_hydrophobic"],
                        "holo_frac_polar": chem["frac_polar"],
                        "holo_avg_hydropathy": best_pocket.get("avg_hydropathy", ""),
                        "delta_frac_charged": round(chem["frac_charged"] - apo_c["apo_frac_charged"], 3),
                        "delta_frac_hydrophobic": round(chem["frac_hydrophobic"] - apo_c["apo_frac_hydrophobic"], 3),
                        "delta_frac_polar": round(chem["frac_polar"] - apo_c["apo_frac_polar"], 3),
                        "holo_ligand_contacts": n_contacts,
                        "holo_contact_fraction": contact_frac,
                        **conf,
                    }
                    all_chem_comparison_rows.append(chem_comparison_row)

                # Compare geometry with apo
                if uid in apo_data:
                    apo = apo_data[uid]
                    jaccard, n_shared, n_apo_only_total, n_holo_only_total = compute_residue_overlap(
                        apo["lining_residues"], holo_residues
                    )

                    vol_change = best_pocket["volume_A3"] - apo["volume_A3"]
                    vol_pct = round(100 * vol_change / apo["volume_A3"], 1) if apo["volume_A3"] > 0 else 0
                    area_change = best_pocket["area_A2"] - apo["area_A2"]
                    apo_shape = classify_pocket_shape(apo["volume_A3"], apo["area_A2"])

                    comparison_row = {
                        "project": project_name,
                        "uid": uid,
                        "apo_volume_A3": apo["volume_A3"],
                        "holo_volume_A3": best_pocket["volume_A3"],
                        "volume_change_A3": round(vol_change, 1),
                        "volume_change_pct": vol_pct,
                        "apo_area_A2": apo["area_A2"],
                        "holo_area_A2": best_pocket["area_A2"],
                        "area_change_A2": round(area_change, 1),
                        "apo_n_residues": apo["n_residues"],
                        "holo_n_residues": best_pocket["n_residues"],
                        "apo_shape": apo_shape,
                        "holo_shape": shape,
                        "residue_jaccard": jaccard,
                        "shared_residues": n_shared,
                        "holo_ligand_contacts": n_contacts,
                        "holo_contact_fraction": contact_frac,
                        **conf,
                    }
                    all_comparison_rows.append(comparison_row)

                    direction = "expanded" if vol_change > 0 else "contracted"
                    print(f" V={best_pocket['volume_A3']}A3 ({direction} {abs(vol_pct)}%), "
                          f"Jaccard={jaccard}, contacts={n_contacts}")
                else:
                    print(f" V={best_pocket['volume_A3']}A3, contacts={n_contacts} (no apo data)")

            finally:
                if protein_pdb and os.path.exists(protein_pdb):
                    os.remove(protein_pdb)

    # ── Write outputs ──
    print(f"\n{'='*60}")
    print("Writing outputs...\n")

    # Holo pocket analysis CSV
    holo_csv = os.path.join(args.output_dir, "holo_pocket_analysis.csv")
    if all_holo_rows:
        with open(holo_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_holo_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_holo_rows)
        print(f"  {holo_csv} ({len(all_holo_rows)} entries)")

    # Comparison CSV
    comp_csv = os.path.join(args.output_dir, "holo_vs_apo_comparison.csv")
    if all_comparison_rows:
        with open(comp_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_comparison_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_comparison_rows)
        print(f"  {comp_csv} ({len(all_comparison_rows)} entries)")

    # Holo pocket chemistry CSV
    chem_csv = os.path.join(args.output_dir, "holo_pocket_chemistry.csv")
    if all_chem_rows:
        with open(chem_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_chem_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_chem_rows)
        print(f"  {chem_csv} ({len(all_chem_rows)} entries)")

    # Chemistry comparison CSV
    chem_comp_csv = os.path.join(args.output_dir, "holo_vs_apo_pocket_chemistry.csv")
    if all_chem_comparison_rows:
        with open(chem_comp_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_chem_comparison_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_chem_comparison_rows)
        print(f"  {chem_comp_csv} ({len(all_chem_comparison_rows)} entries)")

    # Holo pocket JSON (with lining residues for visualization)
    holo_json_path = os.path.join(args.output_dir, "holo_pocket_residues.json")
    with open(holo_json_path, "w") as f:
        json.dump(holo_pocket_json, f, indent=2)
    print(f"  {holo_json_path} ({len(holo_pocket_json)} entries)")

    # Summary statistics
    summary_path = os.path.join(args.output_dir, "holo_vs_apo_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Holo vs Apo Pocket Analysis Summary\n")
        f.write("=" * 60 + "\n\n")

        for project_name in set(r["project"] for r in all_comparison_rows):
            rows = [r for r in all_comparison_rows if r["project"] == project_name]
            if not rows:
                continue

            f.write(f"Project: {project_name} ({len(rows)} enzymes compared)\n")
            f.write("-" * 50 + "\n\n")

            vol_changes = [r["volume_change_pct"] for r in rows]
            jaccards = [r["residue_jaccard"] for r in rows]
            contact_fracs = [r["holo_contact_fraction"] for r in rows]

            expanded = sum(1 for v in vol_changes if v > 10)
            contracted = sum(1 for v in vol_changes if v < -10)
            similar = len(vol_changes) - expanded - contracted

            f.write("Volume changes (holo vs apo):\n")
            f.write(f"  Mean change: {np.mean(vol_changes):+.1f}%\n")
            f.write(f"  Median change: {np.median(vol_changes):+.1f}%\n")
            f.write(f"  Std dev: {np.std(vol_changes):.1f}%\n")
            f.write(f"  Range: {min(vol_changes):+.1f}% to {max(vol_changes):+.1f}%\n")
            f.write(f"  Expanded (>10%): {expanded}/{len(rows)}\n")
            f.write(f"  Contracted (<-10%): {contracted}/{len(rows)}\n")
            f.write(f"  Similar (within 10%): {similar}/{len(rows)}\n\n")

            f.write("Residue overlap (Jaccard similarity):\n")
            f.write(f"  Mean: {np.mean(jaccards):.3f}\n")
            f.write(f"  Median: {np.median(jaccards):.3f}\n")
            f.write(f"  Range: {min(jaccards):.3f} to {max(jaccards):.3f}\n\n")

            f.write("Ligand contact fraction:\n")
            f.write(f"  Mean: {np.mean(contact_fracs):.3f}\n")
            f.write(f"  Median: {np.median(contact_fracs):.3f}\n\n")

            # Shape change analysis
            shape_changed = sum(1 for r in rows if r["apo_shape"] != r["holo_shape"])
            f.write(f"Shape changed: {shape_changed}/{len(rows)} enzymes\n\n")

            # Confidence-correlated analysis
            if any(r.get("iptm", 0) > 0 for r in rows):
                high_conf = [r for r in rows if r.get("iptm", 0) >= 0.5]
                low_conf = [r for r in rows if r.get("iptm", 0) < 0.5]
                f.write("By cofolding confidence (ipTM):\n")
                if high_conf:
                    f.write(f"  High (ipTM >= 0.5, n={len(high_conf)}): "
                            f"mean vol change {np.mean([r['volume_change_pct'] for r in high_conf]):+.1f}%, "
                            f"mean Jaccard {np.mean([r['residue_jaccard'] for r in high_conf]):.3f}\n")
                if low_conf:
                    f.write(f"  Low  (ipTM <  0.5, n={len(low_conf)}):  "
                            f"mean vol change {np.mean([r['volume_change_pct'] for r in low_conf]):+.1f}%, "
                            f"mean Jaccard {np.mean([r['residue_jaccard'] for r in low_conf]):.3f}\n")
                f.write("\n")

            # Top differences
            sorted_by_change = sorted(rows, key=lambda r: abs(r["volume_change_pct"]), reverse=True)
            f.write("Largest volume differences:\n")
            for r in sorted_by_change[:5]:
                f.write(f"  {r['uid']}: apo={r['apo_volume_A3']}A3, holo={r['holo_volume_A3']}A3 "
                        f"({r['volume_change_pct']:+.1f}%), Jaccard={r['residue_jaccard']}\n")
            f.write("\n")

            # Lowest residue overlap
            sorted_by_jaccard = sorted(rows, key=lambda r: r["residue_jaccard"])
            f.write("Lowest residue overlap (most different pockets):\n")
            for r in sorted_by_jaccard[:5]:
                f.write(f"  {r['uid']}: Jaccard={r['residue_jaccard']}, "
                        f"shared={r['shared_residues']}, vol change={r['volume_change_pct']:+.1f}%\n")
            f.write("\n\n")

    print(f"  {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
