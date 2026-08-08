#!/usr/bin/env python3
"""
Systematic analysis of Boltz-2 cofolding results for enzyme-substrate complexes.

Compares amide synthetase vs amide hydrolase cofolding quality, binding geometry,
and structural metrics to understand differential catalytic outcomes.

Usage:
    python analyze_cofolding.py
    python analyze_cofolding.py --output-dir analysis_results

Outputs:
    - cofolding_metrics.csv: Per-candidate metrics for all structures
    - cofolding_summary.txt: Statistical summary and comparison
    - confidence_distribution.png: Confidence score histograms
    - binding_quality.png: Multi-panel binding quality comparison
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

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


def parse_confidence_json(json_path):
    """Parse Boltz-2 confidence JSON into a flat dict of metrics."""
    with open(json_path) as f:
        data = json.load(f)

    metrics = {
        "confidence_score": data.get("confidence_score", None),
        "ptm": data.get("ptm", None),
        "iptm": data.get("iptm", None),
        "ligand_iptm": data.get("ligand_iptm", None),
        "protein_iptm": data.get("protein_iptm", None),
        "complex_plddt": data.get("complex_plddt", None),
        "complex_iplddt": data.get("complex_iplddt", None),
        "complex_pde": data.get("complex_pde", None),
        "complex_ipde": data.get("complex_ipde", None),
    }

    # Per-chain PTM scores
    chains_ptm = data.get("chains_ptm", {})
    metrics["protein_chain_ptm"] = chains_ptm.get("0", None)
    metrics["n_chains"] = len(chains_ptm)

    # Ligand chain PTM scores (chains 1+)
    ligand_ptms = [v for k, v in chains_ptm.items() if k != "0"]
    metrics["mean_ligand_chain_ptm"] = np.mean(ligand_ptms) if ligand_ptms else None

    # Pairwise interface scores: protein-ligand interactions
    pair_iptm = data.get("pair_chains_iptm", {})
    protein_to_ligand_iptms = []
    ligand_to_protein_iptms = []
    for lig_chain_id in [k for k in pair_iptm.keys() if k != "0"]:
        # Ligand -> protein iptm
        val = pair_iptm.get(lig_chain_id, {}).get("0", None)
        if val is not None:
            ligand_to_protein_iptms.append(val)
        # Protein -> ligand iptm
        val = pair_iptm.get("0", {}).get(lig_chain_id, None)
        if val is not None:
            protein_to_ligand_iptms.append(val)

    metrics["mean_protein_to_ligand_iptm"] = (
        np.mean(protein_to_ligand_iptms) if protein_to_ligand_iptms else None
    )
    metrics["mean_ligand_to_protein_iptm"] = (
        np.mean(ligand_to_protein_iptms) if ligand_to_protein_iptms else None
    )
    metrics["max_ligand_to_protein_iptm"] = (
        max(ligand_to_protein_iptms) if ligand_to_protein_iptms else None
    )

    # Store raw pairwise iPTM matrix for substrate-specific analysis downstream
    metrics["pair_chains_iptm"] = pair_iptm

    return metrics


def parse_pdb_structure(pdb_path):
    """Parse PDB to extract structural metrics for binding analysis."""
    protein_atoms = []
    ligand_atoms = defaultdict(list)
    protein_residues = set()

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66])
                resname = line[17:20].strip()
                resid = int(line[22:26])
                chain = line[21]
                atom_name = line[12:16].strip()
                protein_atoms.append({
                    "xyz": np.array([x, y, z]),
                    "bfactor": bfactor,
                    "resname": resname,
                    "resid": resid,
                    "chain": chain,
                    "atom_name": atom_name,
                })
                protein_residues.add((chain, resid))

            elif line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66])
                resname = line[17:20].strip()
                chain = line[21]
                atom_name = line[12:16].strip()
                ligand_atoms[chain].append({
                    "xyz": np.array([x, y, z]),
                    "bfactor": bfactor,
                    "resname": resname,
                    "atom_name": atom_name,
                })

    return {
        "protein_atoms": protein_atoms,
        "ligand_atoms": dict(ligand_atoms),
        "protein_residue_count": len(protein_residues),
        "ligand_chain_count": len(ligand_atoms),
    }


def compute_sasa_burial(pdb_path):
    """Compute SASA-based burial for each ligand chain."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except Exception:
        return {}
        
    model = structure[0]

    protein_chain_ids = set()
    ligand_chain_ids = set()
    for chain in model:
        for residue in chain:
            hetflag = residue.get_id()[0]
            if hetflag == " ":  
                protein_chain_ids.add(chain.id)
            else:  
                ligand_chain_ids.add(chain.id)
            break  

    ligand_chain_ids -= protein_chain_ids
    if not ligand_chain_ids:
        return {}

    sr = ShrakeRupley()
    sr.compute(structure, level="A")
    complex_sasa_per_chain = {}
    for chain_id in ligand_chain_ids:
        chain = model[chain_id]
        chain_sasa = sum(atom.sasa for residue in chain for atom in residue)
        complex_sasa_per_chain[chain_id] = chain_sasa

    results = {}
    for chain_id in ligand_chain_ids:
        iso_parser = PDBParser(QUIET=True)
        iso_structure = iso_parser.get_structure("iso", pdb_path)
        iso_model = iso_structure[0]
        chains_to_remove = [c.id for c in iso_model if c.id != chain_id]
        for cid in chains_to_remove:
            iso_model.detach_child(cid)

        sr_iso = ShrakeRupley()
        sr_iso.compute(iso_structure, level="A")
        iso_sasa = sum(atom.sasa for residue in iso_model[chain_id] for atom in residue)
        complex_sasa = complex_sasa_per_chain[chain_id]

        burial = 1.0 - (complex_sasa / iso_sasa) if iso_sasa > 0 else 0.0
        if burial < 0.0:
            print(f"  Warning: chain {chain_id} has negative SASA burial ({burial:.3f}), "
                  f"complex_sasa={complex_sasa:.1f} > iso_sasa={iso_sasa:.1f}; clamping to 0",
                  file=sys.stderr)
        results[chain_id] = {
            "sasa_isolated": iso_sasa,
            "sasa_complex": complex_sasa,
            "sasa_burial": max(0.0, min(1.0, burial)),
        }

    return results


def compute_binding_metrics(pdb_data, contact_cutoff=5.0):
    """Compute binding quality metrics from parsed PDB data."""
    protein_coords = np.array([a["xyz"] for a in pdb_data["protein_atoms"]])
    protein_bfactors = np.array([a["bfactor"] for a in pdb_data["protein_atoms"]])

    if len(protein_coords) == 0:
        return {}

    metrics = {
        "protein_residue_count": pdb_data["protein_residue_count"],
        "ligand_chain_count": pdb_data["ligand_chain_count"],
        "mean_protein_plddt": np.mean(protein_bfactors),
        "median_protein_plddt": np.median(protein_bfactors),
    }

    all_contact_plddts = []
    all_min_dists = []
    per_chain_metrics = {}

    for chain_id, lig_atoms in pdb_data["ligand_atoms"].items():
        lig_coords = np.array([a["xyz"] for a in lig_atoms])
        lig_bfactors = np.array([a["bfactor"] for a in lig_atoms])
        n_lig_atoms = len(lig_coords)

        if n_lig_atoms == 0:
            continue

        min_dists_to_protein = np.full(n_lig_atoms, np.inf)
        contact_protein_indices = set()
        contact_protein_residues = set()

        chunk_size = 5000
        for i in range(0, len(protein_coords), chunk_size):
            prot_chunk = protein_coords[i:i + chunk_size]
            diff = lig_coords[:, None, :] - prot_chunk[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            chunk_min = np.min(dists, axis=1)
            min_dists_to_protein = np.minimum(min_dists_to_protein, chunk_min)

            contact_mask = dists < contact_cutoff
            for lig_idx in range(n_lig_atoms):
                for prot_offset in np.where(contact_mask[lig_idx])[0]:
                    abs_idx = i + prot_offset
                    contact_protein_indices.add(abs_idx)
                    atom = pdb_data["protein_atoms"][abs_idx]
                    contact_protein_residues.add((atom["chain"], atom["resid"]))

        if contact_protein_indices:
            contact_plddts = protein_bfactors[list(contact_protein_indices)]
            mean_contact_plddt = np.mean(contact_plddts)
        else:
            mean_contact_plddt = 0.0

        mean_ligand_plddt = np.mean(lig_bfactors)
        contact_residue_info = []
        for chain, resid in contact_protein_residues:
            matching = [a for a in pdb_data["protein_atoms"] if a["chain"] == chain and a["resid"] == resid]
            if matching: contact_residue_info.append(matching[0]["resname"])

        charged_resnames = {"ARG", "LYS", "ASP", "GLU", "HIS"}
        hydrophobic_resnames = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
        polar_resnames = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "GLY"}

        n_contact_res = len(contact_residue_info)
        if n_contact_res > 0:
            frac_charged = sum(1 for r in contact_residue_info if r in charged_resnames) / n_contact_res
            frac_hydrophobic = sum(1 for r in contact_residue_info if r in hydrophobic_resnames) / n_contact_res
            frac_polar = sum(1 for r in contact_residue_info if r in polar_resnames) / n_contact_res
        else:
            frac_charged = frac_hydrophobic = frac_polar = 0.0

        per_chain_metrics[chain_id] = {
            "n_atoms": n_lig_atoms,
            "n_contact_residues": n_contact_res,
            "mean_contact_plddt": mean_contact_plddt,
            "mean_ligand_plddt": mean_ligand_plddt,
            "min_distance": np.min(min_dists_to_protein),
            "mean_min_distance": np.mean(min_dists_to_protein),
            "frac_charged": frac_charged,
            "frac_hydrophobic": frac_hydrophobic,
            "frac_polar": frac_polar,
        }
        all_contact_plddts.append(mean_contact_plddt)
        all_min_dists.append(np.min(min_dists_to_protein))

    metrics["per_chain"] = per_chain_metrics
    metrics["mean_contact_plddt"] = np.mean(all_contact_plddts) if all_contact_plddts else 0.0
    metrics["mean_min_distance"] = np.mean(all_min_dists) if all_min_dists else 999.0
    return metrics


def parse_input_yaml(yaml_path):
    """Parse the cofolding input YAML to determine strategy and chain layout."""
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return {"strategy": "unknown", "n_chains": 0}

    chains = data.get("sequences", [])
    has_cofactor = False
    has_metal = False
    cofactor_ccd = None
    metal_ccds = []
    substrate_smiles = []
    protein_length = 0

    for entry in chains:
        if "protein" in entry:
            protein_length = len(entry["protein"].get("sequence", ""))
        elif "ligand" in entry:
            lig = entry["ligand"]
            if "ccd" in lig:
                ccd = lig["ccd"]
                if ccd in ("MG", "MN", "ZN", "CA", "FE", "CO", "NI", "CU"):
                    has_metal = True
                    metal_ccds.append(ccd)
                else:
                    has_cofactor = True
                    cofactor_ccd = ccd
            elif "smiles" in lig:
                substrate_smiles.append(lig["smiles"])

    has_constraints = "constraints" in data and len(data["constraints"]) > 0
    strategy = "cofactor_guided" if has_cofactor else ("pocket_constrained" if has_constraints else "blind")

    return {
        "strategy": strategy,
        "n_chains": len(chains),
        "has_cofactor": has_cofactor,
        "has_metal": has_metal,
        "cofactor_ccd": cofactor_ccd,
        "metal_ccds": metal_ccds,
        "n_substrates": len(substrate_smiles),
        "protein_length": protein_length,
        "has_constraints": has_constraints,
    }


def load_protein_names(project_dir):
    """Parse all_sequences.fasta to build a mapping from UniProt ID to gene/protein name."""
    fasta_path = os.path.join(project_dir, "all_sequences.fasta")
    if not os.path.exists(fasta_path): return {}

    names = {}
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                parts = header.split()
                if not parts: continue
                uid = parts[0]
                
                # 1. Prioritize GN=... (gene name)
                gn_match = re.search(r"GN=([^\s]+)", header)
                if gn_match:
                    names[uid] = gn_match.group(1)
                    continue
                
                # 2. Try second field if it's a short symbol
                if len(parts) > 1 and not parts[1].startswith("sp|") and not parts[1].startswith("tr|") and len(parts[1]) < 10:
                    names[uid] = parts[1]
                    continue

                # 3. Fallback to descriptive name
                name_part = header[len(uid):].strip() if header.startswith(uid) else header
                name_part = re.sub(r"(sp|tr)\|[^|]+\|[^ ]+ ", "", name_part)
                if " OS=" in name_part: name_part = name_part.split(" OS=")[0]
                names[uid] = name_part.strip()
    return names


def analyze_candidate(uid, project_dir, project_name, enzyme_name, pocket_data, cat_data, cofold_dir="cofold_output"):
    """Process a single candidate UniProt ID."""
    uid_dir = os.path.join(project_dir, cofold_dir, uid)
    pred_dir = os.path.join(uid_dir, f"boltz_results_{uid}", "predictions", uid)
    conf_json = os.path.join(pred_dir, f"confidence_{uid}_model_0.json")
    pdb_file = os.path.join(pred_dir, f"{uid}_model_0.pdb")
    plddt_npz = os.path.join(pred_dir, f"plddt_{uid}_model_0.npz")

    if not os.path.exists(conf_json) or not os.path.exists(pdb_file):
        return None

    conf_metrics = parse_confidence_json(conf_json)
    yaml_path = os.path.join(project_dir, "cofold_input", f"{uid}.yaml")
    yaml_info = parse_input_yaml(yaml_path) if os.path.exists(yaml_path) else {"strategy": "unknown"}

    # Pocket Source Logic
    pocket_source = "none"
    if yaml_info.get("has_cofactor"):
        pocket_source = "alphafill_cofactor"
    else:
        uid_pocket = pocket_data.get(uid, {})
        if uid_pocket:
            ev = uid_pocket.get("evidence", {})
            pocket_source = ev.get("type", "p2rank")
        else:
            pocket_source = "p2rank" if yaml_info.get("has_constraints") else "none"

    # Catalytic residue counts (for reference; recall scores removed as they
    # are circular — pocket was selected to overlap these same residues)
    cat_entry = cat_data.get(uid, {})
    n_core = cat_entry.get("n_core", 0)
    n_shell = cat_entry.get("n_shell", 0)

    pdb_data = parse_pdb_structure(pdb_file)
    binding_metrics = compute_binding_metrics(pdb_data)
    sasa_burial = compute_sasa_burial(pdb_file)

    # Chain classification
    all_chains = sorted(binding_metrics.get("per_chain", {}).keys())
    n_special = (1 if yaml_info.get("has_cofactor") else 0) + len(yaml_info.get("metal_ccds", []))
    
    cof_ids = all_chains[:n_special]
    sub_ids = all_chains[n_special:]
    
    substrate_chains = {c: binding_metrics["per_chain"][c] for c in sub_ids if c in binding_metrics["per_chain"]}
    cofactor_chains = {c: binding_metrics["per_chain"][c] for c in cof_ids if c in binding_metrics["per_chain"]}

    # Substrate metrics
    sub_burial = sub_contact_res = sub_plddt = sub_contact_plddt = 0.0
    sub_min_dist = 999.0
    sub_charged = sub_hydrophobic = sub_polar = 0.0
    if substrate_chains:
        sub_burials = [sasa_burial[c]["sasa_burial"] for c in substrate_chains if c in sasa_burial]
        sub_burial = np.mean(sub_burials) if sub_burials else 0.0
        sub_contact_res = sum(c["n_contact_residues"] for c in substrate_chains.values())
        sub_plddt = np.mean([c["mean_ligand_plddt"] for c in substrate_chains.values()])
        sub_contact_plddt = np.mean([c["mean_contact_plddt"] for c in substrate_chains.values()])
        sub_min_dist = min(c["min_distance"] for c in substrate_chains.values())
        sub_charged = np.mean([c["frac_charged"] for c in substrate_chains.values()])
        sub_hydrophobic = np.mean([c["frac_hydrophobic"] for c in substrate_chains.values()])
        sub_polar = np.mean([c["frac_polar"] for c in substrate_chains.values()])

    # Cofactor metrics
    cof_burial = np.mean([sasa_burial[c]["sasa_burial"] for c in cofactor_chains if c in sasa_burial]) if cofactor_chains else None
    cof_plddt = np.mean([c["mean_ligand_plddt"] for c in cofactor_chains.values()]) if cofactor_chains else None

    # Per-interface iPTM: split cofactor vs substrate contributions
    # PDB ligand chains (B, C, D, E...) map to confidence JSON chains (1, 2, 3, 4...)
    pair_iptm = conf_metrics.get("pair_chains_iptm", {})
    sub_iptm_to_prot = sub_iptm_from_prot = cof_iptm_to_prot = None
    if pair_iptm and all_chains:
        chain_to_conf = {ch: str(i + 1) for i, ch in enumerate(all_chains)}
        # Substrate -> protein iPTM
        sub_to_prot = [pair_iptm.get(chain_to_conf[c], {}).get("0", None)
                       for c in sub_ids if c in chain_to_conf]
        sub_to_prot = [v for v in sub_to_prot if v is not None]
        sub_iptm_to_prot = np.mean(sub_to_prot) if sub_to_prot else None
        # Protein -> substrate iPTM
        sub_from_prot = [pair_iptm.get("0", {}).get(chain_to_conf[c], None)
                         for c in sub_ids if c in chain_to_conf]
        sub_from_prot = [v for v in sub_from_prot if v is not None]
        sub_iptm_from_prot = np.mean(sub_from_prot) if sub_from_prot else None
        # Cofactor -> protein iPTM
        cof_to_prot = [pair_iptm.get(chain_to_conf[c], {}).get("0", None)
                       for c in cof_ids if c in chain_to_conf]
        cof_to_prot = [v for v in cof_to_prot if v is not None]
        cof_iptm_to_prot = np.mean(cof_to_prot) if cof_to_prot else None

    return {
        "project": project_name, "uid": uid, "enzyme_name": enzyme_name,
        "protein_length": yaml_info.get("protein_length", pdb_data["protein_residue_count"]),
        "strategy": yaml_info.get("strategy"), "pocket_source": pocket_source,
        "n_core": n_core, "n_shell": n_shell,
        "confidence_score": conf_metrics["confidence_score"], "iptm": conf_metrics["iptm"], "complex_plddt": conf_metrics["complex_plddt"],
        "substrate_sasa_burial": sub_burial, "substrate_n_contact_residues": sub_contact_res,
        "substrate_mean_plddt": sub_plddt, "substrate_contact_plddt": sub_contact_plddt, "substrate_min_distance": sub_min_dist,
        "pocket_frac_charged": sub_charged, "pocket_frac_hydrophobic": sub_hydrophobic, "pocket_frac_polar": sub_polar,
        "cofactor_sasa_burial": cof_burial, "cofactor_mean_plddt": cof_plddt, "mean_protein_plddt": binding_metrics.get("mean_protein_plddt", 0),
        "cofactor_ccd": yaml_info.get("cofactor_ccd", ""),
        "ptm": conf_metrics["ptm"], "ligand_iptm": conf_metrics["ligand_iptm"],
        "complex_iplddt": conf_metrics["complex_iplddt"], "complex_pde": conf_metrics["complex_pde"], "complex_ipde": conf_metrics["complex_ipde"],
        "substrate_iptm_to_protein": sub_iptm_to_prot, "substrate_iptm_from_protein": sub_iptm_from_prot,
        "cofactor_iptm_to_protein": cof_iptm_to_prot,
    }


def compute_statistics(results, project_name):
    """Compute summary statistics for a project's results."""
    data = [r for r in results if r["project"] == project_name]
    if not data: return {}

    def safe_val(func, vals):
        v = [x for x in vals if x is not None]
        return func(v) if v else None

    return {
        "n_candidates": len(data),
        "confidence_mean": safe_val(np.mean, [d["confidence_score"] for d in data]),
        "confidence_median": safe_val(np.median, [d["confidence_score"] for d in data]),
        "iptm_mean": safe_val(np.mean, [d["iptm"] for d in data]),
        "substrate_iptm_to_protein_mean": safe_val(np.mean, [d["substrate_iptm_to_protein"] for d in data]),
        "sasa_burial_mean": safe_val(np.mean, [d["substrate_sasa_burial"] for d in data]),
        "n_high_confidence": sum(1 for d in data if d["confidence_score"] and d["confidence_score"] > 0.65),
        "n_high_iptm": sum(1 for d in data if d["iptm"] and d["iptm"] > 0.6),
    }


def generate_plots(all_results, output_dir):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError: return

    syn_data = [r for r in all_results if r["project"] == "synthetase"]
    hyd_data = [r for r in all_results if r["project"] == "hydrolase"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, key, title in [(axes[0], "confidence_score", "Confidence Distribution"), (axes[1], "iptm", "iPTM Distribution")]:
        ax.hist([d[key] for d in syn_data if d[key]], bins=15, alpha=0.6, label="Synthetase", color="#2196F3")
        ax.hist([d[key] for d in hyd_data if d[key]], bins=15, alpha=0.6, label="Hydrolase", color="#FF5722")
        ax.set_title(title); ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150)
    plt.close()


def write_summary(all_results, syn_stats, hyd_stats, output_dir):
    """Write text summary report."""
    out_path = os.path.join(output_dir, "cofolding_summary.txt")
    lines = ["="*80, "SYSTEMATIC COFOLDING ANALYSIS: SYNTHETASE vs HYDROLASE", "="*80, ""]
    
    header = f"{'Metric':<40} {'Synthetase':>12} {'Hydrolase':>12}"
    lines.append(header); lines.append("-" * 60)
    for label, key, fmt in [("N candidates", "n_candidates", "d"), ("Mean confidence", "confidence_mean", ".3f"),
                            ("Mean iPTM", "iptm_mean", ".3f"), ("Mean substrate iPTM→protein", "substrate_iptm_to_protein_mean", ".3f"),
                            ("Mean SASA burial", "sasa_burial_mean", ".3f")]:
        sv, hv = syn_stats.get(key), hyd_stats.get(key)
        lines.append(f"{label:<40} {f'{sv:{fmt}}' if sv is not None else 'N/A':>12} {f'{hv:{fmt}}' if hv is not None else 'N/A':>12}")

    for proj_name, label in [("synthetase", "SYNTHETASE"), ("hydrolase", "HYDROLASE")]:
        lines.append(f"\n{label} CANDIDATES (All)")
        lines.append("-" * 120)
        lines.append(f"{'Rank':<5} {'UID':<15} {'Name':<30} {'Conf':>6} {'iPTM':>6} {'Sub→P':>6} {'SASA%':>6} {'Strategy':<20} {'Pocket Source':<22}")
        proj_data = sorted([r for r in all_results if r["project"] == proj_name], key=lambda x: x["confidence_score"] or 0, reverse=True)
        for i, d in enumerate(proj_data):
            name = (d["enzyme_name"][:27] + "...") if len(d["enzyme_name"]) > 30 else d["enzyme_name"]
            sub_iptm_str = f"{d['substrate_iptm_to_protein']:>6.3f}" if d.get("substrate_iptm_to_protein") is not None else "   N/A"
            lines.append(f"{i+1:<5} {d['uid']:<15} {name:<30} {d['confidence_score']:>6.3f} {d['iptm']:>6.3f} {sub_iptm_str} {d['substrate_sasa_burial']*100:>5.1f}% {d['strategy']:<20} {d['pocket_source']:<22}")

    with open(out_path, "w") as f: f.write("\n".join(lines))
    print(f"  Saved full list to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "analysis_results"))
    parser.add_argument("--cofold-dir", default="cofold_output",
                        help="Name of cofolding output directory within each project (default: cofold_output)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for p_dir, p_name in load_projects():
        cofold_path = os.path.join(p_dir, args.cofold_dir)
        if not os.path.isdir(cofold_path):
            print(f"Skipping {p_name}: {args.cofold_dir} not found in {p_dir}")
            continue
        print(f"Analyzing {p_name} using {args.workers} workers...")
        p_names = load_protein_names(p_dir)
        uids = sorted(os.listdir(cofold_path))
        
        # Load project-level data
        pocket_data = {}
        try:
            with open(os.path.join(p_dir, "pocket_analysis.json")) as f:
                pocket_data = json.load(f)
        except FileNotFoundError:
            print(f"  Warning: pocket_analysis.json not found in {p_dir}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: failed to load pocket_analysis.json: {e}")

        cat_data = {}
        try:
            with open(os.path.join(p_dir, "catalytic_site_validation.json")) as f:
                cat_data = json.load(f)
        except FileNotFoundError:
            print(f"  Warning: catalytic_site_validation.json not found in {p_dir}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: failed to load catalytic_site_validation.json: {e}")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(analyze_candidate, uid, p_dir, p_name, p_names.get(uid, "unknown"), pocket_data, cat_data, args.cofold_dir) for uid in uids]
            for future in as_completed(futures):
                res = future.result()
                if res: all_results.append(res)
    
    if all_results:
        with open(os.path.join(args.output_dir, "cofolding_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader(); writer.writerows(all_results)
        
        syn_stats = compute_statistics(all_results, "synthetase")
        hyd_stats = compute_statistics(all_results, "hydrolase")
        write_summary(all_results, syn_stats, hyd_stats, args.output_dir)
        generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
