#!/usr/bin/env python3
"""
Analyze MD trajectories for binding stability validation.

Computes protein RMSD, per-ligand RMSD, contact persistence, Mg2+ coordination,
and hydrogen bonds. Classifies each ligand as stable/partially_stable/unstable.

Usage:
    python analyze_md.py <project_dir>
    python analyze_md.py <project_dir> --candidates P31663
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    import mdtraj
except ImportError:
    sys.exit("ERROR: mdtraj not installed. Run: conda install -c conda-forge mdtraj")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_trajectory(prmtop_path, dcd_path):
    """Load trajectory with MDTraj."""
    traj = mdtraj.load(dcd_path, top=prmtop_path)
    return traj


def identify_components(traj, project_dir, uid):
    """Map topology atoms to protein/ligand components using cofold YAML.

    Returns dict of component_name -> atom_indices (numpy array).
    """
    top = traj.topology
    components = {}

    # Protein: standard amino acid residues
    protein_indices = top.select("protein")
    if len(protein_indices) > 0:
        components["protein"] = protein_indices
        components["backbone"] = top.select("backbone")

    # Non-protein, non-water, non-ion residues are ligands
    # Build chain mapping from cofold YAML
    yaml_path = os.path.join(project_dir, "cofold_input", f"{uid}.yaml")
    chain_names = {}
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        for entry in data.get("sequences", []):
            if "ligand" in entry:
                info = entry["ligand"]
                chain_id = info["id"]
                if "ccd" in info:
                    chain_names[chain_id] = info["ccd"]
                elif "smiles" in info:
                    smiles = info["smiles"]
                    if "NH3" in smiles:
                        chain_names[chain_id] = "hexamethylenediamine"
                    elif "C(=O)" in smiles:
                        chain_names[chain_id] = "adipic_acid"
                    else:
                        chain_names[chain_id] = f"ligand_{chain_id}"

    # Identify ligand residues by exclusion (not protein, water, or common ions)
    standard_res = set()
    for r in top.residues:
        if r.is_protein or r.is_water or r.name in {"Na+", "Cl-", "K+"}:
            standard_res.add(r.index)

    # Group remaining residues as ligands
    # AMBER topology preserves order: protein chains first, then ligands in order
    ligand_residues = [r for r in top.residues
                       if r.index not in standard_res]

    # Map ligand residues to chain names based on order
    # Chain order in YAML: B, C, D, E -> ligand residues in topology order
    chain_ids = sorted(chain_names.keys())
    for i, chain_id in enumerate(chain_ids):
        if i < len(ligand_residues):
            res = ligand_residues[i]
            indices = np.array([a.index for a in res.atoms])
            name = chain_names[chain_id]
            key = f"{chain_id}_{name}"
            components[key] = indices

    return components


def compute_protein_rmsd(traj, backbone_indices):
    """Backbone RMSD vs frame 0 after alignment (in Angstroms)."""
    if backbone_indices is None or len(backbone_indices) == 0:
        return None
    rmsd = mdtraj.rmsd(traj, traj, frame=0, atom_indices=backbone_indices)
    return rmsd * 10.0  # nm -> Angstrom


def compute_ligand_rmsd(traj, ligand_indices, backbone_indices):
    """Ligand RMSD after aligning on protein backbone (in Angstroms)."""
    if ligand_indices is None or len(ligand_indices) == 0:
        return None
    if backbone_indices is None or len(backbone_indices) == 0:
        return None

    # Align on backbone
    traj_aligned = traj.superpose(traj, frame=0, atom_indices=backbone_indices)

    # Compute ligand RMSD on aligned trajectory
    rmsd = mdtraj.rmsd(traj_aligned, traj_aligned, frame=0,
                       atom_indices=ligand_indices)
    return rmsd * 10.0  # nm -> Angstrom


def compute_min_distance(traj, protein_indices, ligand_indices):
    """Minimum protein-ligand distance over trajectory (in Angstroms).

    Uses a sampled approach for efficiency: picks up to 50 protein atoms
    closest to the ligand in frame 0.
    """
    if protein_indices is None or ligand_indices is None:
        return None
    if len(protein_indices) == 0 or len(ligand_indices) == 0:
        return None

    # In frame 0, find protein atoms closest to any ligand atom
    pos0 = traj.xyz[0]
    prot_pos = pos0[protein_indices]
    lig_pos = pos0[ligand_indices]

    # Compute distance matrix (protein x ligand) for frame 0
    # Use broadcasting
    diff = prot_pos[:, np.newaxis, :] - lig_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_per_prot = dists.min(axis=1)

    # Select top 50 closest protein atoms
    n_select = min(50, len(protein_indices))
    closest_prot = protein_indices[np.argsort(min_per_prot)[:n_select]]

    # Build atom pairs
    pairs = []
    for p in closest_prot:
        for l in ligand_indices:
            pairs.append((p, l))
    pairs = np.array(pairs)

    # Compute distances over trajectory
    distances = mdtraj.compute_distances(traj, pairs)  # (n_frames, n_pairs)
    min_dist = distances.min(axis=1)  # (n_frames,)

    return min_dist * 10.0  # nm -> Angstrom


def compute_contact_persistence(traj, protein_indices, ligand_indices,
                                cutoff_nm=0.4):
    """Fraction of initial contacts maintained over trajectory."""
    if protein_indices is None or ligand_indices is None:
        return None
    if len(protein_indices) == 0 or len(ligand_indices) == 0:
        return None

    # Build all protein-ligand pairs
    pairs = []
    for p in protein_indices:
        for l in ligand_indices:
            pairs.append((p, l))
    pairs = np.array(pairs)

    # Limit to reasonable number of pairs
    if len(pairs) > 50000:
        # Sample protein atoms near ligand in frame 0
        pos0 = traj.xyz[0]
        prot_pos = pos0[protein_indices]
        lig_center = pos0[ligand_indices].mean(axis=0)
        dists = np.sqrt(np.sum((prot_pos - lig_center)**2, axis=1))
        nearby = protein_indices[dists < 1.5]  # within 15 A
        if len(nearby) == 0:
            nearby = protein_indices[np.argsort(dists)[:100]]
        pairs = []
        for p in nearby:
            for l in ligand_indices:
                pairs.append((p, l))
        pairs = np.array(pairs)

    distances = mdtraj.compute_distances(traj, pairs)

    # Initial contacts (frame 0)
    initial_contacts = distances[0] < cutoff_nm
    n_initial = initial_contacts.sum()

    if n_initial == 0:
        return np.zeros(len(traj))

    # Persistence per frame
    maintained = (distances[:, initial_contacts] < cutoff_nm).sum(axis=1)
    persistence = maintained / n_initial

    return persistence


def compute_mg_coordination(traj, mg_indices):
    """Coordination number for Mg2+ (atoms within 2.5 A)."""
    if mg_indices is None or len(mg_indices) == 0:
        return None

    mg_idx = mg_indices[0]  # Assume single Mg2+
    all_atoms = np.arange(traj.n_atoms)
    others = all_atoms[all_atoms != mg_idx]

    # Build pairs (mg, every other atom)
    pairs = np.column_stack([np.full(len(others), mg_idx), others])

    distances = mdtraj.compute_distances(traj, pairs)
    coordination = (distances < 0.25).sum(axis=1)  # 2.5 A = 0.25 nm

    return coordination.astype(float)


def compute_hbonds_summary(traj, protein_indices, ligand_indices):
    """Count protein-ligand hydrogen bonds per frame."""
    if protein_indices is None or ligand_indices is None:
        return None, None
    if len(protein_indices) == 0 or len(ligand_indices) == 0:
        return None, None

    try:
        # Use baker_hubbard for H-bond detection
        # This finds all H-bonds across the trajectory
        hbonds = mdtraj.baker_hubbard(traj, freq=0.0)

        if len(hbonds) == 0:
            return np.zeros(len(traj)), []

        prot_set = set(protein_indices.tolist())
        lig_set = set(ligand_indices.tolist())

        # Filter for protein-ligand H-bonds
        pl_hbonds = []
        for d, h, a in hbonds:
            d_in_prot = d in prot_set
            a_in_prot = a in prot_set
            d_in_lig = d in lig_set
            a_in_lig = a in lig_set
            if (d_in_prot and a_in_lig) or (d_in_lig and a_in_prot):
                pl_hbonds.append((d, h, a))

        if not pl_hbonds:
            return np.zeros(len(traj)), []

        # Count per frame using distance criterion
        n_hbonds = np.zeros(len(traj))
        for d, h, a in pl_hbonds:
            pairs = np.array([[d, a]])
            dists = mdtraj.compute_distances(traj, pairs).flatten()
            n_hbonds += (dists < 0.35).astype(float)  # 3.5 A

        # Format persistent hbonds
        top = traj.topology
        persistent = []
        for d, h, a in pl_hbonds:
            donor = str(top.atom(d))
            acceptor = str(top.atom(a))
            persistent.append(f"{donor}--{acceptor}")

        return n_hbonds, persistent

    except Exception as e:
        print(f"    Warning: H-bond analysis failed: {e}")
        return np.zeros(len(traj)), []


def classify_stability(rmsd_angstrom, contact_persistence):
    """Classify binding stability for a ligand."""
    if rmsd_angstrom is None:
        return "unknown"

    mean_rmsd = float(np.mean(rmsd_angstrom))
    mean_contacts = float(np.mean(contact_persistence)) if contact_persistence is not None else 0.0

    if mean_rmsd < 3.0 and mean_contacts > 0.50:
        return "stable"
    elif mean_rmsd < 5.0 or mean_contacts > 0.30:
        return "partially_stable"
    else:
        return "unstable"


def generate_rmsd_plot(uid, components, rmsd_data, time_ns, output_dir):
    """Generate RMSD time series plot."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(len(rmsd_data), 1, figsize=(10, 3*len(rmsd_data)),
                              sharex=True)
    if len(rmsd_data) == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for i, (name, rmsd) in enumerate(rmsd_data.items()):
        if rmsd is not None:
            axes[i].plot(time_ns, rmsd, color=colors[i % len(colors)],
                        linewidth=0.5, alpha=0.7)
            # Running average
            window = min(50, len(rmsd) // 5)
            if window > 1:
                avg = np.convolve(rmsd, np.ones(window)/window, mode="valid")
                t_avg = time_ns[window-1:]
                axes[i].plot(t_avg, avg, color=colors[i % len(colors)],
                            linewidth=2, label=f"mean: {np.mean(rmsd):.1f} A")
            axes[i].set_ylabel("RMSD (A)")
            axes[i].set_title(name)
            axes[i].legend(loc="upper right")
            axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ns)")
    fig.suptitle(f"{uid} — Binding Stability", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{uid}_rmsd_timeseries.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def analyze_one_uid(uid, project_dir, prep_dir, md_output_dir):
    """Full analysis for one UID."""
    print(f"\n  Analyzing {uid}...", flush=True)

    prmtop = os.path.join(project_dir, prep_dir, uid, "system.prmtop")
    dcd = os.path.join(md_output_dir, uid, "trajectory.dcd")

    if not os.path.exists(prmtop) or not os.path.exists(dcd):
        print(f"    Missing files for {uid}")
        return None

    # Load trajectory
    print(f"    Loading trajectory...", flush=True)
    traj = load_trajectory(prmtop, dcd)
    print(f"    {traj.n_frames} frames, {traj.n_atoms} atoms", flush=True)

    # Identify components
    components = identify_components(traj, project_dir, uid)
    backbone = components.get("backbone")

    # Time axis
    sim_info_path = os.path.join(md_output_dir, uid, "simulation_info.json")
    if os.path.exists(sim_info_path):
        with open(sim_info_path) as f:
            sim_info = json.load(f)
        sim_time = sim_info.get("sim_time_ns", traj.n_frames * 0.01)
    else:
        sim_time = traj.n_frames * 0.01  # assume 10 ps/frame

    time_ns = np.linspace(0, sim_time, traj.n_frames)

    # Compute metrics
    result = {
        "uid": uid,
        "simulation_time_ns": sim_time,
        "n_frames": traj.n_frames,
    }

    # Protein RMSD
    prot_rmsd = compute_protein_rmsd(traj, backbone)
    if prot_rmsd is not None:
        result["protein"] = {
            "backbone_rmsd_mean_angstrom": round(float(np.mean(prot_rmsd)), 2),
            "backbone_rmsd_std_angstrom": round(float(np.std(prot_rmsd)), 2),
            "backbone_rmsd_final_angstrom": round(float(prot_rmsd[-1]), 2),
        }
        print(f"    Protein backbone RMSD: {np.mean(prot_rmsd):.1f} +/- {np.std(prot_rmsd):.1f} A")

    # Per-ligand analysis
    protein_indices = components.get("protein")
    rmsd_data = {"Protein backbone": prot_rmsd}
    result["ligands"] = {}

    for comp_name, comp_indices in components.items():
        if comp_name in ("protein", "backbone"):
            continue

        print(f"    Analyzing {comp_name}...", flush=True)
        lig_result = {}

        # RMSD
        lig_rmsd = compute_ligand_rmsd(traj, comp_indices, backbone)
        if lig_rmsd is not None:
            lig_result["rmsd_mean_angstrom"] = round(float(np.mean(lig_rmsd)), 2)
            lig_result["rmsd_std_angstrom"] = round(float(np.std(lig_rmsd)), 2)
            rmsd_data[comp_name] = lig_rmsd

        # Min distance
        min_dist = compute_min_distance(traj, protein_indices, comp_indices)
        if min_dist is not None:
            lig_result["min_distance_mean_angstrom"] = round(float(np.mean(min_dist)), 2)

        # Contact persistence
        contacts = compute_contact_persistence(traj, protein_indices, comp_indices)
        if contacts is not None:
            lig_result["contact_persistence_mean"] = round(float(np.mean(contacts)), 3)

        # Mg coordination
        if "MG" in comp_name.upper():
            coord = compute_mg_coordination(traj, comp_indices)
            if coord is not None:
                lig_result["coordination_number_mean"] = round(float(np.mean(coord)), 1)

        # H-bonds
        n_hbonds, persistent = compute_hbonds_summary(traj, protein_indices, comp_indices)
        if n_hbonds is not None:
            lig_result["hbonds_mean"] = round(float(np.mean(n_hbonds)), 1)
            lig_result["persistent_hbonds"] = persistent[:10]  # top 10

        # Stability classification
        lig_result["stability"] = classify_stability(lig_rmsd, contacts)

        result["ligands"][comp_name] = lig_result
        print(f"      RMSD: {lig_result.get('rmsd_mean_angstrom', '?')} A, "
              f"Contacts: {lig_result.get('contact_persistence_mean', '?')}, "
              f"Stability: {lig_result.get('stability', '?')}")

    # Generate plot
    uid_output = os.path.join(md_output_dir, uid)
    plot_path = generate_rmsd_plot(uid, components, rmsd_data, time_ns,
                                   uid_output)
    if plot_path:
        result["plot"] = plot_path

    # Save per-UID analysis
    analysis_path = os.path.join(uid_output, "md_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def write_summary_csv(results, output_path):
    """Write aggregate metrics CSV."""
    if not results:
        return

    rows = []
    for r in results:
        row = {
            "uid": r["uid"],
            "sim_time_ns": r["simulation_time_ns"],
            "n_frames": r["n_frames"],
        }

        # Protein metrics
        prot = r.get("protein", {})
        row["protein_rmsd_mean"] = prot.get("backbone_rmsd_mean_angstrom")
        row["protein_rmsd_std"] = prot.get("backbone_rmsd_std_angstrom")

        # Per-ligand metrics
        for lig_name, lig_data in r.get("ligands", {}).items():
            prefix = lig_name.replace(" ", "_").lower()
            row[f"{prefix}_rmsd_mean"] = lig_data.get("rmsd_mean_angstrom")
            row[f"{prefix}_rmsd_std"] = lig_data.get("rmsd_std_angstrom")
            row[f"{prefix}_contacts"] = lig_data.get("contact_persistence_mean")
            row[f"{prefix}_stability"] = lig_data.get("stability")
            if "coordination_number_mean" in lig_data:
                row[f"{prefix}_coordination"] = lig_data["coordination_number_mean"]
            if "hbonds_mean" in lig_data:
                row[f"{prefix}_hbonds"] = lig_data["hbonds_mean"]

        rows.append(row)

    # Collect all field names
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MD trajectories for binding stability")
    parser.add_argument("project_dir",
                        help="Project directory")
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Specific UIDs to analyze")
    parser.add_argument("--prep-dir", default="md_prep",
                        help="Preparation directory (default: md_prep)")
    parser.add_argument("--md-dir", default="md_output",
                        help="MD output directory (default: md_output)")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    md_output_dir = os.path.join(project_dir, args.md_dir)

    print(f"\n{'='*60}")
    print(f"  Phase 6c: MD Trajectory Analysis")
    print(f"{'='*60}\n", flush=True)

    # Find completed simulations
    uids = []
    if os.path.isdir(md_output_dir):
        for entry in sorted(os.listdir(md_output_dir)):
            if args.candidates and entry not in args.candidates:
                continue
            traj = os.path.join(md_output_dir, entry, "trajectory.dcd")
            if os.path.exists(traj):
                uids.append(entry)

    if not uids:
        sys.exit("ERROR: No completed MD trajectories found")

    print(f"  Trajectories to analyze: {len(uids)} ({', '.join(uids)})\n")

    # Analyze each
    results = []
    for uid in uids:
        result = analyze_one_uid(uid, project_dir, args.prep_dir, md_output_dir)
        if result:
            results.append(result)

    # Write aggregate CSV
    csv_path = os.path.join(md_output_dir, "md_stability_metrics.csv")
    write_summary_csv(results, csv_path)
    print(f"\n  Metrics saved: {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  MD Analysis Summary")
    print(f"{'='*60}")

    for r in results:
        print(f"\n  {r['uid']}:")
        prot = r.get("protein", {})
        if prot:
            print(f"    Protein RMSD: {prot.get('backbone_rmsd_mean_angstrom', '?')} A")
        for lig_name, lig_data in r.get("ligands", {}).items():
            stability = lig_data.get("stability", "?")
            rmsd = lig_data.get("rmsd_mean_angstrom", "?")
            contacts = lig_data.get("contact_persistence_mean", "?")
            print(f"    {lig_name}: {stability} "
                  f"(RMSD={rmsd} A, contacts={contacts})")

    print()


if __name__ == "__main__":
    main()
