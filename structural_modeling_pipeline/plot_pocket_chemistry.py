#!/usr/bin/env python3
"""
Plot binding pocket composition (hydrophobic vs charged fraction) for
synthetases and hydrolases, comparing apo and holo structures.

Outputs:
    analysis_results/pocket_chemistry_landscape.png

Usage:
    python plot_pocket_chemistry.py
"""

import csv
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "GLY"}

# Notable hydrolases to label (UniProt annotated + discussed in email)
NOTABLE_HYDROLASES = {"Q8KQN8", "Q77WJ4", "Q852V1", "Q7LYX0", "A0A919JC44"}

# NylC benchmark values
NYLC_APO = {"h": 40.5, "c": 8.1}
NYLC_HOLO = {"h": 42.1, "c": 5.3}

# Color schemes
SYN_COLORS = {
    "NIS synthetase": "#e74c3c",
    "ATP-grasp": "#3498db",
    "ANL superfamily": "#9b59b6",
    "Pantoate ligase": "#2ecc71",
    "Mur ligase": "#f39c12",
    "GS superfamily": "#1abc9c",
    "GSP synthetase": "#e67e22",
    "Cyanophycin synthetase": "#c0392b",
    "PylC family": "#34495e",
    "Other synthetase": "#95a5a6",
}
HYD_COLORS = {
    "Polyglutamate hydrolase": "#e74c3c", "Cyanophycinase": "#3498db",
    "Pseudomurein hydrolase": "#2ecc71", "No annotation": "#d5d8dc", "Other": "#aab7b8",
}

# adjustText parameters
AT_KWARGS = dict(
    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
    force_points=(1.5, 1.5), force_text=(1.2, 1.2),
    expand=(1.8, 1.8), min_arrow_len=5,
)


def load_pfam_families():
    """Load Pfam-based family assignments."""
    pfam_fam = {}
    path = os.path.join(BASE_DIR, "analysis_results", "enzyme_family_classification.csv")
    if not os.path.exists(path):
        return pfam_fam
    with open(path) as f:
        for r in csv.DictReader(f):
            uid = r["uniprot_id"]
            fam = r.get("assigned_family", "")
            if r["project"] == "hydrolase":
                if fam == "Gamma_PGA_hydro": pfam_fam[uid] = "Polyglutamate hydrolase"
                elif fam == "Peptidase_S51": pfam_fam[uid] = "Cyanophycinase"
                elif fam in ("Transglut_core", "Peptidase_C71", "Pseudomurein endoisopeptidase"):
                    pfam_fam[uid] = "Pseudomurein hydrolase"
                elif fam == "no_domain_annotation": pfam_fam[uid] = "No annotation"
                else: pfam_fam[uid] = "Other"
            elif r["project"] == "synthetase":
                if fam == "FhuF": pfam_fam[uid] = "NIS synthetase"
                elif "ATP" in fam or "grasp" in fam.lower(): pfam_fam[uid] = "ATP-grasp"
                elif fam.startswith("AMP-binding") or "CapK" in fam: pfam_fam[uid] = "ANL superfamily"
                elif fam == "Pantoate_ligase": pfam_fam[uid] = "Pantoate ligase"
                elif fam in ("Mur_ligase_C", "Mur_ligase_M"): pfam_fam[uid] = "Mur ligase"
                elif fam == "Gln_synth/guanido_kin_cat_dom": pfam_fam[uid] = "GS superfamily"
                elif fam in ("GSP_synth", "CHAP"): pfam_fam[uid] = "GSP synthetase"
                elif fam == "Cyanophycin_syn": pfam_fam[uid] = "Cyanophycin synthetase"
                elif fam == "PylC-like_N": pfam_fam[uid] = "PylC family"
                else: pfam_fam[uid] = "Other synthetase"
    return pfam_fam


def is_labelable_syn(name):
    """Filter for synthetase names worth labeling."""
    if not name:
        return False
    if name.startswith("UPI") or len(name) > 12:
        return False
    return True


def pocket_chemistry(lining_residues):
    """Compute hydrophobic/charged fractions from lining residues."""
    rn = [r[0] for r in lining_residues]
    n = len(rn)
    if n == 0:
        return None, None
    return (sum(1 for r in rn if r in HYDROPHOBIC) / n * 100,
            sum(1 for r in rn if r in CHARGED) / n * 100)


def main():
    # Load data
    with open(os.path.join(BASE_DIR, "amide_synthetase", "pocket_analysis.json")) as f:
        pa_syn = json.load(f)
    with open(os.path.join(BASE_DIR, "amide_hydrolase", "pocket_analysis.json")) as f:
        pa_hyd = json.load(f)

    csv_data = {}
    with open(os.path.join(BASE_DIR, "analysis_results", "comprehensive_enzyme_table.csv")) as f:
        for r in csv.DictReader(f):
            csv_data[r["uniprot_id"].strip()] = r

    pfam_fam = load_pfam_families()

    # ── Synthetase APO ──
    syn_apo = []
    for uid, entry in pa_syn.items():
        if not isinstance(entry, dict) or entry.get("role") == "reference":
            continue
        if uid not in csv_data:
            continue
        bp = entry.get("binding_pocket") or {}
        lining = bp.get("lining_residues", [])
        if not lining:
            continue
        h, c = pocket_chemistry(lining)
        if h is None:
            continue
        syn_apo.append({"uid": uid, "h": h, "c": c,
                         "fam": pfam_fam.get(uid, "Other synthetase"),
                         "name": csv_data[uid].get("enzyme_name", uid)})

    # ── Synthetase HOLO ──
    # Exclude UIDs where apo pocket detection failed (binding_pocket: null in
    # pocket_analysis.json) to keep apo/holo counts consistent.
    apo_failed = {uid for uid, entry in pa_syn.items()
                  if isinstance(entry, dict) and entry.get("role") != "reference"
                  and uid in csv_data
                  and not (entry.get("binding_pocket") or {}).get("lining_residues")}
    syn_holo = []
    for uid, r in csv_data.items():
        if r["project"] != "synthetase":
            continue
        if uid in apo_failed:
            continue
        h = r.get("holo_frac_hydrophobic", "").replace("%", "").strip()
        c = r.get("holo_frac_charged", "").replace("%", "").strip()
        if not h or not c:
            continue
        syn_holo.append({"uid": uid, "h": float(h), "c": float(c),
                          "fam": pfam_fam.get(uid, "Other synthetase"),
                          "name": r.get("enzyme_name", uid)})

    # ── Hydrolase APO (evidence-guided only) ──
    hyd_cofolded = set(
        d for d in os.listdir(os.path.join(BASE_DIR, "amide_hydrolase", "cofold_output"))
        if os.path.isdir(os.path.join(BASE_DIR, "amide_hydrolase", "cofold_output", d))
        and "autocleavage" not in d and "native" not in d and d != "Q1EPR4"
    )

    hyd_apo = []
    for uid in hyd_cofolded:
        entry = pa_hyd.get(uid, {})
        if not isinstance(entry, dict):
            continue
        if entry.get("evidence", {}).get("type", "none") == "none":
            continue
        bp = entry.get("binding_pocket") or {}
        lining = bp.get("lining_residues", [])
        if not lining:
            continue
        h, c = pocket_chemistry(lining)
        if h is None:
            continue
        hyd_apo.append({"uid": uid, "h": h, "c": c,
                         "fam": pfam_fam.get(uid, "Other"),
                         "name": csv_data.get(uid, {}).get("enzyme_name", uid),
                         "notable": uid in NOTABLE_HYDROLASES})

    # ── Hydrolase HOLO (all with evidence + detectable pocket) ──
    hyd_holo = []
    for uid_dir in sorted(Path(BASE_DIR, "amide_hydrolase", "cofold_output").iterdir()):
        if not uid_dir.is_dir() or "autocleavage" in uid_dir.name or "_native_" in uid_dir.name:
            continue
        uid = uid_dir.name
        if uid == "Q1EPR4":
            continue
        pe = pa_hyd.get(uid, {})
        if not isinstance(pe, dict):
            continue
        if pe.get("evidence", {}).get("type", "none") == "none":
            continue
        bp = pe.get("binding_pocket") or {}
        al = set(r[1] for r in bp.get("lining_residues", []))
        if not al:
            continue
        pdbs = sorted(uid_dir.glob("**/*_best_model.pdb"))
        if not pdbs:
            pdbs = sorted(uid_dir.glob("**/*_model_0.pdb"))
        if not pdbs:
            continue

        p3, s3 = [], []
        with open(pdbs[0]) as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                ch = line[21]
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                ri = int(line[22:26])
                rn_ = line[17:20].strip()
                if ch == "A":
                    p3.append((np.array([x, y, z]), ri, rn_))
                elif ch == "B":
                    s3.append(np.array([x, y, z]))
        if not s3:
            continue
        s3 = np.array(s3)

        cr, crn, seen = set(), [], set()
        for coord, ri, rn_ in p3:
            if ri in seen:
                continue
            if np.min(np.sqrt(np.sum((s3 - coord) ** 2, axis=1))) < 5.0:
                cr.add(ri)
                crn.append(rn_)
                seen.add(ri)
        if not crn:
            continue

        nn = len(crn)
        hyd_holo.append({
            "uid": uid,
            "h": sum(1 for r in crn if r in HYDROPHOBIC) / nn * 100,
            "c": sum(1 for r in crn if r in CHARGED) / nn * 100,
            "fam": pfam_fam.get(uid, "Other"),
            "name": csv_data.get(uid, {}).get("enzyme_name", uid),
            "notable": uid in NOTABLE_HYDROLASES,
        })

    # ── PLOT (2x2) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (A) Synthetases APO
    ax = axes[0][0]
    for fam in SYN_COLORS:
        pts = [p for p in syn_apo if p["fam"] == fam]
        if pts:
            ax.scatter([p["h"] for p in pts], [p["c"] for p in pts],
                       c=SYN_COLORS[fam], alpha=0.8, s=60, edgecolors="white",
                       linewidth=0.6, label=fam)
    texts = [ax.text(p["h"], p["c"], p["name"], fontsize=9, fontweight="bold", alpha=0.9)
             for p in syn_apo if is_labelable_syn(p["name"])]
    adjust_text(texts, ax=ax, **AT_KWARGS)
    ax.set_xlabel("Hydrophobic fraction (%)", fontsize=14)
    ax.set_ylabel("Charged fraction (%)", fontsize=14)
    ax.set_title(f"(A) Synthetases - apo (n={len(syn_apo)})", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.set_xlim(-2, 60); ax.set_ylim(-3, 55)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    # (B) Synthetases HOLO
    ax = axes[0][1]
    for fam in SYN_COLORS:
        pts = [p for p in syn_holo if p["fam"] == fam]
        if pts:
            ax.scatter([p["h"] for p in pts], [p["c"] for p in pts],
                       c=SYN_COLORS[fam], alpha=0.8, s=60, edgecolors="white",
                       linewidth=0.6, label=fam)
    texts = [ax.text(p["h"], p["c"], p["name"], fontsize=9, fontweight="bold", alpha=0.9)
             for p in syn_holo if is_labelable_syn(p["name"])]
    adjust_text(texts, ax=ax, **AT_KWARGS)
    ax.set_xlabel("Hydrophobic fraction (%)", fontsize=14)
    ax.set_ylabel("Charged fraction (%)", fontsize=14)
    ax.set_title(f"(B) Synthetases - holo (n={len(syn_holo)})", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.set_xlim(-2, 60); ax.set_ylim(-3, 55)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    # (C) Hydrolases APO
    ax = axes[1][0]
    for fam in ["No annotation", "Other", "Pseudomurein hydrolase", "Cyanophycinase",
                "Polyglutamate hydrolase"]:
        pts = [p for p in hyd_apo if p["fam"] == fam]
        if pts:
            ax.scatter([p["h"] for p in pts], [p["c"] for p in pts],
                       c=HYD_COLORS[fam], alpha=0.8, s=60, edgecolors="white",
                       linewidth=0.6, label=fam)
    ax.scatter(NYLC_APO["h"], NYLC_APO["c"], c="darkgreen", marker="*", s=320, zorder=5,
               edgecolors="black", linewidth=1.0, label="NylC (benchmark)")
    texts = [ax.text(NYLC_APO["h"], NYLC_APO["c"], "NylC",
                     fontsize=12, fontweight="bold", color="darkgreen")]
    adjust_text(texts, ax=ax, **AT_KWARGS)
    ax.set_xlabel("Hydrophobic fraction (%)", fontsize=14)
    ax.set_ylabel("Charged fraction (%)", fontsize=14)
    ax.set_title(f"(C) Hydrolases - apo (n={len(hyd_apo)})", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.set_xlim(-2, 60); ax.set_ylim(-3, 55)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    # (D) Hydrolases HOLO
    ax = axes[1][1]
    for fam in ["No annotation", "Other", "Pseudomurein hydrolase", "Cyanophycinase",
                "Polyglutamate hydrolase"]:
        pts = [p for p in hyd_holo if p["fam"] == fam]
        if pts:
            ax.scatter([p["h"] for p in pts], [p["c"] for p in pts],
                       c=HYD_COLORS[fam], alpha=0.8, s=60, edgecolors="white",
                       linewidth=0.6, label=fam)
    ax.scatter(NYLC_HOLO["h"], NYLC_HOLO["c"], c="darkgreen", marker="*", s=320, zorder=5,
               edgecolors="black", linewidth=1.0, label="NylC (autocleavage)")
    texts = [ax.text(NYLC_HOLO["h"], NYLC_HOLO["c"], "NylC",
                     fontsize=12, fontweight="bold", color="darkgreen")]
    adjust_text(texts, ax=ax, **AT_KWARGS)
    ax.set_xlabel("Hydrophobic fraction (%)", fontsize=14)
    ax.set_ylabel("Charged fraction (%)", fontsize=14)
    ax.set_title(f"(D) Hydrolases - holo (n={len(hyd_holo)})", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.set_xlim(-2, 60); ax.set_ylim(-3, 55)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "analysis_results", "pocket_chemistry_landscape.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"  syn_apo={len(syn_apo)}, syn_holo={len(syn_holo)}, "
          f"hyd_apo={len(hyd_apo)}, hyd_holo={len(hyd_holo)}")


if __name__ == "__main__":
    main()
