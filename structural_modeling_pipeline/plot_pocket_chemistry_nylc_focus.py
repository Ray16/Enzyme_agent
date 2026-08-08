#!/usr/bin/env python3
"""
Focused plot of hydrolase holo pocket chemistry showing NylC as an outlier.
Single panel, no data-point labels, enlarged text.

Usage:
    python plot_pocket_chemistry_nylc_focus.py
"""

import csv
import json
import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}

NYLC_HOLO = {"h": 42.1, "c": 5.3}

HYD_COLORS = {
    "Polyglutamate hydrolase": "#e74c3c", "Cyanophycinase": "#3498db",
    "Pseudomurein hydrolase": "#2ecc71", "No annotation": "#d5d8dc", "Other": "#aab7b8",
}


def load_pfam_families():
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
    return pfam_fam


def main():
    with open(os.path.join(BASE_DIR, "amide_hydrolase", "pocket_analysis.json")) as f:
        pa_hyd = json.load(f)

    csv_data = {}
    with open(os.path.join(BASE_DIR, "analysis_results", "comprehensive_enzyme_table.csv")) as f:
        for r in csv.DictReader(f):
            csv_data[r["uniprot_id"].strip()] = r

    pfam_fam = load_pfam_families()

    # ── Hydrolase HOLO ──
    hyd_holo = []
    cofold_dir = Path(BASE_DIR, "amide_hydrolase", "cofold_output")
    if not cofold_dir.exists():
        cofold_dir = Path(BASE_DIR, "amide_hydrolase", "cofold_output_no_msa")
    for uid_dir in sorted(cofold_dir.iterdir()):
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
        })

    # ── Single-panel plot ──
# ── Single-panel plot ──
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.annotate("NylC", (NYLC_HOLO["h"], NYLC_HOLO["c"]),
                textcoords="offset points", xytext=(-40, 12),
                fontsize=18, fontweight="bold", color="darkgreen",
                arrowprops=dict(arrowstyle="-", color="darkgreen", lw=1.2))

    # NylC first so it heads the legend
    ax.scatter(NYLC_HOLO["h"], NYLC_HOLO["c"], c="darkgreen", marker="D", s=120, zorder=5,
               edgecolors="black", linewidth=1.0, label="NylC")

    for fam in ["Pseudomurein hydrolase", "Cyanophycinase",
                "Polyglutamate hydrolase", "Other", "No annotation"]:
        pts = [p for p in hyd_holo if p["fam"] == fam]
        if pts:
            ax.scatter([p["h"] for p in pts], [p["c"] for p in pts],
                       c=HYD_COLORS[fam], alpha=0.8, s=90, edgecolors="white",
                       linewidth=0.6, label=fam)

    ax.set_xlabel("Hydrophobic fraction (%)", fontsize=18)
    ax.set_ylabel("Charged fraction (%)", fontsize=18)
    #ax.set_title(f"Hydrolases, holo pocket chemistry (n={len(hyd_holo)})",
    #             fontsize=20, fontweight="bold")
    ax.tick_params(labelsize=16)
    ax.set_xlim(-8, 65)
    ax.set_ylim(-3, 75)
    ax.legend(fontsize=14, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "analysis_results", "pocket_chemistry_nylc_focus.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"  hyd_holo={len(hyd_holo)}")


if __name__ == "__main__":
    main()
