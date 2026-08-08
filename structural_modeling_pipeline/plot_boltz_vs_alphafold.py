#!/usr/bin/env python3
"""Plot AlphaFold vs Boltz-2 apo pocket chemistry comparison.

Creates a 2x2 figure:
  (A) Synthetases - AlphaFold apo
  (B) Synthetases - Boltz-2 apo
  (C) Hydrolases  - AlphaFold apo
  (D) Hydrolases  - Boltz-2 apo

Output:
    analysis_results/boltz_vs_alphafold_pocket_chemistry.png
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}

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
    "Polyglutamate hydrolase": "#e74c3c",
    "Cyanophycinase": "#3498db",
    "Pseudomurein hydrolase": "#2ecc71",
    "No annotation": "#d5d8dc",
    "Other": "#aab7b8",
}

AT_KWARGS = dict(
    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
    force_points=(1.5, 1.5), force_text=(1.2, 1.2),
    expand=(1.8, 1.8), min_arrow_len=5,
)


def load_pfam_families():
    pfam = {}
    path = os.path.join(BASE_DIR, "analysis_results", "enzyme_family_classification.csv")
    if not os.path.exists(path):
        return pfam
    with open(path) as f:
        for r in csv.DictReader(f):
            uid = r["uniprot_id"]
            fam = r.get("assigned_family", "")
            if r["project"] == "hydrolase":
                if fam == "Gamma_PGA_hydro": pfam[uid] = "Polyglutamate hydrolase"
                elif fam == "Peptidase_S51": pfam[uid] = "Cyanophycinase"
                elif fam in ("Transglut_core", "Peptidase_C71", "Pseudomurein endoisopeptidase"):
                    pfam[uid] = "Pseudomurein hydrolase"
                elif fam == "no_domain_annotation": pfam[uid] = "No annotation"
                else: pfam[uid] = "Other"
            elif r["project"] == "synthetase":
                if fam == "FhuF": pfam[uid] = "NIS synthetase"
                elif "ATP" in fam or "grasp" in fam.lower(): pfam[uid] = "ATP-grasp"
                elif fam.startswith("AMP-binding") or "CapK" in fam: pfam[uid] = "ANL superfamily"
                elif fam == "Pantoate_ligase": pfam[uid] = "Pantoate ligase"
                elif fam in ("Mur_ligase_C", "Mur_ligase_M"): pfam[uid] = "Mur ligase"
                elif fam == "Gln_synth/guanido_kin_cat_dom": pfam[uid] = "GS superfamily"
                elif fam in ("GSP_synth", "CHAP"): pfam[uid] = "GSP synthetase"
                elif fam == "Cyanophycin_syn": pfam[uid] = "Cyanophycin synthetase"
                elif fam == "PylC-like_N": pfam[uid] = "PylC family"
                else: pfam[uid] = "Other synthetase"
    return pfam


def pocket_chem(lining):
    names = [r[0] for r in lining]
    n = len(names)
    if n == 0:
        return None, None
    return (sum(1 for r in names if r in HYDROPHOBIC) / n * 100,
            sum(1 for r in names if r in CHARGED) / n * 100)


def is_labelable_syn(name):
    if not name:
        return False
    if name.startswith("UPI") or len(name) > 12:
        return False
    return True


def load_alphafold_apo(pa_path, csv_data, pfam, project):
    with open(pa_path) as f:
        pa = json.load(f)

    points = []
    for uid, entry in pa.items():
        if not isinstance(entry, dict) or entry.get("role") == "reference":
            continue
        if uid not in csv_data:
            continue
        bp = entry.get("binding_pocket") or {}
        lining = bp.get("lining_residues", [])
        if not lining:
            continue

        if project == "hydrolase":
            ev = entry.get("evidence", {})
            if ev.get("type", "none") == "none":
                continue

        h, c = pocket_chem(lining)
        if h is None:
            continue

        default_fam = "Other synthetase" if project == "synthetase" else "Other"
        points.append({
            "uid": uid, "h": h, "c": c,
            "fam": pfam.get(uid, default_fam),
            "name": csv_data.get(uid, {}).get("enzyme_name", uid),
        })
    return points


def load_boltz_apo(boltz_pa_path, csv_data, pfam, project):
    if not os.path.exists(boltz_pa_path):
        return []
    with open(boltz_pa_path) as f:
        data = json.load(f)

    points = []
    for uid, entry in data.items():
        if entry.get("project", "").startswith("hydrolase_reference"):
            continue
        if entry.get("project") != project:
            continue
        chem = entry.get("chemistry", {})
        h = chem.get("frac_hydrophobic")
        c = chem.get("frac_charged")
        if h is None or c is None:
            continue

        default_fam = "Other synthetase" if project == "synthetase" else "Other"
        points.append({
            "uid": uid, "h": h, "c": c,
            "fam": pfam.get(uid, default_fam),
            "name": csv_data.get(uid, {}).get("enzyme_name", uid),
        })
    return points


def get_nylc_boltz(boltz_pa_path):
    if not os.path.exists(boltz_pa_path):
        return None
    with open(boltz_pa_path) as f:
        data = json.load(f)
    entry = data.get("Q1EPR4")
    if not entry:
        return None
    chem = entry.get("chemistry", {})
    h = chem.get("frac_hydrophobic")
    c = chem.get("frac_charged")
    if h is None:
        return None
    return {"h": h, "c": c}


def _auto_lims(points, nylc=None, margin=5):
    hs = [p["h"] for p in points]
    cs = [p["c"] for p in points]
    if nylc:
        hs.append(nylc["h"])
        cs.append(nylc["c"])
    if not hs:
        return (-2, 60), (-3, 55)
    return (min(0, min(hs) - margin), max(hs) + margin), \
           (min(0, min(cs) - margin), max(cs) + margin)


def plot_synthetase_panel(ax, points, title, colors, label_filter):
    for fam in colors:
        pts = [p for p in points if p["fam"] == fam]
        if pts:
            ax.scatter([p["h"] for p in pts], [p["c"] for p in pts],
                       c=colors[fam], alpha=0.8, s=60, edgecolors="white",
                       linewidth=0.6, label=fam)
    texts = [ax.text(p["h"], p["c"], p["name"], fontsize=9, fontweight="bold", alpha=0.9)
             for p in points if label_filter(p["name"])]
    if texts:
        adjust_text(texts, ax=ax, **AT_KWARGS)
    ax.set_xlabel("Hydrophobic fraction (%)", fontsize=14)
    ax.set_ylabel("Charged fraction (%)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    xlim, ylim = _auto_lims(points)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)


def plot_hydrolase_panel(ax, points, title, colors, nylc, nylc_label):
    for fam in ["No annotation", "Other", "Pseudomurein hydrolase",
                "Cyanophycinase", "Polyglutamate hydrolase"]:
        pts = [p for p in points if p["fam"] == fam]
        if pts:
            ax.scatter([p["h"] for p in pts], [p["c"] for p in pts],
                       c=colors[fam], alpha=0.8, s=60, edgecolors="white",
                       linewidth=0.6, label=fam)
    if nylc:
        ax.scatter(nylc["h"], nylc["c"], c="darkgreen", marker="*", s=320,
                   zorder=5, edgecolors="black", linewidth=1.0, label=nylc_label)
        texts = [ax.text(nylc["h"], nylc["c"], "NylC",
                         fontsize=12, fontweight="bold", color="darkgreen")]
        adjust_text(texts, ax=ax, **AT_KWARGS)
    ax.set_xlabel("Hydrophobic fraction (%)", fontsize=14)
    ax.set_ylabel("Charged fraction (%)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    xlim, ylim = _auto_lims(points, nylc)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)


def main():
    pfam = load_pfam_families()

    csv_data = {}
    csv_path = os.path.join(BASE_DIR, "analysis_results", "comprehensive_enzyme_table.csv")
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            csv_data[r["uniprot_id"].strip()] = r

    boltz_pa = os.path.join(BASE_DIR, "analysis_results", "boltz_apo_pocket_analysis.json")

    # AlphaFold apo data
    syn_af = load_alphafold_apo(
        os.path.join(BASE_DIR, "amide_synthetase", "pocket_analysis.json"),
        csv_data, pfam, "synthetase")
    hyd_af = load_alphafold_apo(
        os.path.join(BASE_DIR, "amide_hydrolase", "pocket_analysis.json"),
        csv_data, pfam, "hydrolase")

    # Boltz-2 apo data
    syn_b2 = load_boltz_apo(boltz_pa, csv_data, pfam, "synthetase")
    hyd_b2 = load_boltz_apo(boltz_pa, csv_data, pfam, "hydrolase")

    # NylC
    nylc_af = {"h": 40.5, "c": 8.1}
    nylc_b2 = get_nylc_boltz(boltz_pa)

    print(f"AlphaFold:  syn_apo={len(syn_af)}, hyd_apo={len(hyd_af)}")
    print(f"Boltz-2:    syn_apo={len(syn_b2)}, hyd_apo={len(hyd_b2)}")
    print(f"NylC Boltz-2: {nylc_b2}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    plot_synthetase_panel(
        axes[0][0], syn_af,
        f"(A) Synthetases - AlphaFold apo (n={len(syn_af)})",
        SYN_COLORS, is_labelable_syn)

    plot_synthetase_panel(
        axes[0][1], syn_b2,
        f"(B) Synthetases - Boltz-2 apo (n={len(syn_b2)})",
        SYN_COLORS, is_labelable_syn)

    plot_hydrolase_panel(
        axes[1][0], hyd_af,
        f"(C) Hydrolases - AlphaFold apo (n={len(hyd_af)})",
        HYD_COLORS, nylc_af, "NylC (benchmark)")

    plot_hydrolase_panel(
        axes[1][1], hyd_b2,
        f"(D) Hydrolases - Boltz-2 apo (n={len(hyd_b2)})",
        HYD_COLORS, nylc_b2, "NylC (Boltz-2)")

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "analysis_results",
                            "boltz_vs_alphafold_pocket_chemistry.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
