#!/usr/bin/env python3
"""Do structural proxies surface the top nylon-HYDROLASE catalysts? (Boltz-2, 50-sample)

Hydrolase counterpart of make_feature_surfacing_figure.py. Same heatmap grammar
(features x surfacing-AUROC, sequential blue, white=random 0.5), but the message
is the CONTRAST: for this single-fold Ntn-hydrolase panel (Peptidase S58, the NylC
family) assayed on crystalline nylon POWDER, no catalytic-geometry proxy surfaces
activity. Directions are FIXED by mechanism (not picked from a near-zero rho), so
the cells cannot be inflated by post-hoc sign-flipping.

Columns show both experimental readouts: L2 = PA66 dimer release (38/96 active),
L1 = PA66 monomer release (15/96). Rows grouped by catalytic determinant, with the
model-confidence NULLs (pLDDT / ipTM) as a separate block at the bottom -- they are
the only features with correct-direction signal, which is the point.

Usage: python make_hydrolase_feature_surfacing_figure.py
"""
import os, csv, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS = 18
BASE = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/nylon_hydrolase"

# (ranking-CSV feature key, display label). None = blank separator between blocks.
# Block 1 = catalytic GEOMETRY (mechanistic); Block 2 = active-site / autocleavage
# CONFIDENCE (mechanistically motivated for these Ntn autoprocessing enzymes);
# Block 3 = GLOBAL confidence (foldability reference).
ROWS = [
    ("nac:joint_frac",       "Near-attack joint gate (dist & Bürgi–Dunitz)"),   # ii
    ("ti:rmsd_min",          "Tetrahedral-intermediate template RMSD"),         # ii
    ("oxh:inplane_frac",     "Oxyanion-hole in-plane donors"),                  # iii
    ("base:hb_frac",         "Nucleophile activation (α-amine⋯Oγ)"),            # iv
    ("preorg:og1_rmsf",      "Reactive-center positional RMSF"),                # vi
    ("esp:oxyanion_O_mean",  "Oxyanion electrostatic potential"),              # iii
    (None, None),
    ("as:shell_plddt",       "Active-site shell pLDDT"),
    ("conf:thr_plddt",       "Catalytic-Thr pLDDT"),
    ("conf:bchain_ptm",      "Nucleophile β-chain pTM"),
    ("conf:ab_iptm",         "Autocleavage-junction ipTM (A|B)"),
    (None, None),
    ("conf:complex_plddt",   "Complex pLDDT (global)"),
    ("conf:ligand_iptm",     "Ligand ipTM (global)"),
]
# (ranking column key, header)
METRICS = [("robust_meanAUROC", "mean"), ("top5_auroc", "top-5"), ("binary_auroc_sig", ">ctrl")]
LABELS = [("L2", "L2  (PA66 dimer)"), ("L1", "L1  (PA66 monomer)")]


def fnum(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return math.nan


def load(label):
    p = f"{BASE}/analysis/feature_surfacing_ranking_{label}.csv"
    return {r["feature"]: r for r in csv.DictReader(open(p))}


def main():
    data = {lab: load(lab) for lab, _ in LABELS}
    cols = [(lab, m) for lab, _ in LABELS for m, _ in METRICS]

    M, ylabels, disagree = [], [], []
    for key, lbl in ROWS:
        if key is None:
            M.append([math.nan] * len(cols)); ylabels.append(""); disagree.append([False] * len(cols))
            continue
        row, drow = [], []
        for lab, m in cols:
            r = data[lab].get(key, {})
            row.append(fnum(r.get(m)))
            drow.append(r.get("mech_disagree") == "*")
        M.append(row); ylabels.append(lbl); disagree.append(drow)
    M = np.array(M)

    fig, ax = plt.subplots(figsize=(13.5, 0.66 * len(ROWS) + 2.2))
    im = ax.imshow(M, cmap=plt.get_cmap("Blues").copy(), vmin=0.40, vmax=0.90, aspect="auto")

    for i in range(len(ROWS)):
        for j in range(len(cols)):
            v = M[i, j]
            if math.isnan(v):
                continue
            txt = f"{v:.2f}" + ("*" if disagree[i][j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=FS - 3,
                    color="white" if v >= 0.70 else "black")

    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ylabels, fontsize=FS - 2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([m for _, m in [(l, dict(METRICS)[mm]) for (l, mm) in cols]], fontsize=FS - 3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # group headers over the two label blocks
    for b, (lab, name) in enumerate(LABELS):
        x0 = b * len(METRICS); xc = x0 + (len(METRICS) - 1) / 2
        ax.text(xc, -0.9, name, ha="center", va="bottom", fontsize=FS - 1, fontweight="bold")
    ax.axvline(len(METRICS) - 0.5, color="white", lw=4)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_ylim(len(ROWS) - 0.5, -1.4)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("AUROC  (0.5 = random)", fontsize=FS - 4)
    cb.ax.tick_params(labelsize=FS - 5)

    fig.text(0.01, 0.005,
             "Directions fixed by mechanism; * = data correlation runs opposite to mechanism "
             "(sign-flip artifact). No catalytic-geometry proxy surfaces activity; the only "
             "correct-direction signal is active-site / autocleavage confidence — weak (ρ≈0.1–0.2) "
             "but partly independent of global foldability.", fontsize=FS - 7, color="#555")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = f"{BASE}/figures/hydrolase_feature_surfacing.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
