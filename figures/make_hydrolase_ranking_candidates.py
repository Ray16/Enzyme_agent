#!/usr/bin/env python3
"""Candidate ranking visualizations for the nylon-hydrolase panel (user to pick one).

All show the same question as the synthetase figure: how well does the best structural
proxy rank the enzymes vs measured activity? Primary readout = L2 (PA66 dimer); best
proxy = autocleavage-junction ipTM (conf:ab_iptm). Writes 5 styles to
exp_data/nylon_hydrolase/figures/ranking_candidates/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
CSV = f"{ROOT}/exp_data/nylon_hydrolase/analysis/hydrolase_proxies.csv"
OUT = f"{ROOT}/exp_data/nylon_hydrolase/figures/ranking_candidates"
os.makedirs(OUT, exist_ok=True)

RED, GREY, BROWN, HILITE = "#d62728", "#c8cdd4", "#8b5a2b", "#3060c0"
# individually highlighted members (override RED/GREY), each a distinct color
HIGHLIGHT = {"Nyl10": "#1f77b4", "Nyl12": "#2ca02c", "Nyl50": "#9467bd"}
FS = 17
plt.rcParams.update({"font.size": FS, "axes.titlesize": FS, "axes.titleweight": "bold",
                     "axes.labelsize": FS, "xtick.labelsize": FS - 2, "ytick.labelsize": FS - 3,
                     "legend.fontsize": FS - 3, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})

df = pd.read_csv(CSV)
for c in ["L1", "L2", "L1sig", "L2sig"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["L2"]).reset_index(drop=True)
PROXY, PNAME = "conf:ab_iptm", "autocleavage-junction ipTM"


def tier(iid, sig):
    if iid == "NylC":
        return BROWN
    return RED if sig == 1 else GREY


# ============================================================= 1. two-panel ranked bars
def fig_bars():
    a = df.sort_values("L2", ascending=False).reset_index(drop=True)
    b = df.sort_values(PROXY, ascending=False).reset_index(drop=True)
    n = len(a)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 15), gridspec_kw={"wspace": 0.26})

    floor = a.L2[a.L2 > 0].min()
    yA = np.arange(n)
    axA.barh(yA, a.L2.clip(lower=floor), color=[tier(i, s) for i, s in zip(a.id, a.L2sig)],
             edgecolor="k", linewidth=0.3, height=0.82, zorder=2)
    axA.set_xscale("log")
    top = a.head(10)
    axA.set_yticks(list(top.index) + ([n - 1] if "NylC" not in top.id.values else []))
    labs = list(top.id)
    ticks = list(top.index)
    if "NylC" not in top.id.values:
        ticks.append(int(a.index[a.id == "NylC"][0])); labs.append("NylC")
    axA.set_yticks(ticks); axA.set_yticklabels(labs)
    for lab, iid in zip(axA.get_yticklabels(), labs):
        r = a[a.id == iid].iloc[0]; lab.set_color(tier(iid, r.L2sig)); lab.set_fontweight("bold")
    axA.set_ylim(n - 0.5, -0.5)
    axA.set_xlabel("PA66 dimer release (µg/mL, log)")
    axA.set_ylabel("panel member (ranked by measured activity)")
    axA.set_title("(a) Measured activity", loc="left")

    yB = np.arange(n)
    axB.barh(yB, b[PROXY], color=[tier(i, s) for i, s in zip(b.id, b.L2sig)],
             edgecolor="k", linewidth=0.3, height=0.82, zorder=2)
    for i, (iid, s, v) in enumerate(zip(b.id, b.L2sig, b[PROXY])):
        if iid in set(a.head(8).id) or iid == "NylC":
            axB.text(v + 0.002, i, iid, va="center", ha="left", fontsize=FS - 6,
                     fontweight="bold", color=tier(iid, s))
    axB.set_ylim(n - 0.5, -0.5)
    axB.set_xlim(float(b[PROXY].min()) - 0.02, float(b[PROXY].max()) + 0.06)
    ticks = list(range(0, n, 10))
    axB.set_yticks(ticks); axB.set_yticklabels([str(t + 1) for t in ticks])
    axB.set_xlabel(f"{PNAME}")
    axB.set_ylabel("rank by structural proxy")
    axB.set_title("(b) Top structural proxy", loc="left")

    fig.legend(handles=[Patch(color=RED, label="active (significant dimer release)"),
                        Patch(color=BROWN, label="NylC (reference)"),
                        Patch(color=GREY, label="inactive")],
               loc="lower center", ncol=3, frameon=True, framealpha=0.95, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.07, top=0.96, left=0.10, right=0.98)
    p = f"{OUT}/cand1_ranked_bars.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ============================================================= 2. ROC curves
def fig_roc():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"wspace": 0.25})
    panels = [("L2sig", [("conf:ab_iptm", "autocleavage-junction ipTM", +1),
                         ("nac:joint_frac", "near-attack gate", +1),
                         ("conf:complex_plddt", "complex pLDDT (global)", +1)],
               "L2 · PA66 dimer"),
              ("L1sig", [("conf:thr_plddt", "catalytic-Thr pLDDT", +1),
                         ("conf:bchain_ptm", "β-chain pTM", +1),
                         ("nac:joint_frac", "near-attack gate", +1)],
               "L1 · PA66 monomer")]
    for ax, (sig, feats, title) in zip(axes, panels):
        y = df[sig].fillna(0).astype(int).values
        for col, name, s in feats:
            x = pd.to_numeric(df[col], errors="coerce").values * s
            m = ~np.isnan(x)
            fpr, tpr, _ = roc_curve(y[m], x[m]); au = roc_auc_score(y[m], x[m])
            ax.plot(fpr, tpr, lw=2.5, label=f"{name}  ({au:.2f})")
        ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1.5, label="random (0.50)")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title(title, loc="left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=FS - 5)
    p = f"{OUT}/cand2_roc.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ============================================================= 3. score distributions
def fig_dist():
    feats = [("conf:ab_iptm", "autocleavage\nipTM"), ("conf:complex_plddt", "complex\npLDDT"),
             ("nac:joint_frac", "near-attack\ngate"), ("oxh:inplane_frac", "oxyanion\nhole"),
             ("ti:rmsd_min", "TI-template\nRMSD"), ("esp:oxyanion_O_mean", "oxyanion\nESP")]
    fig, ax = plt.subplots(figsize=(14, 7))
    for j, (col, name) in enumerate(feats):
        x = pd.to_numeric(df[col], errors="coerce")
        xn = (x - x.min()) / (x.max() - x.min() + 1e-9)          # normalize 0..1 for shared axis
        act = df.L2sig == 1
        for grp, off, cpt in [(act, +0.16, RED), (~act, -0.16, GREY)]:
            vals = xn[grp].dropna().values
            parts = ax.violinplot([vals], positions=[j + off], widths=0.3, showmeans=True)
            for b in parts["bodies"]:
                b.set_facecolor(cpt); b.set_alpha(0.7); b.set_edgecolor("k")
            for k in ("cbars", "cmins", "cmaxes", "cmeans"):
                parts[k].set_edgecolor("#333")
    ax.set_xticks(range(len(feats))); ax.set_xticklabels([n for _, n in feats])
    ax.set_ylabel("normalized proxy score (0–1)")
    ax.set_title("Active vs inactive proxy-score distributions (L2) — overlap = weak separation", loc="left")
    ax.legend(handles=[Patch(color=RED, label="active"), Patch(color=GREY, label="inactive")],
              loc="upper right")
    p = f"{OUT}/cand3_distributions.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ============================================================= 4. rank-enrichment
def fig_enrich():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"wspace": 0.25})
    panels = [("L2sig", [("conf:ab_iptm", "autocleavage-junction ipTM"),
                         ("nac:joint_frac", "near-attack gate")], "L2 · PA66 dimer"),
              ("L1sig", [("conf:thr_plddt", "catalytic-Thr pLDDT"),
                         ("conf:bchain_ptm", "β-chain pTM")], "L1 · PA66 monomer")]
    for ax, (sig, feats, title) in zip(axes, panels):
        y = df[sig].fillna(0).astype(int).values
        n, npos = len(y), int(y.sum())
        for col, name in feats:
            order = np.argsort(-pd.to_numeric(df[col], errors="coerce").values)
            cum = np.cumsum(y[order])
            ax.plot(np.arange(1, n + 1), cum, lw=2.5, label=name)
        ax.plot([1, n], [0, npos], ls="--", color="#888", lw=1.5, label="random")
        ax.plot([1, npos, n], [0, npos, npos], ls=":", color="#2a2", lw=1.5, label="perfect")
        ax.set_xlabel("enzymes screened (by proxy rank)")
        ax.set_ylabel("cumulative actives recovered")
        ax.set_title(title, loc="left"); ax.legend(loc="lower right", fontsize=FS - 5)
    p = f"{OUT}/cand4_enrichment.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ============================================================= 5. waterfall
def fig_waterfall():
    b = df.sort_values(PROXY, ascending=False).reset_index(drop=True)
    floor = b.L2[b.L2 > 0].min()
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(b))
    ax.bar(x, b.L2.clip(lower=floor), color=[tier(i, s) for i, s in zip(b.id, b.L2sig)],
           edgecolor="none", width=0.9, zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"enzyme rank by {PNAME}  (1 = best proxy score)")
    ax.set_ylabel("PA66 dimer release (µg/mL, log)")
    ax.set_title("Waterfall: measured activity of enzymes ordered by the best proxy", loc="left")
    ax.set_xlim(-1, len(b))
    ax.legend(handles=[Patch(color=RED, label="active"), Patch(color=BROWN, label="NylC"),
                       Patch(color=GREY, label="inactive")], loc="upper right")
    p = f"{OUT}/cand5_waterfall.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (fig_bars, fig_roc, fig_dist, fig_enrich, fig_waterfall):
        print("wrote", f())
