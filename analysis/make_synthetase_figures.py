#!/usr/bin/env python3
"""The one figure for the amide-synthetase diversity panel: the validated proxy.

Proxy = structural_modeling/scripts/angle_gate_proxy.py. Distance <4A to the mechanism
phosphate (Pa for this ANL/AMP-binding panel) is a GATE (keep docked poses; >=5 required
or the enzyme is INCONCLUSIVE); the SCORE is the MEAN in-line attack angle among gated
poses. Distance never enters the score. Raw NAC (d AND angle) and best-single-pose angle
were shown to fail (rho ~0.09) and are not used.

Regenerate the proxy CSV first:
  python structural_modeling/scripts/angle_gate_proxy.py \
      exp_data/amide_synthetase/boltz2/step1/results/output_configs \
      --phosphate pa --out exp_data/amide_synthetase/analysis/angle_gate_step1_pa.csv
then:  python analysis/make_synthetase_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
AS = f"{ROOT}/exp_data/amide_synthetase"
PROXY_CSV = f"{AS}/analysis/angle_gate_step1_pa.csv"
FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)

ACTIVES = ["AS9", "AS17", "AS23"]          # the three clear hits (>0.2 ug/ml)
ACT_THR = 0.2
A_CUT = 150.0                              # in-line angle cut used by the proxy (deg)
RED, BLUE, GREEN, GREY = "#d62728", "#3060c0", "#2e8b57", "#9aa0a6"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
})


def load():
    act = pd.read_csv(f"{AS}/assay_data/panel_activity.csv")
    act = act[["id", "activity_N66L1_ugml_avg"]].rename(columns={"activity_N66L1_ugml_avg": "act"})
    prox = pd.read_csv(PROXY_CSV)
    df = act.merge(prox, on="id", how="inner")
    df["is_active"] = (df["act"] > ACT_THR).astype(int)
    df["concl"] = df["docking_ok"].astype(bool)
    return df


def boot_auc_ci(y, s, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 101)
    curves, aucs = [], []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        if len(np.unique(y[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[idx], s[idx])
        curves.append(np.interp(grid, fpr, tpr)); aucs.append(auc(fpr, tpr))
    C = np.vstack(curves)
    return grid, np.percentile(C, 2.5, 0), np.percentile(C, 97.5, 0), np.percentile(aucs, [2.5, 97.5])


def fig_proxy_ranking(df):
    """Validated proxy: ranked bar of mean in-line angle + ROC (conclusive enzymes)."""
    c = df[df.concl].copy()
    d = c.sort_values("mean_inline_angle", ascending=False).reset_index(drop=True)
    y = c.is_active.values
    s = c.mean_inline_angle.values
    rho, pval = spearmanr(c.mean_inline_angle, np.log10(c.act + 1e-3))
    fpr, tpr, _ = roc_curve(y, s); a = auc(fpr, tpr)
    grid, lo, hi, aci = boot_auc_ci(y, s)
    ranks = {r.id: i + 1 for i, r in d.iterrows() if r.is_active}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Validated proxy — mean in-line attack angle (distance-gated poses): "
                 "ranks the hits to the top", fontsize=13, fontweight="bold")
    # A. ranked bar
    x = np.arange(len(d))
    colors = [RED if a_ else "#cfe0f5" for a_ in d.is_active]
    axA.bar(x, d.mean_inline_angle, color=colors, edgecolor="k", linewidth=0.3)
    axA.axhline(A_CUT, color=GREY, ls=":", lw=1)
    axA.text(len(d) - 0.5, A_CUT - 1.5, f"{A_CUT:.0f}° in-line cut", color="#666",
             ha="right", va="top", fontsize=8)
    callout = "actives by angle rank:\n" + "\n".join(f"  {k}  →  #{ranks[k]}" for k in ACTIVES)
    axA.text(0.97, 0.96, callout, transform=axA.transAxes, ha="right", va="top",
             fontsize=9.5, color=RED, fontweight="bold",
             bbox=dict(boxstyle="round", fc="#fff0f0", ec=RED, alpha=0.9))
    axA.set_xlabel(f"rank by mean in-line angle ({len(d)} conclusive enzymes)")
    axA.set_ylabel("mean in-line attack angle (°)")
    axA.set_title("A. All 3 hits in the top 7 (AS9 = #1)")
    axA.set_ylim(min(d.mean_inline_angle) - 4, 180)
    axA.legend(handles=[Patch(color=RED, label="active"), Patch(color="#cfe0f5", label="inactive")],
               loc="lower left", frameon=False)
    # B. ROC + CI
    axB.fill_between(grid, lo, hi, color=BLUE, alpha=0.15, label="95% CI (bootstrap)")
    axB.plot(fpr, tpr, color=BLUE, lw=2.2, label=f"ROC  AUROC={a:.2f}")
    axB.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="random")
    axB.set_xlabel("false positive rate"); axB.set_ylabel("true positive rate")
    axB.set_title(f"B. AUROC={a:.2f}  (95% CI [{aci[0]:.2f}, {aci[1]:.2f}], 3 positives)")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.legend(loc="lower right", frameon=False)
    axB.text(0.97, 0.42, f"Spearman(angle, log-act)\n= {rho:+.2f}  (p = {pval:.2f})",
             transform=axB.transAxes, ha="right", va="top", fontsize=9, color="#333",
             bbox=dict(boxstyle="round", fc="#eafaef", ec=GREEN))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = f"{FIG}/synthetase_angle_proxy_ranking.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    df = load()
    assert set(ACTIVES) <= set(df.id), "actives missing from merged data"
    assert df[df.is_active == 1].concl.all(), "an active is inconclusive — check phosphate/gate"
    print(f"n={len(df)}  conclusive={int(df.concl.sum())}  inconclusive={int((~df.concl).sum())} "
          f"({df[~df.concl].id.tolist()})")
    print("wrote:", fig_proxy_ranking(df))


if __name__ == "__main__":
    main()
