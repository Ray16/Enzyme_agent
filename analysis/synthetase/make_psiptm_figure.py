#!/usr/bin/env python3
"""Counterpart to figures/synthetase_angle_proxy_ranking.png, but the SCORE is
enzyme<->diacid ps_iptm (cofactor ATP/Mg excluded) instead of the in-line angle.
Shows ps_iptm does NOT rank the actives to the top (unlike the angle proxy)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
AS = f"{ROOT}/exp_data/amide_synthetase"
FIG = f"{ROOT}/figures"
SCORE = "se_mean"     # pair_chains_iptm[diacid][enzyme] mean = repo's enzyme->substrate
                      # direction; the best-behaving of all ps_iptm variants tested
LABEL = "mean enzyme–diacid ps_iptm  (best variant; ATP/Mg excluded)"
RED, ORANGE, GREY = "#d62728", "#f39c12", "#c8cdd4"
FS = 18
plt.rcParams.update({"font.size": FS, "axes.titlesize": FS, "axes.titleweight": "bold",
    "axes.labelsize": FS, "xtick.labelsize": FS-3, "ytick.labelsize": FS-3,
    "legend.fontsize": FS-3, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 300})

df = pd.read_csv(f"{AS}/analysis/psiptm_enzyme_diacid.csv")
rho, pv = spearmanr(df[SCORE], np.log10(df["act"] + 1e-3))
au = roc_auc_score(df["is_active"], df[SCORE])

# tiers match the angle-proxy figure: active (>0.2), marginal (> SfaB baseline, <0.2)
_a = pd.read_csv(f"{AS}/assay_data/panel_activity.csv")
_sf = _a.loc[_a.id == "SfaB", "activity_N66L1_ugml_avg"]
SF = float(_sf.iloc[0]) if len(_sf) else 0.0489

d = df.sort_values(SCORE, ascending=False).reset_index(drop=True)
rank = {r.id: i + 1 for i, r in d.iterrows()}
above = df[df.act > SF].sort_values("act", ascending=False).id.tolist()   # active + marginal
strong = df[df.act > 0.2].id.tolist()
near = [e for e in above if e not in strong]

fig, ax = plt.subplots(figsize=(11, 8.5))
fig.suptitle("Enzyme–diacid ps_iptm does NOT rank the active enzymes to the top",
             fontweight="bold")
x = np.arange(len(d))
colors = [RED if i in strong else ORANGE if i in near else GREY for i in d.id]
ax.bar(x, d[SCORE], color=colors, edgecolor="k", linewidth=0.3, zorder=2)
lines = ["above SfaB baseline  →  ps_iptm rank"]
for i, e in enumerate(above, 1):
    lines.append(f"{i}. {e}  →  #{rank[e]}")
lines.append("")
lines.append(f"Spearman ρ = {rho:+.2f}  (p={pv:.2f}, n.s.)")
lines.append(f"AUROC = {au:.2f}   (best of all")
lines.append("  directions/aggregations; all ≤0.63)")
lines.append("(angle proxy: ρ=+0.30, AUROC 0.95,")
lines.append("  actives #1/#4/#7)")
ax.text(0.985, 0.985, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
        fontsize=FS - 4, color="#222", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fbfbfb", ec="#888", lw=1.2))
ax.legend(handles=[Patch(color=RED, label="active  (>0.2 µg/ml)"),
                   Patch(color=ORANGE, label="marginal  (> SfaB, < 0.2)"),
                   Patch(color=GREY, label="≤ SfaB")],
          loc="upper left", bbox_to_anchor=(0.01, 1.0),
          frameon=True, framealpha=0.95, edgecolor="#ccc")
ax.set_xlabel(f"rank by mean enzyme–diacid ps_iptm  ({len(d)} enzymes)")
ax.set_ylabel(LABEL)
ax.set_ylim(min(d[SCORE]) - 0.02, max(d[SCORE]) + 0.04)
ax.set_xlim(-1, len(d))
plt.tight_layout(rect=[0, 0, 1, 0.95])
import os; os.makedirs(FIG, exist_ok=True)
p = f"{FIG}/synthetase_psiptm_ranking.png"
plt.savefig(p, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
print(f"{SCORE}: rho={rho:+.3f} p={pv:.3f} AUROC={au:.3f}")
print("active ranks:", {a: rank[a] for a in ['AS9','AS17','AS23']})
