#!/usr/bin/env python3
"""How does the best proxy actually RANK the nylon-hydrolase panel vs measured activity?

Two panels (L2 = PA66 dimer, L1 = PA66 monomer). For each readout we show its best
proxy: enzymes are ordered left->right by the proxy score (rank 1 = best proxy value),
y = measured product release (log). Active enzymes (significant vs no-enzyme control)
are highlighted; the top-5 by activity are labeled. A perfect proxy would put all the
coloured (active) points at the far left; the spread here shows the ranking is weak.

Best proxy per readout (from feature_surfacing_ranking_*.csv, mechanism-fixed dir):
  L2 -> conf:ab_iptm   (autocleavage-junction A|B ipTM),  top-5 AUROC 0.81, rho 0.18
  L1 -> conf:thr_plddt (catalytic-Thr pLDDT),             top-5 AUROC 0.81, rho 0.12
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS = 18
BASE = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/nylon_hydrolase"
CSV = f"{BASE}/analysis/hydrolase_proxies.csv"
PANELS = [
    ("L2", "L2sig", "conf:ab_iptm", "Autocleavage-junction ipTM (A|B)",
     "L2  ·  PA66 dimer release", "top-5 AUROC 0.81   ρ = 0.18"),
    ("L1", "L1sig", "conf:thr_plddt", "Catalytic-Thr pLDDT",
     "L1  ·  PA66 monomer release", "top-5 AUROC 0.81   ρ = 0.12"),
]
ACT = "#c0392b"    # active
INA = "#b8c4d0"    # inactive


def main():
    df = pd.read_csv(CSV)
    for c in ["L1", "L2", "L1sig", "L2sig"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
    for ax, (val, sig, proxy, pname, title, stat) in zip(axes, PANELS):
        d = df.dropna(subset=[proxy, val]).copy()
        # rank 1 = best proxy score (higher ipTM/pLDDT = predicted more active)
        d = d.sort_values(proxy, ascending=False).reset_index(drop=True)
        rk = np.arange(1, len(d) + 1)
        act = (d[sig] == 1).to_numpy()
        y = d[val].clip(lower=d[val][d[val] > 0].min()).to_numpy()

        ax.scatter(rk[~act], y[~act], s=55, c=INA, edgecolor="none",
                   label="inactive", zorder=2)
        ax.scatter(rk[act], y[act], s=95, c=ACT, edgecolor="black",
                   linewidth=0.4, label="active (> control)", zorder=3)
        ax.set_yscale("log")

        # label the 5 most active enzymes where they land in the proxy ranking
        top5 = d.sort_values(val, ascending=False).head(5)
        for _, r in top5.iterrows():
            ax.annotate(r["id"], (r["rank"], r[val]), fontsize=FS - 7,
                        xytext=(4, 4), textcoords="offset points", color=ACT)

        ax.set_title(title, fontsize=FS + 1, fontweight="bold", pad=10)
        ax.set_xlabel(f"enzyme rank by {pname}  (1 = best)", fontsize=FS - 3)
        ax.set_ylabel("measured release  (µg/mL, log)", fontsize=FS - 3)
        ax.tick_params(labelsize=FS - 5)
        ax.text(0.97, 0.05, stat, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS - 5, color="#333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))
        ax.legend(fontsize=FS - 6, loc="upper right", framealpha=0.9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", ls=":", color="#ddd", zorder=0)

    fig.suptitle("Best proxy vs measured activity — actives are only weakly enriched at good ranks",
                 fontsize=FS, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f"{BASE}/figures/hydrolase_proxy_ranking.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
