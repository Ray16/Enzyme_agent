#!/usr/bin/env python
"""Evaluate amide-hydrolysis capability proxies: AUROC (with explicit mechanistic
direction + direction audit), pLDDT confound check, partial correlation controlling
for pLDDT, INC20 spotlight, and a distribution figure.

Reads exp_data/amide_hydrolase_capability/capability_scores.csv
Writes RESULTS.md and figures/capability_auroc.png
"""
import os, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_hydrolase_capability"
CSV = os.path.join(BASE, "capability_scores.csv")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# proxy: (column, mechanistic orientation: +1 higher=more capable, -1 lower=more capable,
#         optional transform for "distance-to-ideal")
PROXIES = [
    ("near_attack_fraction", +1, None, "near-attack fraction"),
    ("min_nuc_dist", -1, None, "min nucleophile->C dist (A)"),
    ("best_BD_angle", +1, "bd105", "best Burgi-Dunitz angle (|dev from 105|)"),
    ("mean_oxyanion_donors", +1, None, "mean oxyanion donors"),
    ("complex_plddt", +1, None, "complex pLDDT (confound)"),
]


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if int(r.get("n_poses", 0) or 0) == 0:
                continue
            rows.append(r)
    return rows


def getval(r, col, transform):
    v = r.get(col, "")
    if v == "" or v is None:
        return None
    v = float(v)
    if transform == "bd105":
        v = -abs(v - 105.0)   # closer to 105 -> larger (more capable)
    return v


def auroc(scores, labels):
    """AUROC with label==1 positive. Higher score -> more positive."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    # Mann-Whitney U with tie handling via rank-sum
    allv = sorted(set(pos + neg))
    # assign average ranks
    data = sorted(pairs, key=lambda x: x[0])
    ranks = {}
    i = 0
    n = len(data)
    vals = [d[0] for d in data]
    ranks_arr = [0.0]*n
    while i < n:
        j = i
        while j+1 < n and vals[j+1] == vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j+1):
            ranks_arr[k] = avg
        i = j + 1
    sum_pos = sum(ranks_arr[k] for k in range(n) if data[k][1] == 1)
    np_, nn = len(pos), len(neg)
    U = sum_pos - np_*(np_+1)/2.0
    return U/(np_*nn), np_, nn


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return float("nan")
    def rankify(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0]*len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j+1 < len(vals) and vals[order[j+1]] == vals[order[i]]:
                j += 1
            avg = (i+j)/2.0 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j+1
        return r
    xr = rankify([p[0] for p in pairs]); yr = rankify([p[1] for p in pairs])
    n = len(pairs)
    mx = sum(xr)/n; my = sum(yr)/n
    cov = sum((xr[i]-mx)*(yr[i]-my) for i in range(n))
    vx = math.sqrt(sum((xr[i]-mx)**2 for i in range(n)))
    vy = math.sqrt(sum((yr[i]-my)**2 for i in range(n)))
    if vx == 0 or vy == 0:
        return float("nan")
    return cov/(vx*vy)


def partial_spearman(x, y, z):
    """partial corr of x,y controlling z (Spearman-based)."""
    common = [(a, b, c) for a, b, c in zip(x, y, z)
              if a is not None and b is not None and c is not None]
    if len(common) < 4:
        return float("nan")
    xs = [c[0] for c in common]; ys = [c[1] for c in common]; zs = [c[2] for c in common]
    rxy = spearman(xs, ys); rxz = spearman(xs, zs); ryz = spearman(ys, zs)
    denom = math.sqrt((1-rxz**2)*(1-ryz**2))
    if denom == 0 or any(math.isnan(v) for v in (rxy, rxz, ryz)):
        return float("nan")
    return (rxy - rxz*ryz)/denom


def main():
    rows = load()
    labels = [1 if r["label"] == "capable" else 0 for r in rows]
    plddt = [getval(r, "complex_plddt", None) for r in rows]

    lines = []
    lines.append("# Amide-hydrolysis CAPABILITY test - results\n")
    n_cap = sum(labels); n_inc = len(labels)-n_cap
    lines.append(f"Enzymes scored (>=1 pose): {len(rows)}  (capable={n_cap}, incapable={n_inc})\n")

    lines.append("## AUROC per proxy (oriented so mechanistically-competent = more capable)\n")
    lines.append("| proxy | AUROC (mech dir) | direction | audit |")
    lines.append("|---|---|---|---|")
    auroc_store = {}
    for col, orient, tr, label in PROXIES:
        raw = [getval(r, col, tr) for r in rows]
        # mechanistic-direction score: multiply by orient (transform already flips bd)
        mech = [None if v is None else orient*v for v in raw]
        a, npos, nneg = auroc(mech, labels)
        auroc_store[col] = (a, mech)
        dirtxt = "higher=capable" if orient > 0 else "lower=capable"
        if tr == "bd105":
            dirtxt = "closer-to-105=capable"
        audit = ""
        if not math.isnan(a) and a < 0.5:
            audit = f"**BACKWARDS**: data favors opposite dir (mech AUROC {a:.2f}<0.5); flipped AUROC={1-a:.2f}"
        lines.append(f"| {col} | {a:.3f} | {dirtxt} | {audit} |")
    lines.append("")

    # pLDDT confound
    ap, _ = auroc_store["complex_plddt"]
    lines.append("## pLDDT confound\n")
    lines.append(f"complex_plddt alone separates classes with AUROC = {ap:.3f} "
                 f"(higher pLDDT = more capable orientation).\n")
    lines.append("Partial Spearman correlation of each geometric proxy with the capability "
                 "label, controlling for complex_plddt (does the proxy survive the confound?):\n")
    lines.append("| proxy | Spearman(proxy,label) | partial (control pLDDT) |")
    lines.append("|---|---|---|")
    for col, orient, tr, label in PROXIES:
        if col == "complex_plddt":
            continue
        raw = [getval(r, col, tr) for r in rows]
        mech = [None if v is None else orient*v for v in raw]
        r_full = spearman(mech, [float(l) for l in labels])
        r_part = partial_spearman(mech, [float(l) for l in labels], plddt)
        lines.append(f"| {col} | {r_full:.3f} | {r_part:.3f} |")
    lines.append("")

    # INC20 spotlight
    lines.append("## INC20 spotlight (parathion hydrolase / phosphotriesterase)\n")
    lines.append("Same amidohydrolase TIM-barrel fold as CAP09/CAP10 but incapable of amide "
                 "hydrolysis - tests fold-vs-true-capability.\n")
    inc20 = next((r for r in rows if r["id"] == "INC20"), None)
    caps = {r["id"]: r for r in rows}
    if inc20:
        naf20 = float(inc20.get("near_attack_fraction") or 0)
        # rank of INC20 near_attack_fraction among all
        nafs = sorted(((float(r.get("near_attack_fraction") or 0), r["id"]) for r in rows), reverse=True)
        rank20 = [i for i, (_, i2) in enumerate(nafs, 1) if i2 == "INC20"][0]
        lines.append(f"- INC20 near_attack_fraction = {naf20:.3f}, min_nuc_dist = "
                     f"{inc20.get('min_nuc_dist')}, plddt = {inc20.get('complex_plddt')}. "
                     f"Rank {rank20}/{len(rows)} by near-attack fraction (1=most amide-competent).")
        for cid in ("CAP09", "CAP10"):
            if cid in caps:
                lines.append(f"- {cid} (same fold, capable): near_attack_fraction="
                             f"{caps[cid].get('near_attack_fraction')}, min_nuc_dist="
                             f"{caps[cid].get('min_nuc_dist')}.")
    lines.append("")

    # metallo note
    lines.append("## Note on metallo-amidases\n")
    lines.append("The Boltz inputs contained NO metal ions (only protein + N-methylacetamide), "
                 "so metallo-amidases (aspartoacylase, aminoacylase, carboxypeptidase, thermolysin, "
                 "dihydro-orotase/pyrimidinase, and INC20) have no modeled metal-activated water. "
                 "The Ser/Thr/Cys residue-nucleophile proxy is expected to MISS these; metal_near_frac "
                 "is ~0 for all as a result. This is a known limitation of the residue-nucleophile proxy.\n")

    with open(os.path.join(BASE, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # figure
    plot_proxies = [p for p in PROXIES]
    fig, axes = plt.subplots(1, len(plot_proxies), figsize=(4.2*len(plot_proxies), 4.6))
    if len(plot_proxies) == 1:
        axes = [axes]
    for ax, (col, orient, tr, label) in zip(axes, plot_proxies):
        cap_v = [getval(r, col, None) for r in rows if r["label"] == "capable"]
        inc_v = [getval(r, col, None) for r in rows if r["label"] == "incapable"]
        cap_v = [v for v in cap_v if v is not None]
        inc_v = [v for v in inc_v if v is not None]
        a, _ = auroc_store[col]
        parts = ax.boxplot([cap_v, inc_v], labels=["capable", "incapable"],
                           widths=0.5, showfliers=False, patch_artist=True)
        for patch, c in zip(parts["boxes"], ["#3b6ea5", "#b5b5b5"]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        import random
        for i, vals in enumerate([cap_v, inc_v], 1):
            xs = [i + (random.random()-0.5)*0.15 for _ in vals]
            ax.scatter(xs, vals, s=18, color="#1f3b57" if i == 1 else "#555555", zorder=3)
        ax.set_title(f"{col}\nAUROC={a:.2f}", fontsize=11)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=9)
    fig.suptitle("Amide-hydrolysis capability proxies: capable vs incapable", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, "capability_auroc.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
