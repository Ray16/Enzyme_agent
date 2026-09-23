#!/usr/bin/env python3
"""Confound disentanglement for the synthetase structural proxies.

A proxy can show good AUROC for the wrong reason: a hidden variable correlated with
BOTH the proxy and activity (a confound). The prime suspect is model confidence
(complex pLDDT) -- in the nylon-hydrolase panel, reactive-center RMSF turned out to
be a pLDDT confound (foldability), not catalytic rigidity. This script tests whether
the synthetase proxies survive controlling for pLDDT.

For each enzyme: mean complex_plddt over its 50 cofold models (from the Boltz
confidence_*.json). Then:
  [1] does pLDDT itself track activity? (Spearman + tier AUROC + concordance)
  [2] per proxy: r(proxy,activity), r(proxy,pLDDT), and the PARTIAL Spearman
      correlation of proxy vs activity controlling for pLDDT. A proxy whose signal
      collapses toward 0 under the partial is confounded by foldability.

CPU-only, read-only. Usage: python analysis/synthetase/confound_check.py
"""
import json, glob, csv, os
import numpy as np
from scipy.stats import spearmanr

B = "exp_data/amide_synthetase"
RES = f"{B}/boltz2/step1_1mg/results"
AN = f"{B}/analysis"

# curated proxy panel: (csv, column, label)
SRC = [("reactive_preorganization.csv", "posn_rmsf", "RMSF (preorg)"),
       ("esp_reactive.csv", "lef_pa_std", "LEF preorg (std)"),
       ("esp_reactive.csv", "esp_carbonyl", "carbonyl ESP"),
       ("battery_1mg.csv", "mean_inline_angle", "in-line angle"),
       ("reactive_sasa_intermediate.csv", "distalC_sasa", "distal-carboxyl SASA"),
       ("hypothesis_battery.csv", "H7_adenineBurial", "ATP adenine burial")]


def au(y, x):
    p = [x[k] for k in range(len(y)) if y[k] == 1]
    n = [x[k] for k in range(len(y)) if y[k] == 0]
    return sum((a > b) + 0.5 * (a == b) for a in p for b in n) / (len(p) * len(n)) if p and n else float("nan")


def cidx(f, a):
    c = d = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] == a[j]:
                continue
            hi, lo = (i, j) if a[i] > a[j] else (j, i)
            c += f[hi] > f[lo]; d += f[hi] < f[lo]
    return c / (c + d) if c + d else float("nan")


def load(fn, col):
    m = {}
    for r in csv.DictReader(open(f"{AN}/{fn}")):
        try:
            m[r["id"]] = float(r[col])
        except (ValueError, TypeError, KeyError):
            pass
    return m


def main():
    plddt = {}
    for d in sorted(glob.glob(f"{RES}/*")):
        e = os.path.basename(d)
        js = glob.glob(f"{d}/boltz_results_step1_{e}/predictions/step1_{e}/confidence_*.json")
        vals = []
        for f in js:
            try:
                vals.append(json.load(open(f))["complex_plddt"])
            except Exception:
                pass
        if vals:
            plddt[e] = float(np.mean(vals))

    act, lab = {}, {}
    for r in csv.DictReader(open(f"{AN}/crossmodel_nearattack.csv")):
        act[r["id"]] = float(r["activity"]); lab[r["id"]] = int(r["label_gt_sfab"])
    ids = [i for i in act if i in plddt]
    top3 = set(sorted(ids, key=lambda i: -act[i])[:3])
    marg = {i for i in ids if lab[i] == 1 and i not in top3}

    def tiers(feat, name):
        x = [feat[i] for i in ids]; a = [act[i] for i in ids]
        rho = spearmanr(x, a).correlation or 0
        xo = x if rho >= 0 else [-v for v in x]
        sig = au([1 if i in top3 else 0 for i in ids], xo)
        sub = [(i, v) for i, v in zip(ids, xo) if i not in top3]
        mg = au([1 if i in marg else 0 for i, _ in sub], [v for _, v in sub])
        print(f"  {name:24s} rho={rho:+.2f}  signif={sig:.2f} marg={mg:.2f} overall={cidx(xo, a):.2f}")

    print(f"n={len(ids)} enzymes with pLDDT + activity\n")
    print("[1] Does pLDDT (foldability) itself predict activity?")
    tiers(plddt, "complex pLDDT")

    print("\n[2] Confound test: r(proxy,act), r(proxy,pLDDT), partial(act|pLDDT)")
    print(f"  {'proxy':24s} {'r(pxy,act)':>10} {'r(pxy,pLDDT)':>13} {'partial':>9}")
    for fn, col, name in SRC:
        m = load(fn, col); comm = [i for i in ids if i in m]
        if len(comm) < 20:
            print(f"  {name:24s} (missing)"); continue
        p = [m[i] for i in comm]; a = [act[i] for i in comm]; pl = [plddt[i] for i in comm]
        r_pa = spearmanr(p, a).correlation or 0
        r_pp = spearmanr(p, pl).correlation or 0
        r_ap = spearmanr(a, pl).correlation or 0
        den = ((1 - r_pp ** 2) * (1 - r_ap ** 2)) ** 0.5
        part = (r_pa - r_pp * r_ap) / den if den > 0 else float("nan")
        print(f"  {name:24s} {r_pa:>+10.2f} {r_pp:>+13.2f} {part:>+9.2f}")


if __name__ == "__main__":
    main()
