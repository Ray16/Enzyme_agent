#!/usr/bin/env python3
"""Head-to-head: unrestrained vs pocket-restrained cofolds, SAME enzyme set, SAME metrics.
Answers: which set predicts measured activity better, for NAC, angle-only, and iptm-gated angle?"""
import glob, sys, csv, json, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "common"))
import step1_nac as s

ACT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/assay_data/panel_activity.csv"
UNR = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
RES = os.path.join(os.path.dirname(__file__), "results", "output_configs")


def spear(x, y):
    x = np.array(x, float); y = np.array(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), len(x)
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]), len(x)


def metrics(pred, gate=0.6):
    per = []
    for pdb in sorted(glob.glob(f"{pred}/*model_*.pdb")):
        g = s.geom_model(pdb)
        if not g:
            continue
        mn = pdb.split("model_")[-1].split(".pdb")[0]
        cj = glob.glob(f"{pred}/confidence_*model_{mn}.json")
        ip = json.load(open(cj[0]))["pair_chains_iptm"]["1"]["4"] if cj else np.nan
        per.append((g["ang_pa"], g["d_pa"], ip))
    if not per:
        return None
    a = np.array([p[0] for p in per]); d = np.array([p[1] for p in per]); ip = np.array([p[2] for p in per])
    gm = ip > gate
    return {
        "nac": float(np.mean((d < 4) & (a > 150))),
        "angle": float(np.mean(a > 150)),
        "angle_gated": float(np.mean(a[gm] > 150)) if gm.any() else float("nan"),
    }


def collect(root, prefix):
    out = {}
    for cfg in sorted(glob.glob(f"{root}/{prefix}*")):
        eid = os.path.basename(cfg)[len(prefix):]
        pred = glob.glob(f"{cfg}/boltz_results_*/predictions/*")
        if not pred:
            continue
        m = metrics(pred[0])
        if m:
            out[eid] = m
    return out


def main():
    act = {r["id"]: float(r["activity_N66L1_ugml_avg"]) for r in csv.DictReader(open(ACT))}
    U = collect(UNR, "step1_")
    R = collect(RES, "pk_")
    common = sorted(set(U) & set(R) & set(act))
    print(f"common enzymes (both runs + activity): {len(common)}\n")
    A = [act[i] for i in common]
    print(f"{'metric':16} {'UNRESTRAINED ρ':>16} {'RESTRAINED ρ':>16}   winner")
    for k in ("nac", "angle", "angle_gated"):
        ru, _ = spear([U[i][k] for i in common], A)
        rr, _ = spear([R[i][k] for i in common], A)
        win = "unrestr" if ru > rr else "restr" if rr > ru else "tie"
        print(f"{k:16} {ru:>+16.3f} {rr:>+16.3f}   {win}")
    # non-circularity: dead-enzyme angle in each set
    dead = [i for i in common if act[i] < 0.001]
    actv = [i for i in common if act[i] > 0.1]
    print(f"\nnon-circularity (mean angle-frac): dead vs active")
    for k in ("angle", "angle_gated"):
        du = np.nanmean([U[i][k] for i in dead]); au = np.nanmean([U[i][k] for i in actv])
        dr = np.nanmean([R[i][k] for i in dead]); ar = np.nanmean([R[i][k] for i in actv])
        print(f"  {k:12} UNRESTR dead={du:.2f} active={au:.2f} (gap {au-du:+.2f}) | RESTR dead={dr:.2f} active={ar:.2f} (gap {ar-dr:+.2f})")


if __name__ == "__main__":
    main()
