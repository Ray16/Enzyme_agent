#!/usr/bin/env python3
"""Evaluate an angle-gate proxy CSV (from angle_gate_proxy.py) against measured
N66-L1 activity for the amide-synthetase diversity panel.

PRIMARY evaluation = Spearman rho of each proxy vs the CONTINUOUS activity, over the
WHOLE panel (no binary active/inactive cutoff). AUROC/P@k under binary labels (top3,
gtSfaB) are reported secondarily for reference.

Proxy scores evaluated:
  * near_attack_frac  : fraction of ALL poses that are BOTH docked (d<4A) AND in-line
                        (>150 deg) = n_gated/n_models * inline_frac_gated. Accounts for
                        docking FREQUENCY (out-of-site poses lower it) AND geometry;
                        defined for every enzyme incl. INCONCLUSIVE (-> low). [preferred]
  * mean_inline_angle : mean in-line angle among docked poses (conclusive enzymes only;
                        ignores docking frequency).
  * inline_frac_gated : fraction of docked poses that are in-line (conclusive only).

Usage: evaluate_angle_gate.py <proxy_csv> [--label top3|gtSfaB|both] [--n_models 50]
"""
import argparse, csv, random
from scipy.stats import spearmanr
random.seed(0)
ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"


def load_activity():
    act = {}
    for r in csv.DictReader(open(ACT)):
        try:
            act[r["id"]] = float(r["activity_N66L1_ugml_avg"])
        except (ValueError, KeyError):
            pass
    return act


def auroc(scores, labels):
    P = [s for s, l in zip(scores, labels) if l]
    N = [s for s, l in zip(scores, labels) if not l]
    if not P or not N:
        return None
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in P for n in N) / (len(P) * len(N))


def mc_p(scores, labels, B=50000):
    o = auroc(scores, labels)
    ge = sum(auroc(scores, random.sample(labels, len(labels))) >= o - 1e-12 for _ in range(B))
    return o, (ge + 1) / (B + 1)


def evaluate(csv_path, which):
    act = load_activity()
    rows = list(csv.DictReader(open(csv_path)))
    # candidates only (drop SfaB reference); keep those present in activity table
    data = []
    for r in rows:
        e = r["id"]
        if e == "SfaB" or e not in act:
            continue
        # primary score: mean_inline_angle; INCONCLUSIVE (blank/None) -> ranked last
        def num(col):
            v = r.get(col)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        data.append((e, act[e], num("mean_inline_angle"), num("inline_frac_gated")))
    cands = [e for e, *_ in data]
    sfab = act["SfaB"]
    labels = {
        "top3": set(sorted(cands, key=lambda e: -act[e])[:3]),
        "gtSfaB": set(e for e in cands if act[e] > sfab),
    }
    print(f"\n### {csv_path.split('/')[-1]}  (n={len(cands)} candidates, SfaB={sfab:.4f})")
    for lname in (["top3", "gtSfaB"] if which == "both" else [which]):
        S = labels[lname]
        for col, idx in (("mean_inline_angle", 2), ("inline_frac_gated", 3)):
            scored = [(e, (v if v is not None else -1.0), e in S) for e, a, *rest in data
                      for v in [ (rest[0] if col == "mean_inline_angle" else rest[1]) ]]
            s = [v for _, v, _ in scored]
            l = [lab for _, _, lab in scored]
            o, p = mc_p(s, l)
            order = sorted(scored, key=lambda r: -r[1])
            p3 = sum(lab for *_, lab in order[:3])
            p5 = sum(lab for *_, lab in order[:5])
            p10 = sum(lab for *_, lab in order[:10])
            flag = "  <== significant" if p < 0.05 else ""
            print(f"  {col:18} label={lname:7} n_act={sum(l)}  AUROC={o:.3f} p={p:.4f}"
                  f"  P@3={p3} P@5={p5} P@10={p10}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--label", default="both", choices=["top3", "gtSfaB", "both"])
    a = ap.parse_args()
    evaluate(a.csv, a.label)
