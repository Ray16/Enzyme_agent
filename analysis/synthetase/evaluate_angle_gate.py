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


def evaluate(csv_path, which, n_models):
    act = load_activity()
    rows = list(csv.DictReader(open(csv_path)))

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # per-candidate proxy scores (drop SfaB reference)
    data = []  # (id, activity, near_attack_frac, mean_inline_angle_or_None, inline_frac_gated_or_None)
    for r in rows:
        e = r["id"]
        if e == "SfaB" or e not in act:
            continue
        ng = num(r.get("n_gated")) or 0.0
        ifg = num(r.get("inline_frac_gated")) or 0.0
        naf = (ng / n_models) * ifg            # docked AND in-line fraction of ALL poses
        mia = num(r.get("mean_inline_angle"))  # None if INCONCLUSIVE
        data.append((e, act[e], naf, mia, ifg))
    cands = [e for e, *_ in data]
    sfab = act["SfaB"]
    acts = [a for _, a, *_ in data]

    print(f"\n### {csv_path.split('/')[-1]}  (n={len(cands)} candidates, SfaB act={sfab:.4f})")

    # --- PRIMARY: Spearman vs the WHOLE continuous activity distribution ---
    print("  Spearman rho vs continuous activity (whole panel):")
    metrics = [
        ("near_attack_frac", [naf for *_, in [()] for (_, _, naf, _, _) in [d]] if False else [d[2] for d in data]),
        ("mean_inline_angle", [d[3] for d in data]),   # None -> handled below
        ("inline_frac_gated", [d[4] for d in data]),
    ]
    for name, vals in metrics:
        if name == "mean_inline_angle":
            pairs = [(a, v) for a, v in zip(acts, vals) if v is not None]  # conclusive only
            note = f"(conclusive n={len(pairs)})"
        else:
            pairs = [(a, (v if v is not None else 0.0)) for a, v in zip(acts, vals)]
            note = f"(all n={len(pairs)})"
        aa = [p[0] for p in pairs]; vv = [p[1] for p in pairs]
        rho, pval = spearmanr(vv, aa)
        star = "  <== significant" if pval < 0.05 else ""
        print(f"    {name:18} rho={rho:+.3f} p={pval:.4f} {note}{star}")

    # --- SECONDARY: binary AUROC/P@k for reference ---
    labels = {"top3": set(sorted(cands, key=lambda e: -act[e])[:3]),
              "gtSfaB": set(e for e in cands if act[e] > sfab)}
    print("  AUROC (binary label, for reference):")
    for lname in (["top3", "gtSfaB"] if which == "both" else [which]):
        S = labels[lname]
        for name, idx in (("near_attack_frac", 2), ("mean_inline_angle", 3)):
            scored = [(e, (d[idx] if d[idx] is not None else -1.0), e in S) for e, d in zip(cands, data)]
            s = [v for _, v, _ in scored]; l = [x for _, _, x in scored]
            o, p = mc_p(s, l)
            order = sorted(scored, key=lambda r: -r[1])
            p3 = sum(x for *_, x in order[:3]); p10 = sum(x for *_, x in order[:10])
            print(f"    {name:18} {lname:7} AUROC={o:.3f} p={p:.4f} P@3={p3} P@10={p10}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--label", default="both", choices=["top3", "gtSfaB", "both"])
    ap.add_argument("--n_models", type=int, default=50)
    a = ap.parse_args()
    evaluate(a.csv, a.label, a.n_models)
