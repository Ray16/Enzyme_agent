"""Pre-registered statistics for the selectivity proxy (PROTOCOL.md e3259ae + amendment 1).
Primary: Spearman(asymmetry, frac_L1) one-sided (higher asym -> higher frac_L1), n=38 cohort.
Also: n=26 sensitivity, selection-corrected max-null over descriptors, alpha-length partial, LOO.
"""
import sys, csv
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent")
HERE = ROOT / "exp_data/nylon_hydrolase/analysis/selectivity_proxy_20261009"
ACT = ROOT / "exp_data/nylon_hydrolase/amide_hydrolase_biorep_activity.csv"
FEAT = ROOT / "exp_data/nylon_hydrolase/analysis/proxy_verification_20261008/qc/features_complete50.csv"
DESC_FILE = sys.argv[1] if len(sys.argv) > 1 else "ligand_axis_per_enzyme.csv"

# activity
act = {}
for r in csv.DictReader(open(ACT)):
    if r["exp"] == "65" and r["type"] == "Powder":
        l1, l2 = float(r["L1_ugml"]), float(r["L2_ugml"])
        act[r["enzyme"]] = dict(l1=l1, l2=l2, tot=l1 + l2,
                                frac=l1 / (l1 + l2) if (l1 + l2) > 0 else np.nan,
                                l2sig=r["L2_significant"] in ("1", "True", "TRUE"))
# alpha length (broad-search feature table covers all 95; take first row per id)
BROAD = ROOT / "exp_data/nylon_hydrolase/analysis/broad_proxy_search_20261008/features/features.csv"
alen = {}
for r in csv.DictReader(open(BROAD)):
    k = r.get("id") or r.get("enzyme")
    if k not in alen and r.get("alpha_len"):
        alen[k] = float(r["alpha_len"])
# descriptor
desc = {r["id"]: dict(asymmetry=float(r["asymmetry"]), distal_capacity=float(r["distal_capacity"]))
        for r in csv.DictReader(open(HERE / DESC_FILE))}

DESCS = {"asymmetry": +1, "distal_capacity": -1}  # sign = pre-registered direction wrt frac_L1


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([rz, np.ones(len(rz))])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]; by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    return stats.pearsonr(rx - Z @ bx, ry - Z @ by)


def oriented(name, v):
    return np.array(v) * DESCS[name]  # so positive corr = pre-registered direction


def run_cohort(ids, tag):
    ids = [i for i in ids if i in desc and not np.isnan(act[i]["frac"])]
    y = np.array([act[i]["frac"] for i in ids])
    al = np.array([alen[i] for i in ids])
    print(f"\n=== cohort {tag}: n={len(ids)} ===")
    rhos = {}
    for name in DESCS:
        x = oriented(name, [desc[i][name] for i in ids])
        rho, p2 = stats.spearmanr(x, y)
        p1 = p2 / 2 if rho > 0 else 1 - p2 / 2  # one-sided in pre-reg direction
        pr, pp = partial_spearman(x, y, al)
        loo = [stats.spearmanr(np.delete(x, k), np.delete(y, k))[0] for k in range(len(x))]
        print(f"  {name:16s} rho={rho:+.3f} p1={p1:.4f}  partial|alen rho={pr:+.3f} p={pp:.4f}  "
              f"LOO[{min(loo):+.3f},{max(loo):+.3f}]")
        rhos[name] = (rho, ids, x, y)
    # selection-corrected max-null over descriptors (primary = asymmetry)
    xa = oriented("asymmetry", [desc[i]["asymmetry"] for i in ids])
    xd = oriented("distal_capacity", [desc[i]["distal_capacity"] for i in ids])
    obs = stats.spearmanr(xa, y)[0]
    rng = np.random.default_rng(0)
    null = []
    for _ in range(20000):
        yp = rng.permutation(y)
        null.append(max(stats.spearmanr(xa, yp)[0], stats.spearmanr(xd, yp)[0]))
    null = np.array(null)
    p_sel = (np.sum(null >= obs) + 1) / 20001
    print(f"  selection-corrected (max over 2 descriptors, 20k perms): obs_asym_rho={obs:+.3f} "
          f"null95={np.quantile(null, 0.95):.3f}  p={p_sel:.4f}")
    return len(ids)


def main():
    sig = [i for i, a in act.items() if a["l2sig"]]
    tot1 = [i for i, a in act.items() if a["tot"] >= 1.0]
    n38 = run_cohort(sig, "L2_significant (primary)")
    n26 = run_cohort(tot1, "total>=1.0 (sensitivity)")
    print(f"\nrealized cohorts (intersected with cofold panel): primary n={n38}, sensitivity n={n26}")


if __name__ == "__main__":
    main()
