#!/usr/bin/env python3
"""Apply the standard proxy-evaluation block (AUROC + CI + perm-p + precision@k)
consistently to every geometric proxy we've computed, across both panels."""
import csv, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from proxy_eval import evaluate, fmt

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"


def _eval(job):
    scores, labels, name, hb = job
    return evaluate(scores, labels, name, hb)


def load(path):
    return list(csv.DictReader(open(path)))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


jobs = []

# ---- SYNTHETASE (active = N66 L1 activity >= 0.5) ----
s1 = load(f"{ROOT}/exp_data/sfab_step1_results/nac_step1.csv")
lab1 = [1 if (fnum(r["activity_N66L1"]) or 0) >= 0.5 else 0 for r in s1]
jobs.append(([fnum(r["nac_frac"]) for r in s1], lab1, "synth step-1 NAC fraction", True))
jobs.append(([fnum(r["min_d"]) for r in s1], lab1, "synth step-1 adenyl min_d", False))

ps = load(f"{ROOT}/exp_data/sfab_step1_results/ps_iptm_step1.csv")
labp = [1 if (fnum(r["activity_N66L1"]) or 0) >= 0.5 else 0 for r in ps]
jobs.append(([fnum(r["ps_iptm_mean"]) for r in ps], labp, "synth step-1 ps_iptm (prot-adipate)", True))

s2 = load(f"{ROOT}/exp_data/sfab_step2_results/nac_step2.csv")
lab2 = [1 if (fnum(r["activity_N66L1"]) or 0) >= 0.5 else 0 for r in s2]
jobs.append(([fnum(r["min_d"]) for r in s2], lab2, "synth step-2 attack dist [loose restraint]", False))

# ---- HYDROLASE (active = L1 or L2 significant) ----
hy = load(f"{ROOT}/exp_data/nylon_hydrolase_results/hydrolase_nac.csv")
labh = [1 if (r.get("L1sig") == "1" or r.get("L2sig") == "1") else 0 for r in hy]
jobs.append(([fnum(r["nac_frac"]) for r in hy], labh, "hydrolase NAC fraction", True))
jobs.append(([fnum(r["min_d"]) for r in hy], labh, "hydrolase min_d", False))

# proxies are independent -> evaluate them in parallel
with Pool(min(len(jobs), 8)) as pool:
    results = pool.map(_eval, jobs)

print("=" * 108)
print("STANDARD PROXY EVALUATION  (AUROC [95% boot CI]  permutation-p  precision@k)")
print("=" * 108)
for r in results:
    print(fmt(r))

# save flat CSV
out = f"{ROOT}/exp_data/proxy_evaluation_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["proxy", "n", "n_active", "auroc", "ci_lo", "ci_hi", "perm_p", "P@3", "P@5", "P@10"])
    for r in results:
        pk = r["precision_at_k"]
        w.writerow([r["name"], r["n"], r["n_active"], r["auroc"], r["ci_lo"], r["ci_hi"], r["perm_p"],
                    f"{pk[3][0]}/3", f"{pk[5][0]}/5", f"{pk[10][0]}/10"])
print("\nWrote", out)
