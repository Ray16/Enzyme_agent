#!/usr/bin/env python3
"""How reliable is the step-1 adenylation-NAC signal for the amide-synthetase panel?

Uses the existing 49 x 50 step-1 cofolds. Tests:
  1. Test-retest (split-half): NAC fraction from odd vs even models -> is 50 samples stable?
  2. Sample-count convergence: AUROC & NAC using first k=5,10,25,50 models.
  3. NAC-definition robustness: AUROC across distance/angle cutoff grid.
  4. Leave-one-active-out: does the signal survive dropping AS9 (the dominant active)?
  5. Per-enzyme binomial SE: are actives separable from high-NAC inactives given sampling noise?
"""
import csv, glob, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from proxy_eval import auroc

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/near_attack_geometry.csv"


def angle(a, b, c):
    v1, v2 = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1)))


def nac_model(pdb):
    at = {}
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; an = ln[12:16].strip(); el = (ln[76:78].strip() or an[0]).upper()
        x = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        at.setdefault((ch, ln.startswith("HETATM")), []).append((an, el, x))
    pa = o3a = None
    for k, lst in at.items():
        nm = {a: x for a, e, x in lst}
        if "PA" in nm and "O3A" in nm:
            pa, o3a = nm["PA"], nm["O3A"]; break
    sub = None
    for (ch, het), lst in at.items():
        if not het:
            continue
        if any(a == "PA" for a, e, x in lst):
            continue
        Os = [x for a, e, x in lst if e == "O"]; Cs = [x for a, e, x in lst if e == "C"]
        cb = [ox for cx in Cs for ox in Os if False]  # placeholder
        carbox = [ox for cx in Cs if sum(np.linalg.norm(cx-ox) < 1.6 for ox in Os) >= 2 for ox in Os if np.linalg.norm(cx-ox) < 1.6]
        if carbox:
            sub = min(carbox, key=lambda o: np.linalg.norm(o-pa)); break
    if pa is None or sub is None:
        return None
    return float(np.linalg.norm(sub-pa)), float(angle(sub, pa, o3a))


def process(cd):
    name = os.path.basename(cd)[len("step1_"):]; base = os.path.basename(cd)
    pred = f"{cd}/boltz_results_{base}/predictions/{base}"
    rows = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        r = nac_model(pdb)
        if r:
            rows.append(r)
    return name, np.array(rows) if rows else None


def frac(arr, dcut=4.0, acut=150.0):
    return float(((arr[:, 0] < dcut) & (arr[:, 1] > acut)).mean())


def main():
    act = {r["id"]: float(r["activity_N66L1"]) for r in csv.DictReader(open(ACT))}
    with Pool(32) as p:
        data = {n: a for n, a in p.map(process, sorted(glob.glob(f"{RES}/*"))) if a is not None}
    ids = list(data)
    lab = np.array([1 if act[i] >= 0.5 else 0 for i in ids])
    actives = [i for i in ids if act[i] >= 0.5]

    def spearman(x, y):
        from scipy.stats import rankdata
        x, y = rankdata(x), rankdata(y); x = x-x.mean(); y = y-y.mean()
        return float((x@y)/(np.sqrt((x@x)*(y@y))+1e-12))

    # 1. split-half test-retest
    odd = np.array([frac(data[i][0::2]) for i in ids]); even = np.array([frac(data[i][1::2]) for i in ids])
    print(f"1) TEST-RETEST (odd vs even 25 models): Spearman r = {spearman(odd, even):.3f}  Pearson = {np.corrcoef(odd,even)[0,1]:.3f}")

    # 2. sample-count convergence
    print("\n2) SAMPLE-COUNT CONVERGENCE:")
    for k in (5, 10, 25, 50):
        f = np.array([frac(data[i][:k]) for i in ids])
        print(f"   first {k:2d} models: AUROC={auroc(f, lab):.3f}")

    # 3. NAC-definition robustness
    print("\n3) NAC CUTOFF ROBUSTNESS (AUROC):")
    print("        angle>140  angle>150  angle>160")
    for dcut in (3.5, 4.0, 4.5):
        aus = [auroc(np.array([frac(data[i], dcut, ac) for i in ids]), lab) for ac in (140, 150, 160)]
        print(f"   d<{dcut}:  " + "     ".join(f"{a:.3f}" for a in aus))

    # 4. leave-one-active-out
    print("\n4) LEAVE-ONE-ACTIVE-OUT (drop each active, AUROC on the rest):")
    f_all = np.array([frac(data[i]) for i in ids])
    for drop in actives:
        keep = [j for j, i in enumerate(ids) if i != drop]
        print(f"   drop {drop} (act={act[drop]:.1f}): AUROC={auroc(f_all[keep], lab[keep]):.3f}")

    # 5. per-enzyme binomial SE (npass/50)
    print("\n5) PER-ENZYME NAC +/- binomial SE (actives vs top-NAC inactives):")
    rows = sorted(ids, key=lambda i: -frac(data[i]))
    for i in rows[:8]:
        p = frac(data[i]); se = (p*(1-p)/len(data[i]))**0.5
        tag = "ACTIVE" if act[i] >= 0.5 else ""
        print(f"   {i:6s} NAC={p:.2f}±{se:.2f}  act={act[i]:.2f} {tag}")


if __name__ == "__main__":
    main()
