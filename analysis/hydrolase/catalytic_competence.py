#!/usr/bin/env python3
"""Paper-motivated JOINT catalytic-competence proxy for the nylon-hydrolase panel.

Motivation (Dishner/Bourgery et al., JACS 2026, 148, 18285; the ORNL group's own
Boltz-2 recipe on THIS panel). Their per-pose competence filter ties, in the SAME
model, the nucleophile geometry AND the oxyanion hole to the SAME scissile carbonyl:
  d1  = catalytic Thr OG1 .. scissile amide carbonyl C           (< 3.2 A)
  aBD = Burgi-Dunitz  OG1-C=O                                     (80-120 deg)
  aFL = Flippin-Lodge lateral approach (0 = symmetric here)       (near-symmetric)
  oxy = oxyanion O stabilised by a protein H-bond donor          (backbone N or
        Asn ND2 / Gln NE2, < 3.5 A)                              (oxyanion hole formed)

My current battery tests these SEPARATELY and counts ANY sidechain donor. This tests
the CONJUNCTIVE gate (all satisfied for one carbonyl in one pose) aggregated as the
fraction of the 50 models passing, and compares AUROC vs L2 activity against my
current single-criterion best and the pLDDT/ipTM foldability nulls.
"""
import csv, glob, os
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/nylon_hydrolase/results/output_configs"
ACT = f"{ROOT}/exp_data/nylon_hydrolase/amide_hydrolase_biorep_activity.csv"
OUT = f"{ROOT}/exp_data/nylon_hydrolase/results/catalytic_competence.csv"

BB_N = "N"                                   # protein backbone amide N
SC_DONOR = {"ASN": "ND2", "GLN": "NE2"}      # canonical oxyanion-hole sidechain N


def parse(path):
    """Return og1, thrN, list of substrate atoms (elem,name,xyz), protein donor xyz list."""
    og1 = thrN = None
    sub = []          # chain C substrate atoms
    donors = []       # protein oxyanion-hole donor N atoms (backbone N + Asn/Gln sc N)
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; an = ln[12:16].strip()
        elem = (ln[76:78].strip() or an[0]).upper()
        res = ln[17:20].strip()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ln.startswith("ATOM") and ch == "B" and int(ln[22:26]) == 1:
            if an == "OG1":
                og1 = xyz
            elif an == "N":
                thrN = xyz
        if ch == "C":
            sub.append((elem, xyz))
        elif ln.startswith("ATOM") and ch in ("A", "B"):
            if an == BB_N:
                donors.append(xyz)
            elif res in SC_DONOR and an == SC_DONOR[res]:
                donors.append(xyz)
    return og1, thrN, sub, donors


def angle(a, b, c):
    v1 = a - b; v2 = c - b
    return float(np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))


def flippin_lodge(og1, C, O, N, Cacyl):
    """Lateral approach deviation of the nucleophile from the symmetric plane.
    0 deg = approaches symmetrically between the two carbonyl substituents."""
    z = (O - C); z /= np.linalg.norm(z) + 1e-9
    n = np.cross(N - C, Cacyl - C); n /= np.linalg.norm(n) + 1e-9   # sp2 normal
    y = np.cross(n, z); y /= np.linalg.norm(y) + 1e-9               # in-plane, perp C=O
    t = (og1 - C); t /= np.linalg.norm(t) + 1e-9
    return float(np.degrees(np.arcsin(np.clip(np.dot(t, y), -1, 1))))


def best_scissile(og1, sub):
    """Nearest scissile amide carbonyl to OG1: return (d1, C, O, N, Cacyl)."""
    Os = [x for (e, x) in sub if e == "O"]
    Ns = [x for (e, x) in sub if e == "N"]
    Cs = [x for (e, x) in sub if e == "C"]
    best = None
    for cx in Cs:
        o_nb = [ox for ox in Os if np.linalg.norm(cx - ox) < 1.6]
        n_nb = [nx for nx in Ns if np.linalg.norm(cx - nx) < 1.6]
        c_nb = [cc for cc in Cs if 0.1 < np.linalg.norm(cx - cc) < 1.7]
        if len(o_nb) == 1 and len(n_nb) >= 1 and len(c_nb) >= 1:   # amide carbonyl
            d = float(np.linalg.norm(og1 - cx))
            if best is None or d < best[0]:
                best = (d, cx, o_nb[0], n_nb[0], c_nb[0])
    return best


def process(cfgdir):
    name = os.path.basename(cfgdir)
    pred = f"{cfgdir}/boltz_results_{name}/predictions/{name}"
    rows = []
    for pdb in sorted(glob.glob(f"{pred}/{name}_model_*.pdb")):
        og1, thrN, sub, donors = parse(pdb)
        if og1 is None or not sub or not donors:
            continue
        b = best_scissile(og1, sub)
        if b is None:
            continue
        d1, C, O, N, Cacyl = b
        aBD = angle(og1, C, O)
        aFL = flippin_lodge(og1, C, O, N, Cacyl)
        d_oxy = min(np.linalg.norm(O - dz) for dz in donors)
        rows.append((d1, aBD, aFL, d_oxy))
    if not rows:
        return None
    r = np.array(rows)
    d1, aBD, aFL, d_oxy = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    n = len(r)
    g_bd = (d1 < 3.2) & (aBD >= 80) & (aBD <= 120)                       # nucleophile only
    g_oxy = g_bd & (d_oxy < 3.5)                                         # + oxyanion hole
    g_full = g_oxy & (np.abs(aFL) < 40)                                  # + lateral align
    return {"id": name, "n": n,
            "cc_bd_frac": round(g_bd.mean(), 3),
            "cc_oxy_frac": round(g_oxy.mean(), 3),
            "cc_full_frac": round(g_full.mean(), 3),
            "min_d1": round(float(d1.min()), 2),
            "min_d_oxy": round(float(d_oxy.min()), 2),
            "med_aFL": round(float(np.median(np.abs(aFL))), 1)}


def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))


def main():
    dirs = sorted(d for d in glob.glob(f"{RES}/*") if os.path.isdir(d))
    with Pool(min(32, os.cpu_count())) as p:
        res = [x for x in p.map(process, dirs) if x]
    by = {r["id"]: r for r in res}

    act = {}
    for r in csv.DictReader(open(ACT)):
        act[r["enzyme"]] = r
    def norm(name):
        return "Nyl" + str(int(name[3:])) if name.startswith("Nyl") and name[3:].isdigit() else name

    merged = []
    for iid, r in by.items():
        a = act.get(norm(iid))
        if not a or a["L2_significant"] == "":
            continue
        r["L2"] = float(a["L2_ugml"]); r["L2sig"] = int(a["L2_significant"])
        r["L1sig"] = int(a["L1_significant"]) if a["L1_significant"] else 0
        merged.append(r)

    keys = ["id", "n", "cc_bd_frac", "cc_oxy_frac", "cc_full_frac",
            "min_d1", "min_d_oxy", "med_aFL", "L1sig", "L2sig", "L2"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in keys})

    L2sig = [r["L2sig"] for r in merged]
    L1sig = [r["L1sig"] for r in merged]
    L2 = [r["L2"] for r in merged]
    print(f"\n{len(merged)} enzymes with activity labels; "
          f"actives L2={sum(L2sig)}  L1={sum(L1sig)}\n")
    print(f"{'proxy':<16}{'AUROC(L2)':>11}{'AUROC(L1)':>11}{'rho(L2)':>10}")
    for col in ["cc_bd_frac", "cc_oxy_frac", "cc_full_frac"]:
        s = [r[col] for r in merged]
        print(f"{col:<16}{auroc(s, L2sig):>11.3f}{auroc(s, L1sig):>11.3f}{spearman(s, L2):>10.3f}")
    # negated distances (smaller = better) as scores
    for col in ["min_d1", "min_d_oxy"]:
        s = [-r[col] for r in merged]
        print(f"{'-'+col:<16}{auroc(s, L2sig):>11.3f}{auroc(s, L1sig):>11.3f}{spearman(s, L2):>10.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
