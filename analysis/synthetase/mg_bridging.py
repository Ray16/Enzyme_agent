#!/usr/bin/env python3
"""Mechanistically-correct metal proxy: Mg2+ bridging of the beta,gamma-diphosphate.

In ANL adenylation the carboxylate attacks ATP-Palpha and the LEAVING GROUP is
pyrophosphate (PPi = Pbeta-Pgamma). The catalytic Mg2+ stabilises that departing
PPi by BRIDGING beta- and gamma-phosphate oxygens (inner-sphere). The coarse
"min Mg-to-any-P distance" proxy is placement-sanity only and ranks backwards;
this descriptor instead asks whether a single Mg2+ simultaneously coordinates a
beta-O AND a gamma-O (the physical leaving-group-stabilisation geometry).

Per step-1 pose (chain-agnostic HETATM parse): find PB, PG and their bonded O
(<1.8 A). For each Mg, dB = nearest Mg-(beta-O), dG = nearest Mg-(gamma-O);
the bridging Mg minimises worse=max(dB,dG). Aggregate over the 50 poses:
  bridge_worse_mean : mean of worse   (LOWER = tighter bridging)
  frac_bridge_{c}   : fraction of poses with worse < c  (HIGHER = better)
  mg_pb_mean/mg_pg_mean : Mg-Pbeta / Mg-Pgamma distances (context)
Writes mg_bridging_bg.csv and self-evaluates surfacing AUROC vs activity
(top-3/5/10 + >SfaB) so it can be judged before wiring into feature_surfacing.
"""
import os, csv, glob, math, warnings
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
PANEL = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1_1mg/results"
XM = f"{ROOT}/exp_data/amide_synthetase/analysis/crossmodel_nearattack.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/mg_bridging_bg.csv"
CUTS = [2.4, 2.8, 3.2]
_P = PDBParser(QUIET=True)


def pred_dir(e):
    return f"{PANEL}/{e}/boltz_results_step1_{e}/predictions/step1_{e}"


def pose(pdb):
    m = _P.get_structure("x", pdb)[0]
    het = [a for a in m.get_atoms() if a.get_parent().id[0] != " "]
    pb = [a.coord for a in het if a.get_name() == "PB"]
    pg = [a.coord for a in het if a.get_name() == "PG"]
    mgs = [a.coord for a in het if a.element == "MG" or a.get_name() == "MG"]
    if not pb or not pg or not mgs:
        return None
    pb, pg = pb[0], pg[0]
    Os = [a.coord for a in het if a.element == "O"]
    bO = [o for o in Os if np.linalg.norm(o - pb) < 1.8]
    gO = [o for o in Os if np.linalg.norm(o - pg) < 1.8]
    if not bO or not gO:
        return None
    best = None
    for mg in mgs:
        dB = min(float(np.linalg.norm(mg - o)) for o in bO)
        dG = min(float(np.linalg.norm(mg - o)) for o in gO)
        w = max(dB, dG)
        if best is None or w < best[0]:
            best = (w, dB, dG)
    w, dB, dG = best
    return dict(worse=w, dB=dB, dG=dG,
                mg_pb=min(float(np.linalg.norm(mg - pb)) for mg in mgs),
                mg_pg=min(float(np.linalg.norm(mg - pg)) for mg in mgs))


def enzyme(e):
    pdbs = sorted(glob.glob(f"{pred_dir(e)}/*model_*.pdb"))
    per = [g for p in pdbs if (g := pose(p))]
    if not per:
        return None
    W = np.array([p["worse"] for p in per])
    row = {"id": e, "n": len(per),
           "bridge_worse_mean": round(float(W.mean()), 3),
           "bridge_worse_min": round(float(W.min()), 3),
           "mg_pb_mean": round(float(np.mean([p["mg_pb"] for p in per])), 3),
           "mg_pg_mean": round(float(np.mean([p["mg_pg"] for p in per])), 3)}
    for c in CUTS:
        row[f"frac_bridge_{c}"] = round(float(np.mean(W < c)), 3)
    return row


# ---- inline evaluation (mirror feature_surfacing.py) ----
def auroc(y, x):
    pos = [x[i] for i in range(len(y)) if y[i] == 1]
    neg = [x[i] for i in range(len(y)) if y[i] == 0]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def spear(x, a):
    x, a = np.asarray(x, float), np.asarray(a, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rx -= rx.mean(); ra -= ra.mean()
    d = math.sqrt((rx @ rx) * (ra @ ra)) + 1e-12
    return float((rx @ ra) / d)


def main():
    enz = sorted([os.path.basename(d) for d in glob.glob(f"{PANEL}/*") if os.path.isdir(d)])
    with Pool(6) as pool:
        rows = [r for r in pool.map(enzyme, enz) if r]
    cols = list(rows[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} enzymes)\n")

    act, lab = {}, {}
    for r in csv.DictReader(open(XM)):
        act[r["id"]] = float(r["activity"]); lab[r["id"]] = int(r["label_gt_sfab"])
    ids = [r["id"] for r in rows if r["id"] in act]
    topk = {k: set(sorted(ids, key=lambda i: -act[i])[:k]) for k in (3, 5, 10)}
    feats = [c for c in cols if c not in ("id", "n")]
    print(f"{'feature':20s} {'dir':>4} {'top3':>5} {'top5':>5} {'top10':>5} {'>SfaB':>6} {'rho':>6} {'AS9rk':>6}")
    for fcol in feats:
        vals = {r["id"]: r[fcol] for r in rows}
        x = [vals[i] for i in ids]; a = [act[i] for i in ids]
        rho = spear(x, a); desc = rho >= 0
        xo = x if desc else [-v for v in x]
        au = {k: auroc([1 if i in topk[k] else 0 for i in ids], xo) for k in (3, 5, 10)}
        aub = auroc([lab[i] for i in ids], xo)
        order = sorted(ids, key=lambda i: -vals[i] if desc else vals[i])
        as9 = order.index("AS9") + 1 if "AS9" in order else -1
        print(f"{fcol:20s} {'+' if desc else '-':>4} {au[3]:5.2f} {au[5]:5.2f} {au[10]:5.2f} "
              f"{aub:6.2f} {rho:+6.2f} {as9:6d}")


if __name__ == "__main__":
    main()
