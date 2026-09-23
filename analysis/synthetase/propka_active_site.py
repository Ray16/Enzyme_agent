#!/usr/bin/env python3
"""Active-site pKa proxy (nucleophile/base-activation determinant) via PROPKA.

Fills the ONE catalytic determinant our current structural-proxy panel does not
touch: activation of the nucleophile / a general acid-base. For each Boltz-2 pose
we run PROPKA (empirical structure-based pKa) and summarise the protonation
environment of ionizable residues in a 10 A shell around the reactive atom:
  step-1 (adenylation)  reactive atom = ATP Pa  (carboxylate -> Pa attack)
  step-2 (aminolysis)   reactive atom = acyl-AMP carbonyl C (amine -> C=O attack)

Per-pose summaries (mechanistic rationale):
  ag_shift  : mean pKa shift (pKa - model_pKa) of nearby Asp/Glu. An UP-shifted
              carboxylate is protonatable at physiological pH -> can act as a
              general acid/base (proton shuttle for the leaving group / nucleophile).
  ag_maxpka : max pKa among nearby Asp/Glu (the most up-shifted carboxylate).
  lys_minpka: min pKa among nearby Lys. A DOWN-shifted Lys (pKa < ~10.5) is
              partly deprotonated -> candidate general base to activate the amine
              nucleophile in aminolysis.
  his_dev7  : min |pKa - 7| among nearby His. A His poised near physiological pH
              is ideal for proton shuttling.
  nion      : count of ionizable residues in the shell (active-site titratable density).
Averaged (nanmean) over the 50 poses; std also written. Self-evaluates AUROC/
Spearman vs measured N66-L1 activity (does the pKa environment surface AS9/17/23?).

CPU-only. Usage: python propka_active_site.py [--n 50] [--pool 6]
"""
import os, csv, glob, argparse, io, contextlib, tempfile, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
from propka.run import single

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
BASE = f"{ROOT}/exp_data/amide_synthetase"
S1 = f"{BASE}/boltz2/step1_1mg/results"
S2 = f"{BASE}/boltz2/step2_1mg/results"
OUT = f"{BASE}/analysis/propka_active_site.csv"
ACT_CSV = f"{BASE}/analysis/crossmodel_nearattack.csv"
SHELL = 10.0
_P = PDBParser(QUIET=True)
IONIZ = {"ASP", "GLU", "HIS", "LYS", "ARG", "CYS", "TYR"}
FUNC = {"ASP": ["OD1", "OD2", "CG"], "GLU": ["OE1", "OE2", "CD"],
        "HIS": ["NE2", "ND1"], "LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE", "CZ"],
        "CYS": ["SG"], "TYR": ["OH"]}
N_MODELS = 50


def s1_dir(e): return f"{S1}/{e}/boltz_results_step1_{e}/predictions/step1_{e}"
def s2_dir(e): return f"{S2}/{e}/boltz_results_step2_{e}/predictions/step2_{e}"


def het(m): return [a for a in m.get_atoms() if a.get_parent().id[0] != " "]


def reactive_atom(m, step):
    """Pa for step-1; acyl-AMP carbonyl C for step-2."""
    h = het(m)
    if step == 1:
        for a in h:
            if a.get_name() == "PA":
                return a.coord
        Ps = [a for a in h if a.element == "P"]
        return Ps[0].coord if len(Ps) == 1 else None
    Ps = [a for a in h if a.element == "P"]
    Os = [a for a in h if a.element == "O"]
    Cs = [a for a in h if a.element == "C"]
    for c in Cs:
        onb = [o for o in Os if np.linalg.norm(c.coord - o.coord) < 1.6]
        if len(onb) < 2:
            continue
        ester_to_P = any(any(np.linalg.norm(o.coord - p.coord) < 1.9 for p in Ps) for o in onb)
        carbonyl = any(np.linalg.norm(c.coord - o.coord) < 1.30 for o in onb)
        if ester_to_P and carbonyl:
            return c.coord      # mixed-anhydride acyl carbon (the electrophile)
    return None


def propka_pkas(pdb):
    """{(chain,resnum): (restype, pka, model_pka)} for ionizable groups."""
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd(); os.chdir(td)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                mol = single(pdb, optargs=["--quiet"], write_pka=False)
        except Exception:
            os.chdir(cwd); return {}
        finally:
            os.chdir(cwd)
    out = {}
    grp = mol.conformations[mol.conformation_names[0]].groups
    for g in grp:
        if g.type in IONIZ:
            out[(g.atom.chain_id, int(g.atom.res_num))] = (g.type, float(g.pka_value), float(g.model_pka))
    return out


def pose_summary(pdb, step):
    m = _P.get_structure("x", pdb)[0]
    rc = reactive_atom(m, step)
    if rc is None:
        return None
    pk = propka_pkas(pdb)
    if not pk:
        return None
    carbox_shift, carbox_pka, lys_pka, his_dev, nion = [], [], [], [], 0
    for res in m.get_residues():
        rn = res.resname
        if rn not in IONIZ:
            continue
        key = (res.get_parent().id, res.id[1])
        if key not in pk:
            continue
        atoms = [res[n] for n in FUNC[rn] if n in res]
        if not atoms:
            continue
        d = min(float(np.linalg.norm(a.coord - rc)) for a in atoms)
        if d > SHELL:
            continue
        _, pka, model = pk[key]
        nion += 1
        if rn in ("ASP", "GLU"):
            carbox_shift.append(pka - model); carbox_pka.append(pka)
        elif rn == "LYS":
            lys_pka.append(pka)
        elif rn == "HIS":
            his_dev.append(abs(pka - 7.0))
    return dict(
        ag_shift=np.mean(carbox_shift) if carbox_shift else np.nan,
        ag_maxpka=np.max(carbox_pka) if carbox_pka else np.nan,
        lys_minpka=np.min(lys_pka) if lys_pka else np.nan,
        his_dev7=np.min(his_dev) if his_dev else np.nan,
        nion=float(nion),
    )


def enzyme(args):
    e, step, d = args
    pdbs = sorted(glob.glob(f"{d}/*model_*.pdb"))[:N_MODELS]
    if not pdbs:
        return e, step, None
    rows = [s for p in pdbs if (s := pose_summary(p, step))]
    if not rows:
        return e, step, None
    keys = rows[0].keys()
    agg = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        v = v[~np.isnan(v)]
        agg[k] = float(v.mean()) if v.size else np.nan
        agg[k + "_std"] = float(v.std()) if v.size else np.nan
    agg["_n"] = len(rows)
    return e, step, agg


# ---------------- inline evaluation (mirror feature_surfacing.py) ----------------
def evaluate(rows_by_id):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    act, lab = {}, {}
    for r in csv.DictReader(open(ACT_CSV)):
        act[r["id"]] = float(r["activity"]); lab[r["id"]] = int(r["label_gt_sfab"])
    ids = [i for i in act]
    topk = {k: set(sorted(ids, key=lambda i: -act[i])[:k]) for k in (3, 5, 10)}

    def au(y, x):
        return roc_auc_score(y, x) if len(set(y)) == 2 else float("nan")
    cols = sorted({c for v in rows_by_id.values() for c in v if not c.startswith("_")})
    print(f"\n{'feature':22s} {'n':>3s} {'dir':>3s} {'mean':>5s} {'top3':>5s} {'top5':>5s} {'top10':>5s} {'>SfaB':>6s} {'rho':>6s}")
    res = []
    for col in cols:
        xs = [(i, rows_by_id[i][col]) for i in ids
              if i in rows_by_id and col in rows_by_id[i] and not np.isnan(rows_by_id[i][col])]
        if len(xs) < 20:
            continue
        oi = [i for i, _ in xs]; x = [v for _, v in xs]; a = [act[i] for i in oi]
        rho = spearmanr(x, a).correlation
        rho = 0.0 if rho is None or np.isnan(rho) else rho
        xo = x if rho >= 0 else [-v for v in x]
        aur = {k: au([1 if i in topk[k] else 0 for i in oi], xo) for k in (3, 5, 10)}
        aub = au([lab[i] for i in oi], xo)
        vals = [v for v in (aur[3], aur[5], aur[10], aub) if not np.isnan(v)]
        mean = sum(vals) / len(vals) if vals else float("nan")
        res.append((mean, col, len(xs), rho, aur, aub))
    for mean, col, n, rho, aur, aub in sorted(res, reverse=True):
        d = "+" if rho >= 0 else "-"
        print(f"{col:22s} {n:>3d} {d:>3s} {mean:>5.2f} {aur[3]:>5.2f} {aur[5]:>5.2f} {aur[10]:>5.2f} {aub:>6.2f} {rho:>6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--pool", type=int, default=6)
    a = ap.parse_args()
    global N_MODELS
    N_MODELS = a.n
    enz = sorted(os.path.basename(d) for d in glob.glob(f"{S1}/*") if os.path.isdir(d))
    tasks = []
    for e in enz:
        tasks.append((e, 1, s1_dir(e)))
        tasks.append((e, 2, s2_dir(e)))
    print(f"{len(enz)} enzymes x 2 steps = {len(tasks)} jobs, {N_MODELS} poses each, Pool({a.pool})")
    merged = {}
    cov = {1: 0, 2: 0}
    with Pool(a.pool) as pool:
        for e, step, agg in pool.imap_unordered(enzyme, tasks):
            if agg is None:
                continue
            cov[step] += 1
            pre = "s1_" if step == 1 else "s2_"
            merged.setdefault(e, {"id": e})
            for k, v in agg.items():
                if k == "_n":
                    merged[e][pre + "n"] = v
                else:
                    merged[e][pre + k] = v
    rows = list(merged.values())
    allcols = ["id"] + sorted({k for r in rows for k in r if k != "id"})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allcols); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT}  ({len(rows)} enzymes; step1 cov {cov[1]}, step2 cov {cov[2]})")
    rbi = {r["id"]: {k: v for k, v in r.items() if k != "id"} for r in rows}
    evaluate(rbi)


if __name__ == "__main__":
    main()
