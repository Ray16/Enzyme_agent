#!/usr/bin/env python3
"""Acceptor-tunnel geometry hypothesis (SfaB panel, step-1 cofolds).

Q: does an ENCLOSED channel of the right length (~8-9A, a C6 diamine's reach) run from
the reactive center's acceptor face out to solvent? Productive tunnel = has an exit
(room for the incoming diamine) AND is walled (excludes water, suppresses hydrolysis).

Done with a DEFINED PROBE (not raw contact counts; the contact-count/approach-cone
version already died from param instability -> here we sweep params and DEMAND stability).

STEP-1: A=prot B=ATP C/D=Mg E=adipate. Reactive carboxyl = carboxyl whose O is nearest
an ATP P; distal carboxyl = the other. Tunnel axis = reactive_C -> distal_C (the adipate
long axis; the acceptor amide forms at the distal end). Gate: reactive carboxyl-O within
4A of an ATP P (docked), >=5 poses, else INCONCLUSIVE.

Per pose:
  L_open  : distance walked outward from distal_C along axis (1A steps) before a 1.4A
            probe clashes (min dist to protein < CLASH) -- i.e. how far the channel is open.
  exit    : 1 if the walk reaches SOLVENT (no protein within 8A in the forward hemisphere)
            before clashing -> a real opening; 0 if it dead-ends.
  encl    : mean over the first WIN A (1A steps) of the walled fraction of 8 lateral rays
            (ring radius R, ⟂ axis): walled if a protein atom is within WALL of the ring
            point. High = tube-like (walled); low = open bowl.
Aggregated per enzyme over gated poses (mean). Hypotheses:
  encl  higher -> active (walled water-excluding tube)
  L_open higher -> active (room for the diamine)   [non-monotonic caveat noted]
  exit  higher -> active (opening for the distal amine)
Sensitivity: encl at tight/mid/loose params; L_open at 3 clash cutoffs. Sign/AUROC must
be STABLE across params or the metric is dead (contact-count disease).
"""
import csv, glob, os, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
S1 = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/acceptor_tunnel_geometry.csv"
ACT_THR = 0.2
GATE = 4.0
NMIN = 5
STEP = 1.0
SMAX = 15.0
SOLV = 8.0          # forward-hemisphere solvent radius
ENCL = [("tight", 3.0, 2.7, 4.0), ("mid", 3.5, 3.0, 6.0), ("loose", 4.0, 3.5, 8.0)]  # (name,R,WALL,WIN)
CLASHES = [2.5, 2.7, 3.0]
_P = PDBParser(QUIET=True)


def _d(a, b):
    return float(np.linalg.norm(a.coord - b.coord))


def carboxyls(chain):
    at = list(chain.get_atoms())
    out = []
    for c in at:
        if c.element != "C":
            continue
        onb = [o for o in at if o.element == "O" and _d(c, o) < 1.36]
        if len(onb) >= 2:
            out.append((c, onb[:2]))
    return out


def ortho_basis(axis):
    tmp = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, tmp); e1 /= (np.linalg.norm(e1) + 1e-9)
    e2 = np.cross(axis, e1); e2 /= (np.linalg.norm(e2) + 1e-9)
    return e1, e2


def pose(pdb):
    m = _P.get_structure("x", pdb)[0]
    if "A" not in m or "B" not in m or "E" not in m:
        return None
    prot = np.array([a.coord for a in m["A"].get_atoms() if a.element != "H"])
    Ps = [a for a in m["B"].get_atoms() if a.element == "P"]
    cbx = carboxyls(m["E"])
    if not len(prot) or not Ps or len(cbx) < 2:
        return None
    keyfn = lambda co: min(_d(o, p) for o in co[1] for p in Ps)
    react = min(cbx, key=keyfn); distal = max(cbx, key=keyfn)
    dP = keyfn(react)
    rc = np.array(react[0].coord); dc = np.array(distal[0].coord)
    v = dc - rc; n = np.linalg.norm(v)
    if n < 1e-3:
        return None
    axis = v / n
    tree = cKDTree(prot)
    e1, e2 = ortho_basis(axis)
    dirs = [np.cos(t) * e1 + np.sin(t) * e2 for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]

    # forward walk from distal_C: L_open per clash cutoff, plus exit (clash=2.7 primary)
    steps = np.arange(STEP, SMAX + 1e-6, STEP)
    pts = np.array([dc + s * axis for s in steps])
    dmin_fwd = tree.query(pts)[0]                         # nearest protein to each walk point
    Lopen = {}
    for c in CLASHES:
        blocked = np.where(dmin_fwd < c)[0]
        Lopen[c] = float(steps[blocked[0]] - STEP) if len(blocked) else float(SMAX)
    # exit: first solvent step (no protein within SOLV forward) before first clash(2.7)
    exit_flag = 0
    firstclash = np.where(dmin_fwd < 2.7)[0]
    fc = firstclash[0] if len(firstclash) else len(steps)
    for i in range(len(steps)):
        if i >= fc:
            break
        p = pts[i]
        idx = tree.query_ball_point(p, SOLV)
        fwd = [j for j in idx if np.dot(prot[j] - p, axis) > 0]
        if not fwd:
            exit_flag = 1
            break

    # enclosure at each param setting (lateral rays)
    enc = {}
    maxwin = max(w for _, _, _, w in ENCL)
    wsteps = np.arange(STEP, maxwin + 1e-6, STEP)
    for name, R, WALL, WIN in ENCL:
        fr = []
        for s in wsteps:
            if s > WIN:
                break
            p = dc + s * axis
            ring = np.array([p + R * dd for dd in dirs])
            wd = tree.query(ring)[0]
            fr.append(float(np.mean(wd < WALL)))
        enc[name] = float(np.mean(fr)) if fr else np.nan
    out = {"dP": dP, "exit": exit_flag}
    out.update({f"L{c}": Lopen[c] for c in CLASHES})
    out.update({f"enc_{name}": enc[name] for name, *_ in ENCL})
    return out


def _safe(pdb):
    try:
        return pose(pdb)
    except Exception:
        return None


def process(cfgdir):
    base = os.path.basename(cfgdir); name = base[len("step1_"):]
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = [r for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")) for r in [_safe(pdb)] if r]
    if not per:
        return None
    g = np.array([p["dP"] for p in per]) < GATE
    row = {"id": name, "n": len(per), "n_gated": int(g.sum())}
    if g.sum() >= NMIN:
        idx = np.where(g)[0]
        keys = ["exit"] + [f"L{c}" for c in CLASHES] + [f"enc_{n}" for n, *_ in ENCL]
        for k in keys:
            row[k] = round(float(np.nanmean([per[i][k] for i in idx])), 3)
    return row


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))


def auroc(s, l):
    s, l = np.asarray(s, float), np.asarray(l, int)
    m = ~np.isnan(s); s, l = s[m], l[m]
    pos, neg = s[l == 1], s[l == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def perm_p(s, l, n=20000, seed=0):
    s, l = np.asarray(s, float), np.asarray(l, int)
    m = ~np.isnan(s); s, l = s[m], l[m]
    obs = auroc(s, l); rng = np.random.default_rng(seed)
    return (sum(auroc(s, rng.permutation(l)) >= obs for _ in range(n)) + 1) / (n + 1)


DIRN = {"exit": +1, "L2.5": +1, "L2.7": +1, "L3.0": +1,
        "enc_tight": +1, "enc_mid": +1, "enc_loose": +1}


def report(rows):
    m = [r for r in rows if r.get("activity") is not None and r.get("n_gated", 0) >= NMIN]
    y = [r["activity"] for r in m]
    lab = [1 if v > ACT_THR else 0 for v in y]
    actives = [r["id"] for r, l in zip(m, lab) if l]
    print(f"n_scored={len(m)}  actives={sum(lab)} ({actives})  "
          f"INCONCLUSIVE={[r['id'] for r in rows if r.get('n_gated',0)<NMIN]}\n")
    print(f"{'metric':11} {'dir':>3} {'Spearman':>9} {'AUROC':>7} {'perm_p':>7}  active-ranks")
    res = {}
    for k, d in DIRN.items():
        x = [r.get(k, np.nan) for r in m]
        rho = spearman(x, y) * d
        oriented = [d * (v if v is not None else np.nan) for v in x]
        au = auroc(oriented, lab); pp = perm_p(oriented, lab)
        order = sorted([r for r in m if r.get(k) is not None], key=lambda r: -d * r[k])
        ranks = sorted(order.index(r) + 1 for r in order if r["id"] in actives)
        res[k] = au
        print(f"{k:11} {('+' if d>0 else '-'):>3} {rho:>+9.3f} {au:>7.3f} {pp:>7.4f}  {ranks}")
    # sensitivity
    encs = [res[k] for k in ("enc_tight", "enc_mid", "enc_loose")]
    Ls = [res[k] for k in ("L2.5", "L2.7", "L3.0")]
    print(f"\nSENSITIVITY  enclosure AUROC across params (tight/mid/loose): "
          f"{[round(e,3) for e in encs]}  range={max(encs)-min(encs):.3f}")
    print(f"             L_open AUROC across clash (2.5/2.7/3.0): "
          f"{[round(e,3) for e in Ls]}  range={max(Ls)-min(Ls):.3f}")
    best = max(res, key=lambda k: res[k] if not np.isnan(res[k]) else -1)
    stable_enc = (max(encs) - min(encs)) < 0.10
    survivor = (res[best] >= 0.85) and stable_enc
    print(f"\nbest metric={best} AUROC={res[best]:.3f}  enclosure_stable={stable_enc}")
    print(f"angle proxy benchmark: AUROC 0.95, ranks [1,4,7]")
    print("VERDICT: " + ("SURVIVOR" if survivor else "DEAD (weak/unstable)"))
    return survivor


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    with Pool(6) as p:
        rows = [r for r in p.map(process, sorted(glob.glob(f"{S1}/step1_*"))) if r]
    for r in rows:
        r["activity"] = act.get(r["id"])
    cols = ["id", "activity", "n_gated", "exit", "L2.5", "L2.7", "L3.0",
            "enc_tight", "enc_mid", "enc_loose"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    report(rows)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
