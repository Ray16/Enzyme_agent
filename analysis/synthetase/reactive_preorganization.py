#!/usr/bin/env python3
"""Hypothesis family: REACTIVE-CARBOXYLATE PRE-ORGANIZATION & BIDENTATE CLAMP.

Beyond the mean in-line angle (the validated proxy, AUROC 0.95, ranks [1,4,7]), is the
reactive carboxylate RIGIDLY, PRODUCTIVELY anchored across the Boltz pose ensemble?

STEP-1 structures (A=prot B=ATP C/D=Mg E=adipate). Gate: reactive carboxyl-O <4A of an
ATP P (near-attack), >=5 gated poses or INCONCLUSIVE.

  (a) POSITIONAL PRE-ORG  = RMSF of the reactive-carboxylate O-centroid across gated poses,
      after superposing each pose's protein CA onto a reference pose (frame = the pocket).
      Low spread = preorganized.                                    dir: lower -> active
  (b) BIDENTATE CLAMP     = fraction of gated poses where BOTH reactive O's each have >=1
      protein H-bond DONOR (backbone N, or Arg/Lys/Ser/Thr/Tyr/Asn/Gln/His/Trp sidechain
      donor) within 3.2A -> a two-point grip.                       dir: higher -> active
  (c) APPROACH-DIR PRE-ORG= angular std of the (reactive-O-centroid -> nearest-P) attack
      vector across gated poses, in the same superposed frame.      dir: lower -> active

CRITICAL orthogonality check: pose-consistency was archived once as "just restates the
angle metric." So each metric's Spearman corr to mean_inline_angle is reported; |corr|>0.6
=> NON-ORTHOGONAL => archive regardless of AUROC. Only 3 actives -> be skeptical.
"""
import csv, glob, os, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, Superimposer

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
S1 = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
ANGLE = f"{ROOT}/exp_data/amide_synthetase/analysis/angle_gate_step1_pa.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/reactive_preorganization.csv"
ACT_THR = 0.2
GATE = 4.0
N_MIN = 5
_P = PDBParser(QUIET=True)

DONOR_SC = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"},
            "TYR": {"OH"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "HIS": {"ND1", "NE2"},
            "TRP": {"NE1"}}


def _d(a, b):
    return float(np.linalg.norm(a - b))


def carboxyls(chain):
    at = list(chain.get_atoms())
    out = []
    for c in at:
        if c.element != "C":
            continue
        onb = [o for o in at if o.element == "O" and np.linalg.norm(c.coord - o.coord) < 1.36]
        if len(onb) >= 2:
            out.append((c, onb[:2]))
    return out


def is_donor(atom):
    if atom.get_name() == "N":                       # backbone amide
        return True
    return atom.get_parent().get_resname() in DONOR_SC and atom.get_name() in DONOR_SC[atom.get_parent().get_resname()]


def pose_data(pdb):
    """Return dict for a single pose or None: gated?, reactive O-centroid, both-O coords,
    nearest-P coord, CA list, clamp(bool)."""
    m = _P.get_structure("x", pdb)[0]
    if "A" not in m or "B" not in m or "E" not in m:
        return None
    Ps = [a for a in m["B"].get_atoms() if a.element == "P"]
    cbx = carboxyls(m["E"])
    if not Ps or len(cbx) < 2:
        return None
    Pc = np.array([p.coord for p in Ps])
    keyfn = lambda co: min(np.linalg.norm(o.coord - p) for o in co[1] for p in Pc)
    react = min(cbx, key=keyfn)
    ro = react[1]
    dmat = [[np.linalg.norm(o.coord - p) for p in Pc] for o in ro]
    dP = min(min(r) for r in dmat)                    # near-attack O...P
    ocent = np.mean([o.coord for o in ro], axis=0)
    # nearest P to the centroid
    nearP = Pc[int(np.argmin([np.linalg.norm(ocent - p) for p in Pc]))]
    prot = list(m["A"].get_atoms())
    # (b) bidentate clamp: each reactive O has a donor within 3.2A
    clamp = all(any(is_donor(a) and np.linalg.norm(o.coord - a.coord) < 3.2 for a in prot) for o in ro)
    ca = [a for a in prot if a.get_name() == "CA"]
    return dict(dP=dP, ocent=ocent, nearP=nearP, clamp=clamp, ca=ca, catom=[c.coord for c in ca])


def process(cfgdir):
    base = os.path.basename(cfgdir)
    name = base[len("step1_"):] if base.startswith("step1_") else base
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    poses = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        try:
            r = pose_data(pdb)
        except Exception:
            r = None
        if r:
            poses.append(r)
    if not poses:
        return None
    gated = [p for p in poses if p["dP"] < GATE]
    row = {"id": name, "n": len(poses), "n_gated": len(gated)}
    if len(gated) < N_MIN:
        row.update(posn_rmsf=None, clamp_frac=None, approach_std=None, status="INCONCLUSIVE")
        return row
    # superpose each gated pose's CA onto the reference (first gated), map ligand atoms
    ref = gated[0]
    n_ca = min(len(p["catom"]) for p in gated)
    ref_ca = ref["ca"][:n_ca]
    ocents, avecs = [], []
    for p in gated:
        si = Superimposer()
        try:
            si.set_atoms(ref_ca, p["ca"][:n_ca])
            rot, tran = si.rotran
            oc = p["ocent"] @ rot + tran
            pp = p["nearP"] @ rot + tran
        except Exception:
            oc, pp = p["ocent"], p["nearP"]
        ocents.append(oc)
        v = oc - pp
        nv = np.linalg.norm(v)
        avecs.append(v / nv if nv > 1e-6 else v)
    ocents = np.array(ocents)
    posn_rmsf = float(np.sqrt(np.mean(np.sum((ocents - ocents.mean(0)) ** 2, axis=1))))
    avecs = np.array(avecs)
    mv = avecs.mean(0); mv /= (np.linalg.norm(mv) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(avecs @ mv, -1, 1)))
    approach_std = float(np.std(ang))
    clamp_frac = float(np.mean([p["clamp"] for p in gated]))
    row.update(posn_rmsf=round(posn_rmsf, 3), clamp_frac=round(clamp_frac, 3),
               approach_std=round(approach_std, 2), status="scored")
    return row


# ---- stats (from hypothesis_battery.py) ----
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


DIRN = {"posn_rmsf": -1, "clamp_frac": +1, "approach_std": -1}


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    ang = {}
    for r in csv.DictReader(open(ANGLE)):
        v = r.get("mean_inline_angle", "")
        ang[r["id"]] = float(v) if v not in ("", "None", None) else None

    cfgdirs = sorted(glob.glob(f"{S1}/step1_*"))
    with Pool(6) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        r["activity"] = act.get(r["id"])
        r["angle"] = ang.get(r["id"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["id", "n_gated", "posn_rmsf", "clamp_frac", "approach_std", "status", "activity", "angle"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["status"] == "scored" and r["activity"] is not None]
    y = [r["activity"] for r in m]
    lab = [1 if v > ACT_THR else 0 for v in y]
    actives = [r["id"] for r, l in zip(m, lab) if l]
    print(f"n_scored={len(m)}  actives={sum(lab)} ({actives})  "
          f"INCONCLUSIVE={[r['id'] for r in rows if r['status']!='scored']}\n")
    print(f"{'metric':14} {'dir':>3} {'Spearman':>9} {'AUROC':>7} {'perm_p':>7} {'corr2angle':>10}  ranks  disposition")
    keep = []
    for k, d in DIRN.items():
        xs = [r[k] for r in m]
        rho = spearman(xs, y) * d
        oriented = [d * v for v in xs]
        au = auroc(oriented, lab); pp = perm_p(oriented, lab)
        c2a = spearman(xs, [r["angle"] for r in m])
        order = sorted(m, key=lambda r: -d * r[k])
        ranks = sorted(order.index(r) + 1 for r in order if r["id"] in actives)
        if abs(c2a) > 0.6:
            disp = "ARCHIVE (non-orthogonal to angle)"
        elif np.isnan(au) or au < 0.85 or pp >= 0.05:
            disp = "ARCHIVE (weak/wrong-signed)"
        else:
            disp = "*** SURVIVOR"; keep.append(k)
        print(f"{k:14} {('+' if d>0 else '-'):>3} {rho:>+9.3f} {au:>7.3f} {pp:>7.4f} {c2a:>+10.3f}  {ranks}  {disp}")
    print(f"\nSURVIVORS: {keep or 'NONE'}   (benchmark angle proxy: AUROC 0.95, ranks [1,4,7])")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
