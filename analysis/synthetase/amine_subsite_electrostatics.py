#!/usr/bin/env python3
"""Hypothesis family: AMINE-SUBSITE ELECTROSTATICS & GENERAL-BASE GEOMETRY.

Rate-limiting step of the repurposed diacid+HMDA amidation is aminolysis: the incoming
hexamethylenediamine (HMDA) ammonium must be bound/deprotonated and positioned to attack
the activated carbonyl. On the STEP-1 cofolds (A=prot, B=ATP, C/D=Mg, E=adipate) the amine
is absent, so we PROBE the acceptor region: a point placed `offset` A beyond the DISTAL
carboxyl of adipate, along the proximal->distal long axis (offset swept 6/7/8/9 A).

Metrics per probe (decomposed, no composites), mean over docked poses:
  (a) acidO_<off>  : # acidic sidechain O (ASP OD1/OD2, GLU OE1/OE2) within 6 A of probe
                     -> electrostatic complementarity to the ammonium / candidate base.  higher->active
  (b) nearBase_<off>: distance from probe to nearest general-base O/N (ASP/GLU carboxyl O,
                      HIS ND1/NE2).                                              lower->active
  (c) netChg_<off> : net formal charge of residues with ANY sidechain heavy atom within 6 A
                     (ASP/GLU=-1, ARG/LYS=+1, HIS=0). functional amine pocket net-negative. lower->active

Gate: reactive carboxyl-O within 4 A of an ATP P (docked); >=5 gated poses else INCONCLUSIVE.
Benchmark = distance-gated in-line ANGLE (AUROC 0.95, active ranks [1,4,7]). Only 3 actives
(AS9/AS17/AS23) -> treat any hit skeptically.
"""
import csv, glob, os, sys, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hypothesis_battery import spearman, auroc, perm_p, carboxyls, _d, _safe

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
S1 = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/amine_subsite_electrostatics.csv"
ACT_THR = 0.2
OFFSETS = (6.0, 7.0, 8.0, 9.0)
RADIUS = 6.0
CHG = {"ASP": -1, "GLU": -1, "ARG": +1, "LYS": +1, "HIS": 0}
ACID_O = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BASE_ATOMS = ACID_O | {("HIS", "ND1"), ("HIS", "NE2")}
BACKBONE = {"N", "CA", "C", "O", "OXT"}
_P = PDBParser(QUIET=True)


def pose(pdb):
    m = _P.get_structure("x", pdb)[0]
    if "A" not in m or "B" not in m or "E" not in m:
        return None
    Ps = [a for a in m["B"].get_atoms() if a.element == "P"]
    cbx = carboxyls(m["E"])
    if not Ps or len(cbx) < 2:
        return None
    keyfn = lambda co: min(_d(o, p) for o in co[1] for p in Ps)
    react = min(cbx, key=keyfn)
    distal = max(cbx, key=keyfn)
    dP = min(_d(o, p) for o in react[1] for p in Ps)          # near-attack gate
    v = distal[0].coord - react[0].coord
    n = np.linalg.norm(v)
    if n < 1e-3:
        return None
    u = v / n

    # protein sidechain atoms with metadata
    coords, is_acidO, is_base, res_charge_key, sc_key = [], [], [], [], []
    for res in m["A"].get_residues():
        rn = res.get_resname()
        for a in res.get_atoms():
            if a.element == "H":
                continue
            nm = a.get_name()
            coords.append(a.coord)
            is_acidO.append((rn, nm) in ACID_O)
            is_base.append((rn, nm) in BASE_ATOMS)
            side = nm not in BACKBONE
            sc_key.append((rn, res.get_id(), side))
            res_charge_key.append((res.get_id(), CHG.get(rn, 0)) if side else None)
    coords = np.asarray(coords)
    is_acidO = np.asarray(is_acidO); is_base = np.asarray(is_base)

    out = {"dP": dP}
    for off in OFFSETS:
        probe = distal[0].coord + off * u
        dist = np.linalg.norm(coords - probe, axis=1)
        within = dist < RADIUS
        out[f"acidO_{int(off)}"] = int(np.sum(within & is_acidO))
        bd = dist[is_base]
        out[f"nearBase_{int(off)}"] = float(bd.min()) if bd.size else np.nan
        # net charge of residues with any sidechain heavy atom within radius
        resq = {}
        for i in np.where(within)[0]:
            rk = res_charge_key[i]
            if rk is not None:
                resq[rk[0]] = rk[1]
        out[f"netChg_{int(off)}"] = float(sum(resq.values()))
    return out


METRIC_KEYS = [f"{m}_{int(o)}" for o in OFFSETS for m in ("acidO", "nearBase", "netChg")]
DIRN = {}
for o in OFFSETS:
    DIRN[f"acidO_{int(o)}"] = +1
    DIRN[f"nearBase_{int(o)}"] = -1
    DIRN[f"netChg_{int(o)}"] = -1


def process(cfgdir):
    base = os.path.basename(cfgdir); name = base[len("step1_"):]
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = [r for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb"))
           for r in [_safe(pose, pdb)] if r]
    if not per:
        return None
    g = np.array([p["dP"] for p in per]) < 4.0
    row = {"id": name, "n_gated": int(g.sum())}
    if g.sum() >= 5:
        idx = np.where(g)[0]
        for k in METRIC_KEYS:
            vals = [per[i][k] for i in idx]
            row[k] = round(float(np.nanmean(vals)), 3)
    return row


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    with Pool(6) as p:
        res = {r["id"]: r for r in p.map(process, sorted(glob.glob(f"{S1}/step1_*"))) if r}
    ids = sorted(res)
    rows = [{"id": i, "activity": act.get(i), **{k: v for k, v in res[i].items() if k != "id"}} for i in ids]
    cols = ["id", "activity", "n_gated"] + METRIC_KEYS
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["activity"] is not None and r.get("n_gated", 0) >= 5]
    y = [r["activity"] for r in m]
    lab = [1 if v > ACT_THR else 0 for v in y]
    actives = [r["id"] for r, l in zip(m, lab) if l]
    print(f"n scored={len(m)}  actives={sum(lab)} ({actives})  "
          f"[INCONCLUSIVE excluded: {[r['id'] for r in rows if r.get('n_gated',0)<5]}]\n")
    print(f"{'metric':14} {'dir':>3} {'Spearman':>9} {'AUROC':>7} {'perm_p':>7}  active-ranks")
    survivors = []
    for k in METRIC_KEYS:
        d = DIRN[k]
        xn = [r.get(k, np.nan) for r in m]
        xn = [np.nan if v is None else v for v in xn]
        rho = spearman(xn, y) * d
        oriented = [d * v for v in xn]
        au = auroc(oriented, lab); pp = perm_p(oriented, lab)
        order = sorted([r for r in m if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))],
                       key=lambda r: -d * r[k])
        ranks = sorted(order.index(r) + 1 for r in order if r["id"] in actives)
        flag = "  *** SURVIVOR" if (not np.isnan(au) and au >= 0.85 and pp < 0.05) else (
               "  ~ weak" if (not np.isnan(au) and au >= 0.75) else "")
        if flag.strip().startswith("***"):
            survivors.append(k)
        print(f"{k:14} {('+' if d>0 else '-'):>3} {rho:>+9.3f} {au:>7.3f} {pp:>7.4f}  {ranks}{flag}")
    print(f"\nSURVIVORS (AUROC>=0.85 & perm_p<0.05): {survivors or 'NONE'}")
    print(f"benchmark angle proxy: AUROC 0.95, active ranks [1,4,7]")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
