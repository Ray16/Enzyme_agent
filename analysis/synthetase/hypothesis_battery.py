#!/usr/bin/env python3
"""Broad battery of NEW geometric hypotheses vs measured N66 activity (49-member
SfaB panel, 3 actives AS9/AS17/AS23). Each metric is decomposed (no composites).
Weak / wrong-signed metrics get archived; only survivors are promoted.

Already dead (not retested): raw NAC, distance-as-score, pocket-restrained cofold,
iptm, enzyme-diacid ps_iptm, pose consistency, approach-cone openness, Mg-carboxylate
coordination, step-1 adipate burial (overfit), step-2 intermediate burial (dead),
step-2 amine near-attack gate. Survivor: distance-gated in-line ANGLE.

NEW hypotheses tested here:
 STEP-1 (A=prot B=ATP C/D=Mg E=adipate); gate: reactive carboxyl C <4A of an ATP P (docked), >=5 poses
  H1 oxyanion H-bonds   : # protein N/O within 3.5A of the REACTIVE carboxyl O's
                          (TS/oxyanion stabilization). higher -> active?
  H2 distal-COO SASA    : solvent exposure of the DISTAL carboxyl (marks the acceptor
                          end). higher(exposed) -> active? (room for the amine)
  H3 acceptor channel   : open space 3A beyond the distal carboxyl along the adipate
                          long axis (min dist to protein). higher(open) -> active?
  H4 distal-proximal    : distal_SASA - reactive_SASA (asymmetric presentation). higher -> active?
 STEP-2 (A=prot B=adipyl-AMP C=HMDA D=Mg); gate: acyl C <5A of protein (intermediate bound)
  H5 amine min-approach : min over poses of (nearest HMDA N -> acyl C) dist. lower -> active?
  H6 amine approach frac: fraction of poses with an HMDA N within 6A of acyl C. higher -> active?
  H7 adenine burial     : mean SASA of the 5 adenine N atoms (AMP pocket engagement,
                          HMDA removed). lower(buried) -> active?
  H8 acyl oxyanion      : # protein N/O within 3.5A of the acyl CARBONYL O. higher -> active?
"""
import csv, glob, os, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
S1 = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
S2 = f"{ROOT}/exp_data/amide_synthetase/boltz2/step2/unrestrained/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/hypothesis_battery.csv"
ACT_THR = 0.2
_P = PDBParser(QUIET=True)
_SR = ShrakeRupley()


def _d(a, b):
    return float(np.linalg.norm(a.coord - b.coord))


def carboxyls(chain):
    """list of (C, [O,O]) for carboxyl-type carbons (>=2 O neighbours <1.36A)."""
    at = list(chain.get_atoms())
    out = []
    for c in at:
        if c.element != "C":
            continue
        onb = [o for o in at if o.element == "O" and _d(c, o) < 1.36]
        if len(onb) >= 2:
            out.append((c, onb[:2]))
    return out


# ---------------- STEP 1 ----------------
def pose_s1(pdb):
    m = _P.get_structure("x", pdb)[0]
    if "A" not in m or "B" not in m or "E" not in m:
        return None
    prot = list(m["A"].get_atoms())
    Ps = [a for a in m["B"].get_atoms() if a.element == "P"]
    cbx = carboxyls(m["E"])
    if not Ps or len(cbx) < 2:
        return None
    keyfn = lambda co: min(_d(o, p) for o in co[1] for p in Ps)   # nearest carboxyl-O to any P
    react = min(cbx, key=keyfn)
    distal = max(cbx, key=keyfn)
    rc, ro = react; dc, do = distal
    dP = min(_d(o, p) for o in ro for p in Ps)            # near-attack O...P gate (matches angle proxy)
    # H1 oxyanion: protein N/O within 3.5A of reactive carboxyl O's
    oxy = sum(1 for a in prot if a.element in ("N", "O")
              for o in ro if _d(a, o) < 3.5)
    # H3 acceptor channel: probe 3A beyond distal C along proximal->distal axis
    v = dc.coord - rc.coord
    n = np.linalg.norm(v)
    chan = np.nan
    if n > 1e-3:
        probe = dc.coord + 3.0 * v / n
        chan = float(min(np.linalg.norm(a.coord - probe) for a in prot))
    _SR.compute(m, level="A")
    react_sasa = float(sum(o.sasa for o in ro))
    distal_sasa = float(sum(o.sasa for o in do))
    return dict(dP=dP, oxy=oxy, react_sasa=react_sasa, distal_sasa=distal_sasa, chan=chan)


# ---------------- STEP 2 ----------------
def find_acyl(B):
    Ps = [a for a in B if a.element == "P"]
    if not Ps:
        return None
    P = Ps[0]
    for o in [o for o in B if o.element == "O" and _d(o, P) < 1.75]:
        for c in [c for c in B if c.element == "C" and _d(c, o) < 1.6]:
            dbl = [oo for oo in B if oo.element == "O" and oo is not o and _d(c, oo) < 1.36]
            if dbl:
                return c, dbl[0]
    return None


def pose_s2(pdb):
    m = _P.get_structure("x", pdb)[0]
    if "A" not in m or "B" not in m:
        return None
    B = list(m["B"].get_atoms())
    r = find_acyl(B)
    if not r:
        return None
    acyl, carO = r
    prot = list(m["A"].get_atoms())
    dprot = min(_d(acyl, a) for a in prot)                # bound gate
    amineN = [a for a in m["C"].get_atoms() if a.element == "N"] if "C" in m else []
    d_amine = min((_d(acyl, nn) for nn in amineN), default=np.nan)
    oxy = sum(1 for a in prot if a.element in ("N", "O") and _d(a, carO) < 3.5)
    adenN = [a for a in B if a.element == "N"]            # 5 adenine N (adipate has none)
    if "C" in m:
        m.detach_child("C")                               # AMP pocket burial w/o mobile amine
    _SR.compute(m, level="A")
    aden_sasa = float(np.mean([a.sasa for a in adenN])) if adenN else np.nan
    return dict(dprot=dprot, d_amine=d_amine, oxy=oxy, aden_sasa=aden_sasa)


def process_s1(cfgdir):
    base = os.path.basename(cfgdir); name = base[len("step1_"):]
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = [r for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb"))
           for r in [_safe(pose_s1, pdb)] if r]
    if not per:
        return None
    g = np.array([p["dP"] for p in per]) < 4.0
    row = {"id": name, "n1": len(per), "n1_gated": int(g.sum())}
    if g.sum() >= 5:
        idx = np.where(g)[0]
        row["H1_oxyanion"] = round(np.mean([per[i]["oxy"] for i in idx]), 2)
        row["H2_distalSASA"] = round(np.mean([per[i]["distal_sasa"] for i in idx]), 2)
        row["H3_channel"] = round(np.nanmean([per[i]["chan"] for i in idx]), 2)
        row["H4_distMinusReact"] = round(np.mean([per[i]["distal_sasa"] - per[i]["react_sasa"] for i in idx]), 2)
    return row


def process_s2(cfgdir):
    base = os.path.basename(cfgdir); name = base[len("step2u_"):]
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = [r for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb"))
           for r in [_safe(pose_s2, pdb)] if r]
    if not per:
        return None
    g = np.array([p["dprot"] for p in per]) < 5.0
    row = {"id": name, "n2": len(per), "n2_bound": int(g.sum())}
    if g.sum() >= 5:
        idx = np.where(g)[0]
        row["H5_amineMinApproach"] = round(float(np.nanmin([per[i]["d_amine"] for i in idx])), 2)
        row["H6_amineFrac6A"] = round(float(np.mean([per[i]["d_amine"] < 6.0 for i in idx])), 3)
        row["H7_adenineBurial"] = round(float(np.nanmean([per[i]["aden_sasa"] for i in idx])), 2)
        row["H8_acylOxyanion"] = round(float(np.mean([per[i]["oxy"] for i in idx])), 2)
    return row


def _safe(fn, pdb):
    try:
        return fn(pdb)
    except Exception:
        return None


# ---------------- stats ----------------
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


# metric -> +1 if higher predicts active, -1 if lower predicts active
DIRN = {"H1_oxyanion": +1, "H2_distalSASA": +1, "H3_channel": +1, "H4_distMinusReact": +1,
        "H5_amineMinApproach": -1, "H6_amineFrac6A": +1, "H7_adenineBurial": -1, "H8_acylOxyanion": +1}


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    with Pool(24) as p:
        r1 = {r["id"]: r for r in p.map(process_s1, sorted(glob.glob(f"{S1}/step1_*"))) if r}
        r2 = {r["id"]: r for r in p.map(process_s2, sorted(glob.glob(f"{S2}/step2u_*"))) if r}
    ids = sorted(set(r1) | set(r2))
    rows = []
    for i in ids:
        row = {"id": i, "activity": act.get(i)}
        row.update({k: v for k, v in r1.get(i, {}).items() if k != "id"})
        row.update({k: v for k, v in r2.get(i, {}).items() if k != "id"})
        rows.append(row)
    cols = ["id", "activity", "n1_gated", "H1_oxyanion", "H2_distalSASA", "H3_channel",
            "H4_distMinusReact", "n2_bound", "H5_amineMinApproach", "H6_amineFrac6A",
            "H7_adenineBurial", "H8_acylOxyanion"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["activity"] is not None]
    y = [r["activity"] for r in m]
    lab = [1 if v > ACT_THR else 0 for v in y]
    actives = [r["id"] for r, l in zip(m, lab) if l]
    print(f"n={len(m)}  actives={sum(lab)} ({actives})\n")
    print(f"{'hypothesis':20} {'dir':>3} {'Spearman':>9} {'AUROC':>7} {'perm_p':>7}  active-ranks(best->worst dir)")
    survivors = []
    for k in DIRN:
        d = DIRN[k]
        x = [r.get(k, np.nan) for r in m]
        xn = [(v if v is not None else np.nan) for v in x]
        rho = spearman(xn, y) * d
        oriented = [d * (v if v is not None else np.nan) for v in xn]
        au = auroc(oriented, lab); pp = perm_p(oriented, lab)
        order = sorted([r for r in m if r.get(k) is not None],
                       key=lambda r: -d * r[k])            # best (predicted-active) first
        ranks = sorted(order.index(r) + 1 for r in order if r["id"] in actives)
        flag = ""
        if not np.isnan(au) and (au >= 0.85 and pp < 0.05):
            flag = "  *** SURVIVOR"; survivors.append(k)
        elif not np.isnan(au) and au >= 0.75:
            flag = "  ~ weak"
        print(f"{k:20} {('+' if d>0 else '-'):>3} {rho:>+9.3f} {au:>7.3f} {pp:>7.4f}  {ranks}{flag}")
    print(f"\nSURVIVORS (AUROC>=0.85 & perm_p<0.05): {survivors or 'NONE'}")
    print(f"angle proxy benchmark: AUROC 0.95, ranks [1,4,7]")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
