#!/usr/bin/env python3
"""Autocleavage near-attack geometry for Ntn-hydrolase precursors.

Hypothesis: nylon-hydrolase activity is gated by AUTOCLEAVAGE (maturation). The catalytic
Thr's own Oγ attacks the carbonyl of the PRECEDING residue (the Asn-Thr scissile bond),
an intramolecular N->O acyl shift that exposes the mature N-terminal nucleophile.

This is INTRAMOLECULAR -> enzyme-intrinsic, so it sidesteps the substrate-docking problem
that killed the substrate near-attack proxy. We measure, in the uncleaved precursor cofold:
  d  = Thr(sp) OG1  ->  C (carbonyl) of residue sp-1
  bd = Buergi-Dunitz angle OG1 - C - O(carbonyl of sp-1), ideal ~107 deg
Near-attack: d < 3.5 A and 90 < bd < 120.  Aggregated over the 50 diffusion poses.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "common"))
from geometry import angle_deg

ROOT = os.path.dirname(__file__)
RES = f"{ROOT}/results/output_configs"
META = json.load(open(f"{ROOT}/meta.json"))
D_CUT = 3.5
BD_LO, BD_HI = 90.0, 120.0


def atoms_by_res(pdb):
    """chain A: resseq -> {atomname: xyz}"""
    d = {}
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")) or ln[21] != "A":
            continue
        try:
            rs = int(ln[22:26])
        except ValueError:
            continue
        nm = ln[12:16].strip()
        d.setdefault(rs, {})[nm] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return d


def geom(pdb, sp):
    a = atoms_by_res(pdb)
    thr = a.get(sp, {}); prev = a.get(sp - 1, {})
    if "OG1" not in thr or "C" not in prev or "O" not in prev:
        return None
    og1 = thr["OG1"]; c = prev["C"]; o = prev["O"]
    return float(np.linalg.norm(og1 - c)), angle_deg(og1, c, o)


def main():
    print(f"{'enzyme':8} {'L2act':>6} {'Thr':>4} {'n':>3} {'minD':>5} {'medD':>5} {'bd@minD':>8} {'NAfrac':>6} {'meanBD(d<3.5)':>13}")
    for cfg in sorted(glob.glob(f"{RES}/prec_*")):
        e = os.path.basename(cfg)[len("prec_"):]
        if e not in META:
            continue
        sp = META[e]["thr"]
        pred = glob.glob(f"{cfg}/boltz_results_*/predictions/*")
        if not pred:
            continue
        rows = [g for pdb in sorted(glob.glob(f"{pred[0]}/*model_*.pdb")) if (g := geom(pdb, sp))]
        if not rows:
            print(f"{e:8} {META[e]['act']:>6} {sp:>4}  no geometry (Thr/prev atoms missing)")
            continue
        d = np.array([r[0] for r in rows]); bd = np.array([r[1] for r in rows])
        na = (d < D_CUT) & (bd > BD_LO) & (bd < BD_HI)
        close = d < D_CUT
        print(f"{e:8} {META[e]['act']:>6} {sp:>4} {len(d):>3} {d.min():>5.2f} {np.median(d):>5.2f} "
              f"{bd[d.argmin()]:>8.1f} {np.mean(na):>6.2f} "
              f"{np.mean(bd[close]) if close.any() else float('nan'):>13.1f}")


if __name__ == "__main__":
    main()
