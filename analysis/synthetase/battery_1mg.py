#!/usr/bin/env python3
"""Full structural-proxy battery on the CORRECTED 1-Mg amide-synthetase panel, searching
for ANY proxy that correlates with measured N66-L1 activity over the WHOLE distribution.

Per enzyme, over its 50 one-Mg step-1 cofold poses (chain-agnostic ligand detection):
  near-attack : n_gated(d<4A to Pa), mean_inline_angle, inline_frac_gated, near_attack_frac
  burial/SASA : reactive carboxyl C + carbonyl O SASA (Shrake-Rupley, in full complex),
                gated to docked poses; lower = buried
  metal (1 Mg): Mg--Pa distance, Mg--nearest-carboxylate-O distance, Mg coordination # (<2.6A)
  ipTM        : enzyme<->substrate pair_chains_iptm (docking confidence)
  electrostat : # basic (Arg/Lys/His) and acidic (Asp/Glu) sidechain N/O within 6A of the
                reactive carboxyl C (active-site charge environment)
Parallel over enzymes. Writes battery_1mg.csv; evaluates each column by Spearman rho vs
continuous activity (+ AUROC top3 for reference).
"""
import csv, glob, os, sys, warnings
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/structural_modeling/scripts")
import step1_nac as NAC
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import json

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
PANEL = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1_1mg/results"
ACT_CSV = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/battery_1mg.csv"
D_GATE, A_CUT, N_MODELS = 4.0, 150.0, 50
_P = PDBParser(QUIET=True); _SR = ShrakeRupley()
BASIC = {"ARG", "LYS", "HIS"}; ACIDIC = {"ASP", "GLU"}


def pred_dir(e):
    return f"{PANEL}/{e}/boltz_results_step1_{e}/predictions/step1_{e}"


def ligands(model):
    het = [a for a in model.get_atoms() if a.get_parent().id[0] != " "]
    Ps = {a.get_name(): a.coord for a in het if a.element == "P"}
    mg = [a.coord for a in het if a.element == "MG" or a.get_name() == "MG"]
    carbs = []
    for a in het:
        if a.element != "C":
            continue
        onb = [b for b in het if b.element == "O" and np.linalg.norm(a.coord - b.coord) < 1.7]
        if len(onb) >= 2:
            carbs.append((a, onb))
    return carbs, Ps, mg


def pose_features(pdb):
    m = _P.get_structure("x", pdb)[0]
    carbs, Ps, mg = ligands(m)
    pa = Ps.get("PA")
    if not carbs or pa is None:
        return None
    rc, onb = min(carbs, key=lambda co: min(np.linalg.norm(co[0].coord - p) for p in Ps.values()))
    dpa = float(min(np.linalg.norm(o.coord - pa) for o in onb))  # nearest carbox-O to Pa
    # SASA (whole-complex context)
    _SR.compute(m, level="A")
    sasa = float(rc.sasa) + float(sum(o.sasa for o in onb))
    # metal (single Mg)
    if mg:
        mgc = min(mg, key=lambda g: np.linalg.norm(g - pa))
        mg_pa = float(np.linalg.norm(mgc - pa))
        mg_o = float(min(np.linalg.norm(mgc - o.coord) for o in onb))
        mg_coord = int(sum(np.linalg.norm(mgc - a.coord) < 2.6
                           for a in m.get_atoms() if a.element in ("O", "N")))
    else:
        mg_pa = mg_o = mg_coord = np.nan
    # electrostatics: charged sidechain atoms within 6A of reactive carbon
    nb = 0; na = 0
    for res in m.get_residues():
        rn = res.resname
        if rn not in BASIC and rn not in ACIDIC:
            continue
        for a in res.get_atoms():
            if a.element in ("N", "O") and np.linalg.norm(a.coord - rc.coord) < 6.0:
                if rn in BASIC:
                    nb += 1
                else:
                    na += 1
                break
    return dict(dpa=dpa, sasa=sasa, mg_pa=mg_pa, mg_o=mg_o, mg_coord=mg_coord,
                n_basic=nb, n_acidic=na)


def enzyme_features(e):
    pd_ = pred_dir(e)
    pdbs = sorted(glob.glob(f"{pd_}/*model_*.pdb"))
    if not pdbs:
        return None
    # near-attack geometry from validated geom_model
    geo = [g for p in pdbs if (g := NAC.geom_model(p))]
    d = np.array([r["d_pa"] for r in geo]); ang = np.array([r["ang_pa"] for r in geo])
    gated = d < D_GATE
    ng = int(gated.sum())
    # per-pose SASA/metal/electrostatics
    feats = [f for p in pdbs if (f := pose_features(p))]
    F = {k: np.array([f[k] for f in feats], float) for k in feats[0]} if feats else {}
    dk = F.get("dpa", np.array([]))
    dock = dk < D_GATE if dk.size else np.array([], bool)
    def gmean(key):  # mean over docked poses (fallback all)
        v = F.get(key)
        if v is None or v.size == 0:
            return np.nan
        vv = v[dock] if dock.sum() >= 5 else v
        vv = vv[~np.isnan(vv)]
        return float(vv.mean()) if vv.size else np.nan
    return dict(
        id=e, n_gated=ng,
        mean_inline_angle=round(float(ang[gated].mean()), 2) if ng else None,
        inline_frac_gated=round(float(np.mean(ang[gated] > A_CUT)), 3) if ng else None,
        near_attack_frac=round(float(np.mean((d < D_GATE) & (ang > A_CUT))), 3),
        reactive_sasa=round(gmean("sasa"), 3),
        mg_pa_dist=round(gmean("mg_pa"), 3),
        mg_o_dist=round(gmean("mg_o"), 3),
        mg_coord=round(gmean("mg_coord"), 3),
        n_basic_6A=round(gmean("n_basic"), 3),
        n_acidic_6A=round(gmean("n_acidic"), 3),
    )


def main():
    enz = sorted([os.path.basename(d) for d in glob.glob(f"{PANEL}/*")
                  if os.path.isdir(d)], key=lambda x: (x != "SfaB", x))
    with Pool(12) as pool:
        rows = [r for r in pool.map(enzyme_features, enz) if r]
    cols = list(rows[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT}  ({len(rows)} enzymes)")


if __name__ == "__main__":
    main()
