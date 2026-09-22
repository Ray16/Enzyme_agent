#!/usr/bin/env python3
"""Re-evaluate structural proxies on the 1-Mg ANL panel re-fold vs the 2-Mg original,
for the amide-synthetase step-1 (adenylation) cofolds.

Goal (user): surface candidates MORE ACTIVE THAN SfaB (label = activity > SfaB), with
emphasis on recovering the top-ranked few. Question: is the previously-best proxy
(near-attack: adipate carboxyl-O within 4 A of ATP Pa, ranked by in-line angle) still
the best on the corrected 1-Mg structures, and how do SASA/burial and active-site
electrostatics compare?

Proxies (all computed over the 50-pose ensemble, chain-agnostic ligand detection):
  near-attack fraction : frac of poses with d(O..Pa)<4 A AND in-line angle>150 deg   [primary]
  mean in-line angle   : mean in-line attack angle among gated (d<4) poses
  reactive-center SASA : burial of adipate reactive carboxyl C + carbonyl O's (lower=buried)
  amine-subsite basic  : # basic-residue N within 6 A of the amine subsite (electrostatics)

Metrics vs the binary "> SfaB" label: AUROC, exact permutation p (few positives), P@k.
Writes a tidy CSV per run + prints a 1-Mg vs 2-Mg comparison and top-k recovery.
"""
import csv, glob, os, sys, warnings
from itertools import combinations
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
sys.path.insert(0, f"{ROOT}/structural_modeling/scripts")
import step1_nac as S  # geom_model: chain-agnostic near-attack geometry
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

ACT_CSV = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
PRED = {
    "1Mg": f"{ROOT}/exp_data/amide_synthetase/boltz2/step1_1mg/results/{{e}}/boltz_results_step1_{{e}}/predictions/step1_{{e}}",
    "2Mg": f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs/step1_{{e}}/boltz_results_step1_{{e}}/predictions/step1_{{e}}",
}
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/reanalysis_1mg_vs_2mg.csv"
D_GATE, ANG_MIN, N_MIN, SASA_GATE = 4.0, 150.0, 5, 6.0
_P = PDBParser(QUIET=True)
_SR = ShrakeRupley()


# ---------- labels ----------
def load_activity():
    act = {}
    for r in csv.DictReader(open(ACT_CSV)):
        try:
            act[r["id"]] = float(r["activity_N66L1_ugml_avg"])
        except (ValueError, KeyError):
            pass
    return act


# ---------- proxy: near-attack geometry (reuse validated code) ----------
def near_attack(pred_dir):
    rows = [g for p in sorted(glob.glob(f"{pred_dir}/*model_*.pdb")) if (g := S.geom_model(p))]
    if not rows:
        return None
    d = np.array([r["d_pa"] for r in rows]); a = np.array([r["ang_pa"] for r in rows])
    gated = d < D_GATE
    return dict(
        n=len(d), min_d=float(d.min()), n_gated=int(gated.sum()),
        na_frac=float(np.mean((d < D_GATE) & (a > ANG_MIN))),
        mean_angle=float(a[gated].mean()) if gated.any() else float("nan"),
    )


# ---------- proxy: reactive-center SASA (burial), composition-based ----------
def _ligands(model):
    """Return (adipate_atoms, phosphate_coords) from a Boltz model, chain-agnostic."""
    het = [a for a in model.get_atoms() if a.get_parent().id[0] != " "]
    Ps = [a.coord for a in het if a.element == "P"]
    carbs = []  # adipate carboxyl C with >=2 close O
    for a in het:
        if a.element != "C":
            continue
        onb = [b for b in het if b.element == "O" and np.linalg.norm(a.coord - b.coord) < 1.7]
        if len(onb) >= 2:
            carbs.append((a, onb))
    return carbs, Ps


def sasa_pose(pdb):
    m = _P.get_structure("x", pdb)[0]
    carbs, Ps = _ligands(m)
    if not carbs or not Ps:
        return None
    rc, onb = min(carbs, key=lambda co: min(np.linalg.norm(co[0].coord - p) for p in Ps))
    dP = float(min(np.linalg.norm(rc.coord - p) for p in Ps))
    _SR.compute(m, level="A")
    return dP, float(rc.sasa) + float(sum(o.sasa for o in onb))


def reactive_sasa(pred_dir):
    per = []
    for pdb in sorted(glob.glob(f"{pred_dir}/*model_*.pdb")):
        try:
            r = sasa_pose(pdb)
        except Exception:
            r = None
        if r:
            per.append(r)
    if not per:
        return None
    gated = [s for (d, s) in per if d < SASA_GATE]
    use = gated if len(gated) >= N_MIN else [s for _, s in per]
    return dict(center_sasa=float(np.mean(use)), n_docked=len(gated))


# ---------- metrics ----------
def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    c = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return c / (len(pos) * len(neg))


def exact_perm_p(scores, labels):
    obs = auroc(scores, labels)
    if obs is None:
        return None
    n, k = len(labels), int(sum(labels))
    ge = tot = 0
    for combo in combinations(range(n), k):
        cs = set(combo)
        lab = [1 if i in cs else 0 for i in range(n)]
        a = auroc(scores, lab)
        tot += 1
        if a >= obs - 1e-12:
            ge += 1
    return obs, ge / tot


def precision_at_k(scores, labels, ks=(3, 5, 10)):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {k: sum(labels[i] for i in order[:k]) for k in ks}


# ---------- driver ----------
def collect(run):
    act = load_activity()
    cands = [e for e in act if e != "SfaB"]
    sfab = act["SfaB"]
    rows = []
    for e in sorted(cands, key=lambda x: int(x[2:]) if x.startswith("AS") else 0):
        pd = PRED[run].format(e=e)
        na = near_attack(pd)
        sa = reactive_sasa(pd)
        if na is None:
            continue
        rows.append(dict(id=e, activity=act[e], label=int(act[e] > sfab),
                         **{f"na_{k}": v for k, v in na.items()},
                         center_sasa=(sa or {}).get("center_sasa", float("nan"))))
    return rows, sfab


def report(run):
    rows, sfab = collect(run)
    if not rows:
        print(f"[{run}] no structures yet"); return None
    n_act = sum(r["label"] for r in rows)
    print(f"\n===== {run}  (n={len(rows)}, active[>SfaB={sfab:.3f}]={n_act}) =====")
    proxies = [
        ("near-attack frac [PRIMARY]", "na_na_frac", True),
        ("mean in-line angle", "na_mean_angle", True),
        ("reactive-center SASA (burial)", "center_sasa", False),
    ]
    labels = [r["label"] for r in rows]
    for name, key, higher in proxies:
        sc = [(r[key] if higher else -r[key]) for r in rows]
        good = [(s, l) for s, l in zip(sc, labels) if s == s]  # drop nan
        if len({l for _, l in good}) < 2:
            print(f"  {name:34s} n/a"); continue
        s2 = [s for s, _ in good]; l2 = [l for _, l in good]
        pr = exact_perm_p(s2, l2)
        pk = precision_at_k(s2, l2)
        au, p = pr
        print(f"  {name:34s} AUROC={au:.3f} perm_p={p:.4f}  "
              f"P@3={pk[3]}/{min(3,n_act)} P@5={pk[5]} P@10={pk[10]}  (n={len(good)})")
    return rows


if __name__ == "__main__":
    r1 = report("1Mg")
    r2 = report("2Mg")
    # write tidy CSV (1Mg)
    if r1:
        cols = list(r1[0].keys())
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(r1)
        print(f"\nwrote {OUT}")
        # top-k recovery of >SfaB actives by primary proxy
        order = sorted(r1, key=lambda r: -r["na_na_frac"])
        print("Top-10 by near-attack frac (1Mg):",
              ", ".join(f"{r['id']}{'*' if r['label'] else ''}" for r in order[:10]))
