#!/usr/bin/env python3
"""Mechanism-aware step-1 near-attack geometry for the discovery candidate panel.

All 81 candidates are ATP-dependent amide ligases (EC 6.3.-), but they split into two
activation sub-mechanisms that attack DIFFERENT ATP phosphates:

  acyl-adenylate (-> acyl-AMP + PPi):  electrophile = Pa (PA),  scissile bridge = O3A
      families: Aerobactin NIS synthetases (IucA/IucC), any ANL/AMP-binding
  acyl-phosphate (-> acyl-P + ADP):    electrophile = Pg (PG),  scissile bridge = O3B
      families: glutamine synthetase (all types), ATP-grasp, RimK/LysX, Pur*, PylC, ...

Nucleophile = an adipate (LIG chain E) carboxylate oxygen (found geometrically, so
naming-independent). Near-attack: d(O_nuc, P) < 4.0 A AND angle(O_nuc, P, bridgeO) > 150.

For every model we compute BOTH (Pa and Pg) distance+angle; the mechanism-appropriate
one is the PRIMARY score, the other is reported so the geometry can validate the
annotation-based mechanism call. Aggregates over the 50 diffusion samples per enzyme.
No experimental activity exists for these candidates -> output is a ranked prediction.
"""
import csv, glob, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "common"))
from geometry import angle_deg  # shared helper

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/structural_modeling"
RES = f"{ROOT}/results/output_configs"          # pulled cofold outputs
REPS = f"{ROOT}/representatives.csv"
OUT = f"{ROOT}/results/step1_nac"
D_CUT, A_CUT = 4.0, 150.0                        # DesD convention

# InterPro-family -> activation mechanism (which ATP phosphate is attacked)
ADENYLATE_KEYS = ("iuca", "iucc", "aerobactin", "amp-binding", "anl")  # -> Pa / O3A


def mechanism(family):
    f = (family or "").lower()
    return "adenylate" if any(k in f for k in ADENYLATE_KEYS) else "phosphate"


def parse_pdb(path):
    atoms = {}
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        name = ln[12:16].strip(); resname = ln[17:20].strip(); chain = ln[21]
        elem = (ln[76:78].strip() or name[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        atoms.setdefault((chain, resname), []).append((name, elem, xyz))
    return atoms


def carboxylate_oxygens(lst):
    C = [(n, x) for (n, e, x) in lst if e == "C"]
    O = [x for (n, e, x) in lst if e == "O"]
    out = []
    for _, cx in C:
        nb = [ox for ox in O if np.linalg.norm(cx - ox) < 1.6]
        if len(nb) >= 2:
            out.extend(nb)
    return out


def geom_model(pdb):
    """Return dict of both-phosphate distance+angle for one model, or None."""
    atoms = parse_pdb(pdb)
    P = {}
    for (ch, rn), lst in atoms.items():
        nm = {n: x for (n, e, x) in lst}
        if "PA" in nm and "PG" in nm:               # ATP
            P = nm; break
    sub = None
    for (ch, rn), lst in atoms.items():
        if rn != "LIG":
            continue
        if any(n == "PA" for (n, e, x) in lst):      # skip ATP
            continue
        if carboxylate_oxygens(lst):                 # adipate
            sub = lst; break
    if not P or sub is None or "PA" not in P or "PG" not in P:
        return None
    cb = carboxylate_oxygens(sub)
    res = {}
    for tag, pk, bk in (("pa", "PA", "O3A"), ("pg", "PG", "O3B")):
        p = P.get(pk); b = P.get(bk)
        if p is None:
            continue
        on = min(cb, key=lambda ox: np.linalg.norm(ox - p))
        res[f"d_{tag}"] = float(np.linalg.norm(on - p))
        res[f"ang_{tag}"] = angle_deg(on, p, b) if b is not None else float("nan")
    return res or None


def process_config(args):
    cfgdir, mech = args
    name = os.path.basename(cfgdir)[len("step1_"):]
    pred = f"{cfgdir}/boltz_results_step1_{name}/predictions/step1_{name}"
    rows = [g for pdb in sorted(glob.glob(f"{pred}/*model_*.pdb"))
            if (g := geom_model(pdb)) is not None]
    if not rows:
        return None
    tag = "pa" if mech == "adenylate" else "pg"
    d = np.array([r[f"d_{tag}"] for r in rows])
    a = np.array([r[f"ang_{tag}"] for r in rows])
    # the OTHER phosphate, for validating the mechanism call
    otag = "pg" if tag == "pa" else "pa"
    od = np.array([r.get(f"d_{otag}", np.nan) for r in rows])
    npass = int(np.sum((d < D_CUT) & (a > A_CUT)))
    return {
        "id": name, "mechanism": mech, "reactive_P": tag.upper(), "n_models": len(d),
        "nac_pass": npass, "nac_frac": round(npass / len(d), 3),
        "min_d": round(float(d.min()), 2), "median_d": round(float(np.median(d)), 2),
        "ang_at_min_d": round(float(a[int(d.argmin())]), 1), "max_angle": round(float(a.max()), 1),
        "other_P_min_d": round(float(np.nanmin(od)), 2) if np.isfinite(od).any() else None,
    }


def main():
    fam = {r["accession"]: r.get("interpro_family", "") for r in csv.DictReader(open(REPS))}
    gene = {r["accession"]: r.get("gene", "") for r in csv.DictReader(open(REPS))}
    cfgdirs = sorted(glob.glob(f"{RES}/step1_*"))
    work = [(c, mechanism(fam.get(os.path.basename(c)[len("step1_"):], ""))) for c in cfgdirs]
    with Pool(min(len(work), 32) or 1) as pool:
        rows = [r for r in pool.map(process_config, work) if r is not None]
    for r in rows:
        r["gene"] = gene.get(r["id"], "")
    # rank by predicted competence: NAC fraction, then closest approach
    rows.sort(key=lambda r: (-r["nac_frac"], r["min_d"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["id", "gene", "mechanism", "reactive_P", "n_models", "nac_pass", "nac_frac",
            "min_d", "median_d", "ang_at_min_d", "max_angle", "other_P_min_d"]
    with open(f"{OUT}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    n_ad = sum(r["mechanism"] == "adenylate" for r in rows)
    print(f"Analyzed {len(rows)} enzymes ({n_ad} adenylate/Pa, {len(rows)-n_ad} phosphate/Pg); "
          f"{sum(r['n_models'] for r in rows)} models.")
    print(f"NAC pass: d<{D_CUT} & angle>{A_CUT} vs mechanism-appropriate phosphate\n")
    print(f"{'id':11s} {'gene':9s} {'mech':10s} {'P':3s} {'nac':>5s} {'min_d':>6s} {'ang':>6s} {'other_d':>7s}")
    for r in rows[:20]:
        print(f"{r['id']:11s} {r['gene']:9s} {r['mechanism']:10s} {r['reactive_P']:3s} "
              f"{r['nac_frac']:>5.2f} {r['min_d']:>6.2f} {r['ang_at_min_d']:>6.1f} "
              f"{str(r['other_P_min_d']):>7s}")
    print(f"\nWrote {OUT}.csv")


if __name__ == "__main__":
    main()
