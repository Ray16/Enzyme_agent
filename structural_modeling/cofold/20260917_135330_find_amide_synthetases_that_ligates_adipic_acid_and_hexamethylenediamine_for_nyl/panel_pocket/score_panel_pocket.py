#!/usr/bin/env python3
"""Full-panel pocket-restrained NAC vs measured activity.

The decisive test: does the ATP-anchored pocket-restrained NAC correlate with the
50-enzyme panel's measured N66 (adipate) activity, where UNRESTRAINED NAC gave rho~0.09?

Also reports the non-circularity check (do dead enzymes stay low?) and the
confidence-gated correlation (drop low ligand-iptm poses).
"""
import glob, json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "common"))
import step1_nac as s

RES = os.path.join(os.path.dirname(__file__), "results", "output_configs")
PANEL = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/assay_data/panel_activity.csv"
UNRESTR = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1/results/near_attack_geometry.csv"
D_CUT, A_CUT = 4.0, 150.0


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def score(cfgdir):
    inner = glob.glob(f"{cfgdir}/boltz_results_*/predictions/*")
    if not inner:
        return None
    pred = inner[0]
    rows = [g for p in sorted(glob.glob(f"{pred}/*model_*.pdb")) if (g := s.geom_model(p))]
    if not rows:
        return None
    d = np.array([r["d_pa"] for r in rows]); a = np.array([r["ang_pa"] for r in rows])
    cj = sorted(glob.glob(f"{pred}/confidence_*model_*.json"))
    ai = [json.load(open(c)).get("pair_chains_iptm", {}).get("1", {}).get("4") for c in cj]
    ai = np.array([x for x in ai if x is not None])
    return {
        "nac": round(float(np.sum((d < D_CUT) & (a > A_CUT)) / len(d)), 3),
        "min_d": round(float(d.min()), 2),
        "iptm": round(float(ai.mean()), 3) if len(ai) else None,
    }


def main():
    act = {r["id"]: float(r["activity_N66L1_ugml_avg"]) for r in csv.DictReader(open(PANEL))}
    unr = {r["id"]: float(r["nac_frac"]) for r in csv.DictReader(open(UNRESTR))}
    data = []
    for cfg in sorted(glob.glob(f"{RES}/pk_*")):
        eid = os.path.basename(cfg)[len("pk_"):]
        r = score(cfg)
        if r and eid in act:
            data.append((eid, act[eid], r))
    if not data:
        print("no scored configs yet"); return

    ids = [d[0] for d in data]
    a = np.array([d[1] for d in data])
    nac = np.array([d[2]["nac"] for d in data])
    iptm = np.array([d[2]["iptm"] if d[2]["iptm"] is not None else 0 for d in data])
    unr_nac = np.array([unr.get(i, np.nan) for i in ids])

    print(f"n = {len(data)} panel enzymes (pocket-restrained, adipate)\n")
    print("=== KEY: does restrained NAC correlate with measured activity? ===")
    print(f"  UNRESTRAINED NAC vs activity : rho = {spearman(unr_nac, a):+.3f}   (baseline)")
    print(f"  RESTRAINED   NAC vs activity : rho = {spearman(nac, a):+.3f}")
    gate = iptm > 0.6
    if gate.sum() > 5:
        print(f"  RESTRAINED NAC, iptm-gated   : rho = {spearman(nac[gate], a[gate]):+.3f}  (n={int(gate.sum())})")

    # precision@k
    order = np.argsort(-nac)
    active = a > 0.1
    for k in (5, 8, 10):
        print(f"  precision@{k} (restrained NAC): {int(active[order[:k]].sum())}/{k}  (base {int(active.sum())}/{len(a)})")

    # non-circularity: dead-enzyme NAC distribution
    dead = a < 0.001
    print(f"\n=== non-circularity: do DEAD enzymes stay low? ===")
    print(f"  dead (act<0.001, n={int(dead.sum())}): mean restrained NAC = {nac[dead].mean():.2f}, max = {nac[dead].max():.2f}")
    print(f"  active (act>0.1,  n={int(active.sum())}): mean restrained NAC = {nac[active].mean():.2f}")

    # top by activity, restrained vs unrestrained
    print(f"\n{'id':6} {'activity':>9} {'unrestr':>8} {'RESTR_nac':>10} {'min_d':>6} {'iptm':>6}")
    for eid, av, r in sorted(data, key=lambda x: -x[1])[:15]:
        print(f"{eid:6} {av:>9.3f} {str(unr.get(eid,'?')):>8} {r['nac']:>10} {r['min_d']:>6} {str(r['iptm']):>6}")


if __name__ == "__main__":
    main()
