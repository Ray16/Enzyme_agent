#!/usr/bin/env python3
"""Score the native-substrate control: does NAC reproduce the Fig 2C substrate order
WITHIN each enzyme? Same protein/MSA/ATP/Mg/affinity across a config set; only the
substrate SMILES changes, so this isolates substrate geometry from all cross-enzyme
confounders (expression, stability, kcat).

For each config nat_<enz>_<substrate> we compute both-phosphate near-attack geometry
(reusing step1_nac.geom_model) over the 50 diffusion samples, plus the ATP x ligand
interface iptm for confidence gating (SfaB taught us a low-iptm pose = docking failure).
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "common"))
import step1_nac as s

RES = os.path.join(os.path.dirname(__file__), "results", "output_configs")
D_CUT, A_CUT = 4.0, 150.0

# Fig 2C measured molar yields (approx). Diacid preference (collapsing diamine):
#   DdaG: succinate >> glutarate > adipate ; native acyl donor = fumarate (C4 unsat)
#   SfaB: glutarate >= succinate > adipate ; native acyl donor = 5-chlorovalerate (C5 mono)
GROUND_TRUTH = {
    "DdaG": {"fumarate": "native", "succinate": 80, "glutarate": 19, "adipate": 1},
    "SfaB": {"chlorovalerate": "native", "glutarate": 34, "succinate": 31, "adipate": 6},
}


def score_config(cfgdir):
    name = os.path.basename(cfgdir)  # nat_<enz>_<sub>
    inner = glob.glob(f"{cfgdir}/boltz_results_*/predictions/*")
    if not inner:
        return None
    pred = inner[0]
    pdbs = sorted(glob.glob(f"{pred}/*model_*.pdb"))
    rows = [g for p in pdbs if (g := s.geom_model(p)) is not None]
    if not rows:
        return None
    out = {"config": name, "n": len(rows)}
    for tag in ("pa", "pg"):
        d = np.array([r[f"d_{tag}"] for r in rows])
        a = np.array([r[f"ang_{tag}"] for r in rows])
        out[f"{tag}_frac"] = round(float(np.sum((d < D_CUT) & (a > A_CUT)) / len(d)), 3)
        out[f"{tag}_mind"] = round(float(d.min()), 2)
    # ATP(chain1) x ligand(chain4) interface iptm, averaged over models
    cj = sorted(glob.glob(f"{pred}/confidence_*model_*.json"))
    ai = [json.load(open(c)).get("pair_chains_iptm", {}).get("1", {}).get("4") for c in cj]
    ai = [x for x in ai if x is not None]
    out["atp_lig_iptm"] = round(float(np.mean(ai)), 3) if ai else None
    return out


def main():
    cfgs = sorted(glob.glob(f"{RES}/nat_*"))
    res = {}
    for c in cfgs:
        r = score_config(c)
        if not r:
            continue
        _, enz, sub = r["config"].split("_", 2)
        res.setdefault(enz, {})[sub] = r

    for enz, subs in res.items():
        gt = GROUND_TRUTH.get(enz, {})
        print(f"\n===== {enz} — does NAC track the measured substrate order? =====")
        print(f"{'substrate':16} {'yield%':>7} {'Pa_frac':>8} {'Pa_mind':>8} {'Pg_frac':>8} {'atp_lig_iptm':>12}")
        # sort by measured yield desc (native first)
        def key(sub):
            v = gt.get(sub, -1)
            return (0, 0) if v == "native" else (1, -(v if isinstance(v, (int, float)) else -1))
        for sub in sorted(subs, key=key):
            r = subs[sub]
            y = gt.get(sub, "?")
            print(f"{sub:16} {str(y):>7} {r['pa_frac']:>8} {r['pa_mind']:>8} {r['pg_frac']:>8} {str(r['atp_lig_iptm']):>12}")
        # verdict: Spearman between measured yield and Pa_frac (diacids only)
        diac = [(sub, gt[sub], subs[sub]["pa_frac"]) for sub in subs
                if isinstance(gt.get(sub), (int, float))]
        if len(diac) >= 2:
            ys = np.array([d[1] for d in diac]); ns = np.array([d[2] for d in diac])
            rx = np.argsort(np.argsort(ys)); ry = np.argsort(np.argsort(ns))
            rho = np.corrcoef(rx, ry)[0, 1] if len(diac) > 1 else float("nan")
            print(f"  -> yield-vs-NAC(Pa) Spearman over {len(diac)} diacids: rho = {rho:+.2f}")


if __name__ == "__main__":
    main()
