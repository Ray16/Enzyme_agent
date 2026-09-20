#!/usr/bin/env python3
"""Pocket-restrained NAC validation.

Question: does a loose ATP-anchored pocket constraint (substrate forced into the
pocket, but reactive geometry NOT imposed) rescue docking for genuinely-active
enzyme+substrate pairs WITHOUT also handing near-attack to inactive ones?

Non-circularity test: POSITIVES (real activity, failed unrestrained) should improve;
NEGATIVES (dead enzymes) must stay poor. If everything gets good NAC, the constraint
manufactures the answer and restrained NAC is worthless.
"""
import glob, json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "common"))
import step1_nac as s

RES = os.path.join(os.path.dirname(__file__), "results", "output_configs")
D_CUT, A_CUT = 4.0, 150.0

# (config, label, known_activity, unrestrained_note)
META = {
    "pk_SfaB_glutarate": ("POS", 34,    "unrestr: FAILED dock (10.0A, iptm0.42)"),
    "pk_SfaB_succinate": ("POS", 31,    "unrestr: FAILED dock (10.0A, iptm0.42)"),
    "pk_AS9_adipate":    ("POS", 9.28,  "unrestr: NAC 0.72 (3.37A)"),
    "pk_AS31_adipate":   ("NEG", 0.0,   "unrestr: dead"),
    "pk_AS20_adipate":   ("NEG", 0.0001,"unrestr: dead"),
    "pk_AS28_adipate":   ("NEG", 0.0001,"unrestr: dead"),
}


def score(cfgdir):
    inner = glob.glob(f"{cfgdir}/boltz_results_*/predictions/*")
    if not inner:
        return None
    pred = inner[0]
    pdbs = sorted(glob.glob(f"{pred}/*model_*.pdb"))
    rows = [g for p in pdbs if (g := s.geom_model(p)) is not None]
    if not rows:
        return None
    d = np.array([r["d_pa"] for r in rows]); a = np.array([r["ang_pa"] for r in rows])
    cj = sorted(glob.glob(f"{pred}/confidence_*model_*.json"))
    ai = [json.load(open(c)).get("pair_chains_iptm", {}).get("1", {}).get("4") for c in cj]
    ai = [x for x in ai if x is not None]
    close = a[d < D_CUT]
    return {
        "pa_frac": round(float(np.sum((d < D_CUT) & (a > A_CUT)) / len(d)), 3),
        "min_d": round(float(d.min()), 2),
        "best_ang_close": round(float(close.max()), 1) if len(close) else None,
        "atp_lig_iptm": round(float(np.mean(ai)), 3) if ai else None,
        "n": len(rows),
    }


def main():
    rows = []
    for cfg in sorted(glob.glob(f"{RES}/pk_*")):
        name = os.path.basename(cfg)
        r = score(cfg)
        if r:
            rows.append((name, r))
    print("POCKET-RESTRAINED NAC — does forcing the substrate in rescue actives WITHOUT rescuing dead enzymes?\n")
    print(f"{'config':20} {'label':5} {'act':>7} {'RESTR pa_frac':>13} {'min_d':>6} {'ang':>6} {'iptm':>6}  unrestrained")
    for name, r in rows:
        label, act, note = META.get(name, ("?", "?", ""))
        print(f"{name:20} {label:5} {str(act):>7} {r['pa_frac']:>13} {r['min_d']:>6} "
              f"{str(r['best_ang_close']):>6} {str(r['atp_lig_iptm']):>6}  {note}")
    # verdict
    pos = [r["pa_frac"] for n, r in rows if META.get(n, ("?",))[0] == "POS"]
    neg = [r["pa_frac"] for n, r in rows if META.get(n, ("?",))[0] == "NEG"]
    print(f"\nPOS mean NAC = {np.mean(pos):.2f}  |  NEG mean NAC = {np.mean(neg):.2f}")
    print("VERDICT:")
    if pos and neg:
        if np.mean(pos) > np.mean(neg) + 0.2:
            print("  POS > NEG -> constraint places substrate & geometry DISCRIMINATES (restrained NAC viable)")
        elif max(neg) > 0.5:
            print("  NEG also high -> constraint MANUFACTURES near-attack (restrained NAC NOT valid)")
        else:
            print("  POS not clearly > NEG -> restraint doesn't rescue actives either (dead end)")


if __name__ == "__main__":
    main()
