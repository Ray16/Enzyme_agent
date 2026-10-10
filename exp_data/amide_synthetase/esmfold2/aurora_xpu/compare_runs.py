#!/usr/bin/env python
"""Compare the Aurora stock/XPU fold of a target against its fast-mode CUDA fold.

The two runs differ in two ways at once -- backend (SDPA/float32 on PVC vs the
kit's bf16 re-association on A100) and nothing else: same checkpoint pin, same
MSA, same seed, same loops/steps/msa_max_depth. So a disagreement here is the
numerics, not the setup.

ESMFold2 is stochastic at inference even with a fixed seed (`lm_dropout` 0.3 is
drawn inside every recycle and the MSA is row-subsampled per loop), so the
honest comparison is ensemble-vs-ensemble: the spread *within* the fast-mode
5-sample set is the yardstick the cross-run distance has to be read against. A
cross-run RMSD inside the within-run spread means the two configurations agree
to the model's own seed-to-seed noise.

usage: compare_runs.py <fast_cif_dir> <stock_cif_dir> [--out report.json]
"""
import argparse
import glob
import itertools
import json
import os

import gemmi
import numpy as np


def load(path):
    """Return (Ca coords by (chain,seqid), {ligand chain: heavy-atom coords})."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    model = st[0]
    ca, lig = {}, {}
    for chain in model:
        poly = chain.get_polymer()
        if len(poly):
            for res in chain:
                at = res.find_atom("CA", "*")
                if at:
                    ca[(chain.name, res.seqid.num)] = [at.pos.x, at.pos.y, at.pos.z]
        else:
            pts = [[a.pos.x, a.pos.y, a.pos.z] for r in chain for a in r
                   if a.element != gemmi.Element("H")]
            if pts:
                lig[chain.name] = np.array(pts)
    return ca, lig


def kabsch_rmsd(p, q):
    """RMSD after optimal superposition of p onto q."""
    p, q = p - p.mean(0), q - q.mean(0)
    v, _, w = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(v @ w))
    r = v @ np.diag([1, 1, d]) @ w
    return float(np.sqrt((((p @ r) - q) ** 2).sum(1).mean()))


def pair(a, b):
    """Ca RMSD on shared residues, plus per-ligand-chain centroid distance
    after superposing on the protein."""
    ca_a, lig_a = a
    ca_b, lig_b = b
    keys = sorted(set(ca_a) & set(ca_b))
    if not keys:
        return None
    pa = np.array([ca_a[k] for k in keys])
    pb = np.array([ca_b[k] for k in keys])
    out = {"n_ca": len(keys), "ca_rmsd": kabsch_rmsd(pa, pb)}

    ca_pa, ca_pb = pa.mean(0), pb.mean(0)
    v, _, w = np.linalg.svd((pa - ca_pa).T @ (pb - ca_pb))
    rot = v @ np.diag([1, 1, np.sign(np.linalg.det(v @ w))]) @ w
    for ch in sorted(set(lig_a) & set(lig_b)):
        ma = (lig_a[ch].mean(0) - ca_pa) @ rot + ca_pb
        out[f"lig_{ch}_centroid_d"] = float(np.linalg.norm(ma - lig_b[ch].mean(0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fast_dir")
    ap.add_argument("stock_dir")
    ap.add_argument("--out")
    args = ap.parse_args()

    fast = [load(p) for p in sorted(glob.glob(os.path.join(args.fast_dir, "*.cif")))]
    stock = [load(p) for p in sorted(glob.glob(os.path.join(args.stock_dir, "*.cif")))]
    if not fast or not stock:
        raise SystemExit(f"need cifs in both dirs (got {len(fast)}, {len(stock)})")

    within = [pair(a, b) for a, b in itertools.combinations(fast, 2)]
    across = [pair(a, b) for a in fast for b in stock]
    within = [w for w in within if w]
    across = [a for a in across if a]

    def summ(rows, key):
        v = [r[key] for r in rows if key in r]
        if not v:
            return None
        return {"n": len(v), "median": round(float(np.median(v)), 3),
                "min": round(float(np.min(v)), 3), "max": round(float(np.max(v)), 3)}

    keys = sorted({k for r in within + across for k in r if k != "n_ca"})
    report = {"n_fast": len(fast), "n_stock": len(stock),
              "within_fast": {k: summ(within, k) for k in keys},
              "fast_vs_stock": {k: summ(across, k) for k in keys}}
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)

    w, a = report["within_fast"].get("ca_rmsd"), report["fast_vs_stock"].get("ca_rmsd")
    if w and a:
        verdict = ("within the fast run's own seed-to-seed spread"
                   if a["median"] <= w["max"] else
                   "OUTSIDE the fast run's seed-to-seed spread")
        print(f"\nCa RMSD fast-vs-stock median {a['median']} A is {verdict} "
              f"(within-fast {w['min']}-{w['max']} A).")


if __name__ == "__main__":
    main()
