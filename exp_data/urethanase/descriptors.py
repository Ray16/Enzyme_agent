#!/usr/bin/env python3
"""The five pre-registered structural descriptors, computed per enzyme.

Defined in PREREGISTRATION.md before any structure was folded. This module only
implements them; it must not add, drop or retune one, because the selection
budget (five) is what makes the corrected null affordable at n=11.

Atoms, per pose:
  O_nuc   catalytic Ser OG / Thr OG1, from catalytic_residues.json
  C_carb  the carbamate carbonyl carbon  (C bonded to >=2 O and >=1 N)
  O_carb  its carbonyl oxygen            (the double-bonded O on C_carb)
  O_lg    the alkoxy leaving-group oxygen (the O on C_carb that carries carbon)

The leaving group is the ALKOXY oxygen, not the nitrogen: the alcohol departs
first and the carbamic acid decomposes afterwards.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

try:
    import gemmi
except ImportError as exc:  # pragma: no cover
    raise SystemExit("gemmi is required") from exc

BOND = 1.65     # A, generous covalent cutoff for ligand connectivity
NEAR_D = 3.5    # A, near-attack distance ceiling (pre-registered)
BD_LO, BD_HI = 95.0, 125.0   # deg, Burgi-Dunitz window (pre-registered)
HB = 3.5        # A, oxyanion-hole donor cutoff


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _pose_atoms(path: str, nuc_resnum: int):
    """Return (O_nuc, C_carb, O_carb, O_lg, donors) or None if not resolvable."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    o_nuc = None
    lig = []       # (element, pos)
    donors = []    # candidate H-bond donor heavy atoms (N, O) on the protein
    for ch in st[0]:
        for res in ch:
            if res.het_flag == "H" and res.name != "HOH":
                lig += [(a.element.name, np.array(a.pos.tolist())) for a in res]
                continue
            for a in res:
                if res.seqid.num == nuc_resnum and a.name in ("OG", "OG1"):
                    o_nuc = np.array(a.pos.tolist())
                if a.element.name in ("N", "O"):
                    donors.append(np.array(a.pos.tolist()))
    if o_nuc is None or not lig:
        return None

    def nbrs(i):
        p = lig[i][1]
        return [(e, q) for j, (e, q) in enumerate(lig)
                if j != i and np.linalg.norm(q - p) < BOND]

    c_idx = None
    for i, (e, _) in enumerate(lig):
        if e != "C":
            continue
        nb = nbrs(i)
        if sum(1 for e2, _ in nb if e2 == "O") >= 2 and sum(1 for e2, _ in nb if e2 == "N") >= 1:
            c_idx = i
    if c_idx is None:
        return None
    c_carb = lig[c_idx][1]

    # carbonyl O is terminal (only the acyl carbon within bonding distance);
    # the alkoxy O additionally carries a carbon substituent.
    o_carb = o_lg = None
    for e, q in nbrs(c_idx):
        if e != "O":
            continue
        heavy = [1 for e2, r in lig
                 if not np.allclose(r, q) and np.linalg.norm(r - q) < BOND]
        if len(heavy) <= 1:
            o_carb = q
        else:
            o_lg = q
    if o_carb is None or o_lg is None:
        return None
    return o_nuc, c_carb, o_carb, o_lg, donors


def find_poses(name: str, out_root: str) -> list[str]:
    """Every pose for one enzyme, across all output layouts used by the fleet.

    The fold fleet writes each attempt to its own directory (a retry must not
    inherit a half-written `processed/` cache), and earlier runs used a shared
    root. Globbing only one layout silently returns zero poses for an enzyme
    that is in fact complete, which reads downstream as "not folded yet" --
    indistinguishable from a real gap. Resolve all layouts, then de-duplicate by
    pose filename so a re-fold cannot inflate the ensemble past its true size.
    """
    pats = [
        f"{out_root}/boltz_results_{name}/predictions/{name}/*.cif",         # shared
        f"{out_root}/{name}/boltz_results_{name}/predictions/{name}/*.cif",  # per-enzyme
        f"{out_root}/{name}/*/boltz_results_{name}/predictions/{name}/*.cif",  # per-attempt
    ]
    by_pose: dict[str, str] = {}
    for p in pats:
        for f in sorted(glob.glob(p)):
            # keep the newest attempt's copy of each pose
            by_pose[Path(f).name] = f
    return [by_pose[k] for k in sorted(by_pose)]


def descriptors_for(name: str, out_root: str, nuc_resnum: int) -> dict:
    poses = find_poses(name, out_root)
    d_nuc, bd, lg, oxy = [], [], [], []
    for p in poses:
        got = _pose_atoms(p, nuc_resnum)
        if got is None:
            continue
        o_nuc, c_carb, o_carb, o_lg, donors = got
        d = float(np.linalg.norm(c_carb - o_nuc))
        d_nuc.append(d)
        bd.append(_angle(o_nuc, c_carb, o_carb))        # Burgi-Dunitz
        lg.append(_angle(o_nuc, c_carb, o_lg))          # anti-periplanar departure
        oxy.append(sum(1 for q in donors if np.linalg.norm(q - o_carb) < HB))
    n = len(d_nuc)
    if n == 0:
        return {"enzyme": name, "n_poses": 0, "n_resolved": 0}
    a_d, a_bd = np.array(d_nuc), np.array(bd)
    near = (a_d <= NEAR_D) & (a_bd >= BD_LO) & (a_bd <= BD_HI)
    # D2: proximity-weighted attack angle -- angle weighted toward poses whose
    # nucleophile is close, so a good angle in a far pose cannot dominate.
    w = np.exp(-np.maximum(a_d - NEAR_D, 0.0))
    return {
        "enzyme": name,
        "n_poses": len(poses),
        "n_resolved": n,
        "D1_near_attack_fraction": float(near.mean()),
        "D2_proximity_weighted_angle": float(np.sum(w * a_bd) / np.sum(w)),
        "D3_d_nuc_carbonyl": float(np.median(a_d)),
        "D4_leaving_group_alignment": float(np.median(lg)),
        "D5_oxyanion_occupancy": float(np.mean(oxy)),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="exp_data/urethanase/panel/cofold_out")
    ap.add_argument("--residues", default="exp_data/urethanase/panel/catalytic_residues.json")
    ap.add_argument("--min-poses", type=int, default=50,
                    help="refuse an enzyme with fewer poses than this")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cat = json.loads(Path(args.residues).read_text())
    rows = []
    # Enumerate from the catalytic-residue table, not from whatever directories
    # happen to exist: the panel is defined by the enzymes we resolved a
    # nucleophile for, so an enzyme missing from disk must show up as absent
    # rather than quietly dropping out of the cohort.
    for name in sorted(cat):
        info = cat.get(name, {})
        if not info.get("position"):
            print(f"  SKIP {name}: no catalytic residue assigned")
            continue
        r = descriptors_for(name, args.out_root, info["position"])
        if r["n_poses"] < args.min_poses:
            print(f"  SKIP {name}: {r['n_poses']} poses < {args.min_poses} (incomplete ensemble)")
            continue
        r["catalytic_residue"] = f"{info['residue']}{info['position']}"
        r["residue_confidence"] = info.get("confidence")
        rows.append(r)
    for r in rows:
        print(f"  {r['enzyme']:30s} poses={r['n_poses']} resolved={r['n_resolved']:>3} "
              f"D1={r['D1_near_attack_fraction']:.3f} D2={r['D2_proximity_weighted_angle']:6.1f} "
              f"D3={r['D3_d_nuc_carbonyl']:5.2f} D4={r['D4_leaving_group_alignment']:6.1f} "
              f"D5={r['D5_oxyanion_occupancy']:.2f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {len(rows)} rows -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
