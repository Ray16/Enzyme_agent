#!/usr/bin/env python
"""Score step-2 (amidation) attack geometry: HMDA nitrogen onto the adipoyl-AMP acyl carbon.

Endpoints are fixed in PREREGISTRATION.md and are NOT rediscovered here:
  d2      = distance(N_attack, C_acyl)
  BD2     = angle(N_attack, C_acyl, O_carbonyl)   ideal Burgi-Dunitz ~105 deg
  inline2 = angle(N_attack, C_acyl, O_bridge)     ideal in-line displacement ~180 deg
  fc2     = fraction of poses with d2 < 4 A       (PRIMARY)
  A2      = sum(BD2_i / d2_i) / sum(1 / d2_i)     (secondary, proximity-weighted)

The acyl carbon is located topologically from coordinates, never by atom name, because
Boltz names SMILES-ligand atoms arbitrarily: it is the unique carbon bonded to a
phosphate oxygen that also carries a short (<1.35 A) carbonyl oxygen. The ribose C5' is
also bonded to a phosphate oxygen but has no carbonyl, which is what separates them.

`--selftest` validates that identification against the RFD4 generation input, where the
acyl carbon is independently known to be Q30 (metadata encounter_N_C_distance_A = 3.6755
is exactly the R0...Q30 distance) with carbonyl oxygen Q31.
"""
import json
import sys
from pathlib import Path

import numpy as np

D_GATE = 4.0          # pre-registered primary gate, Angstrom
P_O_MAX = 1.80        # P-O bond
O_C_MAX = 1.60        # O-C single bond
CARBONYL_MAX = 1.35   # C=O; a single C-O(H) is ~1.43 and is excluded


def _records(path):
    """Yield (name, element, chain, xyz) from a PDB or an mmCIF atom_site loop."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".cif", ".mmcif"):
        lines = text.splitlines()
        hdr, start = [], None
        for i, l in enumerate(lines):
            if l.startswith("_atom_site."):
                hdr.append(l.strip())
                start = i + 1
        if start is None:
            raise ValueError(f"{path}: no atom_site loop")
        col = {h.split(".")[1]: k for k, h in enumerate(hdr)}
        for l in lines[start:]:
            if l.startswith("#") or not l.strip():
                break
            f = l.split()
            if len(f) < len(hdr):
                break
            yield (f[col["label_atom_id"]].strip('"'), f[col["type_symbol"]],
                   f[col["label_asym_id"]],
                   np.array([float(f[col["Cartn_x"]]), float(f[col["Cartn_y"]]),
                             float(f[col["Cartn_z"]])]))
    else:
        for l in text.splitlines():
            if not l.startswith(("ATOM", "HETATM")):
                continue
            el = l[76:78].strip() or l[12:16].strip()[0]
            yield (l[12:16].strip(), el, l[21].strip(),
                   np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]))


def _angle(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def find_acyl(atoms):
    """Return (C_acyl, O_carbonyl, O_bridge) atom dicts for the mixed anhydride."""
    ps = [a for a in atoms if a["el"] == "P"]
    if len(ps) != 1:
        raise ValueError(f"expected 1 phosphorus, found {len(ps)}")
    p = ps[0]
    phos_o = [a for a in atoms if a["el"] == "O" and np.linalg.norm(a["xyz"] - p["xyz"]) <= P_O_MAX]
    hits = []
    for ob in phos_o:
        for c in atoms:
            if c["el"] != "C" or np.linalg.norm(c["xyz"] - ob["xyz"]) > O_C_MAX:
                continue
            carbonyls = [o for o in atoms
                         if o["el"] == "O" and o is not ob
                         and np.linalg.norm(o["xyz"] - c["xyz"]) <= CARBONYL_MAX]
            if carbonyls:
                oc = min(carbonyls, key=lambda o: np.linalg.norm(o["xyz"] - c["xyz"]))
                hits.append((c, oc, ob))
    if len(hits) != 1:
        raise ValueError(f"acyl carbon ambiguous: {[(h[0]['name']) for h in hits]}")
    return hits[0]


def pose_geometry(pdb):
    raw = [dict(name=n, el=e, ch=c, xyz=x) for n, e, c, x in _records(pdb)]
    bych = {}
    for a in raw:
        bych.setdefault(a["ch"], []).append(a)

    lig = next(at for at in bych.values() if any(a["el"] == "P" for a in at) and len(at) < 60)
    hmda = [at for ch, at in bych.items()
            if len(at) == 8 and sum(a["el"] == "N" for a in at) == 2
            and all(a["el"] in ("C", "N") for a in at)]
    if len(hmda) != 1:
        raise ValueError(f"{pdb}: HMDA chain ambiguous ({len(hmda)} candidates)")
    hmda = hmda[0]

    c_acyl, o_carbonyl, o_bridge = find_acyl(lig)
    ns = [a for a in hmda if a["el"] == "N"]
    n_attack = min(ns, key=lambda a: np.linalg.norm(a["xyz"] - c_acyl["xyz"]))
    n_other = [a for a in ns if a is not n_attack][0]

    return dict(
        d2=float(np.linalg.norm(n_attack["xyz"] - c_acyl["xyz"])),
        BD2=_angle(n_attack["xyz"], c_acyl["xyz"], o_carbonyl["xyz"]),
        inline2=_angle(n_attack["xyz"], c_acyl["xyz"], o_bridge["xyz"]),
        d_other_N=float(np.linalg.norm(n_other["xyz"] - c_acyl["xyz"])),
        acyl_atom=c_acyl["name"], carbonyl_atom=o_carbonyl["name"], bridge_atom=o_bridge["name"],
        attack_atom=n_attack["name"],
    )


def score_candidate(pred_dir, name):
    pred_dir = Path(pred_dir)
    pdbs = sorted(pred_dir.glob(f"{name}_model_*.pdb"),
                  key=lambda p: int(p.stem.split("_model_")[1]))
    rows = []
    for p in pdbs:
        i = int(p.stem.split("_model_")[1])
        g = pose_geometry(p)
        cf = pred_dir / f"confidence_{name}_model_{i}.json"
        conf = json.loads(cf.read_text()) if cf.exists() else {}
        g.update(sample_index=i, ligand_iptm=conf.get("ligand_iptm"),
                 complex_plddt=conf.get("complex_plddt"))
        rows.append(g)
    if not rows:
        return None
    d = np.array([r["d2"] for r in rows])
    bd = np.array([r["BD2"] for r in rows])
    w = 1 / d
    return dict(
        name=name, n_poses=len(rows),
        fc2=float((d < D_GATE).mean()),
        n_gated=int((d < D_GATE).sum()),
        A2=float((bd * w).sum() / w.sum()),
        harmonic_d2=float(len(d) / w.sum()),
        median_d2=float(np.median(d)),
        min_d2=float(d.min()),
        median_BD2=float(np.median(bd)),
        median_inline2=float(np.median([r["inline2"] for r in rows])),
        mean_ligand_iptm=float(np.mean([r["ligand_iptm"] for r in rows if r["ligand_iptm"] is not None]))
        if any(r["ligand_iptm"] is not None for r in rows) else None,
        poses=rows,
    )


def selftest():
    cif = Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_design_agent/campaigns/"
               "diad_AS9_20261026/inputs_restrained_pose_hypotheses/"
               "AS9_diad_pose0_internal_cap_r2.cif")
    raw = [dict(name=n, el=e, ch=c, xyz=x) for n, e, c, x in _records(cif)]
    lig = [a for a in raw if a["ch"] == "L"]
    c, oc, ob = find_acyl(lig)
    print(f"selftest: acyl={c['name']} carbonyl={oc['name']} bridge={ob['name']}")
    assert c["name"] == "Q30", f"expected acyl Q30, got {c['name']}"
    assert oc["name"] == "Q31", f"expected carbonyl Q31, got {oc['name']}"
    hm = [a for a in raw if a["ch"] == "M"]
    n = min([a for a in hm if a["el"] == "N"], key=lambda a: np.linalg.norm(a["xyz"] - c["xyz"]))
    d = float(np.linalg.norm(n["xyz"] - c["xyz"]))
    print(f"selftest: N_attack={n['name']} d={d:.4f} (metadata encounter_N_C_distance_A 3.6755)")
    # the CIF stores 3 decimal places, so a round-trip against the full-precision
    # metadata value agrees only to ~1e-3 A; tighter than that tests the file format.
    assert n["name"] == "R0", f"expected attacking N R0, got {n['name']}"
    assert abs(d - 3.6755118010965493) < 2e-3, f"geometry mismatch: {d}"
    print("SELFTEST PASS")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        selftest()
    else:
        base = Path(sys.argv[1])
        out = {}
        for job in sys.argv[2:]:
            pd = base / "cofold_out" / job / f"boltz_results_{job}" / "predictions" / job
            r = score_candidate(pd, job)
            if r:
                out[job] = r
                print(f"{job:34s} n={r['n_poses']:2d} fc2={r['fc2']:.2f} "
                      f"({r['n_gated']}/50) A2={r['A2']:.1f} med_d2={r['median_d2']:.2f} "
                      f"min_d2={r['min_d2']:.2f} med_BD2={r['median_BD2']:.1f}")
            else:
                print(f"{job:34s} NO POSES YET")
        (base / "step2_scores.json").write_text(json.dumps(out, indent=2) + "\n")
