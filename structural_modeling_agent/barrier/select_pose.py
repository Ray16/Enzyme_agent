"""Uniform cofold pose selection for a reaction-barrier panel. For each enzyme, among its
N cofold poses pick the one best poised for the modeled reaction: the substrate nucleophile
(carboxylate O) closest to the electrophile (ATP Palpha) with a reasonably in-line attack
angle (O...Palpha-O3A). This selects a hypothesis seed; it is not ensemble averaging
and does not include the population cost of reaching the reactive pose. Pure geometry
(CPU, no GPU). Docking propensity (how MANY poses are near-attack) is a separate axis; this
just picks the single representative reactive pose per enzyme.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
from .prep_cofold import _read_pdb_atoms, METALS


def _classify(pdb):
    rows = _read_pdb_atoms(pdb)
    groups = {}
    for r in (x for x in rows if x["rec"] == "HETATM"):
        groups.setdefault(r["chain"], []).append(r)
    atp = metal = sub = None
    for g in groups.values():
        names = {a["name"] for a in g}
        if "PA" in names and "PG" in names:
            atp = g
        elif len(g) == 1 and g[0]["elem"] in METALS:
            metal = g
        else:
            sub = g if sub is None else sub + g
    return atp, metal, sub


def _nac(pdb):
    """Return (min_d, angle_at_min_d) for substrate-O nearest ATP Palpha; None if unclassifiable."""
    atp, metal, sub = _classify(pdb)
    if not (atp and sub):
        return None
    xyz = lambda r: np.array([r["x"], r["y"], r["z"]])
    try:
        pa = next(xyz(r) for r in atp if r["name"] == "PA")
        o3a = next((xyz(r) for r in atp if r["name"] == "O3A"), None)
    except StopIteration:
        return None
    subO = [xyz(r) for r in sub if r["elem"] == "O"]
    if not subO:
        return None
    ds = [float(np.linalg.norm(o - pa)) for o in subO]
    k = int(np.argmin(ds))
    ang = None
    if o3a is not None:
        a = subO[k] - pa; b = o3a - pa
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        ang = float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))
    return ds[k], ang


def select(enzyme_id, results_base, angle_min=120.0):
    """Pick the best near-attack pose for one enzyme. Prefer min d among poses with
    angle >= angle_min (in-line); fall back to global min d if none qualify."""
    patt = (f"{results_base}/{enzyme_id}/boltz_results_step1_{enzyme_id}/"
            f"predictions/step1_{enzyme_id}/step1_{enzyme_id}_model_*.pdb")
    rows = []
    for pdb in sorted(glob.glob(patt)):
        r = _nac(pdb)
        if r is None:
            continue
        d, ang = r
        rows.append(dict(pdb=pdb, model=Path(pdb).stem.split("_model_")[-1], d=d, angle=ang))
    if not rows:
        return None
    inline = [r for r in rows if r["angle"] is not None and r["angle"] >= angle_min]
    pool = inline if inline else rows
    best = min(pool, key=lambda r: r["d"])
    best["n_poses"] = len(rows)
    best["n_inline"] = len(inline)
    best["selector"] = "min_d|angle>=%.0f" % angle_min if inline else "min_d(fallback)"
    return best


if __name__ == "__main__":
    import sys, json
    base = sys.argv[1]
    ids = sys.argv[2:] if len(sys.argv) > 2 else None
    if not ids:
        ids = sorted(p.name for p in Path(base).iterdir() if p.is_dir())
    out = {}
    for i in ids:
        b = select(i, base)
        if b:
            out[i] = b
            print(f"{i:6s} model_{b['model']:>3s}  d={b['d']:.2f}  ang={b['angle'] if b['angle'] else 0:.0f}  "
                  f"({b['n_inline']}/{b['n_poses']} in-line)  {b['selector']}", flush=True)
        else:
            print(f"{i:6s} NO VALID POSE", flush=True)
    print("JSON:" + json.dumps(out))
