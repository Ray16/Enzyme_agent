"""Ligand-axis tunnel-clearance asymmetry (amendment 2).

The bound ligand defines the tunnel axis; we walk the axis in both directions from the
catalytic Thr projection and record how far the tunnel stays open (clearance >= probe)
before it closes. Terminates naturally at tunnel closure -> no length cap, no depth
threshold (the two failure modes of the beam-search descriptor, which saturated on the
ground-truth crystals). Asymmetry is direction-agnostic magnitude, so no leaving-group
labelling is needed.

  clearance(point) = distance to nearest PROTEIN atom surface (vdW), ligand excluded.
  extent(dir)      = arc length from Thr projection until clearance < PROBE (or box edge).
  asymmetry = (ext_long - ext_short)/(ext_long + ext_short);  distal_extent = ext_short.
Direction fixed a priori: higher asymmetry / shorter distal_extent -> higher frac_L1
(exo = short distal tract = monomer release).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_asymmetry as E  # noqa
import thread_chain as tc  # noqa

PROBE = 1.6      # tunnel considered closed when a sphere this big no longer fits (A)
STEP = 0.4
MAXWALK = 30.0   # safety bound (A); real tunnels close well before this


def protein_tree(chains):
    pts, rad = [], []
    for ch, atoms in chains.items():
        for (rs, rn, an, el, xyz) in atoms:
            pts.append(xyz); rad.append(tc.VDW[el])
    return cKDTree(np.array(pts)), np.array(rad)


def ligand_atoms(path, het="LIG"):
    out = {}
    for ln in open(path):
        if ln[:6] == "HETATM" and het in ln[17:20]:
            ch = ln[21]
            el = tc.element(ln[12:16].strip(), ln[76:78].strip())
            if el == "H":
                continue
            out.setdefault(ch, []).append(
                np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
    return {c: np.array(v) for c, v in out.items()}


def clearance(pt, tree, rad):
    d, i = tree.query(pt, workers=1)
    return d - rad[i]


def extent(origin, axis, tree, rad):
    """Walk +axis from origin until clearance < PROBE; return arc length."""
    s = STEP
    while s <= MAXWALK:
        if clearance(origin + axis * s, tree, rad) < PROBE:
            return s
        s += STEP
    return MAXWALK


def site_descriptor(nuc, lig_xyz, tree, rad):
    # axis = principal axis of the ligand copy (PCA)
    c = lig_xyz.mean(0)
    axis = np.linalg.svd(lig_xyz - c)[2][0]
    axis = axis / np.linalg.norm(axis)
    # origin = projection of the catalytic Thr Og1 onto the ligand axis
    origin = c + np.dot(nuc - c, axis) * axis
    e_plus = extent(origin, axis, tree, rad)
    e_minus = extent(origin, -axis, tree, rad)
    lo, hi = min(e_plus, e_minus), max(e_plus, e_minus)
    asym = (hi - lo) / (hi + lo) if (hi + lo) else np.nan
    return dict(e_plus=e_plus, e_minus=e_minus, distal_extent=lo, total_extent=hi + lo,
                asymmetry=asym)


def beta_nuc(atoms):
    first = min(rs for (rs, rn, an, el, xyz) in atoms)
    og = [xyz for (rs, rn, an, el, xyz) in atoms if rs == first and an == "OG1" and rn == "THR"]
    return og[0] if og else None


def score(path, het="LIG"):
    chains = E.load_chains(path)
    tree, rad = protein_tree(chains)
    ligs = ligand_atoms(path, het=het)
    rows = []
    for ch, atoms in chains.items():
        res = {rs for (rs, rn, an, el, xyz) in atoms}
        if len(res) >= 150:
            continue
        nuc = beta_nuc(atoms)
        if nuc is None:
            continue
        # pick the ligand copy nearest this catalytic Thr
        if not ligs:
            continue
        best = min(ligs.values(), key=lambda L: np.linalg.norm(L - nuc, axis=1).min())
        if np.linalg.norm(best - nuc, axis=1).min() > 6.0:
            continue  # no engaged ligand at this site
        d = site_descriptor(nuc, best, tree, rad)
        d["site"] = ch
        rows.append(d)
    return rows


if __name__ == "__main__":
    import json
    p = sys.argv[1]
    het = sys.argv[2] if len(sys.argv) > 2 else "LIG"
    print(json.dumps({"model": Path(p).stem, "rows": score(p, het)}, default=float, indent=2))
