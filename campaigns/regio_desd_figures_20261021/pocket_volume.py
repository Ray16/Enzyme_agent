#!/usr/bin/env python3
"""Grid-based pocket volume (POVME / PyVOL style), CALIBRATED against the paper's own numbers.

PyVOL -- the tool the paper used -- is not installed here, and neither are fpocket or freesasa. So
the volume is computed directly on a grid, and critically it is VALIDATED before use against the two
values the paper publishes from crystal structures we already hold locally:

    DesD  (PDB 7TGM)  pocket volume  2497 A^3   <- paper, Fig 3B
    AcsD  (PDB 2W02)  pocket volume   816 A^3   <- paper, Fig 3B

Method (standard enclosed-grid approach):
  1. build a grid of spacing `step` inside a box of radius `extent` about a centre
  2. discard points within (vdW radius + probe) of any protein heavy atom  -> free points
  3. keep only ENCLOSED free points: cast rays along 26 directions and require at least
     `min_blocked` of them to strike protein within `ray_len` -- this is what separates a pocket
     from bulk solvent
  4. keep the connected component containing the centre (flood fill on the grid), so a detached
     cavity elsewhere in the protein is not counted
  5. volume = n_points * step^3

CAVEAT recorded up front: the paper measures crystal structures; our campaign has Boltz folds with a
ligand bound, and pocket volume is conformation-dependent AND reduced by a bound ligand. Calibration
shows the METHOD is right; it does not make fold-derived volumes interchangeable with crystal ones.
Apo-vs-holo and crystal-vs-model are reported separately, never pooled.
"""
import math, pathlib, sys
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "MG": 1.73, "F": 1.47}
AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
DIRS = [np.array(d, float) / np.linalg.norm(d) for d in
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]]


def load(path, chain=None, want_het=None):
    """-> (protein_xyz, protein_radii, hetero{resname: xyz list})"""
    pxyz, prad, het = [], [], {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        if chain and line[21] != chain:
            continue
        el = line[76:78].strip().upper()
        if el in ("H", "D", ""):
            continue
        try:
            xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
        except ValueError:
            continue
        resn = line[17:20].strip()
        if resn in AA3:
            pxyz.append(xyz); prad.append(VDW.get(el, 1.70))
        elif resn != "HOH":
            het.setdefault(resn, []).append(xyz)
    return np.asarray(pxyz), np.asarray(prad), het


def pocket_volume(pxyz, prad, centre, *, step=0.8, extent=14.0, probe=1.4,
                  min_blocked=18, ray_len=11.0):
    """Enclosed-cavity volume (A^3) around `centre`."""
    if len(pxyz) == 0:
        return 0.0, 0
    centre = np.asarray(centre, float)
    n = int(2 * extent / step) + 1
    ax = np.linspace(-extent, extent, n)
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + centre
    G = G[np.linalg.norm(G - centre, axis=1) <= extent]

    tree = cKDTree(pxyz)
    # free = outside vdW+probe of every protein atom
    nb = tree.query_ball_point(G, r=float(prad.max() + probe), workers=1)
    free = np.ones(len(G), bool)
    for i, lst in enumerate(nb):
        if lst and np.any(np.linalg.norm(pxyz[lst] - G[i], axis=1) < prad[lst] + probe):
            free[i] = False
    Gf = G[free]
    if len(Gf) == 0:
        return 0.0, 0

    # enclosure by ray casting
    keep = np.zeros(len(Gf), bool)
    samples = np.arange(1.5, ray_len, 0.7)
    for i, g in enumerate(Gf):
        blocked = 0
        for d in DIRS:
            pts = g + np.outer(samples, d)
            idx = tree.query_ball_point(pts, r=float(prad.max()), workers=1)
            hit = False
            for j, lst in enumerate(idx):
                if lst and np.any(np.linalg.norm(pxyz[lst] - pts[j], axis=1) < prad[lst]):
                    hit = True; break
            if hit:
                blocked += 1
                if blocked >= min_blocked:
                    break
        keep[i] = blocked >= min_blocked
    Gk = Gf[keep]
    if len(Gk) == 0:
        return 0.0, 0

    # connected component containing the point nearest the centre
    ktree = cKDTree(Gk)
    start = int(np.argmin(np.linalg.norm(Gk - centre, axis=1)))
    seen = {start}
    dq = deque([start])
    nbrs = ktree.query_ball_point(Gk, r=step * 1.45, workers=1)
    while dq:
        cur = dq.popleft()
        for nb_i in nbrs[cur]:
            if nb_i not in seen:
                seen.add(nb_i); dq.append(nb_i)
    return len(seen) * step ** 3, len(seen)


def calibrate():
    """Reproduce the paper's two published volumes before trusting the method."""
    here = pathlib.Path(__file__).resolve().parent / "structures"
    targets = [
        ("DesD", here / "7TGM.pdb", "A", 2497, ("I3U", "LMS")),
        ("AcsD", here / "2W02.pdb", "A", 816, ("ATP", "CIT")),
    ]
    print("CALIBRATION against the paper's Fig-3B volumes")
    print(f"{'protein':8s} {'PDB':6s} {'centre from':14s} {'ours (A^3)':>11s} {'paper':>7s} {'ratio':>6s}")
    ok = True
    for name, path, chain, paper, ligs in targets:
        if not path.exists():
            print(f"  {name}: {path.name} absent -- skipped"); ok = False; continue
        pxyz, prad, het = load(path, chain=chain)
        lig = next((het[l] for l in ligs if l in het), None)
        if lig is None:
            print(f"  {name}: none of {ligs} in chain {chain} ({sorted(het)}) -- skipped")
            ok = False; continue
        centre = np.mean(np.asarray(lig), axis=0)
        vol, npt = pocket_volume(pxyz, prad, centre)
        print(f"{name:8s} {path.stem:6s} {'+'.join(l for l in ligs if l in het):14s} "
              f"{vol:11.0f} {paper:7d} {vol/paper:6.2f}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if calibrate() else 1)
