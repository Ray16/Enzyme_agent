#!/usr/bin/env python3
"""Cross-model step-1 near-attack geometry for the amide-synthetase panel.

Runs the SAME validated near-attack proxy used for Boltz-2 (acyl-adenylate:
nucleophile = adipate proximal carboxyl O, electrophile = ATP Pa, scissile bridge
O3A; gate d(O..Pa)<4.0 A THEN in-line angle(O,Pa,O3A)>150 deg) uniformly across all
cofold models (Boltz-2, protenix, af3, esmfold2), so predictivity can be compared.

Dependency-free (stdlib `math` only) parser for BOTH PDB and mmCIF, so the identical
geometry can be computed locally (Boltz/protenix on lambda) and inside a Globus-Compute
function on Polaris (af3/esmfold2) without numpy/Biopython on the remote worker.

Metric per target = fraction of poses with d<4 & angle>150 (na_frac, PRIMARY), plus
min_d and mean gated in-line angle. Aggregates over all diffusion samples per target.
"""
import glob
import math
import os

D_GATE, ANG_MIN = 4.0, 150.0


# ---------- dependency-free structure parsing ----------
def _parse_pdb(path):
    atoms = []
    with open(path) as fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            name = ln[12:16].strip()
            resn = ln[17:20].strip()
            chain = ln[21]
            elem = (ln[76:78].strip() or name[0]).upper()
            try:
                xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            atoms.append((chain, resn, name, elem, xyz))
    return atoms


def _parse_cif(path):
    """Parse the _atom_site loop of an mmCIF by column header order (model-agnostic)."""
    atoms = []
    with open(path) as fh:
        lines = fh.readlines()
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("_atom_site."):
            cols = []
            while i < n and lines[i].startswith("_atom_site."):
                cols.append(lines[i].strip().split(".", 1)[1])
                i += 1
            idx = {c: k for k, c in enumerate(cols)}
            # required columns (fall back across common aliases)
            def col(*names):
                for nm in names:
                    if nm in idx:
                        return idx[nm]
                return None
            c_group = col("group_PDB")
            c_elem = col("type_symbol")
            c_atom = col("label_atom_id", "auth_atom_id")
            c_comp = col("label_comp_id", "auth_comp_id")
            c_asym = col("auth_asym_id", "label_asym_id")
            c_x = col("Cartn_x"); c_y = col("Cartn_y"); c_z = col("Cartn_z")
            if None in (c_atom, c_comp, c_asym, c_x, c_y, c_z):
                # not the atom loop we can use; skip
                continue
            ncol = len(cols)
            while i < n:
                row = lines[i]
                s = row.strip()
                if s.startswith("#") or s.startswith("loop_") or s.startswith("_") or s == "":
                    break
                parts = s.split()
                i += 1
                if len(parts) < ncol:
                    continue
                try:
                    xyz = (float(parts[c_x]), float(parts[c_y]), float(parts[c_z]))
                except (ValueError, IndexError):
                    continue
                name = parts[c_atom].strip('"')
                resn = parts[c_comp].strip('"')
                chain = parts[c_asym].strip('"')
                elem = (parts[c_elem].strip() if c_elem is not None else name[0]).upper()
                atoms.append((chain, resn, name, elem, xyz))
            continue
        i += 1
    return atoms


def parse_atoms(path):
    if path.endswith(".pdb"):
        return _parse_pdb(path)
    return _parse_cif(path)


def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _angle(a, b, c):
    v1 = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    v2 = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
    n1 = math.sqrt(sum(x*x for x in v1)); n2 = math.sqrt(sum(x*x for x in v2))
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = sum(v1[k]*v2[k] for k in range(3)) / (n1*n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _carboxylate_os(atoms_of_res):
    """O coords whose parent C has >=2 O within 1.6 A (carboxylate)."""
    Os = [x for (_, _, _, e, x) in atoms_of_res if e == "O"]
    Cs = [x for (_, _, _, e, x) in atoms_of_res if e == "C"]
    out = []
    for cx in Cs:
        nb = [ox for ox in Os if _dist(cx, ox) < 1.6]
        if len(nb) >= 2:
            out.extend(nb)
    return out


def near_attack_pose(path):
    """Return (d_pa, ang_pa) for one pose, or None. Chain/naming-agnostic:
    ATP = residue containing atoms PA and PG; adipate = non-ATP ligand residue with a
    carboxylate; nucleophile = its carboxyl O closest to Pa."""
    atoms = parse_atoms(path)
    if not atoms:
        return None
    byres = {}
    for (ch, resn, name, elem, xyz) in atoms:
        byres.setdefault((ch, resn), []).append((ch, resn, name, elem, xyz))
    # ATP phosphate atoms
    P = None
    for key, lst in byres.items():
        nm = {a[2]: a[4] for a in lst}
        if "PA" in nm and "PG" in nm:
            P = nm
            break
    if not P or "PA" not in P or "O3A" not in P:
        return None
    # adipate: a ligand residue (not ATP, not protein backbone) with a carboxylate
    sub = None
    for key, lst in byres.items():
        resn = key[1]
        if resn == "ATP":
            continue
        nm = {a[2] for a in lst}
        if "PA" in nm and "PG" in nm:  # skip any other NTP
            continue
        # skip standard amino acids (they have N, CA, C, O backbone) — adipate is a small HET ligand
        if {"N", "CA", "C", "O"}.issubset(nm):
            continue
        cbx = _carboxylate_os(lst)
        if cbx:
            sub = cbx
            break
    if not sub:
        return None
    pa = P["PA"]; o3a = P["O3A"]
    onuc = min(sub, key=lambda ox: _dist(ox, pa))
    return (_dist(onuc, pa), _angle(onuc, pa, o3a))


def analyze_target(pred_dir, pattern):
    poses = []
    for p in sorted(glob.glob(os.path.join(pred_dir, pattern))):
        try:
            r = near_attack_pose(p)
        except Exception:
            r = None
        if r is not None:
            poses.append(r)
    if not poses:
        return None
    ds = [d for d, a in poses]
    gated = [(d, a) for d, a in poses if d < D_GATE]
    na = sum(1 for d, a in poses if d < D_GATE and a > ANG_MIN)
    return {
        "n_poses": len(poses),
        "na_frac": na / len(poses),
        "min_d": min(ds),
        "mean_gated_angle": (sum(a for _, a in gated) / len(gated)) if gated else float("nan"),
        "n_gated": len(gated),
    }
