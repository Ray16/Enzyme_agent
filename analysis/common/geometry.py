"""Shared geometry helpers for Boltz cofold analysis.

Small, dependency-light utilities that were previously copy-pasted across the
per-analysis scripts (PDB atom iteration, bond angle, carboxylate-oxygen finder).
"""
import numpy as np


def iter_atoms(pdb_path):
    """Yield (chain, is_hetatm, atom_name, element, xyz) for each ATOM/HETATM line."""
    for ln in open(pdb_path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        chain = ln[21]
        an = ln[12:16].strip()
        elem = (ln[76:78].strip() or an[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        yield chain, ln.startswith("HETATM"), an, elem, xyz


def angle_deg(a, b, c):
    """Angle a-b-c at vertex b, in degrees."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def carboxylate_oxygens(atoms):
    """Carboxylate oxygens in a ligand.

    atoms: iterable of (element, xyz). Returns the list of O coords whose parent
    carbon has >=2 O neighbours within 1.6 A (i.e. carboxylate/carboxyl O atoms).
    """
    Os = [x for e, x in atoms if e == "O"]
    Cs = [x for e, x in atoms if e == "C"]
    out = []
    for cx in Cs:
        nb = [ox for ox in Os if np.linalg.norm(cx - ox) < 1.6]
        if len(nb) >= 2:
            out.extend(nb)
    return out
