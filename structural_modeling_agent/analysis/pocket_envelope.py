"""Enzyme- and substrate-agnostic asymmetric pocket-envelope & tail-orientation analysis.

Measures:
  1. Distal-tail vs. reactive-end burial and solvent exposure (whether the
     non-reacting end points into bulk solvent or into a buried wall).
  2. Heavy-atom steric clashes and nearest protein contacts.
  3. Native-vs-target asymmetric envelope comparison (target/native size ratio,
     clash vs. pocket-slack diagnosis, and shared vs. lost contact residues).

Works on any PDB or mmCIF structure without assuming any specific enzyme family,
cofactor, or reaction mechanism.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIE", "HID", "HIP", "HSE", "HSD", "HSP", "CYX", "ASH", "GLH", "LYN", "MSE",
}
_SOLVENT = {"HOH", "WAT", "SOL", "TIP3", "T3P", "DOD", "NA", "CL", "K", "NA+", "CL-", "K+", "SOD", "CLA"}
_METALS = {"MG", "MN", "ZN", "CA", "FE", "NI", "CO", "CU", "CD", "MO", "W", "V"}
VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "MG": 1.73, "MN": 1.73, "ZN": 1.39, "CA": 2.31, "FE": 1.56,
}


def read_heavy_atoms(path: str | Path, *, include_solvent: bool = False) -> list[dict[str, Any]]:
    """Read heavy atoms from PDB or mmCIF (uses gemmi when available, else pure-Python PDB)."""
    p = Path(path)
    if p.suffix.lower() in {".cif", ".mmcif"}:
        import gemmi
        st = gemmi.read_structure(str(p))
        model = st[0]
        atoms = []
        for chain in model:
            for res in chain:
                rn = res.name.strip().upper()
                is_solv = rn in _SOLVENT
                if is_solv and not include_solvent:
                    continue
                resid = f"{res.seqid.num}{res.seqid.icode.strip()}"
                is_prot = rn in _AA3
                for atom in res:
                    if atom.altloc not in ("\x00", " ", "A") or atom.element.is_hydrogen:
                        continue
                    el = atom.element.name.upper()
                    atoms.append({
                        "chain": chain.name,
                        "resid": resid,
                        "resname": rn,
                        "name": atom.name.strip(),
                        "element": el,
                        "xyz": np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float),
                        "protein": is_prot,
                        "solvent": is_solv,
                    })
        return atoms

    atoms = []
    with p.open() as fh:
        for ln in fh:
            if not ln.startswith(("ATOM  ", "HETATM")):
                continue
            alt = ln[16]
            if alt not in (" ", "A"):
                continue
            rn = ln[17:20].strip().upper()
            is_solv = rn in _SOLVENT
            if is_solv and not include_solvent:
                continue
            el = ln[76:78].strip().upper()
            name = ln[12:16].strip()
            if not el:
                el = "".join(c for c in name if c.isalpha()).upper()[:1]
            if el in ("H", "D"):
                continue
            chain = ln[21].strip() or "A"
            resid = ln[22:27].strip()
            atoms.append({
                "chain": chain,
                "resid": resid,
                "resname": rn,
                "name": name,
                "element": el,
                "xyz": np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])], dtype=float),
                "protein": rn in _AA3,
                "solvent": is_solv,
            })
    return atoms


@lru_cache(maxsize=8)
def _fibonacci_sphere(n: int = 128) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    pts = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
    pts.setflags(write=False)
    return pts


def _points_burial_batch(
    points: np.ndarray,
    obs_tree: cKDTree,
    obs_radii: np.ndarray,
    probe: float = 1.4,
    n_dir: int = 128,
    max_len: float = 16.0,
    step: float = 1.0,
) -> np.ndarray:
    """Vectorized ray-cast burial fraction for M points in one cKDTree.query call."""
    m = len(points)
    if m == 0:
        return np.zeros(0, dtype=float)
    if len(obs_radii) == 0:
        return np.zeros(m, dtype=float)
    s = np.arange(step, max_len + step, step)
    dirs = _fibonacci_sphere(n_dir)
    # Shape: (M, n_dir, n_steps, 3) -> (M * n_dir * n_steps, 3)
    offsets = dirs[:, None, :] * s[None, :, None]
    rays = (points[:, None, None, :] + offsets[None, :, :, :]).reshape(-1, 3)
    k = min(4, len(obs_radii))
    d, idx = obs_tree.query(rays, k=k)
    d = d.reshape(len(rays), -1)
    idx = idx.reshape(len(rays), -1)
    gap = (d - obs_radii[idx]).min(axis=1).reshape(m, n_dir, len(s))
    return (gap < probe).any(axis=2).mean(axis=1).astype(float)


def _point_burial(point: np.ndarray, obs_tree: cKDTree, obs_radii: np.ndarray,
                  probe: float = 1.4, n_dir: int = 128, max_len: float = 16.0, step: float = 1.0) -> float:
    """Ray-cast burial fraction of `point` against `obs_tree` (0=solvent-open, 1=fully buried)."""
    return float(_points_burial_batch(point[None, :], obs_tree, obs_radii, probe, n_dir, max_len, step)[0])


def _select_substrate(atoms: list[dict[str, Any]], substrate_chains: list[str] | None = None,
                      substrate_resnames: list[str] | None = None):
    """Split atoms into (protein+other_obstacles, substrate)."""
    sub_ch = set(substrate_chains) if substrate_chains else None
    sub_rn = {r.upper() for r in substrate_resnames} if substrate_resnames else None

    if sub_ch or sub_rn:
        sub = [
            a for a in atoms
            if not a["protein"]
            and (sub_ch is None or a["chain"] in sub_ch)
            and (sub_rn is None or a["resname"] in sub_rn)
        ]
    else:
        # Generic fallback: non-protein, non-metal organic groups; exclude obvious P-cofactors if multiple groups
        by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for a in atoms:
            if not a["protein"]:
                by_group.setdefault((a["chain"], a["resname"]), []).append(a)
        organic = []
        for grp in by_group.values():
            elems = {a["element"] for a in grp}
            if elems <= _METALS:
                continue
            organic.append(grp)
        non_p = [g for g in organic if not ("P" in {a["element"] for a in g} and len(g) > 10)]
        chosen = non_p if non_p else organic
        sub = [a for g in chosen for a in g]

    sub_ids = {id(a) for a in sub}
    obstacles = [a for a in atoms if id(a) not in sub_ids]
    return obstacles, sub


def analyze_pose_envelope(
    pose_path: str | Path,
    *,
    substrate_chains: list[str] | None = None,
    substrate_resnames: list[str] | None = None,
    reaction_center_xyz: list[float] | np.ndarray | None = None,
    clash_cutoff_A: float = 2.2,
    contact_cutoff_A: float = 4.0,
) -> dict[str, Any]:
    """Measure pocket envelope, clashes, and reactive-vs-distal tail exposure on one holo pose."""
    atoms = read_heavy_atoms(pose_path)
    obstacles, sub = _select_substrate(atoms, substrate_chains, substrate_resnames)
    if not sub:
        raise ValueError(f"{pose_path}: no substrate heavy atoms resolved")
    prot = [a for a in obstacles if a["protein"]]
    if not prot:
        raise ValueError(f"{pose_path}: no protein heavy atoms found")

    sub_xyz = np.asarray([a["xyz"] for a in sub], dtype=float)
    obs_xyz = np.asarray([a["xyz"] for a in obstacles], dtype=float)
    obs_rad = np.asarray([VDW.get(a["element"], 1.70) for a in obstacles], dtype=float)
    prot_xyz = np.asarray([a["xyz"] for a in prot], dtype=float)

    obs_tree = cKDTree(obs_xyz)
    prot_tree = cKDTree(prot_xyz)

    # Steric clashes and protein contacts per substrate atom
    min_d_prot, nearest_prot_idx = prot_tree.query(sub_xyz, k=1)
    min_d_prot = np.atleast_1d(min_d_prot)
    n_clashes = int((min_d_prot < clash_cutoff_A).sum())
    n_contacts = int((min_d_prot <= contact_cutoff_A).sum())

    contact_residues = set()
    clashing_residues = set()
    clash_sq = clash_cutoff_A * clash_cutoff_A
    for i_sub, hit_list in enumerate(prot_tree.query_ball_point(sub_xyz, r=contact_cutoff_A)):
        if not hit_list:
            continue
        p_pt = sub_xyz[i_sub]
        for j in hit_list:
            label = f"{prot[j]['chain']}:{prot[j]['resname']}{prot[j]['resid']}"
            contact_residues.add(label)
            dv = prot_xyz[j] - p_pt
            if float(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]) < clash_sq:
                clashing_residues.add(label)

    # Reactive end vs distal tail: rank atoms by distance to reaction center (or cofactor/protein center)
    if reaction_center_xyz is not None:
        rc = np.asarray(reaction_center_xyz, dtype=float)
    else:
        non_prot_obs = [a["xyz"] for a in obstacles if not a["protein"]]
        rc = np.mean(non_prot_obs, axis=0) if non_prot_obs else np.mean(prot_xyz, axis=0)

    d_to_rc = np.linalg.norm(sub_xyz - rc, axis=1)
    order = np.argsort(d_to_rc)
    k_end = max(1, len(sub) // 3)
    reactive_idx = order[:k_end]
    distal_idx = order[-k_end:]

    # Single batched KDTree query for both reactive-end and distal-tail atoms
    eval_idx = np.concatenate([reactive_idx, distal_idx])
    burials = _points_burial_batch(sub_xyz[eval_idx], obs_tree, obs_rad)
    reactive_burial = float(burials[: len(reactive_idx)].mean())
    distal_burial = float(burials[len(reactive_idx) :].mean())
    distal_clashes = int((min_d_prot[distal_idx] < clash_cutoff_A).sum())

    if len(sub_xyz) >= 2:
        extent = float(np.linalg.norm(sub_xyz[:, None, :] - sub_xyz[None, :, :], axis=2).max())
    else:
        extent = 0.0

    tail_class = (
        "solvent_exposed_tail" if distal_burial < 0.65
        else "buried_distal_clash" if distal_clashes > 0
        else "buried_tail_accommodated"
    )

    return {
        "pose": str(pose_path),
        "n_substrate_heavy_atoms": len(sub),
        "substrate_extent_A": round(extent, 2),
        "n_clashing_substrate_atoms": n_clashes,
        "n_contacting_substrate_atoms": n_contacts,
        "contact_fraction": round(n_contacts / len(sub), 3),
        "min_protein_distance_A": round(float(min_d_prot.min()), 2),
        "reactive_end_burial": round(reactive_burial, 3),
        "distal_tail_burial": round(distal_burial, 3),
        "distal_tail_clashes": distal_clashes,
        "tail_orientation": tail_class,
        "contact_residues": sorted(contact_residues),
        "clashing_residues": sorted(clashing_residues),
    }


def _aggregate_envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n_poses": 0}
    res_counts: dict[str, int] = {}
    clash_counts: dict[str, int] = {}
    for r in rows:
        for res in r["contact_residues"]:
            res_counts[res] = res_counts.get(res, 0) + 1
        for res in r["clashing_residues"]:
            clash_counts[res] = clash_counts.get(res, 0) + 1
    return {
        "n_poses": n,
        "n_substrate_heavy_atoms": rows[0]["n_substrate_heavy_atoms"],
        "substrate_extent_A_mean": round(float(np.mean([r["substrate_extent_A"] for r in rows])), 2),
        "clash_atoms_mean": round(float(np.mean([r["n_clashing_substrate_atoms"] for r in rows])), 2),
        "contact_fraction_mean": round(float(np.mean([r["contact_fraction"] for r in rows])), 3),
        "reactive_end_burial_mean": round(float(np.mean([r["reactive_end_burial"] for r in rows])), 3),
        "distal_tail_burial_mean": round(float(np.mean([r["distal_tail_burial"] for r in rows])), 3),
        "distal_tail_clash_poses": sum(r["distal_tail_clashes"] > 0 for r in rows),
        "persistent_contact_residues": sorted(k for k, v in res_counts.items() if v / n >= 0.5),
        "clashing_residues": sorted(k for k, v in clash_counts.items() if v / n >= 0.25),
    }


def compare_native_vs_target(
    target_poses: list[str | Path],
    native_poses: list[str | Path] | None = None,
    *,
    target_substrate_chains: list[str] | None = None,
    target_substrate_resnames: list[str] | None = None,
    native_substrate_chains: list[str] | None = None,
    native_substrate_resnames: list[str] | None = None,
    reaction_center_xyz: list[float] | np.ndarray | None = None,
    clash_cutoff_A: float = 2.2,
) -> dict[str, Any]:
    """Compare target-holo envelope against an optional native-holo control ensemble."""
    t_rows = [
        analyze_pose_envelope(
            p,
            substrate_chains=target_substrate_chains,
            substrate_resnames=target_substrate_resnames,
            reaction_center_xyz=reaction_center_xyz,
            clash_cutoff_A=clash_cutoff_A,
        )
        for p in target_poses
    ]
    target = _aggregate_envelope(t_rows)

    native = None
    size_ratio = None
    shared_anchors = None
    lost_anchors = None
    if native_poses:
        n_rows = [
            analyze_pose_envelope(
                p,
                substrate_chains=native_substrate_chains,
                substrate_resnames=native_substrate_resnames,
                reaction_center_xyz=reaction_center_xyz,
                clash_cutoff_A=clash_cutoff_A,
            )
            for p in native_poses
        ]
        native = _aggregate_envelope(n_rows)
        if native["n_substrate_heavy_atoms"] > 0:
            size_ratio = round(target["n_substrate_heavy_atoms"] / native["n_substrate_heavy_atoms"], 2)
        t_res = set(target["persistent_contact_residues"])
        n_res = set(native["persistent_contact_residues"])
        shared_anchors = sorted(t_res & n_res)
        lost_anchors = sorted(n_res - t_res)

    # Asymmetric pocket-envelope diagnosis
    t_clashes = target["clash_atoms_mean"] >= 1.0
    n_clashes = native is not None and native["clash_atoms_mean"] >= 1.0
    if native is not None and t_clashes and n_clashes:
        diagnosis = "modeling_setup_issue"
    elif t_clashes and target["distal_tail_burial_mean"] >= 0.75:
        diagnosis = "buried_steric_exclusion" if (size_ratio is None or size_ratio > 1.15) else "distal_tail_buried_clash"
    elif size_ratio is not None and size_ratio < 0.75 and target["contact_fraction_mean"] < 0.65:
        diagnosis = "pocket_too_loose_slack"
    elif target["distal_tail_burial_mean"] < 0.65 and not t_clashes:
        diagnosis = "permissive_solvent_exposed_tail"
    else:
        diagnosis = "compatible_envelope"

    return {
        "status": "evaluated",
        "diagnosis": diagnosis,
        "target": target,
        "native": native,
        "target_to_native_heavy_atom_ratio": size_ratio,
        "shared_contact_residues": shared_anchors,
        "lost_native_contact_residues": lost_anchors,
        "clashing_residues": target["clashing_residues"],
        "scope": "Geometric pocket-envelope comparison; does not by itself establish catalytic activity.",
    }
