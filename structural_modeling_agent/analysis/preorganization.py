"""Enzyme- and reaction-agnostic active-site preorganization & TS-accommodation analyzer.

Measures three general physical axes of catalysis:
  1. Apo-vs-holo catalytic-constellation reorganization RMSD (global CA aligned).
  2. Electrostatic preorganization: projected electric field (MV/cm) along any
     declared reaction/dipole axis (from Amber ``prmtop`` charges or explicit
     partial-charge maps), plus top stabilizing/destabilizing residues.
  3. Transition-state / intermediate charge-center stabilization: persistent
     H-bond donors, cationic contacts, and metal coordination around any
     declared ``charge_center`` atom.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .pocket_envelope import _METALS, read_heavy_atoms

# Protein heavy atoms capable of donating an H-bond (excludes backbone carbonyl O)
_HBOND_DONOR_ATOM_NAMES = {
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "OG", "OG1", "OH", "SG",
}

# Electrostatic conversion: 1 e / A^2 = 1439.9645 MV/cm (vacuum permittivity)
_E_PER_A2_TO_MV_CM = 1439.964547842567
# Amber prmtop CHARGE flag stores q * 18.2223 (sqrt(332.0522173))
_AMBER_CHARGE_SCALE = 18.2223


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def apo_holo_reorganization(
    apo_path: str | Path,
    holo_path: str | Path,
    catalytic_residues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute heavy-atom RMSD of declared catalytic residues between apo and holo structures.

    ``catalytic_residues``: list of ``{"chain": "A", "residue": "117"}``.
    Aligns on all matched protein C-alpha atoms first, then measures catalytic heavy-atom shift.
    """
    apo_atoms = read_heavy_atoms(apo_path)
    holo_atoms = read_heavy_atoms(holo_path)

    apo_ca = {(a["chain"], a["resid"]): a["xyz"] for a in apo_atoms if a["protein"] and a["name"] == "CA"}
    holo_ca = {(a["chain"], a["resid"]): a["xyz"] for a in holo_atoms if a["protein"] and a["name"] == "CA"}
    common_ca = sorted(set(apo_ca) & set(holo_ca))
    if len(common_ca) < 3:
        raise ValueError("need at least 3 matched protein C-alpha atoms to align apo and holo structures")

    P = np.asarray([apo_ca[k] for k in common_ca], dtype=float)
    Q = np.asarray([holo_ca[k] for k in common_ca], dtype=float)
    R, t = _kabsch(P, Q)
    global_ca_rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))

    cat_keys = {(str(r["chain"]), str(r["residue"])) for r in catalytic_residues}
    apo_cat = {
        (a["chain"], a["resid"], a["name"]): a["xyz"] @ R.T + t
        for a in apo_atoms
        if a["protein"] and (a["chain"], a["resid"]) in cat_keys
    }
    holo_cat = {
        (a["chain"], a["resid"], a["name"]): a["xyz"]
        for a in holo_atoms
        if a["protein"] and (a["chain"], a["resid"]) in cat_keys
    }
    matched = sorted(set(apo_cat) & set(holo_cat))
    if not matched:
        raise ValueError("no matched catalytic heavy atoms found between apo and holo structures")

    diffs = np.asarray([apo_cat[k] - holo_cat[k] for k in matched], dtype=float)
    sq_dists = np.sum(diffs ** 2, axis=1)
    cat_rmsd = float(np.sqrt(np.mean(sq_dists)))

    per_res: dict[str, list[float]] = {}
    for (ch, res, _), sd in zip(matched, sq_dists):
        per_res.setdefault(f"{ch}:{res}", []).append(float(np.sqrt(sd)))

    return {
        "global_ca_rmsd_A": round(global_ca_rmsd, 3),
        "catalytic_heavy_rmsd_A": round(cat_rmsd, 3),
        "n_matched_catalytic_atoms": len(matched),
        "per_residue_max_shift_A": {k: round(max(v), 2) for k, v in sorted(per_res.items())},
    }


def parse_prmtop_charges(prmtop_path: str | Path) -> np.ndarray:
    """Read atomic partial charges (in elementary charge e) from an Amber .prmtop file."""
    text = Path(prmtop_path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"%FLAG CHARGE\s*\n%FORMAT\([^\)]+\)\s*\n(.*?)(?=%FLAG|\Z)", text, flags=re.S)
    if not m:
        raise ValueError(f"{prmtop_path}: %FLAG CHARGE section not found")
    vals = [float(tok) / _AMBER_CHARGE_SCALE for tok in m.group(1).split()]
    return np.asarray(vals, dtype=float)


def compute_projected_electric_field(
    coords: np.ndarray,
    charges_e: np.ndarray,
    axis_a_xyz: np.ndarray,
    axis_b_xyz: np.ndarray,
    *,
    probe_xyz: np.ndarray | None = None,
    include_mask: np.ndarray | None = None,
    cutoff_A: float = 15.0,
    min_dist_A: float = 0.5,
) -> dict[str, Any]:
    """Compute electrostatic field (MV/cm) projected along unit vector (axis_b - axis_a)."""
    axis = np.asarray(axis_b_xyz, dtype=float) - np.asarray(axis_a_xyz, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        raise ValueError("reaction field axis atoms overlap")
    u_hat = axis / norm
    probe = 0.5 * (np.asarray(axis_a_xyz, dtype=float) + np.asarray(axis_b_xyz, dtype=float)) if probe_xyz is None else np.asarray(probe_xyz, dtype=float)

    if include_mask is not None:
        mask = np.asarray(include_mask, dtype=bool)
        sel_coords = coords[mask]
        sel_charges = charges_e[mask]
    else:
        sel_coords = coords
        sel_charges = charges_e

    if len(sel_coords) == 0:
        return {"projected_field_MV_per_cm": 0.0, "field_vector_MV_per_cm": [0.0, 0.0, 0.0], "n_charges": 0}

    delta = probe[None, :] - sel_coords  # vector from source charge -> probe
    d2 = np.einsum("ij,ij->i", delta, delta)
    keep = (d2 >= min_dist_A * min_dist_A) & (d2 <= cutoff_A * cutoff_A)
    if not keep.any():
        return {"projected_field_MV_per_cm": 0.0, "field_vector_MV_per_cm": [0.0, 0.0, 0.0], "n_charges": 0}

    q = sel_charges[keep]
    d_vec = delta[keep]
    d = np.sqrt(d2[keep])
    # E = sum_i q_i * (r_probe - r_i) / |r_probe - r_i|^3
    e_vec = (q[:, None] * d_vec / (d[:, None] ** 3)).sum(axis=0) * _E_PER_A2_TO_MV_CM
    e_proj = float(np.dot(e_vec, u_hat))
    return {
        "projected_field_MV_per_cm": round(e_proj, 2),
        "field_vector_MV_per_cm": [round(float(x), 2) for x in e_vec],
        "n_charges": int(keep.sum()),
    }


def analyze_charge_center_stabilization(
    pose_paths: list[str | Path],
    charge_center: dict[str, Any],
    *,
    donor_cutoff_A: float = 3.5,
    metal_cutoff_A: float = 2.7,
) -> dict[str, Any]:
    """Count persistent H-bond donors and metal contacts around any TS/intermediate charge center.

    Distinguishes H-bond donors (backbone amide N and polar sidechain donors) from
    backbone carbonyl O acceptors so oxyanion/anionic TS stabilization is not inflated.
    """
    target_key = (str(charge_center["chain"]), str(charge_center["residue"]), str(charge_center["atom"]))
    donor_counts: list[int] = []
    polar_counts: list[int] = []
    metal_hits = 0
    res_freq: dict[str, int] = {}
    n_eval = 0

    for path in pose_paths:
        atoms = read_heavy_atoms(path)
        center_xyz = None
        for a in atoms:
            if (a["chain"], a["resid"], a["name"]) == target_key:
                center_xyz = a["xyz"]
                break
        if center_xyz is None:
            continue
        n_eval += 1
        donors = [
            a for a in atoms
            if a["protein"]
            and a["name"] in _HBOND_DONOR_ATOM_NAMES
            and not (a["resname"] == "PRO" and a["name"] == "N")
        ]
        all_polar = [
            a for a in atoms
            if a["protein"] and a["element"] in ("N", "O", "S")
        ]
        if donors:
            d_xyz = np.asarray([a["xyz"] for a in donors], dtype=float)
            dists = np.linalg.norm(d_xyz - center_xyz[None, :], axis=1)
            hits = np.flatnonzero(dists <= donor_cutoff_A)
            donor_counts.append(int(len(hits)))
            for label in {f"{donors[i]['chain']}:{donors[i]['resname']}{donors[i]['resid']}" for i in hits}:
                res_freq[label] = res_freq.get(label, 0) + 1
        else:
            donor_counts.append(0)
        if all_polar:
            p_xyz = np.asarray([a["xyz"] for a in all_polar], dtype=float)
            polar_counts.append(int((np.linalg.norm(p_xyz - center_xyz[None, :], axis=1) <= donor_cutoff_A).sum()))
        else:
            polar_counts.append(0)
        metals = [a for a in atoms if not a["protein"] and a["element"] in _METALS]
        if metals:
            m_xyz = np.asarray([a["xyz"] for a in metals], dtype=float)
            if float(np.linalg.norm(m_xyz - center_xyz[None, :], axis=1).min()) <= metal_cutoff_A:
                metal_hits += 1

    top_res = [
        {"residue": k, "frequency": round(v / n_eval, 3)}
        for k, v in sorted(res_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    ] if n_eval else []

    return {
        "n_evaluated": n_eval,
        "charge_center": charge_center,
        "hbond_donor_atoms_mean": round(float(np.mean(donor_counts)), 2) if donor_counts else None,
        "polar_donor_atoms_mean": round(float(np.mean(donor_counts)), 2) if donor_counts else None,
        "all_polar_heavy_atoms_mean": round(float(np.mean(polar_counts)), 2) if polar_counts else None,
        "metal_stabilization_freq": round(metal_hits / n_eval, 3) if n_eval else None,
        "stabilizing_residues": top_res,
    }


def analyze_preorganization(
    *,
    apo_path: str | Path | None = None,
    holo_path: str | Path | None = None,
    catalytic_residues: list[dict[str, Any]] | None = None,
    pose_paths: list[str | Path] | None = None,
    charge_center: dict[str, Any] | None = None,
    prmtop_path: str | Path | None = None,
    traj_paths: list[str | Path] | None = None,
    axis_a_sel: str | None = None,
    axis_b_sel: str | None = None,
    environment_sel: str = "protein",
    expected_favorable_sign: int | None = None,
    cutoff_A: float = 15.0,
    stride: int = 1,
) -> dict[str, Any]:
    """Run any combination of apo/holo reorganization, TS charge stabilization, and MD electric field."""
    result: dict[str, Any] = {
        "status": "evaluated",
        "scope": "Structural and electrostatic preorganization diagnostics; not a reaction barrier.",
    }
    if apo_path and holo_path and catalytic_residues:
        result["apo_holo_reorganization"] = apo_holo_reorganization(apo_path, holo_path, catalytic_residues)

    if pose_paths and charge_center:
        result["charge_center_stabilization"] = analyze_charge_center_stabilization(pose_paths, charge_center)

    if prmtop_path and traj_paths and axis_a_sel and axis_b_sel:
        import MDAnalysis as mda

        charges = parse_prmtop_charges(prmtop_path)
        u = mda.Universe(str(prmtop_path), [str(t) for t in traj_paths])
        if len(charges) != len(u.atoms):
            raise ValueError(f"prmtop charge count ({len(charges)}) != topology atom count ({len(u.atoms)})")
        sel_a = u.select_atoms(axis_a_sel)
        sel_b = u.select_atoms(axis_b_sel)
        if len(sel_a) != 1 or len(sel_b) != 1:
            raise ValueError("axis_a_sel and axis_b_sel must each select exactly 1 atom")
        idx_a, idx_b = int(sel_a.indices[0]), int(sel_b.indices[0])
        env_atoms = u.select_atoms(environment_sel)
        # Exclude the entire residue(s) containing the reaction axis atoms so
        # covalently adjacent ligand/substrate atoms cannot produce unphysical field spikes.
        excl_res_indices = set(u.atoms[idx_a].residue.atoms.indices) | set(u.atoms[idx_b].residue.atoms.indices)
        env_idx = np.asarray([i for i in env_atoms.indices if i not in excl_res_indices], dtype=int)
        env_charges = charges[env_idx]
        # Precompute residue label per environment atom for O(N_env) per-residue decomposition
        env_res_labels = [f"{a.segid or 'A'}:{a.resname}{a.resid}" for a in u.atoms[env_idx]]
        unique_res = sorted(set(env_res_labels))
        res_to_id = {r: k for k, r in enumerate(unique_res)}
        env_res_ids = np.asarray([res_to_id[r] for r in env_res_labels], dtype=int)
        res_field_sums = np.zeros(len(unique_res), dtype=float)
        cut_sq = cutoff_A * cutoff_A
        min_sq = 0.25

        from MDAnalysis.lib.distances import minimize_vectors
        fields = []
        for ts in u.trajectory[::stride]:
            pos = u.atoms.positions
            box = ts.dimensions if ts.dimensions is not None and np.all(ts.dimensions[:3] > 0) else None
            pa = pos[idx_a].astype(float)
            pb = pos[idx_b].astype(float)
            axis = pb - pa
            if box is not None:
                axis = minimize_vectors(np.ascontiguousarray(axis[None, :], dtype=np.float32), box)[0].astype(float)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-8:
                continue
            u_hat = axis / norm
            probe = pa + 0.5 * axis
            env_coords = pos[env_idx].astype(float)
            delta = probe[None, :] - env_coords
            if box is not None:
                delta = minimize_vectors(np.ascontiguousarray(delta, dtype=np.float32), box).astype(float)
            d2 = np.einsum("ij,ij->i", delta, delta)
            keep = (d2 >= min_sq) & (d2 <= cut_sq)
            if not keep.any():
                fields.append(0.0)
                continue
            d_vec = delta[keep]
            d = np.sqrt(d2[keep])
            atom_proj = (env_charges[keep] * (d_vec @ u_hat) / (d ** 3)) * _E_PER_A2_TO_MV_CM
            fields.append(float(atom_proj.sum()))
            np.add.at(res_field_sums, env_res_ids[keep], atom_proj)

        n_f = len(fields)
        per_res_means = (
            [(unique_res[k], float(res_field_sums[k] / n_f)) for k in range(len(unique_res)) if abs(res_field_sums[k]) > 1e-6]
            if n_f else []
        )
        sign = expected_favorable_sign if expected_favorable_sign in (-1, 1) else 1
        per_res_means.sort(key=lambda kv: -(kv[1] * sign))
        top_pos = [{"residue": r, "field_MV_per_cm": round(v, 2)} for r, v in per_res_means[:5] if v * sign > 0]
        top_neg = [{"residue": r, "field_MV_per_cm": round(v, 2)} for r, v in reversed(per_res_means[-5:]) if v * sign < 0]

        result["electric_field"] = {
            "n_frames": n_f,
            "projected_field_MV_per_cm_mean": round(float(np.mean(fields)), 2) if fields else None,
            "projected_field_MV_per_cm_sd": round(float(np.std(fields)), 2) if fields else None,
            "expected_favorable_sign": expected_favorable_sign,
            "cutoff_A": cutoff_A,
            "environment_sel": environment_sel,
            "top_stabilizing_residues": top_pos,
            "top_destabilizing_residues": top_neg,
        }

    return result
