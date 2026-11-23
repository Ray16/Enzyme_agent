"""Reaction- and enzyme-agnostic competing side-reaction geometry analyzer.

Evaluates any user/agent-declared list of competing reaction pathways on a single
structure, a pose ensemble, or a solvated MD trajectory. Examples across enzyme
classes:
  * Amide synthetases / acyltransferases: solvent water attacking an acyl-X
    carbonyl (e.g. d < 4.0 A, angle 107 +/- 20 deg) or intramolecular cyclization.
  * Kinases / ATPases / phosphatases: water attacking ATP Pgamma inline
    (e.g. d < 4.0 A, angle 180 +/- 30 deg -> futile ATP hydrolysis).
  * P450s / dioxygenases: competing C-H abstraction sites (distance-only,
    ``angle_reference=None``).
  * Hydrolases / transferases: competing nucleophiles or off-register cleavage.

All chemistry (nucleophile, electrophile, optional angle reference, ideal angle,
and tolerances) is supplied declaratively; no enzyme or mechanism is hardcoded.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .pocket_envelope import read_heavy_atoms

DEFAULT_SOLVENT_RESNAMES = ("HOH", "WAT", "SOL", "TIP3", "T3P", "DOD")


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - b
    v = c - b
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-12:
        return 0.0
    return float(math.degrees(math.acos(float(np.clip(np.dot(u, v) / denom, -1.0, 1.0)))))


def _resolve_atom_xyz(atom_index: dict[tuple[str, str, str], np.ndarray], ref: dict[str, Any]) -> np.ndarray:
    key = (str(ref["chain"]), str(ref["residue"]), str(ref["atom"]))
    if key not in atom_index:
        raise KeyError(f"declared atom {key} not found in structure")
    return atom_index[key]


def _read_all_heavy_incl_solvent(path: str | Path) -> tuple[dict[tuple[str, str, str], np.ndarray], list[dict[str, Any]]]:
    """Read all heavy atoms including solvent via the shared pocket_envelope reader."""
    atoms = read_heavy_atoms(path, include_solvent=True)
    index: dict[tuple[str, str, str], np.ndarray] = {}
    non_protein: list[dict[str, Any]] = []
    for a in atoms:
        index[(a["chain"], a["resid"], a["name"])] = a["xyz"]
        if not a["protein"]:
            non_protein.append({"resname": a["resname"], "atom": a["name"], "xyz": a["xyz"]})
    return index, non_protein


def _eval_pathway_coords(
    nuc_pts: np.ndarray,
    elec_xyz: np.ndarray,
    ref_xyz: np.ndarray | None,
    distance_cutoff_A: float,
    ideal_angle_deg: float | None,
    angle_tolerance_deg: float | None,
    *,
    box: np.ndarray | None = None,
) -> tuple[bool, float | None, float | None]:
    """Check whether any nucleophile in `nuc_pts` meets the competing-attack geometry.

    Uses a single vectorized squared-distance pass (with PBC minimum-image when
    ``box`` is supplied) rather than rebuilding a spatial tree per MD frame.
    """
    if len(nuc_pts) == 0:
        return False, None, None
    diff = (nuc_pts - elec_xyz[None, :]).astype(np.float32 if box is not None else float, copy=False)
    v_ref = (ref_xyz - elec_xyz) if ref_xyz is not None else None
    if box is not None:
        from MDAnalysis.lib.distances import minimize_vectors
        diff = minimize_vectors(np.ascontiguousarray(diff, dtype=np.float32), box).astype(float)
        if v_ref is not None:
            v_ref = minimize_vectors(np.ascontiguousarray(v_ref[None, :], dtype=np.float32), box)[0].astype(float)
    d2 = np.einsum("ij,ij->i", diff, diff)
    min_idx = int(np.argmin(d2))
    min_d = float(math.sqrt(float(d2[min_idx])))
    within = np.flatnonzero(d2 <= distance_cutoff_A * distance_cutoff_A)
    if len(within) == 0:
        if v_ref is None:
            return False, min_d, None
        denom = float(min_d * np.linalg.norm(v_ref))
        ang0 = float(math.degrees(math.acos(float(np.clip(np.dot(diff[min_idx], v_ref) / denom, -1.0, 1.0))))) if denom > 1e-12 else 0.0
        return False, min_d, ang0

    if v_ref is None or ideal_angle_deg is None or angle_tolerance_deg is None:
        return True, min_d, None

    # Vectorized angles for all within-cutoff nucleophiles
    u = diff[within]
    v_norm = float(np.linalg.norm(v_ref))
    if v_norm < 1e-12:
        return False, min_d, 0.0
    u_norms = np.sqrt(d2[within])
    denom = u_norms * v_norm
    cos_vals = np.where(denom > 1e-12, (u @ v_ref) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    angles = np.degrees(np.arccos(np.clip(cos_vals, -1.0, 1.0)))
    devs = np.abs(angles - ideal_angle_deg)
    hits = np.flatnonzero(devs <= angle_tolerance_deg)
    if len(hits) > 0:
        # Report distance and angle for the SAME winning (closest hitting) nucleophile
        best_hit = int(hits[int(np.argmin(u_norms[hits]))])
        return True, float(u_norms[best_hit]), float(angles[best_hit])
    best_i = int(np.argmin(devs))
    return False, float(u_norms[best_i]), float(angles[best_i])


def analyze_side_reactions_poses(
    pose_paths: list[str | Path],
    pathways: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate declared competing-reaction pathways across static PDB/mmCIF poses."""
    if not pathways:
        raise ValueError("declare at least one competing reaction pathway")
    stats = {
        p["name"]: {"hits": 0, "dists": [], "angles": [], "n_evaluated": 0, "spec": p}
        for p in pathways
    }
    for path in pose_paths:
        index, solvent = _read_all_heavy_incl_solvent(path)
        for p in pathways:
            st = stats[p["name"]]
            try:
                elec = _resolve_atom_xyz(index, p["electrophile"])
                ref = _resolve_atom_xyz(index, p["angle_reference"]) if p.get("angle_reference") else None
                kind = p.get("kind", "intermolecular")
                if kind == "solvent":
                    resnames = {r.upper() for r in (p.get("solvent_resnames") or DEFAULT_SOLVENT_RESNAMES)}
                    atom_name = p.get("solvent_atom_name", "O")
                    pts = [s["xyz"] for s in solvent if s["resname"] in resnames and s["atom"].startswith(atom_name)]
                    nuc_pts = np.asarray(pts, dtype=float) if pts else np.zeros((0, 3), dtype=float)
                else:
                    nuc_pts = np.asarray([_resolve_atom_xyz(index, r) for r in p["nucleophiles"]], dtype=float)
            except KeyError:
                continue
            hit, d_min, ang = _eval_pathway_coords(
                nuc_pts, elec, ref,
                float(p["distance_cutoff_A"]),
                p.get("ideal_angle_deg"),
                p.get("angle_tolerance_deg"),
            )
            st["n_evaluated"] += 1
            if hit:
                st["hits"] += 1
            if d_min is not None:
                st["dists"].append(d_min)
            if ang is not None:
                st["angles"].append(ang)

    return _summarize_side_reactions(stats, source="pose_ensemble", n_input=len(pose_paths))


def analyze_side_reactions_trajectory(
    top_path: str | Path,
    traj_paths: list[str | Path],
    pathways: list[dict[str, Any]],
    *,
    stride: int = 1,
) -> dict[str, Any]:
    """Evaluate declared competing-reaction pathways across an MD trajectory using MDAnalysis."""
    import MDAnalysis as mda

    if not pathways:
        raise ValueError("declare at least one competing reaction pathway")
    u = mda.Universe(str(top_path), [str(t) for t in traj_paths])

    # Precompute atom index selections once on frame 0 (fast O(N_frames) traversal)
    resolved = []
    for p in pathways:
        elec_sel = u.select_atoms(p["electrophile_sel"])
        if len(elec_sel) != 1:
            raise ValueError(f"{p['name']}: electrophile_sel must match exactly 1 atom (got {len(elec_sel)})")
        ref_idx = None
        if p.get("angle_reference_sel"):
            ref_sel = u.select_atoms(p["angle_reference_sel"])
            if len(ref_sel) != 1:
                raise ValueError(f"{p['name']}: angle_reference_sel must match exactly 1 atom (got {len(ref_sel)})")
            ref_idx = int(ref_sel.indices[0])
        nuc_sel = u.select_atoms(p["nucleophile_sel"])
        resolved.append({
            "spec": p,
            "elec_idx": int(elec_sel.indices[0]),
            "ref_idx": ref_idx,
            "nuc_idx": np.asarray(nuc_sel.indices, dtype=int),
            "hits": 0,
            "dists": [],
            "angles": [],
            "n_evaluated": 0,
        })

    n_frames = 0
    for ts in u.trajectory[::stride]:
        n_frames += 1
        coords = u.atoms.positions
        box = ts.dimensions if ts.dimensions is not None and np.all(ts.dimensions[:3] > 0) else None
        for item in resolved:
            p = item["spec"]
            elec = coords[item["elec_idx"]]
            ref = coords[item["ref_idx"]] if item["ref_idx"] is not None else None
            nuc_idx = item["nuc_idx"]
            if len(nuc_idx) == 0:
                continue
            nuc_pts = coords[nuc_idx]
            d_cut = float(p["distance_cutoff_A"])
            hit, d_min, ang = _eval_pathway_coords(
                nuc_pts, elec, ref, d_cut, p.get("ideal_angle_deg"), p.get("angle_tolerance_deg"), box=box,
            )
            item["n_evaluated"] += 1
            if hit:
                item["hits"] += 1
            if d_min is not None:
                item["dists"].append(d_min)
            if ang is not None:
                item["angles"].append(ang)

    stats = {item["spec"]["name"]: item for item in resolved}
    return _summarize_side_reactions(stats, source="md_trajectory", n_input=n_frames)


def _summarize_side_reactions(stats: dict[str, dict[str, Any]], *, source: str, n_input: int) -> dict[str, Any]:
    out = []
    for name, st in stats.items():
        n_ev = st["n_evaluated"]
        spec = st["spec"]
        out.append({
            "name": name,
            "kind": spec.get("kind", "intermolecular"),
            "n_evaluated": n_ev,
            "n_hits": st["hits"],
            "occupancy_fraction": round(st["hits"] / n_ev, 4) if n_ev else None,
            "min_distance_A_mean": round(float(np.mean(st["dists"])), 2) if st["dists"] else None,
            "closest_approach_A": round(float(np.min(st["dists"])), 2) if st["dists"] else None,
            "angle_deg_mean": round(float(np.mean(st["angles"])), 1) if st["angles"] else None,
            "distance_cutoff_A": spec["distance_cutoff_A"],
            "ideal_angle_deg": spec.get("ideal_angle_deg"),
            "angle_tolerance_deg": spec.get("angle_tolerance_deg"),
            "warning_threshold": spec.get("warning_threshold", 0.30),
        })
    return {
        "status": "evaluated",
        "source": source,
        "n_structures": n_input,
        "pathways": out,
        "scope": "Geometric competing-pathway occupancy; does not compute kinetic selectivity ratios without barriers.",
    }
