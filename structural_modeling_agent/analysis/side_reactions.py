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
from scipy.spatial import cKDTree

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
    """Read all heavy atoms including solvent (needed for water-attack checks)."""
    p = Path(path)
    index: dict[tuple[str, str, str], np.ndarray] = {}
    solvent: list[dict[str, Any]] = []
    if p.suffix.lower() in {".cif", ".mmcif"}:
        import gemmi
        st = gemmi.read_structure(str(p))
        for chain in st[0]:
            for res in chain:
                rn = res.name.strip().upper()
                resid = f"{res.seqid.num}{res.seqid.icode.strip()}"
                for atom in res:
                    if atom.altloc not in ("\x00", " ", "A") or atom.element.is_hydrogen:
                        continue
                    xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                    name = atom.name.strip()
                    index[(chain.name, resid, name)] = xyz
                    if rn in DEFAULT_SOLVENT_RESNAMES:
                        solvent.append({"resname": rn, "atom": name, "xyz": xyz})
        return index, solvent

    with p.open() as fh:
        for ln in fh:
            if not ln.startswith(("ATOM  ", "HETATM")):
                continue
            if ln[16] not in (" ", "A"):
                continue
            el = ln[76:78].strip().upper()
            name = ln[12:16].strip()
            if not el:
                el = "".join(c for c in name if c.isalpha()).upper()[:1]
            if el in ("H", "D"):
                continue
            rn = ln[17:20].strip().upper()
            chain = ln[21].strip() or "A"
            resid = ln[22:27].strip()
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])], dtype=float)
            index[(chain, resid, name)] = xyz
            if rn in DEFAULT_SOLVENT_RESNAMES:
                solvent.append({"resname": rn, "atom": name, "xyz": xyz})
    return index, solvent


def _eval_pathway_coords(
    nuc_pts: np.ndarray,
    elec_xyz: np.ndarray,
    ref_xyz: np.ndarray | None,
    distance_cutoff_A: float,
    ideal_angle_deg: float | None,
    angle_tolerance_deg: float | None,
) -> tuple[bool, float | None, float | None]:
    """Check whether any nucleophile in `nuc_pts` meets the competing-attack geometry."""
    if len(nuc_pts) == 0:
        return False, None, None
    dists = np.linalg.norm(nuc_pts - elec_xyz[None, :], axis=1)
    min_idx = int(np.argmin(dists))
    min_d = float(dists[min_idx])
    within = np.flatnonzero(dists <= distance_cutoff_A)
    if len(within) == 0:
        best_ang = _angle_deg(nuc_pts[min_idx], elec_xyz, ref_xyz) if ref_xyz is not None else None
        return False, min_d, best_ang

    if ref_xyz is None or ideal_angle_deg is None or angle_tolerance_deg is None:
        return True, min_d, None

    best_ang = None
    best_dev = 1e9
    hit = False
    for idx in within:
        ang = _angle_deg(nuc_pts[idx], elec_xyz, ref_xyz)
        dev = abs(ang - ideal_angle_deg)
        if dev < best_dev:
            best_dev = dev
            best_ang = ang
        if dev <= angle_tolerance_deg:
            hit = True
    return hit, min_d, best_ang


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
    for _ in u.trajectory[::stride]:
        n_frames += 1
        coords = u.atoms.positions
        for item in resolved:
            p = item["spec"]
            elec = coords[item["elec_idx"]]
            ref = coords[item["ref_idx"]] if item["ref_idx"] is not None else None
            nuc_idx = item["nuc_idx"]
            if len(nuc_idx) == 0:
                continue
            nuc_pts = coords[nuc_idx]
            # Fast cKDTree prefilter when nucleophile pool is large (e.g. 10,000+ waters)
            d_cut = float(p["distance_cutoff_A"])
            if len(nuc_pts) > 256:
                tree = cKDTree(nuc_pts)
                near = tree.query_ball_point(elec, r=d_cut)
                if near:
                    sub_pts = nuc_pts[near]
                    hit, d_min, ang = _eval_pathway_coords(
                        sub_pts, elec, ref, d_cut, p.get("ideal_angle_deg"), p.get("angle_tolerance_deg")
                    )
                else:
                    d_min_val, idx0 = tree.query(elec, k=1)
                    hit = False
                    d_min = float(d_min_val)
                    ang = _angle_deg(nuc_pts[int(idx0)], elec, ref) if ref is not None else None
            else:
                hit, d_min, ang = _eval_pathway_coords(
                    nuc_pts, elec, ref, d_cut, p.get("ideal_angle_deg"), p.get("angle_tolerance_deg")
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
