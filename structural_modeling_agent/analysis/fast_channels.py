"""Fast native Python Voronoi/Delaunay substrate-channel finder.

Computes 1D transport pathways from a buried active-site start point to bulk
solvent using 3D Delaunay triangulation (exact Voronoi vertices + face-adjacent
edges with true VDW-subtracted clearances) and CAVER's cost function
``sum(length / r^2)`` via ``scipy.sparse.csgraph.dijkstra``.

Compared with CAVER 3.0:
  * No 12-ball atom expansion and no JVM/Java heap overhead.
  * Embarrassingly parallel across poses via ProcessPoolExecutor.
  * Emits the exact same cluster schema and ``tun_cl_NNN_1.pdb`` sphere-string
    files so ``render_channels.py`` and ``channels_html.py`` work unchanged.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import os
import statistics as st
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import Delaunay, cKDTree

from ..io_utils import bounded_workers, init_single_thread_worker

DEFAULT_PROBE_RADIUS = 0.9
DEFAULT_SHELL_RADIUS = 3.0
DEFAULT_START_OPT_DIST = 3.0
DEFAULT_BOTTLENECK_CONTACT = 3.0
DEFAULT_EXIT_CLUSTER_TOL = 4.0
DEFAULT_MAX_TUNNELS_PER_POSE = 8

VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.10,
    "MG": 1.73, "MN": 1.73, "ZN": 1.39, "CA": 2.31, "NA": 2.27, "K": 2.75, "FE": 1.56,
}


def _read_pdb_atoms(path: str):
    """Fast pure-Python PDB reader returning (xyz, radii, res_labels) for heavy atoms."""
    coords = []
    radii = []
    labels = []
    with open(path) as fh:
        for ln in fh:
            if not ln.startswith(("ATOM  ", "HETATM")):
                continue
            el = ln[76:78].strip().upper()
            if not el:
                name = ln[12:16].strip()
                el = "".join(c for c in name if c.isalpha()).upper()[:1]
            if el in ("H", "D"):
                continue
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
            chain = ln[21].strip() or "A"
            resi = ln[22:26].strip()
            coords.append((x, y, z))
            radii.append(VDW.get(el, 1.70))
            labels.append(f"{chain}:{resi}")
    if not coords:
        return np.zeros((0, 3), float), np.zeros((0,), float), []
    return np.asarray(coords, dtype=float), np.asarray(radii, dtype=float), labels


def _tetra_circumcenters(pts: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Vectorized circumcenters of 3D tetrahedra (returns NaN for degenerate ones)."""
    p0 = pts[simplices[:, 0]]
    a = pts[simplices[:, 1]] - p0
    b = pts[simplices[:, 2]] - p0
    c = pts[simplices[:, 3]] - p0

    b_cross_c = np.cross(b, c)
    c_cross_a = np.cross(c, a)
    a_cross_b = np.cross(a, b)
    denom = 2.0 * np.einsum("ij,ij->i", a, b_cross_c)

    a2 = np.einsum("ij,ij->i", a, a)
    b2 = np.einsum("ij,ij->i", b, b)
    c2 = np.einsum("ij,ij->i", c, c)

    num = a2[:, None] * b_cross_c + b2[:, None] * c_cross_a + c2[:, None] * a_cross_b
    out = np.full_like(p0, np.nan)
    good = np.abs(denom) > 1e-8
    out[good] = p0[good] + num[good] / denom[good, None]
    return out


def _resample_spheres(pts: np.ndarray, radii: np.ndarray, step: float = 0.5) -> list[list[float]]:
    """Resample a polyline of (xyz, r) at ~`step` A for smooth tube visualization."""
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        return [[float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2]), float(radii[0])]]
    out = [[float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2]), float(radii[0])]]
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        r0, r1 = float(radii[i]), float(radii[i + 1])
        seg_len = float(np.linalg.norm(p1 - p0))
        n_sub = max(1, int(math.ceil(seg_len / step)))
        for k in range(1, n_sub + 1):
            t = k / n_sub
            p = p0 + t * (p1 - p0)
            r = r0 + t * (r1 - r0)
            out.append([float(p[0]), float(p[1]), float(p[2]), float(r)])
    return out


def _find_tunnels_one_pose(
    pdb_path: str,
    start_xyz: np.ndarray,
    *,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    shell_radius: float = DEFAULT_SHELL_RADIUS,
    start_opt_dist: float = DEFAULT_START_OPT_DIST,
    bottleneck_contact: float = DEFAULT_BOTTLENECK_CONTACT,
    exit_cluster_tol: float = DEFAULT_EXIT_CLUSTER_TOL,
    max_tunnels: int = DEFAULT_MAX_TUNNELS_PER_POSE,
) -> list[dict[str, Any]]:
    """Find distinct low-cost transport tunnels from `start_xyz` to bulk solvent in one PDB."""
    xyz, radii, res_labels = _read_pdb_atoms(pdb_path)
    if len(xyz) < 4:
        return []

    start_xyz = np.asarray(start_xyz, dtype=float)
    bbox_min = xyz.min(axis=0) - shell_radius - 2.0
    bbox_max = xyz.max(axis=0) + shell_radius + 2.0

    # 1. 3D Delaunay triangulation (C/Qhull) -> Voronoi vertices (circumcenters)
    tri = Delaunay(xyz)
    simplices = tri.simplices
    neighbors = tri.neighbors
    centers = _tetra_circumcenters(xyz, simplices)

    # Filter out degenerate/far-away circumcenters
    finite = np.isfinite(centers).all(axis=1)
    in_box = finite & (centers >= bbox_min).all(axis=1) & (centers <= bbox_max).all(axis=1)
    # Convex-hull boundary tetrahedra have at least one neighbor == -1
    on_hull = (neighbors < 0).any(axis=1)

    atom_tree = cKDTree(xyz)
    valid_idx = np.flatnonzero(in_box)
    if len(valid_idx) == 0:
        return []

    # 2. Exact VDW-subtracted clearance at each Voronoi vertex
    valid_centers = centers[valid_idx]
    k_q = min(8, len(xyz))
    dists, near_atoms = atom_tree.query(valid_centers, k=k_q)
    if k_q == 1:
        dists = dists[:, None]
        near_atoms = near_atoms[:, None]
    v_clear = (dists - radii[near_atoms]).min(axis=1)

    # Keep passable vertices (clearance >= probe_radius)
    pass_mask = v_clear >= probe_radius
    if not pass_mask.any():
        return []

    pass_tet = valid_idx[pass_mask]
    pass_xyz = valid_centers[pass_mask]
    pass_r = v_clear[pass_mask]
    pass_hull = on_hull[pass_tet]
    n_pass = len(pass_tet)

    # Map original tetrahedron index -> compact passable node index (-1 if impassable)
    tet_to_node = np.full(len(simplices), -1, dtype=np.int32)
    tet_to_node[pass_tet] = np.arange(n_pass, dtype=np.int32)

    # Also treat any passable tetrahedron adjacent to an out-of-box tetrahedron as exterior-touching
    adj_tets = neighbors[pass_tet]  # shape (n_pass, 4)
    adj_oob = (adj_tets >= 0) & (~in_box[np.maximum(adj_tets, 0)])
    touches_exterior = pass_hull | adj_oob.any(axis=1)

    # 3. Build passable Voronoi edges between adjacent passable tetrahedra
    src_nodes = np.repeat(np.arange(n_pass, dtype=np.int32), 4)
    dst_tets = adj_tets.ravel()
    valid_e = dst_tets >= 0
    src_nodes = src_nodes[valid_e]
    dst_nodes = tet_to_node[dst_tets[valid_e]]
    keep_e = (dst_nodes >= 0) & (src_nodes < dst_nodes)
    src_nodes = src_nodes[keep_e]
    dst_nodes = dst_nodes[keep_e]

    if len(src_nodes) == 0:
        return []

    # Edge clearance: min of endpoint clearances and midpoint clearance (exact against nearby atoms)
    p_u = pass_xyz[src_nodes]
    p_v = pass_xyz[dst_nodes]
    seg_vec = p_v - p_u
    seg_len = np.linalg.norm(seg_vec, axis=1)
    mid_pts = 0.5 * (p_u + p_v)
    m_dists, m_atoms = atom_tree.query(mid_pts, k=min(4, len(xyz)))
    if m_dists.ndim == 1:
        m_dists = m_dists[:, None]
        m_atoms = m_atoms[:, None]
    mid_clear = (m_dists - radii[m_atoms]).min(axis=1)
    edge_r = np.minimum(np.minimum(pass_r[src_nodes], pass_r[dst_nodes]), mid_clear)

    e_ok = edge_r >= probe_radius
    src_nodes = src_nodes[e_ok]
    dst_nodes = dst_nodes[e_ok]
    seg_len = seg_len[e_ok]
    edge_r = edge_r[e_ok]

    if len(src_nodes) == 0:
        return []

    # 4. Identify bulk-solvent nodes: connected component of exterior through r >= shell_radius
    wide_node = (pass_r >= shell_radius) | touches_exterior
    wide_e = wide_node[src_nodes] & wide_node[dst_nodes] & (edge_r >= shell_radius * 0.85)
    if wide_e.any():
        wide_graph = csr_matrix(
            (np.ones(2 * int(wide_e.sum()), dtype=np.int8),
             (np.concatenate([src_nodes[wide_e], dst_nodes[wide_e]]),
              np.concatenate([dst_nodes[wide_e], src_nodes[wide_e]]))),
            shape=(n_pass, n_pass),
        )
        _, labels = connected_components(wide_graph, directed=False)
        ext_components = set(labels[touches_exterior])
        is_bulk = np.isin(labels, list(ext_components)) & wide_node
    else:
        is_bulk = touches_exterior

    if not is_bulk.any():
        is_bulk = touches_exterior
        if not is_bulk.any():
            return []

    # 5. Build CAVER-weighted passable graph (cost = seg_len / edge_r^2) first so
    # start-point optimization only picks a cavity vertex that can reach bulk solvent.
    # Determine whether the neighborhood of start_xyz already sits in bulk.
    node_tree = cKDTree(pass_xyz)
    raw_cand = node_tree.query_ball_point(start_xyz, r=start_opt_dist)
    if not raw_cand:
        d0, i0 = node_tree.query(start_xyz, k=1)
        if float(d0) > 5.0:
            return []
        raw_cand = [int(i0)]
    cand_arr = np.asarray(raw_cand, dtype=int)

    # If every candidate near start_xyz is already in bulk (e.g. a wide hollow tube > shell_radius),
    # terminate tunnels at the convex hull exterior instead of immediately at start_node.
    start_in_bulk = bool(is_bulk[cand_arr].all())
    target_mask = (touches_exterior if start_in_bulk else is_bulk).copy()
    keep_graph = ~(target_mask[src_nodes] & target_mask[dst_nodes])

    u_e = src_nodes[keep_graph]
    v_e = dst_nodes[keep_graph]
    w_e = seg_len[keep_graph] / (np.maximum(edge_r[keep_graph], 1e-3) ** 2)

    graph = csr_matrix(
        (np.concatenate([w_e, w_e]), (np.concatenate([u_e, v_e]), np.concatenate([v_e, u_e]))),
        shape=(n_pass, n_pass),
    )

    # Restrict start candidates to vertices connected to at least one surface/bulk target
    _, comp_labels = connected_components(graph, directed=False)
    bulk_comps = set(comp_labels[target_mask])
    reachable_cand = cand_arr[np.isin(comp_labels[cand_arr], list(bulk_comps))]
    if len(reachable_cand) == 0:
        return []
    # Prefer widest cavity center within start_opt_dist (mirrors CAVER starting_point_optimization)
    d_cand = np.linalg.norm(pass_xyz[reachable_cand] - start_xyz, axis=1)
    eff_r = np.minimum(pass_r[reachable_cand], 5.0)
    start_node = int(reachable_cand[int(np.argmax(eff_r - 0.05 * d_cand))])
    if not start_in_bulk:
        target_mask[start_node] = False

    # 6. Run Dijkstra once from start_node
    costs, predecessors = dijkstra(graph, directed=False, indices=start_node, return_predecessors=True)

    bulk_nodes = np.flatnonzero(target_mask & np.isfinite(costs))
    if len(bulk_nodes) == 0:
        return []
    order = bulk_nodes[np.argsort(costs[bulk_nodes])]

    # 7. Extract distinct tunnels (greedy by lowest cost, deduplicated by exit & path overlap)
    tunnels: list[dict[str, Any]] = []
    accepted_exits: list[np.ndarray] = []
    accepted_node_sets: list[set[int]] = []
    snap_name = os.path.basename(pdb_path)

    for target in order:
        if len(tunnels) >= max_tunnels:
            break
        exit_pt = pass_xyz[int(target)]
        if any(float(np.linalg.norm(exit_pt - prev)) <= exit_cluster_tol for prev in accepted_exits):
            continue

        # Reconstruct path from start_node -> target
        path_nodes = []
        cur = int(target)
        while cur != -9999 and cur != start_node:
            path_nodes.append(cur)
            cur = int(predecessors[cur])
        if cur != start_node:
            continue
        path_nodes.append(start_node)
        path_nodes.reverse()

        # Trim at the FIRST terminal surface node encountered along the path
        first_term = next((idx for idx, n in enumerate(path_nodes) if target_mask[n] and idx > 0), len(path_nodes) - 1)
        path_nodes = path_nodes[: first_term + 1]
        exit_pt = pass_xyz[path_nodes[-1]]
        if any(float(np.linalg.norm(exit_pt - prev)) <= exit_cluster_tol for prev in accepted_exits):
            continue

        # Collapse surface fan-outs that share >50% of the same internal tunnel path
        inner_nodes = set(path_nodes[1:-1]) if len(path_nodes) > 2 else set(path_nodes)
        if inner_nodes and any(
            len(inner_nodes & prev_set) / min(len(inner_nodes), len(prev_set)) > 0.5
            for prev_set in accepted_node_sets
            if prev_set
        ):
            continue

        pts = pass_xyz[path_nodes]
        rs = pass_r[path_nodes].copy()
        if len(path_nodes) >= 2:
            seg_lens = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            length = float(seg_lens.sum())
            # Compute midpoint clearance directly for the ~20-40 extracted path edges
            # instead of maintaining an O(E) Python edge_r_map dict over all Voronoi edges.
            mids = 0.5 * (pts[:-1] + pts[1:])
            md_edge, mi_edge = atom_tree.query(mids, k=min(4, len(radii)))
            md_edge = np.asarray(md_edge).reshape(len(mids), -1)
            mi_edge = np.asarray(mi_edge).reshape(len(mids), -1)
            mid_r = (md_edge - radii[mi_edge]).min(axis=1)
            e_rs = np.minimum(mid_r, np.minimum(rs[:-1], rs[1:]))
            min_v_idx = int(np.argmin(rs))
            min_e_idx = int(np.argmin(e_rs))
            if e_rs[min_e_idx] < rs[min_v_idx]:
                bottleneck_r = float(e_rs[min_e_idx])
                bottleneck_xyz = 0.5 * (pts[min_e_idx] + pts[min_e_idx + 1])
            else:
                bottleneck_r = float(rs[min_v_idx])
                bottleneck_xyz = pts[min_v_idx]
            cost = float((seg_lens / (np.maximum(e_rs, 1e-3) ** 2)).sum())
        else:
            length = 0.0
            bottleneck_r = float(rs[0])
            bottleneck_xyz = pts[0]
            cost = 0.0

        straight = float(np.linalg.norm(pts[-1] - pts[0]))
        curvature = float(length / straight) if straight > 1e-6 else 1.0
        throughput = float(math.exp(-cost))

        near_bn = atom_tree.query_ball_point(bottleneck_xyz, r=bottleneck_r + bottleneck_contact + 2.5)
        lining_res = sorted({
            res_labels[idx]
            for idx in near_bn
            if float(np.linalg.norm(xyz[idx] - bottleneck_xyz)) - radii[idx] <= bottleneck_r + bottleneck_contact
        })

        accepted_exits.append(exit_pt)
        accepted_node_sets.append(inner_nodes)
        tunnels.append({
            "snapshot": snap_name,
            "throughput": throughput,
            "cost": cost,
            "bottleneck_A": bottleneck_r,
            "length_A": length,
            "curvature": curvature,
            "start_xyz": [float(x) for x in pts[0]],
            "exit_xyz": [float(x) for x in pts[-1]],
            "bottleneck_xyz": [float(x) for x in bottleneck_xyz],
            "lining_residues": lining_res,
            "spheres": _resample_spheres(pts, rs, step=0.5),
        })

    return tunnels


def _worker_pose(args):
    pdb_path, start_xyz, probe_radius, shell_radius = args
    return _find_tunnels_one_pose(
        pdb_path, start_xyz, probe_radius=probe_radius, shell_radius=shell_radius
    )


def _write_tunnel_pdb(path: str, spheres: list[list[float]]) -> None:
    """Write a CAVER-compatible sphere-string PDB (B-factor = local radius in A)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for i, (x, y, z, r) in enumerate(spheres, 1):
            fh.write(
                f"HETATM{i:5d}  H   TUN T{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{r:6.2f}           H\n"
            )
        fh.write("END\n")


def run_fast_channels(
    pdb_dir: str,
    start_xyz: list[float] | np.ndarray,
    *,
    out_dir: str | None = None,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    shell_radius: float = DEFAULT_SHELL_RADIUS,
    exit_cluster_tol: float = DEFAULT_EXIT_CLUSTER_TOL,
    max_workers: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the fast Voronoi/Delaunay channel finder across all PDBs in `pdb_dir`.

    Returns ``(clusters, warnings)`` in the exact same schema as
    ``substrate_channels.parse_caver_output()``. When ``out_dir`` is given, also
    writes CAVER-compatible ``caver_out/data/clusters_timeless/tun_cl_NNN_1.pdb``
    files so ``render_channels.py`` and ``channels_html.py`` work without edits.
    """
    pdb_files = sorted(
        os.path.join(pdb_dir, f)
        for f in os.listdir(pdb_dir)
        if f.startswith("pose_") and f.endswith(".pdb")
    )
    n_poses = len(pdb_files)
    if n_poses == 0:
        return [], []

    tasks = [(p, start_xyz, probe_radius, shell_radius) for p in pdb_files]
    workers = bounded_workers(max_workers, n_items=n_poses, min_items_for_pool=4, max_cap=8)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_single_thread_worker) as pool:
            per_pose_tunnels = list(pool.map(_worker_pose, tasks))
    else:
        per_pose_tunnels = [_worker_pose(t) for t in tasks]

    all_tunnels = [t for lst in per_pose_tunnels for t in lst]
    if not all_tunnels:
        return [], []

    # Cluster tunnels across poses by exit coordinates (average-link, matching CAVER)
    if len(all_tunnels) == 1:
        cluster_ids = np.array([1])
    else:
        exits = np.asarray([t["exit_xyz"] for t in all_tunnels], dtype=float)
        Z = linkage(exits, method="average")
        cluster_ids = fcluster(Z, t=exit_cluster_tol, criterion="distance")

    # One cheapest (highest-throughput) tunnel per snapshot per cluster
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for t, cid in zip(all_tunnels, cluster_ids):
        cid_int = int(cid)
        snap = t["snapshot"]
        by_snap = grouped.setdefault(cid_int, {})
        if snap not in by_snap or t["throughput"] > by_snap[snap]["throughput"]:
            by_snap[snap] = t

    sd = lambda xs: float(st.pstdev(xs)) if len(xs) > 1 else 0.0
    raw_clusters = []
    for by_snap in grouped.values():
        P = list(by_snap.values())
        n = len(P)
        br = [p["bottleneck_A"] for p in P]
        tp = [p["throughput"] for p in P]
        lining_counts: dict[str, int] = {}
        for p in P:
            for res in p["lining_residues"]:
                lining_counts[res] = lining_counts.get(res, 0) + 1
        top = sorted(lining_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        ex = np.asarray([p["exit_xyz"] for p in P], dtype=float)
        c0 = np.asarray([p["start_xyz"] for p in P], dtype=float)
        exit_mean = ex.mean(axis=0)
        rep = max(P, key=lambda p: p["throughput"])
        raw_clusters.append({
            "n_poses": n,
            "frequency": round(n / n_poses, 4),
            "bottleneck_A": {
                "mean": round(st.mean(br), 3),
                "sd": round(sd(br), 3),
                "min": round(min(br), 3),
                "max": round(max(br), 3),
            },
            "length_A": {
                "mean": round(st.mean(p["length_A"] for p in P), 2),
                "sd": round(sd([p["length_A"] for p in P]), 2),
            },
            "curvature_mean": round(st.mean(p["curvature"] for p in P), 3),
            "throughput_mean": round(st.mean(tp), 4),
            "priority": round(sum(tp) / n_poses, 4),
            "exit_xyz": [round(float(x), 2) for x in exit_mean],
            "exit_spread_A": round(float(np.linalg.norm(ex - exit_mean, axis=1).max()), 2),
            "caver_start_xyz": [round(float(x), 2) for x in c0.mean(axis=0)],
            "bottleneck_lining_residues": [{"residue": k, "fraction": round(v / n, 3)} for k, v in top],
            "_rep_spheres": rep["spheres"],
        })

    raw_clusters.sort(key=lambda c: -c["priority"])
    clusters = []
    for rank, c in enumerate(raw_clusters, 1):
        spheres = c.pop("_rep_spheres")
        c["cluster"] = rank
        if out_dir is not None and spheres:
            pdb_out = os.path.join(out_dir, "caver_out", "data", "clusters_timeless", f"tun_cl_{rank:03d}_1.pdb")
            _write_tunnel_pdb(pdb_out, spheres)
        clusters.append(c)

    return clusters, []
