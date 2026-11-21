"""Substrate-channel (tunnel) analysis for the structural modeling agent -- a CAVER 3 wrapper.

Active-site geometry (attack_geometry, substrate_contacts) says whether a substrate sits in a
productive pose. It says nothing about HOW A SUBSTRATE GETS THERE: is there a transport pathway from
bulk solvent to the catalytic center, how narrow is its bottleneck, how long and winding is it, and
which residues line the bottleneck? That is a channel question, and for a step-2 nucleophile (e.g. a
C5 diamine that must reach an already-bound acyl intermediate) it is often the decisive one.

ENGINE: CAVER 3.0 (Chovancova et al. 2012, PLoS Comput Biol 8:e1002708) -- Voronoi-based tunnel
search designed for tunnels from a buried active site to the surface, with built-in clustering across
an ENSEMBLE of structures. We do not reimplement it. A cofold ensemble (50 poses) is exactly its
multi-snapshot input, so every result here is a FREQUENCY over the ensemble, never one hero pose.

DETECTS, DOES NOT JUDGE. Like substrate_contacts, this reports measurements (cluster frequency,
bottleneck radius, length, curvature, throughput, bottleneck-lining residues, exit position) and
renders NO verdict -- no "channel is wide enough", no score, no ranking, no composite. Tunnel topology
was already tried as a ranking proxy in this project and came out null (wrong sign, p=0.99, 7
families; memory `project_selectivity_proxy_null_oct9`). A static bottleneck radius is also NOT
comparable to a ligand's cross-section as pass/fail: GlnA3's native L-glutamate (cross-section radius
~3.3 A) is wider than the 1.4-1.5 A bottlenecks of its own channels, so channels must breathe.

ROBUSTNESS (what makes a "channel" call trustworthy, and what each guard is for):
  * SURFACE-EXPOSED SITES ARE NOT CHANNELS. CAVER applies a start-protection radius (4 A). A cluster
    shorter than that never left the protected zone: it is the solvent surface at the start point, not
    a pathway. Such clusters are labelled `surface_exposed`; if one is persistent the SITE is flagged
    open. (Without this, open sites reported 1.9-3 A "tunnels" with throughput ~0.9 as if they were the
    widest channels.)
  * START-POINT STABILITY. CAVER measures from one coordinate (and silently relocates it). The ensemble
    is re-run from `n_starts-1` perturbed starts (tetrahedral offsets of `start_radius` A). A channel is
    reported with the fraction of perturbed runs in which a channel with the same exit (<= 4 A) recurs.
  * BURIAL. An independent ray-cast burial fraction at the start point (CAVER-free), so openness is
    not read from CAVER alone.
  * ASSEMBLY. Pass the FULL biological assembly. In a controlled test on the same 5 GlnA3 dimer poses
    and start point, dropping the partner subunit turned a protected site into a 1.8 A, 2.5 A-bottleneck
    "tunnel" that disappears when the partner is present; a 16 A channel was unchanged (1.53 vs 1.54 A).
    Single-chain input is flagged.
  * FILTERS ARE EXPLICIT: persistence >= min_freq, length >= protection radius, bottleneck >= probe +
    margin; the thresholds are recorded in the output, and every cluster keeps its raw numbers.

Procedure per ensemble:
  1. superpose every pose onto pose 0 (Kabsch on all protein C-alpha) -- CAVER clusters tunnels across
     snapshots and needs a common frame;
  2. strip the SUBSTRATE only; keep protein (all chains) + nucleotide cofactor + metals, because bound
     ATP/ADP and Mg physically occupy the site the tunnel must route around;
  3. start point = median (across poses) of the chosen start atom in the common frame; the spread across
     poses is reported. Priority: --start-xyz (coordinates) > --start-sel (an atom selection) >
     auto (the substrate functional group of kind --start-group nearest the reaction centre);
  4. run CAVER from the start and from the perturbed starts; validate that each really finished (exit
     code alone is not trusted);
  5. recompute cluster statistics from CAVER's per-snapshot tables (not from its pretty-printed summary).

Scope (general, enzyme-agnostic):
  * step-1 (a carboxylate substrate): the default --start-group carboxylate resolves the start.
  * step-2 (acyl intermediate + amine nucleophile): --start-group amine, or --start-sel on the amine N.
  * any substrate without a standard group, a named catalytic residue, a fixed point: --start-sel /
    --start-xyz. Pin the substrate/cofactor split with --substrate-resnames / --cofactor-resnames and
    the electrophile with --reaction-center-sel when auto-detection would guess wrong.
  * apo / cofactor-free structures: allowed with an explicit start (--start-sel / --start-xyz); whatever
    cofactor + metals are present are kept as obstacles, a bare protein is staged protein-only.

CLI:
  python -m structural_modeling_agent.analysis.substrate_channels run \\
      --poses-json analysis.json --out OUT_DIR            # list of {"path": ...} records
  python -m structural_modeling_agent.analysis.substrate_channels run --poses a.cif b.cif --out OUT_DIR
  # step-2 amine nucleophile:   ... run --poses ... --start-group amine --out OUT_DIR
  # apo enzyme from a residue:  ... run --poses apo.pdb --start-sel 'resid 106 and name OD1 OD2' --out OUT_DIR
  python -m structural_modeling_agent.analysis.substrate_channels parse --caver-out OUT/caver_out --n-poses 50

Needs: Java (>= 8) and CAVER 3 (CAVER_HOME, default below); numpy; scipy; MDAnalysis; gemmi (mmCIF).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import substrate_contacts as sc

CAVER_HOME = os.environ.get(
    "CAVER_HOME",
    "/nfs/lambda_stor_01/homes/rzhu/Tool/processivity-toolchain-20261106/sources/caver/Caver3")
DEFAULT_PROBE_RADIUS = 0.9     # CAVER's own default for protein tunnels
DEFAULT_XMX_GB = 12            # dimers (~7k atoms x 50 poses) peg an 8 GB heap -> GC thrash; 12 clears it
DEFAULT_THREADS = 4            # shared nodes: cap JVM CPU (global courtesy rule)
DEFAULT_N_STARTS = 5           # primary start + 4 perturbed (tetrahedral)
DEFAULT_STABILITY_POSES = 15   # perturbed-start runs use only this many poses (sensitivity, not stats);
#                                None/0 = all poses. The costly 50-pose ensemble runs ONCE (the primary);
#                                the 4 start-sensitivity runs only need to confirm the exit recurs, which
#                                a subset answers -> cuts the 5x start multiplier to ~2x with no stat loss.
DEFAULT_START_RADIUS = 2.0     # A, perturbation of the start point
EXIT_MATCH_TOL = 4.0           # A, two channels are "the same" if their exits are this close
DEFAULT_MIN_FREQ = 0.5         # a channel must occur in at least half the poses
DEFAULT_MARGIN = 0.1           # A, bottleneck must exceed the probe radius by this much
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.10, "MG": 1.73, "MN": 1.73,
       "ZN": 1.39, "CA": 2.31, "NA": 2.27, "K": 2.75, "FE": 1.56}

SINGLE_CHAIN_WARNING = (
    "single protein chain: if this enzyme's active site is capped by a partner subunit (e.g. the GS "
    "fold), a monomer model reads the site as solvent-open and OVERSTATES channel openness. In a "
    "controlled GlnA3 test, removing the partner turned a protected site into a 1.8 A, 2.5 A-bottleneck "
    "'tunnel'. Analyze the biological assembly if one is known; interpret monomer results with care.")
START_INCONSISTENT_NOTE = (
    "start point is NOT consistent across poses (the substrate's reactive carbon is scattered): "
    "channels are measured from a median position that many poses do not occupy; read cluster "
    "frequencies accordingly.")
SURFACE_EXPOSED_NOTE = (
    "SITE IS SURFACE-EXPOSED: a persistent cluster shorter than CAVER's start-protection radius means "
    "the start point already touches bulk solvent. Those short clusters are NOT channels (labelled "
    "surface_exposed); only longer clusters are reported as channels. Often a missing partner subunit, "
    "or a substrate that is not seated in the pocket.")


# --------------------------------------------------------------------------- burial (CAVER-free)
def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def burial_fraction(point, xyz, radii, probe=DEFAULT_PROBE_RADIUS, n_dir=256, max_len=20.0,
                    step=0.5, k=8):
    """Fraction of directions from `point` whose ray cannot carry a probe sphere of radius `probe`
    out `max_len` A without touching the obstacle surface. 1.0 = fully enclosed, ~0.5 = a half-space
    open, 0 = nothing around. Independent of CAVER."""
    from scipy.spatial import cKDTree
    xyz, radii = np.asarray(xyz, float), np.asarray(radii, float)
    if not len(xyz):
        return 0.0
    tree = cKDTree(xyz)
    s = np.arange(step, max_len + step, step)
    P = (np.asarray(point, float) + _fibonacci_sphere(n_dir)[:, None, :] * s[None, :, None]).reshape(-1, 3)
    d, idx = tree.query(P, k=min(k, len(xyz)))
    d, idx = d.reshape(len(P), -1), idx.reshape(len(P), -1)       # k=1 returns 1-D arrays
    gap = (d - radii[idx]).min(axis=1).reshape(n_dir, len(s))
    return float((gap < probe).any(axis=1).mean())


def _elements(ag):
    """Element symbols for an MDAnalysis AtomGroup (element column, else atom-name fallback)."""
    els = getattr(ag, "elements", None)
    if els is not None and len(els) and all(str(e).strip() for e in els):
        return [str(e).strip().upper() for e in els]
    return [sc._element(a) for a in ag]


def _radii_for(elements):
    return np.array([VDW.get(e, 1.70) for e in elements])


# --------------------------------------------------------------------------- staging
def _to_pdb(path, tmpdir):
    """MDAnalysis reads PDB; Boltz/Mol-1 write mmCIF -> convert with gemmi."""
    if path.lower().endswith((".cif", ".mmcif")):
        import gemmi
        st_ = gemmi.read_structure(path)
        st_.setup_entities()
        out = os.path.join(tmpdir, os.path.basename(path) + ".pdb")
        st_.write_pdb(out)
        return out
    return path


def _substrate_extent(sub):
    """Longest heavy-atom distance in the substrate (measured, for context only)."""
    if sub is None or len(sub) < 2:
        return None
    p = sub.positions
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    return float(d.max())


def _pose_start(u, p, start_xyz, start_sel, start_group, reaction_center_sel,
                substrate_resnames, cofactor_resnames, hetero_sel):
    """The start-point contribution of ONE pose, or None when `start_xyz` fixes it globally.

    Priority: explicit coordinates (handled by the caller, returns None here) > an MDAnalysis atom
    selection (its centre of geometry) > auto (the substrate functional group of kind `start_group`
    nearest the reaction centre; the first such group if there is no reaction centre to rank by)."""
    if start_xyz is not None:
        return None
    if start_sel is not None:
        sel = u.select_atoms(start_sel)
        if not len(sel):
            raise ValueError(f"{p}: --start-sel '{start_sel}' matched no atoms")
        return sel.center_of_geometry()
    try:
        g = sc.identify_groups(u, reaction_center_sel=reaction_center_sel,
                               substrate_resnames=substrate_resnames,
                               cofactor_resnames=cofactor_resnames, hetero_sel=hetero_sel)
    except ValueError as e:
        raise ValueError(
            f"{p}: cannot auto-place the start ({e}); pass an explicit start (--start-xyz) or an "
            "atom selection (--start-sel) for apo / step-2 / non-carboxylate substrates") from e
    kind_groups = [fg for fg in g["functional_groups"] if fg["kind"] == start_group]
    if not kind_groups:
        kinds = sorted({fg["kind"] for fg in g["functional_groups"]})
        raise ValueError(
            f"{p}: no '{start_group}' group on the substrate (groups found: {kinds or 'none'}); "
            "choose --start-group from those, or pass --start-sel / --start-xyz")
    ranked = [fg for fg in kind_groups if fg.get("d_to_center") is not None]
    chosen = min(ranked, key=lambda fg: fg["d_to_center"]) if ranked else kind_groups[0]
    return chosen["center"].position.copy()


def stage_ensemble(pose_paths, workdir, start_xyz=None, start_sel=None, start_group="carboxylate",
                   reaction_center_sel=None, substrate_resnames=None, cofactor_resnames=None,
                   hetero_sel=None, probe_radius=DEFAULT_PROBE_RADIUS,
                   keep_cofactor=True, keep_metals=True):
    """Align poses, strip substrate, write one PDB per pose for CAVER. Returns a manifest dict.

    Start point (priority): `start_xyz` (explicit coordinates) > `start_sel` (an MDAnalysis selection;
    start = median centre of geometry across poses) > auto (the substrate functional group of kind
    `start_group` -- default carboxylate for a step-1 acid; e.g. `amine` for a step-2 nucleophile --
    nearest the reaction centre). The substrate/cofactor split and the reaction centre can be pinned
    with `substrate_resnames` / `cofactor_resnames` / `reaction_center_sel` / `hetero_sel`.

    Apo / ligand-free / cofactor-free inputs are allowed WHEN the start is given explicitly (start_xyz
    or start_sel): whatever cofactor + metals are present are kept as obstacles, and a bare protein is
    staged protein-only. Auto start still requires a resolvable substrate.

    Fails loud (never silently degrades): empty input, pose-to-pose CA-count mismatch, a start
    selection that matches no atoms, and an auto start with no substrate / no group of the requested
    kind all raise ValueError."""
    import MDAnalysis as mda
    from MDAnalysis.analysis.align import rotation_matrix
    if not pose_paths:
        raise ValueError("no poses given")
    explicit = start_xyz is not None or start_sel is not None
    pdb_dir = os.path.join(workdir, "pdb")
    shutil.rmtree(pdb_dir, ignore_errors=True)
    os.makedirs(pdb_dir)
    tmp = tempfile.mkdtemp(prefix="substrate_channels_")
    per_pose, starts, obstacles, ref_ca = [], [], [], None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, p in enumerate(pose_paths):
                u = mda.Universe(_to_pdb(p, tmp))
                ca = u.select_atoms("protein and name CA")
                if not len(ca):
                    raise ValueError(f"{p}: no protein C-alpha atoms found")
                if ref_ca is None:
                    ref_ca = ca.positions.copy()
                elif len(ca) != len(ref_ca):
                    raise ValueError(f"{p}: {len(ca)} C-alpha vs {len(ref_ca)} in pose 0 -- not the "
                                     "same system; an ensemble must share one topology")
                R, _ = rotation_matrix(ca.positions - ca.positions.mean(0),
                                       ref_ca - ref_ca.mean(0))
                u.atoms.positions = (u.atoms.positions - ca.positions.mean(0)) @ R.T + ref_ca.mean(0)
                # ligand split (graceful: apo / cofactor-free is allowed when the start is explicit)
                try:
                    cof, metals, substrate = sc.split_ligands(
                        u, hetero_sel=hetero_sel, substrate_resnames=substrate_resnames,
                        cofactor_resnames=cofactor_resnames)
                except ValueError:
                    if not explicit:
                        raise
                    cof = metals = substrate = u.atoms[[]]
                center = _pose_start(u, p, start_xyz, start_sel, start_group, reaction_center_sel,
                                     substrate_resnames, cofactor_resnames, hetero_sel)
                if center is not None:
                    starts.append(center)
                # substrate is ALWAYS stripped (its path is what we trace); cofactor + metals are kept
                # as obstacles by default (tunnels route around bound ATP*Mg), or stripped for the
                # bare-pocket topology
                keep = u.select_atoms("protein")
                if keep_cofactor:
                    keep = keep + cof
                if keep_metals:
                    keep = keep + metals
                keep.write(os.path.join(pdb_dir, f"pose_{i:04d}.pdb"))
                obstacles.append((keep.positions.copy(), _radii_for(_elements(keep))))
                prot = u.select_atoms("protein")
                chain_ids = set(prot.chainIDs) if hasattr(prot, "chainIDs") else set(prot.segids)
                per_pose.append({
                    "pose": p,
                    "n_protein_chains": len(chain_ids) or 1,
                    "n_protein_residues": int(prot.n_residues),
                    "substrate_extent_A": _substrate_extent(substrate),
                    "reactive_center": None if center is None else [round(float(x), 3) for x in center],
                })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if start_xyz is not None:
        start, how, spread = np.asarray(start_xyz, float), "explicit", None
    else:
        S = np.array(starts)
        start = np.median(S, axis=0)
        d = np.linalg.norm(S - start, axis=1)
        how = "explicit_selection" if start_sel is not None else f"median_reactive_{start_group}"
        spread = {
            "max_A": round(float(d.max()), 2), "median_A": round(float(np.median(d)), 2),
            "frac_within_4A": round(float((d <= 4.0).mean()), 3)}
    if len(obstacles) >= 4:
        with ThreadPoolExecutor(max_workers=min(8, len(obstacles))) as ex:
            bur = list(ex.map(lambda item: burial_fraction(start, item[0], item[1], probe=probe_radius), obstacles))
    else:
        bur = [burial_fraction(start, xyz, r, probe=probe_radius) for xyz, r in obstacles]
    for pp, b in zip(per_pose, bur):
        pp["burial_fraction"] = round(b, 3)
    return {"pdb_dir": pdb_dir, "n_poses": len(per_pose), "start_xyz": [round(float(x), 3) for x in start],
            "start_how": how, "start_spread": spread, "per_pose": per_pose,
            "burial": {"mean": round(float(np.mean(bur)), 3), "min": round(float(np.min(bur)), 3),
                       "max": round(float(np.max(bur)), 3)}}


# --------------------------------------------------------------------------- CAVER
from ..io_utils import file_digest as _file_sha256


def protection_radius(caver_home=CAVER_HOME):
    """CAVER's start-protection radius (A) from its shipped config (default 4)."""
    try:
        for ln in open(os.path.join(caver_home, "config.txt")):
            m = re.match(r"\s*starting_point_protection_radius\s+([0-9.]+)", ln)
            if m:
                return float(m.group(1))
    except OSError:
        pass
    return 4.0


def write_caver_config(path, start_xyz, probe_radius=DEFAULT_PROBE_RADIUS, caver_home=CAVER_HOME):
    """CAVER's shipped config with our start point + probe radius; everything else stays at the
    package defaults (shell_radius 3, shell_depth 4, average-link clustering, threshold 3.5)."""
    drop = ("starting_point_atom", "starting_point_residue", "starting_point_coordinates", "probe_radius")
    base = [ln for ln in open(os.path.join(caver_home, "config.txt"))
            if not ln.strip().lstrip("#").strip().startswith(drop)]
    with open(path, "w") as fh:
        fh.writelines(base)
        fh.write(f"\nprobe_radius {probe_radius}\n")
        fh.write(f"starting_point_coordinates {start_xyz[0]:.3f} {start_xyz[1]:.3f} {start_xyz[2]:.3f}\n")


def subset_pdb_dir(pdb_dir, dest, n):
    """A directory of symlinks to the first `n` staged pose PDBs, for a cheaper perturbed-start run
    (CAVER processes every PDB in the -pdb dir, so fewer files = proportionally less work). Symlinks,
    not copies -- the poses are already staged and immutable. Returns (dest, actual_count)."""
    poses = sorted(f for f in os.listdir(pdb_dir) if f.startswith("pose_") and f.endswith(".pdb"))
    keep = poses[:n] if n else poses
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    for f in keep:
        os.symlink(os.path.abspath(os.path.join(pdb_dir, f)), os.path.join(dest, f))
    return dest, len(keep)


def run_caver(workdir, start_xyz, probe_radius=DEFAULT_PROBE_RADIUS, xmx_gb=DEFAULT_XMX_GB,
              threads=DEFAULT_THREADS, caver_home=CAVER_HOME, timeout=7200, tag="", pdb_dir=None):
    """Run CAVER on `pdb_dir` (default workdir/pdb) -> <run_dir>/caver_out. `tag` gives a perturbed-start
    run its own directory (workdir/perturb_<tag>) so concurrent runs never share files; the staged PDBs
    are shared. Exit code alone is NOT trusted: CAVER must print 'Finished successfully' and produce
    tunnel_characteristics.csv. Returns (out_dir, warnings, log)."""
    jar = os.path.join(caver_home, "caver.jar")
    if not os.path.exists(jar):
        raise FileNotFoundError(f"CAVER not found at {caver_home} (set CAVER_HOME)")
    if shutil.which("java") is None:
        raise FileNotFoundError("java not on PATH (CAVER 3 needs a JRE)")
    run_dir = workdir if not tag else os.path.join(workdir, f"perturb_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    conf = os.path.join(run_dir, "caver_config.txt")
    write_caver_config(conf, start_xyz, probe_radius, caver_home)
    out = os.path.join(run_dir, "caver_out")
    shutil.rmtree(out, ignore_errors=True)
    cmd = ["java", f"-Xmx{int(xmx_gb)}g", f"-XX:ActiveProcessorCount={int(threads)}",
           "-cp", f"{jar}:{os.path.join(caver_home, 'lib')}/*", "caver.ui.Launcher",
           "-home", caver_home, "-pdb", pdb_dir or os.path.join(workdir, "pdb"), "-conf", conf, "-out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=run_dir, timeout=timeout)
    log = (r.stdout or "") + (r.stderr or "")
    chars = os.path.join(out, "analysis", "tunnel_characteristics.csv")
    if "Finished successfully" not in log or not os.path.exists(chars):
        raise RuntimeError(f"CAVER did not finish cleanly (rc={r.returncode}):\n{log[-1500:]}")
    warns = sorted({ln.strip() for ln in log.splitlines() if ln.strip().startswith("WARNING")})
    return out, warns, log


# --------------------------------------------------------------------------- parsing
def _rows(path):
    with open(path) as fh:
        return [r for r in csv.reader(fh, skipinitialspace=True) if r]


def _floats(vals):
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except ValueError:
            pass            # '-' / '' placeholders
    return out


def _profile_endpoints(out_dir):
    """(snapshot, cluster) -> (first_xyz, last_xyz) of that snapshot's tunnel centerline, from
    CAVER's tunnel_profiles.csv (one X/Y/Z row each). The LAST point is the exit to solvent; the FIRST
    is where CAVER actually started (it relocates the requested start)."""
    path = os.path.join(out_dir, "analysis", "tunnel_profiles.csv")
    if not os.path.exists(path):
        return {}
    ax = {}
    for r in _rows(path)[1:]:
        if len(r) > 13 and r[12] in ("X", "Y", "Z"):
            ax.setdefault((r[0], int(float(r[1]))), {})[r[12]] = _floats(r[13:])
    out = {}
    for k, a in ax.items():
        if all(a.get(c) for c in "XYZ"):
            out[k] = ([a["X"][0], a["Y"][0], a["Z"][0]], [a["X"][-1], a["Y"][-1], a["Z"][-1]])
    return out


def parse_caver_output(out_dir, n_poses):
    """Per-snapshot table -> per-cluster statistics (recomputed here; CAVER's pretty summary is not
    parsed). Cluster id == CAVER's priority rank (1 = highest mean throughput). Frequency is over ALL
    n_poses, so a cluster seen in 2 of 50 poses reads 0.04, not 1.0. Each cluster also carries its exit
    position (mean over the cluster's snapshots) and how tightly the snapshots agree on it."""
    rows = _rows(os.path.join(out_dir, "analysis", "tunnel_characteristics.csv"))
    hdr = rows[0]
    ix = {k: hdr.index(k) for k in ("Snapshot", "Tunnel cluster", "Throughput", "Bottleneck radius",
                                    "Length", "Curvature")}
    per = {}
    for r in rows[1:]:
        try:
            cl = int(float(r[ix["Tunnel cluster"]]))
            per.setdefault(cl, []).append(dict(
                snapshot=r[ix["Snapshot"]], throughput=float(r[ix["Throughput"]]),
                bottleneck_A=float(r[ix["Bottleneck radius"]]), length_A=float(r[ix["Length"]]),
                curvature=float(r[ix["Curvature"]])))
        except (ValueError, IndexError):
            continue

    # bottleneck-lining residues: fraction of the cluster's snapshots in which each residue lines it
    lining = {}
    bpath = os.path.join(out_dir, "analysis", "bottlenecks.csv")
    if os.path.exists(bpath):
        for r in _rows(bpath)[2:]:           # row 0 = comment, row 1 = header
            try:
                cl = int(float(r[1]))
            except (ValueError, IndexError):
                continue
            for res in r[9:]:
                lining.setdefault(cl, {}).setdefault(res, 0)
                lining[cl][res] += 1

    ends = _profile_endpoints(out_dir)
    clusters = []
    for cl in sorted(per):
        P = per[cl]
        n = len(P)
        br = [p["bottleneck_A"] for p in P]
        tp = [p["throughput"] for p in P]
        sd = lambda xs: float(st.pstdev(xs)) if len(xs) > 1 else 0.0
        top = sorted(lining.get(cl, {}).items(), key=lambda kv: -kv[1])[:8]
        ex = np.array([ends[(p["snapshot"], cl)][1] for p in P if (p["snapshot"], cl) in ends])
        c0 = np.array([ends[(p["snapshot"], cl)][0] for p in P if (p["snapshot"], cl) in ends])
        exit_mean = ex.mean(0) if len(ex) else None
        clusters.append({
            "cluster": cl, "n_poses": n, "frequency": round(n / n_poses, 4),
            "bottleneck_A": {"mean": round(st.mean(br), 3), "sd": round(sd(br), 3),
                             "min": round(min(br), 3), "max": round(max(br), 3)},
            "length_A": {"mean": round(st.mean(p["length_A"] for p in P), 2),
                         "sd": round(sd([p["length_A"] for p in P]), 2)},
            "curvature_mean": round(st.mean(p["curvature"] for p in P), 3),
            "throughput_mean": round(st.mean(tp), 4),
            "priority": round(sum(tp) / n_poses, 4),     # CAVER: mean throughput over ALL snapshots
            "exit_xyz": None if exit_mean is None else [round(float(x), 2) for x in exit_mean],
            "exit_spread_A": None if exit_mean is None else
            round(float(np.linalg.norm(ex - exit_mean, axis=1).max()), 2),
            "caver_start_xyz": None if not len(c0) else [round(float(x), 2) for x in c0.mean(0)],
            "bottleneck_lining_residues": [{"residue": k, "fraction": round(v / n, 3)} for k, v in top],
        })
    clusters.sort(key=lambda c: -c["priority"])
    return clusters


# --------------------------------------------------------------------------- robustness
def classify_clusters(clusters, probe_radius=DEFAULT_PROBE_RADIUS, protect_radius=4.0,
                      min_freq=DEFAULT_MIN_FREQ, margin=DEFAULT_MARGIN):
    """Label every cluster (raw numbers are kept). A CHANNEL is persistent, longer than CAVER's
    start-protection radius (else it is just the solvent surface at the start) and wider than the probe
    by `margin` (else it is at CAVER's detection floor)."""
    for c in clusters:
        c["persistent"] = c["frequency"] >= min_freq
        c["surface_exposed"] = c["length_A"]["mean"] < protect_radius
        c["marginal"] = c["bottleneck_A"]["mean"] < probe_radius + margin
        c["is_channel"] = bool(c["persistent"] and not c["surface_exposed"] and not c["marginal"])
    return clusters


def site_surface_exposed(clusters):
    """True if a persistent cluster is shorter than the protection radius: the start touches bulk."""
    return any(c["persistent"] and c["surface_exposed"] for c in clusters)


def perturbed_starts(center, radius=DEFAULT_START_RADIUS, k=4):
    """k starts on a sphere of `radius` around `center` (tetrahedral directions for k=4; they sum to
    zero, so the perturbed set is centred on `center`)."""
    T = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3.0)
    if k > len(T):
        raise ValueError("at most 4 perturbed starts are supported")
    return [list(np.asarray(center, float) + radius * T[i]) for i in range(k)]


def match_stability(ref_channels, other_runs, tol=EXIT_MATCH_TOL):
    """For each reference channel, the fraction of other runs that contain a channel with an exit within
    `tol` A. `other_runs` = list of channel lists (already filtered to is_channel). Returns a list of
    {matched, of, fraction} aligned with ref_channels; fraction is None if there were no other runs."""
    out = []
    for c in ref_channels:
        if c.get("exit_xyz") is None or not other_runs:
            out.append({"matched": 0, "of": len(other_runs), "fraction": None})
            continue
        hit = 0
        for run in other_runs:
            hit += any(o.get("exit_xyz") is not None and
                       np.linalg.norm(np.array(o["exit_xyz"]) - np.array(c["exit_xyz"])) <= tol
                       for o in run)
        out.append({"matched": hit, "of": len(other_runs), "fraction": round(hit / len(other_runs), 3)})
    return out


# --------------------------------------------------------------------------- orchestration
def build_notes(chains, start_spread, surface_exposed=False):
    """Caveats attached to a result. `chains` = sorted distinct protein-chain counts across poses."""
    notes = []
    if 1 in chains:
        notes.append(SINGLE_CHAIN_WARNING)
    if start_spread and start_spread["frac_within_4A"] < 0.8:
        notes.append(START_INCONSISTENT_NOTE)
    if surface_exposed:
        notes.append(SURFACE_EXPOSED_NOTE)
    return notes


def _java_version():
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=30)
        return ((r.stderr or r.stdout).splitlines() or [""])[0].strip()
    except Exception:      # noqa: BLE001 -- provenance only
        return "unknown"


def analyze_channels(pose_paths, out_dir, start_xyz=None, probe_radius=DEFAULT_PROBE_RADIUS,
                     xmx_gb=DEFAULT_XMX_GB, threads=DEFAULT_THREADS, caver_home=CAVER_HOME, label=None,
                     n_starts=DEFAULT_N_STARTS, start_radius=DEFAULT_START_RADIUS,
                     min_freq=DEFAULT_MIN_FREQ, margin=DEFAULT_MARGIN, start_sel=None,
                     start_group="carboxylate", reaction_center_sel=None, substrate_resnames=None,
                     cofactor_resnames=None, hetero_sel=None, stability_poses=DEFAULT_STABILITY_POSES,
                     keep_cofactor=True, keep_metals=True):
    """Stage -> CAVER (primary + perturbed starts) -> parse -> classify -> stability ->
    out_dir/channels.json. n_starts=1 disables the start-point sensitivity runs. The primary run uses
    ALL poses (full statistics); each perturbed start-sensitivity run uses only `stability_poses` of
    them (None = all) -- the costly ensemble is processed once, not n_starts times. See stage_ensemble
    for how the start point and the substrate/cofactor split are resolved (start_sel, start_group,
    reaction_center_sel, substrate_resnames, cofactor_resnames, hetero_sel)."""
    os.makedirs(out_dir, exist_ok=True)
    man = stage_ensemble(pose_paths, out_dir, start_xyz=start_xyz, start_sel=start_sel,
                         start_group=start_group, reaction_center_sel=reaction_center_sel,
                         substrate_resnames=substrate_resnames, cofactor_resnames=cofactor_resnames,
                         hetero_sel=hetero_sel, probe_radius=probe_radius,
                         keep_cofactor=keep_cofactor, keep_metals=keep_metals)
    prot_r = protection_radius(caver_home)
    start = man["start_xyz"]

    def one(tag, s, pdb_dir=None, n=None):
        out, warns, _ = run_caver(out_dir, s, probe_radius, xmx_gb, threads, caver_home, tag=tag,
                                  pdb_dir=pdb_dir)
        return classify_clusters(parse_caver_output(out, n or man["n_poses"]), probe_radius, prot_r,
                                 min_freq, margin), warns

    primary, warns = one("", start)                       # full ensemble, all poses
    others, failures = [], []
    n_stab = man["n_poses"]
    if n_starts > 1:
        offs = perturbed_starts(start, start_radius, n_starts - 1)
        # perturbed runs only test start-point sensitivity -> a pose subset is enough and far cheaper
        if stability_poses and stability_poses < man["n_poses"]:
            sub_dir, n_stab = subset_pdb_dir(os.path.join(out_dir, "pdb"),
                                             os.path.join(out_dir, "pdb_stability"), stability_poses)
        else:
            sub_dir = os.path.join(out_dir, "pdb")
        with ThreadPoolExecutor(max_workers=len(offs)) as ex:
            futs = {i: ex.submit(one, f"s{i + 1}", s, sub_dir, n_stab) for i, s in enumerate(offs)}
        for i, f in futs.items():
            try:
                others.append(f.result()[0])
            except Exception as e:     # noqa: BLE001 -- a failed perturbation must not hide the primary
                failures.append(f"perturbed start s{i + 1}: {str(e)[:200]}")
    channels = [c for c in primary if c["is_channel"]]
    stab = match_stability(channels, [[c for c in run if c["is_channel"]] for run in others])
    for c, s in zip(channels, stab):
        c["stability"] = s

    chains = sorted({p["n_protein_chains"] for p in man["per_pose"]})
    exposed = site_surface_exposed(primary)
    notes = build_notes(chains, man["start_spread"], exposed)
    if n_starts > 1 and failures:
        notes.append(f"{len(failures)} of {n_starts - 1} perturbed-start runs failed; stability uses "
                     f"{len(others)}: " + "; ".join(failures))
    ext = [p["substrate_extent_A"] for p in man["per_pose"] if p["substrate_extent_A"]]
    res = {
        "label": label, "descriptive_only": True, "n_poses": man["n_poses"],
        "n_protein_chains": chains, "start": {"xyz": man["start_xyz"], "how": man["start_how"],
                                              "selection": start_sel,
                                              "group": None if (start_xyz is not None or start_sel)
                                              else start_group,
                                              "spread_across_poses": man["start_spread"]},
        "site": {"burial_fraction": man["burial"], "surface_exposed": exposed,
                 "protection_radius_A": prot_r},
        "substrate_extent_A_median": round(st.median(ext), 2) if ext else None,
        "obstacles": "protein (all chains)"
        + (" + nucleotide cofactor" if keep_cofactor else "")
        + (" + metals" if keep_metals else "")
        + "; substrate stripped" + ("" if (keep_cofactor and keep_metals) else " (bare-pocket: ligands stripped)"),
        "filters": {"min_frequency": min_freq, "min_length_A": prot_r,
                    "min_bottleneck_A": round(probe_radius + margin, 3),
                    "exit_match_tolerance_A": EXIT_MATCH_TOL},
        "robustness": {"n_starts": n_starts, "start_radius_A": start_radius if n_starts > 1 else None,
                       "stability_poses": n_stab if n_starts > 1 else None,
                       "n_perturbed_runs_ok": len(others), "n_perturbed_runs_failed": len(failures)},
        "n_clusters": len(primary), "n_channels": len(channels),
        "channels": channels, "clusters": primary, "notes": notes, "caver_warnings": warns,
        "provenance": {"engine": "CAVER 3.0 (Chovancova 2012, PLoS Comput Biol 8:e1002708)",
                       "caver_home": caver_home,
                       "caver_jar_sha256": _file_sha256(os.path.join(caver_home, "caver.jar")),
                       "java": _java_version(), "probe_radius_A": probe_radius,
                       "config": "package defaults except probe_radius + starting_point_coordinates",
                       "poses": [p["pose"] for p in man["per_pose"]]},
        "per_pose": man["per_pose"],
    }
    with open(os.path.join(out_dir, "channels.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def _poses_from_json(path):
    d = json.load(open(path))
    recs = d["poses"] if isinstance(d, dict) and "poses" in d else d
    return [r["path"] if isinstance(r, dict) else r for r in recs]


def _cli(argv=None):
    ap = argparse.ArgumentParser(prog="substrate_channels", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="stage + CAVER (+ perturbed starts) + parse an ensemble of poses")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--poses", nargs="+")
    g.add_argument("--poses-json", help="analysis.json-style: a list (or {'poses': [...]}) of {'path': ...}")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, help="use only the first N poses")
    r.add_argument("--start-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="explicit start coordinates (highest priority)")
    r.add_argument("--start-sel", help="MDAnalysis atom selection for the start (its centre of "
                   "geometry); e.g. 'resname LIG and name N1' or 'resid 106 and name OD1 OD2'")
    r.add_argument("--start-group", default="carboxylate",
                   choices=["carboxylate", "phosphate", "amine", "amide_N", "hydroxyl", "thiol"],
                   help="auto start: substrate functional group nearest the reaction centre "
                   "(default carboxylate for a step-1 acid; use 'amine' for a step-2 nucleophile)")
    r.add_argument("--reaction-center-sel", help="MDAnalysis selection for the electrophile used to "
                   "rank which substrate group is reactive (default: cofactor phosphorus)")
    r.add_argument("--substrate-resnames", nargs="+", help="pin these residue names as the substrate")
    r.add_argument("--cofactor-resnames", nargs="+", help="pin these residue names as the cofactor")
    r.add_argument("--hetero-sel", help="MDAnalysis selection for the non-protein atoms to consider")
    r.add_argument("--probe-radius", type=float, default=DEFAULT_PROBE_RADIUS)
    r.add_argument("--xmx-gb", type=int, default=DEFAULT_XMX_GB)
    r.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    r.add_argument("--n-starts", type=int, default=DEFAULT_N_STARTS,
                   help="1 = no start-point sensitivity runs (fastest); default 5 = primary + 4 perturbed")
    r.add_argument("--stability-poses", type=int, default=DEFAULT_STABILITY_POSES,
                   help="poses used by each perturbed-start run (0 = all; default 15). The primary run "
                   "always uses all poses; this only shrinks the cheap sensitivity checks")
    r.add_argument("--start-radius", type=float, default=DEFAULT_START_RADIUS)
    r.add_argument("--min-freq", type=float, default=DEFAULT_MIN_FREQ)
    r.add_argument("--strip-cofactor", action="store_true", help="also strip the cofactor (ATP/ADP/...) obstacle")
    r.add_argument("--strip-metals", action="store_true", help="also strip the metal (Mg/...) obstacle")
    r.add_argument("--strip-all-ligands", action="store_true",
                   help="bare-pocket: strip substrate + cofactor + metals (obstacles = protein only)")
    r.add_argument("--label")
    p = sub.add_parser("parse", help="re-parse an existing CAVER output dir (no JVM)")
    p.add_argument("--caver-out", required=True)
    p.add_argument("--n-poses", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "parse":
        print(json.dumps(parse_caver_output(a.caver_out, a.n_poses), indent=1))
        return 0
    poses = a.poses if a.poses else _poses_from_json(a.poses_json)
    if a.limit:
        poses = poses[:a.limit]
    res = analyze_channels(poses, a.out, a.start_xyz, a.probe_radius, a.xmx_gb, a.threads, label=a.label,
                           n_starts=a.n_starts, start_radius=a.start_radius, min_freq=a.min_freq,
                           start_sel=a.start_sel, start_group=a.start_group,
                           reaction_center_sel=a.reaction_center_sel,
                           substrate_resnames=a.substrate_resnames,
                           cofactor_resnames=a.cofactor_resnames, hetero_sel=a.hetero_sel,
                           stability_poses=a.stability_poses,
                           keep_cofactor=not (a.strip_cofactor or a.strip_all_ligands),
                           keep_metals=not (a.strip_metals or a.strip_all_ligands))
    b = res["site"]["burial_fraction"]
    print(f"{res['label'] or ''} n_poses={res['n_poses']} chains={res['n_protein_chains']} "
          f"start={res['start']['xyz']} burial={b['mean']:.2f} surface_exposed={res['site']['surface_exposed']} "
          f"| {res['n_channels']} channel(s) of {res['n_clusters']} clusters")
    for c in res["channels"][:6]:
        s = c.get("stability") or {}
        sf = "n/a" if s.get("fraction") is None else f"{s['matched']}/{s['of']}"
        print(f"  channel {c['cluster']}: freq {c['frequency']:.2f}  bottleneck {c['bottleneck_A']['mean']:.2f} A  "
              f"length {c['length_A']['mean']:.1f} A  throughput {c['throughput_mean']:.2f}  "
              f"stable {sf}  exit {c['exit_xyz']}")
    for n in res["notes"]:
        print("  NOTE:", n)
    print("wrote", os.path.join(a.out, "channels.json"))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
