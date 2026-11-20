"""Substrate-channel (tunnel) analysis for the structural modeling agent -- a CAVER 3 wrapper.

Active-site geometry (attack_geometry, substrate_contacts) says whether the substrate sits in a
productive pose. It says nothing about HOW A SUBSTRATE GETS THERE: is there a transport pathway from
bulk solvent to the catalytic center, how narrow is its bottleneck, how long and winding is it, and
which residues line the bottleneck? That is a channel question, and for a step-2 nucleophile (e.g. a
C5 diamine that must reach an already-bound acyl intermediate) it is often the decisive one.

ENGINE: CAVER 3.0 (Chovancova et al. 2012, PLoS Comput Biol 8:e1002708) -- Voronoi-based tunnel
search designed for tunnels from a buried active site to the surface, with built-in clustering across
an ENSEMBLE of structures. We do not reimplement it. A cofold ensemble (50 poses) is exactly its
multi-snapshot input, so every result here is a FREQUENCY over the ensemble, never one hero pose.

DETECTS, DOES NOT JUDGE. Like substrate_contacts, this reports measurements (cluster frequency,
bottleneck radius, length, curvature, throughput, bottleneck-lining residues) and renders NO verdict
-- no "channel is wide enough", no score, no ranking, no composite. Tunnel topology was already tried
as a ranking proxy in this project and came out null (wrong sign, p=0.99, 7 families; memory
`project_selectivity_proxy_null_oct9`); it is a descriptive tool here.

ASSEMBLY MATTERS (validated). Pass the FULL biological assembly. In a controlled test on the same 5
GlnA3 dimer poses with the same start point, dropping the partner subunit turned a protected site into
a 1.8 A-long, 2.5 A-bottleneck, throughput-0.94 "tunnel" (i.e. open to solvent) that disappears when
the partner chain is present; a 16 A channel was unchanged (1.53 vs 1.54 A bottleneck). A monomer
model therefore OVERSTATES openness for any enzyme whose site is capped by a partner subunit (the GS
fold is one). The output flags single-chain inputs.

Procedure per ensemble:
  1. superpose every pose onto pose 0 (Kabsch on all protein C-alpha) -- CAVER clusters tunnels across
     snapshots and needs a common frame;
  2. strip the SUBSTRATE only; keep protein (all chains) + nucleotide cofactor + metals, because bound
     ATP/ADP and Mg physically occupy the site the tunnel must route around;
  3. start point = median (across poses) of the substrate's reactive carboxylate carbon, in the common
     frame (or an explicit --start-xyz); the spread across poses is reported;
  4. run CAVER once over all poses; validate that it really finished (exit code alone is not trusted);
  5. recompute cluster statistics from CAVER's per-snapshot table (not from its pretty-printed one).

Scope: step-1 poses (a carboxylate substrate) resolve the start point automatically. For step-2 poses
(acyl intermediate + amine) or any substrate without a carboxylate, give --start-xyz explicitly.

CLI:
  # tunnels over an ensemble of poses (PDB or mmCIF), staged + run + parsed
  python -m structural_modeling_agent.analysis.substrate_channels run \\
      --poses-json analysis.json --out OUT_DIR            # list of {"path": ...} records
  python -m structural_modeling_agent.analysis.substrate_channels run --poses a.cif b.cif --out OUT_DIR
  # re-parse an existing CAVER output directory (no JVM)
  python -m structural_modeling_agent.analysis.substrate_channels parse --caver-out OUT/caver_out --n-poses 50

Needs: Java (>= 8) and CAVER 3 (CAVER_HOME, default below); numpy; MDAnalysis; gemmi (mmCIF input).
"""
from __future__ import annotations

import argparse
import csv
import glob
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

import numpy as np

from . import substrate_contacts as sc

CAVER_HOME = os.environ.get(
    "CAVER_HOME",
    "/nfs/lambda_stor_01/homes/rzhu/Tool/processivity-toolchain-20261106/sources/caver/Caver3")
DEFAULT_PROBE_RADIUS = 0.9     # CAVER's own default for protein tunnels
DEFAULT_XMX_GB = 8
DEFAULT_THREADS = 4            # shared nodes: cap JVM CPU (global courtesy rule)

SINGLE_CHAIN_WARNING = (
    "single protein chain: if this enzyme's active site is capped by a partner subunit (e.g. the GS "
    "fold), a monomer model reads the site as solvent-open and OVERSTATES channel openness. In a "
    "controlled GlnA3 test, removing the partner turned a protected site into a 1.8 A, 2.5 A-bottleneck "
    "'tunnel'. Analyze the biological assembly if one is known; interpret monomer results with care.")


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


def stage_ensemble(pose_paths, workdir, start_xyz=None):
    """Align poses, strip substrate, write one PDB per pose for CAVER. Returns a manifest dict.

    Fails loud (never silently degrades): empty input, pose-to-pose CA-count mismatch, and a missing
    reactive carboxylate with no explicit start point all raise ValueError."""
    import MDAnalysis as mda
    from MDAnalysis.analysis.align import rotation_matrix
    if not pose_paths:
        raise ValueError("no poses given")
    pdb_dir = os.path.join(workdir, "pdb")
    shutil.rmtree(pdb_dir, ignore_errors=True)
    os.makedirs(pdb_dir)
    tmp = tempfile.mkdtemp(prefix="substrate_channels_")
    per_pose, starts, ref_ca = [], [], None
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
                g = sc.identify_groups(u)
                _, _, substrate = sc.split_ligands(u)      # identify_groups does not return it
                # the reactive carboxylate = the carboxylate nearest the electrophile (cofactor P).
                # Taken from the carboxylates specifically: for L-Glu the nearest GROUP overall can be
                # the alpha-amine, which is not where the acid attacks.
                acids = [fg for fg in g["functional_groups"]
                         if fg["kind"] == "carboxylate" and fg.get("d_to_center") is not None]
                center = None
                if acids:
                    center = min(acids, key=lambda fg: fg["d_to_center"])["center"].position.copy()
                    starts.append(center)
                elif start_xyz is None:
                    raise ValueError(
                        f"{p}: no reactive carboxylate found to place the start point -- pass an "
                        "explicit start (--start-xyz) for step-2 / non-carboxylate substrates")
                keep = u.select_atoms("protein") + g["cofactor"] + g["metals"]   # substrate STRIPPED
                keep.write(os.path.join(pdb_dir, f"pose_{i:04d}.pdb"))
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
        how, spread = "median_reactive_carboxylate_carbon", {
            "max_A": round(float(d.max()), 2), "median_A": round(float(np.median(d)), 2),
            "frac_within_4A": round(float((d <= 4.0).mean()), 3)}
    return {"pdb_dir": pdb_dir, "n_poses": len(per_pose), "start_xyz": [round(float(x), 3) for x in start],
            "start_how": how, "start_spread": spread, "per_pose": per_pose}


# --------------------------------------------------------------------------- CAVER
def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def run_caver(workdir, start_xyz, probe_radius=DEFAULT_PROBE_RADIUS, xmx_gb=DEFAULT_XMX_GB,
              threads=DEFAULT_THREADS, caver_home=CAVER_HOME, timeout=7200):
    """Run CAVER on workdir/pdb -> workdir/caver_out. Exit code alone is NOT trusted: CAVER must print
    'Finished successfully' and produce tunnel_characteristics.csv. Returns (out_dir, warnings, log)."""
    jar = os.path.join(caver_home, "caver.jar")
    if not os.path.exists(jar):
        raise FileNotFoundError(f"CAVER not found at {caver_home} (set CAVER_HOME)")
    if shutil.which("java") is None:
        raise FileNotFoundError("java not on PATH (CAVER 3 needs a JRE)")
    conf = os.path.join(workdir, "caver_config.txt")
    write_caver_config(conf, start_xyz, probe_radius, caver_home)
    out = os.path.join(workdir, "caver_out")
    shutil.rmtree(out, ignore_errors=True)
    cmd = ["java", f"-Xmx{int(xmx_gb)}g", f"-XX:ActiveProcessorCount={int(threads)}",
           "-cp", f"{jar}:{os.path.join(caver_home, 'lib')}/*", "caver.ui.Launcher",
           "-home", caver_home, "-pdb", os.path.join(workdir, "pdb"), "-conf", conf, "-out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, timeout=timeout)
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


def parse_caver_output(out_dir, n_poses):
    """Per-snapshot table -> per-cluster statistics (recomputed here; CAVER's pretty summary is not
    parsed). Cluster id == CAVER's priority rank (1 = highest mean throughput). Frequency is over ALL
    n_poses, so a cluster seen in 2 of 50 poses reads 0.04, not 1.0."""
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

    clusters = []
    for cl in sorted(per):
        P = per[cl]
        n = len(P)
        br = [p["bottleneck_A"] for p in P]
        tp = [p["throughput"] for p in P]
        sd = lambda xs: float(st.pstdev(xs)) if len(xs) > 1 else 0.0
        top = sorted(lining.get(cl, {}).items(), key=lambda kv: -kv[1])[:8]
        clusters.append({
            "cluster": cl, "n_poses": n, "frequency": round(n / n_poses, 4),
            "bottleneck_A": {"mean": round(st.mean(br), 3), "sd": round(sd(br), 3),
                             "min": round(min(br), 3), "max": round(max(br), 3)},
            "length_A": {"mean": round(st.mean(p["length_A"] for p in P), 2),
                         "sd": round(sd([p["length_A"] for p in P]), 2)},
            "curvature_mean": round(st.mean(p["curvature"] for p in P), 3),
            "throughput_mean": round(st.mean(tp), 4),
            "priority": round(sum(tp) / n_poses, 4),     # CAVER: mean throughput over ALL snapshots
            "bottleneck_lining_residues": [{"residue": k, "fraction": round(v / n, 3)} for k, v in top],
        })
    clusters.sort(key=lambda c: -c["priority"])
    return clusters


# --------------------------------------------------------------------------- orchestration
START_INCONSISTENT_NOTE = (
    "start point is NOT consistent across poses (the substrate's reactive carbon is scattered): "
    "channels are measured from a median position that many poses do not occupy; read cluster "
    "frequencies accordingly.")


def build_notes(chains, start_spread):
    """Caveats attached to a result. `chains` = sorted distinct protein-chain counts across poses."""
    notes = []
    if 1 in chains:
        notes.append(SINGLE_CHAIN_WARNING)
    if start_spread and start_spread["frac_within_4A"] < 0.8:
        notes.append(START_INCONSISTENT_NOTE)
    return notes


def _java_version():
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=30)
        return ((r.stderr or r.stdout).splitlines() or [""])[0].strip()
    except Exception:      # noqa: BLE001 -- provenance only
        return "unknown"


def analyze_channels(pose_paths, out_dir, start_xyz=None, probe_radius=DEFAULT_PROBE_RADIUS,
                     xmx_gb=DEFAULT_XMX_GB, threads=DEFAULT_THREADS, caver_home=CAVER_HOME, label=None):
    """Stage -> CAVER -> parse -> write out_dir/channels.json. Returns the result dict."""
    os.makedirs(out_dir, exist_ok=True)
    man = stage_ensemble(pose_paths, out_dir, start_xyz=start_xyz)
    caver_out, warns, _ = run_caver(out_dir, man["start_xyz"], probe_radius, xmx_gb, threads, caver_home)
    clusters = parse_caver_output(caver_out, man["n_poses"])
    chains = sorted({p["n_protein_chains"] for p in man["per_pose"]})
    notes = build_notes(chains, man["start_spread"])
    ext = [p["substrate_extent_A"] for p in man["per_pose"] if p["substrate_extent_A"]]
    res = {
        "label": label, "descriptive_only": True, "n_poses": man["n_poses"],
        "n_protein_chains": chains, "start": {"xyz": man["start_xyz"], "how": man["start_how"],
                                              "spread_across_poses": man["start_spread"]},
        "substrate_extent_A_median": round(st.median(ext), 2) if ext else None,
        "obstacles": "protein (all chains) + nucleotide cofactor + metals; substrate stripped",
        "clusters": clusters, "notes": notes, "caver_warnings": warns,
        "provenance": {"engine": "CAVER 3.0 (Chovancova 2012, PLoS Comput Biol 8:e1002708)",
                       "caver_home": caver_home, "caver_jar_sha256": _file_sha256(os.path.join(caver_home, "caver.jar")),
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
    r = sub.add_parser("run", help="stage + CAVER + parse an ensemble of poses")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--poses", nargs="+")
    g.add_argument("--poses-json", help="analysis.json-style: a list (or {'poses': [...]}) of {'path': ...}")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, help="use only the first N poses")
    r.add_argument("--start-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    r.add_argument("--probe-radius", type=float, default=DEFAULT_PROBE_RADIUS)
    r.add_argument("--xmx-gb", type=int, default=DEFAULT_XMX_GB)
    r.add_argument("--threads", type=int, default=DEFAULT_THREADS)
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
    res = analyze_channels(poses, a.out, a.start_xyz, a.probe_radius, a.xmx_gb, a.threads, label=a.label)
    print(f"{res['label'] or ''} n_poses={res['n_poses']} chains={res['n_protein_chains']} "
          f"start={res['start']['xyz']} ({res['start']['how']})")
    for c in res["clusters"][:6]:
        print(f"  cluster {c['cluster']}: freq {c['frequency']:.2f}  bottleneck {c['bottleneck_A']['mean']:.2f} A  "
              f"length {c['length_A']['mean']:.1f} A  throughput {c['throughput_mean']:.2f}")
    for n in res["notes"]:
        print("  NOTE:", n)
    print("wrote", os.path.join(a.out, "channels.json"))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
