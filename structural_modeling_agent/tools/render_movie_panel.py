"""Render reactive-geometry movies for a whole candidate PANEL (companion to render_md_movie.py).

Reads a movie-registry JSON -- a list of {name, prmtop, traj, [nuc], [elec], [stride], [zoom]} -- and for
each entry renders one reactive-geometry movie via render_md_movie, naming the output
`<name>_reactive_geometry.gif` in --out-dir (house convention: Gene_ACCESSION, see
feedback_run_folder_artifacts_naming).

Reaction coordinate is auto-derived when not given in the registry:
  * electrophile phosphorus by Mg count -- >=2 Mg => acyl-phosphate (Pgamma), else adenylate (Palpha)
    (reference_step1_nac_phosphorus_by_mechanism);
  * nucleophile = the substrate O/N/S closest to that P at frame 0;
  * stride auto-set so the movie is ~`--frames` states regardless of trajectory length;
  * zoom widened for substrates that start dissociated.

Rendering is SEQUENTIAL by design: running many headless PyMOL `ray` instances across movies at once
stalls on this cluster (single-movie / sequential always complete). Within-movie parallelism (`--workers`)
is the fast path; a per-movie `--timeout` prevents one hang from blocking the batch. PyMOL `ray` is
CPU-bound -- GPUs do not accelerate it; the lever is cores-per-movie.

Run:
  python -m structural_modeling_agent.tools.render_movie_panel \
      --registry runs/<campaign>/movie_registry.json --out-dir runs/<campaign>/movies \
      [--workers 12 --timeout 300 --frames 50 --repo <root>]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_DEFAULT = str(HERE.parents[2])
MDPY = "/nfs/lambda_stor_01/homes/rzhu/Tool/envs/processivity-md-20261106/bin/python"


def _derive(top, traj, frames):
    """-> (nuc_name, elec_P, stride, zoom) from topology+trajectory."""
    import numpy as np
    import MDAnalysis as mda
    u = mda.Universe(top, traj)
    nmg = len(u.select_atoms("resname MG"))
    P = "PG" if nmg >= 2 else "PA"
    p = u.select_atoms(f"resname ATP and name {P}")
    best = None
    for a in u.select_atoms("resname SUB"):
        if not a.name.startswith("O"):
            continue
        o = u.select_atoms(f"resname SUB and name {a.name}")
        d = float(np.linalg.norm(o.positions[0] - p.positions[0]))
        if best is None or d < best[0]:
            best = (d, a.name)
    d0, nuc = best
    stride = max(1, round(len(u.trajectory) / frames))
    zoom = 3.5 if d0 < 5 else 5.0
    return nuc, P, stride, zoom, d0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", required=True, help="JSON list of {name, prmtop, traj, [nuc,elec,stride,zoom]}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default=REPO_DEFAULT, help="root to resolve relative prmtop/traj paths")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--md-python", default=MDPY)
    a = ap.parse_args(argv)
    reg = json.load(open(a.registry))
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"; work.mkdir(exist_ok=True)

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(a.repo, p)

    ts = time.strftime("%H%M%S"); fails = []
    for e in reg:
        name, top, traj = e["name"], rp(e["prmtop"]), rp(e["traj"])
        if not (os.path.exists(top) and os.path.exists(traj)):
            print(f"SKIP {name}: missing prmtop/traj", flush=True); fails.append(name); continue
        nuc = e.get("nuc"); elec = e.get("elec"); stride = e.get("stride"); zoom = e.get("zoom")
        if not (nuc and elec and stride and zoom):
            dn, de, ds, dz, d0 = _derive(top, traj, a.frames)
            nuc = nuc or f"resname SUB and name {dn}"
            elec = elec or f"resname ATP and name {de}"
            stride = stride or ds; zoom = zoom or dz
        out = str(out_dir / f"{name}_reactive_geometry.gif")
        cmd = [a.md_python, "-m", "structural_modeling_agent.tools.render_md_movie", "all",
               "--workdir", str(work / f"{name}_{ts}.work"), "--prmtop", top, "--traj", traj,
               "--nuc", nuc, "--elec", elec, "--stride", str(stride), "--workers", str(a.workers),
               "--zoom", str(zoom), "--out", out]
        print(f">> {name} nuc='{nuc}' elec='{elec}' stride={stride} zoom={zoom} {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        with open(work / f"{name}.renderlog", "w") as lg:
            try:
                r = subprocess.run(cmd, cwd=a.repo, stdout=lg, stderr=subprocess.STDOUT, timeout=a.timeout)
                ok = r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000
            except subprocess.TimeoutExpired:
                ok = False
        if not ok:
            fails.append(name)
        sz = os.path.getsize(out) // 1024 if os.path.exists(out) else 0
        print(f"   {'OK' if ok else 'FAIL'} {name} ({time.time()-t0:.0f}s) {sz}KB", flush=True)
    print(f"DONE {len(reg)-len(fails)}/{len(reg)}. fails: {fails or 'none'}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
