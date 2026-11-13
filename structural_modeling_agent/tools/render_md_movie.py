"""Render a reactive-geometry-persistence movie from an MD trajectory.

House standard for visualizing enzyme active-site MD (reuse for every enzyme so panels are
comparable). Produces an animated GIF of the active site with the trajectory aligned on the
protein, so the pocket is stationary and substrate motion is what you see.

APPROVED FORMATTING (do not change per-figure; change here so all enzymes stay comparable):
  * protein            -> cartoon ribbon, gray75, transparency 0.35, fancy helices
  * cofactor (ATP)     -> sticks, carbons DARK GRAY (gray25), heteroatoms by element
  * substrate          -> sticks (fatter), carbons ORANGE, heteroatoms by element
  * metals (Mg)        -> forest-green spheres
  * forming bond       -> yellow dashed line (nucleophile -> electrophile), no distance label
  * background white, no text overlays

Stages (each needs a different env, so `all` orchestrates by re-invoking itself):
  frames   -> MDAnalysis: align on protein CA, write stripped multiframe PDB (backbone + ligands)
  render   -> PyMOL: ray-trace states [--lo,--hi] to PNG (fan out many workers = the fast path)
  assemble -> PIL: PNG frames -> GIF
  all      -> frames, then N parallel render workers, then assemble

Example (GlnA3 step-1):
  python -m structural_modeling_agent.tools.render_md_movie all \
    --prmtop runs/<camp>/GlnA3_Q9KZC7/prep/complex_1264.prmtop \
    --traj   runs/<camp>/GlnA3_Q9KZC7/md/production.dcd \
    --nuc "resname SUB and name O3" --elec "resname ATP and name PG" \
    --out runs/<camp>/analysis/GlnA3_reactive_geometry_persistence.gif
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Interpreters per stage (this cluster's envs; override via flags if they move).
MD_PYTHON = "/nfs/lambda_stor_01/homes/rzhu/Tool/envs/processivity-md-20261106/bin/python"
PYMOL = "/homes/rzhu/miniforge3/envs/pymol/bin/pymol"
PIL_PYTHON = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2/bin/python"

STYLE = {
    "protein_color": "gray75", "cartoon_transparency": 0.35,
    "cofactor_carbon": "gray25", "substrate_carbon": "orange", "metal_color": "forest",
    "dash_color": "yellow", "background": "white",
    "width": 1000, "height": 750, "antialias": 1, "duration_ms": 80,
}


def stage_frames(a):
    import numpy as np
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    u = mda.Universe(a.prmtop, a.traj)
    ref = mda.Universe(a.prmtop, a.traj); ref.trajectory[0]
    ligs = " ".join([a.cofactor_resn, a.substrate_resn, a.metal_resn])
    sel = f"(protein and backbone) or (resname {ligs} and not name H*)"
    render = u.select_atoms(sel)
    out = Path(a.workdir) / "frames.pdb"
    n = 0
    with mda.Writer(str(out), multiframe=True, n_atoms=render.n_atoms) as W:
        for i, ts in enumerate(u.trajectory):
            if i % a.stride:
                continue
            align.alignto(u, ref, select="protein and name CA")
            W.write(render); n += 1
    print(f"frames: wrote {n} states ({render.n_atoms} atoms) -> {out}")


def stage_render(a):
    from pymol import cmd
    fr = Path(a.workdir) / "png"; fr.mkdir(parents=True, exist_ok=True)
    cmd.load(str(Path(a.workdir) / "frames.pdb"), "mv")
    cmd.bg_color(STYLE["background"]); cmd.hide("everything")
    cmd.show("cartoon", "polymer"); cmd.dss()
    cmd.color(STYLE["protein_color"], "polymer")
    cmd.set("cartoon_transparency", STYLE["cartoon_transparency"])
    cmd.set("cartoon_fancy_helices", 1)
    core = f"resn {a.cofactor_resn}+{a.substrate_resn}"
    cmd.show("sticks", core); cmd.show("spheres", f"resn {a.metal_resn}")
    cmd.color(STYLE["cofactor_carbon"], f"resn {a.cofactor_resn} and elem C")
    cmd.color(STYLE["substrate_carbon"], f"resn {a.substrate_resn} and elem C")
    cmd.color(STYLE["metal_color"], f"resn {a.metal_resn}")
    cmd.set("stick_radius", 0.17, f"resn {a.cofactor_resn}")
    cmd.set("stick_radius", 0.26, f"resn {a.substrate_resn}")
    cmd.set("sphere_scale", 0.45, f"resn {a.metal_resn}")
    cmd.util.cnc(core)
    cmd.set("ray_opaque_background", 1); cmd.set("ray_shadows", 0)
    cmd.set("antialias", STYLE["antialias"]); cmd.set("label_size", 0)
    cmd.set("dash_color", STYLE["dash_color"]); cmd.set("dash_width", 4.0); cmd.set("dash_gap", 0.35)
    orient_sel = f"resn {a.cofactor_resn}+{a.substrate_resn}+{a.metal_resn}"
    cmd.orient(orient_sel, state=1); cmd.zoom(orient_sel, 3.0, state=1)
    view = cmd.get_view()
    # pymol selection syntax uses `resn`; MDAnalysis-style `resname` is accepted here too.
    nuc = a.nuc.replace("resname", "resn"); elec = a.elec.replace("resname", "resn")
    for s in range(a.lo, a.hi + 1):
        cmd.set_view(view); cmd.delete("fb")
        cmd.distance("fb", nuc, elec, state=s); cmd.hide("labels", "fb")
        cmd.set("state", s)
        cmd.ray(STYLE["width"], STYLE["height"]); cmd.png(str(fr / f"frame_{s:04d}.png"), dpi=150)
    print(f"render: states {a.lo}-{a.hi} done")


def stage_assemble(a):
    import glob
    from PIL import Image
    pngs = sorted(glob.glob(str(Path(a.workdir) / "png" / "frame_*.png")))
    if not pngs:
        sys.exit("assemble: no PNG frames found")
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:],
                 duration=STYLE["duration_ms"], loop=0, optimize=True)
    print(f"assemble: {len(imgs)} frames -> {a.out} ({os.path.getsize(a.out)//1024} KB)")


def stage_all(a):
    common = ["--workdir", a.workdir, "--prmtop", a.prmtop, "--traj", a.traj,
              "--nuc", a.nuc, "--elec", a.elec, "--cofactor-resn", a.cofactor_resn,
              "--substrate-resn", a.substrate_resn, "--metal-resn", a.metal_resn]
    me = ["-m", "structural_modeling_agent.tools.render_md_movie"]
    repo = str(Path(__file__).resolve().parents[2])
    subprocess.run([a.md_python, *me, "frames", *common, "--stride", str(a.stride)],
                   cwd=repo, check=True)
    # count states from the PDB MODEL records
    n_states = sum(1 for line in (Path(a.workdir) / "frames.pdb").open() if line.startswith("MODEL"))
    n_states = n_states or 1
    # PyMOL does not run a script as __main__, so drive the render stage through a shim
    # that executes at the shim's module top level and reads its argv from the environment.
    shim = Path(a.workdir) / "_render_shim.py"
    shim.write_text(
        "import json, os, sys\n"
        "sys.path.insert(0, os.environ['RMM_REPO'])\n"
        "from structural_modeling_agent.tools.render_md_movie import stage_render, build_parser\n"
        "stage_render(build_parser().parse_args(json.loads(os.environ['RMM_ARGV'])))\n")
    workers = max(1, a.workers)
    chunk = -(-n_states // workers)
    procs = []
    for w in range(workers):
        lo, hi = w * chunk + 1, min((w + 1) * chunk, n_states)
        if lo > hi:
            break
        env = dict(os.environ, RMM_REPO=repo,
                   RMM_ARGV=json.dumps(["render", *common, "--lo", str(lo), "--hi", str(hi)]))
        procs.append(subprocess.Popen([a.pymol, "-cq", str(shim)], cwd=repo, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    bad = [p.communicate()[1].decode()[-800:] for p in procs if p.wait() != 0]
    if bad:
        sys.exit("render workers failed:\n" + "\n".join(bad))
    n_png = len(list((Path(a.workdir) / "png").glob("frame_*.png")))
    if n_png != n_states:
        sys.exit(f"render incomplete: {n_png} PNG for {n_states} states")
    subprocess.run([a.pil_python, *me, "assemble", *common, "--out", a.out], cwd=repo, check=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["frames", "render", "assemble", "all"])
    p.add_argument("--workdir", default=None, help="scratch dir for frames.pdb + png/ (default: <out>.work)")
    p.add_argument("--prmtop"); p.add_argument("--traj"); p.add_argument("--out")
    p.add_argument("--nuc", default="resname SUB and name O3", help="nucleophile atom selection")
    p.add_argument("--elec", default="resname ATP and name PG", help="electrophile atom selection")
    p.add_argument("--cofactor-resn", default="ATP"); p.add_argument("--substrate-resn", default="SUB")
    p.add_argument("--metal-resn", default="MG")
    p.add_argument("--stride", type=int, default=10); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lo", type=int, default=1); p.add_argument("--hi", type=int, default=1)
    p.add_argument("--md-python", default=MD_PYTHON); p.add_argument("--pymol", default=PYMOL)
    p.add_argument("--pil-python", default=PIL_PYTHON)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.workdir is None:
        if not a.out:
            sys.exit("--workdir or --out required")
        a.workdir = str(Path(a.out).with_suffix("")) + ".work"
    Path(a.workdir).mkdir(parents=True, exist_ok=True)
    {"frames": stage_frames, "render": stage_render, "assemble": stage_assemble, "all": stage_all}[a.stage](a)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--" in argv:  # pymol -cq script.py -- <args>
        argv = argv[argv.index("--") + 1:]
    main(argv)
