"""Publication-quality figure of the substrate ACCESS CHANNELS a run found (companion to
substrate_channels.py, which produces the numbers). It renders ONE representative pose as a faint
translucent protein body with the detected tunnel(s) threading through it as a string of spheres whose
radius IS the local tunnel radius -- so the bottleneck is visibly the narrowest bead -- coloured by that
radius, with a radius colour-bar and a measurement box (bottleneck, length, burial, stability) in the
house style (gray90 protein, gray40 sticks, white background, bold bottom-right box).

It reads an out_dir produced by `substrate_channels run`: out_dir/channels.json for the statistics and
out_dir/caver_out/data/{pose_0000.pdb, clusters_timeless/tun_cl_00N_1.pdb} for the geometry. Only the
channels classified `is_channel` are drawn (surface-exposed / marginal clusters are omitted); pass
--all-clusters to draw every cluster.

DESCRIPTIVE ONLY, like the detector: the figure reports geometry, it renders no verdict. The narrowest
bead is a static lower bound (channels breathe under dynamics).

Run (env with pymol + matplotlib + PIL):
  python -m structural_modeling_agent.analysis.render_channels --run-dir OUT_DIR \\
      --title "GlnA3 (Q9KZC7)" --out channels.png
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

PYMOL = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/pymol/bin/pymol"
IMG_W, IMG_H = 1500, 1200
RADIUS_VMIN, RADIUS_VMAX = 1.0, 2.6     # A; colour-bar range for tunnel radius
CMAP = "RdYlBu"                          # narrow = red (bottleneck), wide = blue (open to solvent)


def _tunnel_pdb(data_dir, cluster_id):
    """CAVER cluster id N -> its sphere-string PDB (data/clusters_timeless/tun_cl_00N_1.pdb)."""
    p = os.path.join(data_dir, "clusters_timeless", f"tun_cl_{int(cluster_id):03d}_1.pdb")
    return p if os.path.exists(p) else None


def _radii(pdb):
    """Per-sphere tunnel radius (CAVER stores it in the B-factor column)."""
    out = []
    for ln in open(pdb):
        if ln.startswith(("ATOM", "HETATM")):
            try:
                out.append(float(ln[60:66]))
            except ValueError:
                pass
    return out


def _sphere_colors(radii):
    """RGB (0-1) per sphere from the shared colormap, so the render and the colour-bar match exactly."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(RADIUS_VMIN, RADIUS_VMAX)
    sm = cm.ScalarMappable(norm, CMAP)
    return [tuple(sm.to_rgba(r)[:3]) for r in radii]


def _render_pose(run_dir, channels, out_png, draw_residues):
    """PyMOL: translucent protein body + catalytic/lining sticks + tunnels coloured by radius.
    Returns the path to the ray-traced RGBA png (transparent background)."""
    data = os.path.join(run_dir, "caver_out", "data")
    # CAVER writes ONE representative snapshot to data/ (its name varies: pose_0000, pose_0025, ...);
    # it is in the same frame as the tunnels, so prefer it over the staged inputs.
    import glob
    hits = sorted(glob.glob(os.path.join(data, "pose_*.pdb")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(run_dir, "pdb", "pose_*.pdb")))
    if not hits:
        raise FileNotFoundError(f"no representative pose_*.pdb under {data} (was the run staged here?)")
    pose = hits[0]
    # protein = a translucent CARTOON only (no protein surface: a surface would occlude the internal
    # tunnels and wash them out). The tunnels are the subject and must read clearly through the ribbon.
    pml = [f"load {pose}, prot", "hide everything", "bg_color white",
           "set ray_opaque_background, 0", "set antialias, 2", "set ray_shadows, 0",
           "set ambient, 0.45", "set specular, 0.25", "set two_sided_lighting, 1",
           "set cartoon_transparency, 0.6", "set ray_interior_color, grey70",
           "show cartoon, prot", "color gray80, prot",
           "set cartoon_side_chain_helper, 1", "set stick_radius, 0.2"]
    # catalytic / bottleneck-lining residues as sticks (from the main channel)
    for rid in draw_residues:
        pml += [f"select r{rid}, (prot and resi {rid} and not name C+N+O)", f"show sticks, r{rid}",
                f"color gray40, r{rid} and elem C", f"util.cnc r{rid}"]
    tunnel_objs = []
    for ci, ch in enumerate(channels):
        tp = _tunnel_pdb(data, ch["cluster"])
        if not tp:
            continue
        obj = f"t{ci}"
        tunnel_objs.append(obj)
        # a VDW surface (solvent_radius 0) over the radius-sized spheres merges the "string of beads"
        # into ONE smooth tube whose thickness is the local tunnel radius
        pml += [f"load {tp}, {obj}", f"hide everything, {obj}", f"alter {obj}, vdw=b", "rebuild",
                f"set solvent_radius, 0, {obj}", f"set surface_quality, 2, {obj}",
                f"show surface, {obj}"]
        cols = _sphere_colors(_radii(tp))
        for ai, (r, g, b) in enumerate(cols, start=1):
            pml += [f"set_color c_{obj}_{ai}, [{r:.3f},{g:.3f},{b:.3f}]",
                    f"color c_{obj}_{ai}, {obj} and id {ai}"]
    sel = "prot and (" + " or ".join(f"resi {r}" for r in draw_residues) + ")" if draw_residues else "prot"
    pml += ["set field_of_view, 25",
            f"orient ({' or '.join(tunnel_objs)})" if tunnel_objs else "orient prot",
            "turn y, 10", "turn x, -8", "zoom (" + " or ".join(tunnel_objs + ['prot']) + "), -3"
            if tunnel_objs else "zoom prot"]
    pml += [f"png {out_png}, width={IMG_W}, height={IMG_H}, dpi=200, ray=1", "quit"]
    f = tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False)
    f.write("\n".join(pml))
    f.close()
    subprocess.run([PYMOL, "-cq", f.name], capture_output=True, text=True, timeout=600)
    os.unlink(f.name)
    if not os.path.exists(out_png):
        raise RuntimeError("pymol produced no image (see pymol output)")
    return out_png


def _composite(render_png, res, out_png, title):
    """Lay the ray-traced render into a matplotlib canvas with a radius colour-bar + a measurement box."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.asarray(Image.open(render_png).convert("RGBA")) / 255.0
    fig = plt.figure(figsize=(IMG_W / 200, IMG_H / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")

    chans = res.get("channels", [])
    main = chans[0] if chans else None
    # radius colour-bar (matches the sphere colours exactly) -- the only legend on the figure
    cax = fig.add_axes([0.055, 0.12, 0.022, 0.30])
    sm = cm.ScalarMappable(mcolors.Normalize(RADIUS_VMIN, RADIUS_VMAX), CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("tunnel radius (Å)", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    # one compact line of numbers, bottom-right
    if main:
        n = res.get("n_channels", len(chans))
        txt = (f"{n} channel{'s' if n != 1 else ''}   ·   bottleneck "
               f"{main['bottleneck_A']['mean']:.2f} Å   ·   length {main['length_A']['mean']:.1f} Å")
        ax.text(0.985, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
                family="DejaVu Sans Mono", weight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#555555", lw=1.2, alpha=0.9))
    if title:
        ax.text(0.5, 0.985, title, transform=ax.transAxes, ha="center", va="top", fontsize=16,
                weight="bold", color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def render(run_dir, out_png, title=None, all_clusters=False, max_residues=6):
    res = json.load(open(os.path.join(run_dir, "channels.json")))
    channels = res.get("clusters", []) if all_clusters else res.get("channels", [])
    if not channels:
        raise ValueError(f"no channels to draw in {run_dir}/channels.json "
                         f"(n_channels={res.get('n_channels')}); try --all-clusters")
    main = channels[0]
    draw = []
    for r in main.get("bottleneck_lining_residues", [])[:max_residues]:
        rid = "".join(c for c in str(r["residue"]) if c.isdigit())
        if rid and rid not in draw:
            draw.append(rid)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False); tmp.close()
    try:
        _render_pose(run_dir, channels, tmp.name, draw)
        _composite(tmp.name, res, out_png, title)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    return out_png


def _cli(argv=None):
    ap = argparse.ArgumentParser(prog="render_channels", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, help="a substrate_channels output dir (has channels.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title")
    ap.add_argument("--all-clusters", action="store_true", help="draw every cluster, not just channels")
    a = ap.parse_args(argv)
    print("wrote", render(a.run_dir, a.out, a.title, a.all_clusters))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
