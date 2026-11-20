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


def _split_ligand_pdbs(holo, tmpdir):
    """Write the substrate / cofactor / metal atom groups (from substrate_contacts.split_ligands) to
    SEPARATE PDBs so each can be coloured distinctly. Selecting by resname does NOT work: Boltz names
    every ligand 'LIG' (substrate, ATP and Mg share the resname, differing only by chain), so a resname
    selection grabs all three at once. Returns {name: pdb_path} for the non-empty groups, in the holo
    frame (= the tunnel frame)."""
    import warnings
    from . import substrate_channels as _sch, substrate_contacts as _sc
    import MDAnalysis as mda
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            u = mda.Universe(_sch._to_pdb(holo, tmpdir))
            cof, metals, sub = _sc.split_ligands(u)
        except (ValueError, Exception):   # noqa: BLE001 -- ligand overlay is best-effort
            return out
        for name, ag in (("substrate", sub), ("cofactor", cof), ("metal", metals)):
            if len(ag):
                p = os.path.join(tmpdir, f"lig_{name}.pdb")
                ag.write(p)
                out[name] = p
    return out


def _render_pose(run_dir, channels, out_png, draw_residues, holo=None, cutaway=True):
    """PyMOL render. cutaway=True: a SOLID protein surface sliced open (front clipped) so you look into
    the active site, with the SUBSTRATE (from the holo cofold pose, same frame) shown in the pocket and
    the tunnels as opaque radius-coloured tubes. cutaway=False: translucent cartoon + see-through tubes.
    Returns the ray-traced png path. Substrate overlay needs `holo` (an original pose with the ligand;
    the channel run strips it, so it is read back from channels.json provenance)."""
    data = os.path.join(run_dir, "caver_out", "data")
    import glob
    hits = sorted(glob.glob(os.path.join(data, "pose_*.pdb"))) or \
        sorted(glob.glob(os.path.join(run_dir, "pdb", "pose_*.pdb")))
    if not hits:
        raise FileNotFoundError(f"no representative pose_*.pdb under {data} (was the run staged here?)")

    pml = ["bg_color white", "set ray_opaque_background, 0", "set antialias, 2",
           "set two_sided_lighting, 1", "set specular, 0.3", "set field_of_view, 20"]
    lig = {}
    use_holo = bool(holo and os.path.exists(holo))
    if use_holo:
        lig = _split_ligand_pdbs(holo, tempfile.mkdtemp())
        pml += [f"load {holo}, sysm", "hide everything"]   # sysm -> protein only; ligands drawn separately
    else:
        pml += [f"load {hits[0]}, sysm", "hide everything"]
    prot = "sysm and polymer"

    if cutaway:   # solid body, sliced open to look into the pocket
        pml += ["set ray_shadows, 1", "set ambient, 0.45", "set ray_interior_color, grey65",
                f"show surface, ({prot})", "set surface_color, wheat, sysm", "set transparency, 0, sysm"]
    else:         # DEFAULT: faint gray cartoon as light context; tunnels + ligands are the subject
        pml += ["set ray_shadows, 1", "set ambient, 0.45", "set specular, 0.2",
                "set cartoon_transparency, 0.55", "set ray_interior_color, grey80",
                f"show cartoon, ({prot})", "color gray70, sysm", "set cartoon_side_chain_helper, 1"]
    # LIGANDS as separate objects, each a distinct representation/colour:
    if "substrate" in lig:   # SUBSTRATE: magenta ball-and-stick (the acyl-acceptor)
        pml += [f"load {lig['substrate']}, substrate", "hide everything, substrate",
                "show sticks, substrate", "set stick_radius, 0.28, substrate",
                "show spheres, substrate", "set sphere_scale, 0.26, substrate",
                "color magenta, substrate and elem C", "util.cnc substrate"]
    if "cofactor" in lig:    # COFACTOR (ATP): orange sticks
        pml += [f"load {lig['cofactor']}, cofactor", "hide everything, cofactor",
                "show sticks, cofactor", "set stick_radius, 0.18, cofactor",
                "color orange, cofactor and elem C", "util.cnc cofactor"]
    if "metal" in lig:       # METAL (Mg): a few small green spheres -- NOT the ligand
        pml += [f"load {lig['metal']}, metal", "hide everything, metal",
                "show spheres, metal", "color forest, metal", "set sphere_scale, 0.35, metal"]

    tunnel_objs = []
    ttransp = 0.35 if cutaway else 0.6   # cartoon: translucent tubes so the substrate molecule reads inside
    for ci, ch in enumerate(channels):
        tp = _tunnel_pdb(data, ch["cluster"])
        if not tp:
            continue
        obj = f"t{ci}"
        tunnel_objs.append(obj)
        pml += [f"load {tp}, {obj}", f"hide everything, {obj}", f"alter {obj}, vdw=b", "rebuild",
                f"set solvent_radius, 0, {obj}", f"set surface_quality, 2, {obj}",
                f"set transparency, {ttransp}, {obj}", f"show surface, {obj}"]
        for ai, (r, g, b) in enumerate(_sphere_colors(_radii(tp)), 1):
            pml += [f"set_color c_{obj}_{ai}, [{r:.3f},{g:.3f},{b:.3f}]",
                    f"color c_{obj}_{ai}, {obj} and id {ai}"]
    for rid in draw_residues:
        pml += [f"select r{rid}, ({prot} and resi {rid} and not name C+N+O)", f"show sticks, r{rid}",
                f"set stick_radius, 0.3, r{rid}", f"color gray25, r{rid} and elem C", f"util.cnc r{rid}"]

    tun = " or ".join(tunnel_objs) or prot
    site = "substrate" if "substrate" in lig else tun   # centre on the substrate if we have it, else tunnels
    if cutaway:
        # HOLISTIC: frame the WHOLE protein, then cut a slab CENTRED ON THE ACTIVE SITE (clip's optional
        # selection arg) so the whole body is in view but sliced open where the substrate + tunnels are
        pml += [f"orient ({prot})", "turn y, 12", f"zoom ({prot}), 4", f"clip slab, 26, ({site})"]
    else:
        # frame the tunnels + ligands with enough surrounding cartoon for context (readable, not a closeup
        # nor a tiny whole-dimer); clip generously so no cartoon is sliced away
        pml += [f"orient ({tun} or ({site}))", "turn y, 12", f"zoom ({tun} or ({site})), 13",
                "clip slab, 1000"]
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


def render(run_dir, out_png, title=None, all_clusters=False, max_residues=0, cutaway=False, holo=None):
    """Default: a cutaway (solid protein sliced open) with the SUBSTRATE shown in the active site and the
    tunnels as opaque radius-coloured tubes. The substrate is read back from the original cofold pose
    (channels.json provenance, or --holo) because the channel run strips it. cutaway=False gives the
    translucent-cartoon + see-through-tubes view. max_residues>0 overlays bottleneck-lining sticks."""
    res = json.load(open(os.path.join(run_dir, "channels.json")))
    channels = res.get("clusters", []) if all_clusters else res.get("channels", [])
    if not channels:
        raise ValueError(f"no channels to draw in {run_dir}/channels.json "
                         f"(n_channels={res.get('n_channels')}); try --all-clusters")
    if holo is None:        # the original pose (with the ligand) recorded by substrate_channels
        poses = res.get("provenance", {}).get("poses", [])
        holo = next((p for p in poses if os.path.exists(p)), None)
    main = channels[0]
    draw = []
    for r in main.get("bottleneck_lining_residues", [])[:max_residues]:
        rid = "".join(c for c in str(r["residue"]) if c.isdigit())
        if rid and rid not in draw:
            draw.append(rid)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False); tmp.close()
    try:
        _render_pose(run_dir, channels, tmp.name, draw, holo=holo, cutaway=cutaway)
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
    ap.add_argument("--show-residues", type=int, default=0, metavar="N",
                    help="overlay N bottleneck-lining residues as sticks (default 0 = none)")
    ap.add_argument("--cutaway", action="store_true",
                    help="sliced-open solid surface instead of the default coloured cartoon (rough)")
    ap.add_argument("--holo", help="pose with the substrate to overlay (default: from channels.json provenance)")
    a = ap.parse_args(argv)
    print("wrote", render(a.run_dir, a.out, a.title, a.all_clusters, max_residues=a.show_residues,
                          cutaway=a.cutaway, holo=a.holo))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
