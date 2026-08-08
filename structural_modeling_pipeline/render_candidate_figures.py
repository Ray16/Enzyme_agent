#!/usr/bin/env python3
"""
Render publication-quality PyMOL images of holo (co-folded) structures
for top synthetase and hydrolase candidates.

Adapts rendering approach from PMD project (generate_struct_figures.py).

Outputs:
    analysis_results/struct_{gene}_{uid}_holo.png  (per-candidate)
    analysis_results/struct_candidates_grid.png     (composite 2x4)

Usage:
    python render_candidate_figures.py
"""

import sys
import os
import json
import csv

sys.argv = ['pymol', '-cq']
import pymol
pymol.finish_launching(['pymol', '-cq'])
from pymol import cmd, stored

BASE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(BASE, "analysis_results")
SYN_STRUCT = os.path.join(BASE, "all_structures", "amide_synthetases_candidates", "cofolded_structure")
HYD_STRUCT = os.path.join(BASE, "all_structures", "amide_hydrolase_candidates", "cofolded_structure")

WIDTH, HEIGHT = 2400, 1800
DPI = 300

CHAIN_COLORS = {
    "B": "cyan",
    "C": "brightorange",
    "D": "lightorange",
    "E": "lightpink",
    "F": "paleblue",
}

CANDIDATES = [
    {"uid": "P31663", "gene": "PanC", "type": "synthetase",
     "label": "Pantothenate synthetase",
     "pdb": os.path.join(SYN_STRUCT, "panC_P31663.pdb")},
    {"uid": "P39641", "gene": "BacD", "type": "synthetase",
     "label": "Alanine-anticapsin ligase",
     "pdb": os.path.join(SYN_STRUCT, "bacD_P39641.pdb")},
    {"uid": "P0A6W9", "gene": "GshA", "type": "synthetase",
     "label": "Glutamate-cysteine ligase",
     "pdb": os.path.join(SYN_STRUCT, "gshA_P0A6W9.pdb")},
    {"uid": "E2JA31", "gene": "DdaF", "type": "synthetase",
     "label": "Dapdiamide A synthase",
     "pdb": os.path.join(SYN_STRUCT, "ddaF_E2JA31.pdb")},
    {"uid": "Q1EPR4_autocleavage", "gene": "NylC", "type": "hydrolase",
     "label": "Nylon hydrolase (autocleavage)",
     "pdb": os.path.join(BASE, "amide_hydrolase", "cofold_output",
                         "Q1EPR4_autocleavage",
                         "boltz_results_Q1EPR4_autocleavage",
                         "predictions", "Q1EPR4_autocleavage",
                         "Q1EPR4_autocleavage_model_0.pdb")},
    {"uid": "A0A553ZW58", "gene": "FN960_14580", "type": "hydrolase",
     "label": "PGA hydrolase (Gamma_PGA_hydro)",
     "pdb": os.path.join(HYD_STRUCT, "FN960_14580_A0A553ZW58.pdb")},
    {"uid": "A0A7W7H167", "gene": "BJY16_005544", "type": "hydrolase",
     "label": "Peptidase S51 family",
     "pdb": os.path.join(HYD_STRUCT, "BJY16_005544_A0A7W7H167.pdb")},
    {"uid": "F0TA78", "gene": "Metbo_1806", "type": "hydrolase",
     "label": "Transglutaminase-fold",
     "pdb": os.path.join(HYD_STRUCT, "Metbo_1806_F0TA78.pdb")},
]


def setup_common():
    cmd.set("ray_opaque_background", 1)
    cmd.set("bg_rgb", [1.0, 1.0, 1.0])
    cmd.set("ray_trace_mode", 0)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("depth_cue", 0)
    cmd.set("fog", 0)
    cmd.set("specular", 0.3)
    cmd.set("spec_reflect", 0.3)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_flat_sheets", 1)
    cmd.set("stick_radius", 0.18)
    cmd.set("sphere_scale", 0.25)
    cmd.set("label_size", 14)
    cmd.set("label_font_id", 7)
    cmd.set("label_color", "black")
    cmd.set("label_outline_color", "white")
    cmd.set("label_position", [0, 1, 3])


def identify_chains(obj_name):
    """Identify protein and ligand chains in a loaded structure."""
    aa_set = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
              "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
              "THR", "TRP", "TYR", "VAL"}
    stored.chains = []
    cmd.iterate(f"{obj_name}", "stored.chains.append(chain)", quiet=1)
    all_chains = sorted(set(stored.chains))

    protein_chains = []
    ligand_chains = []
    for ch in all_chains:
        if not ch:
            continue
        stored.resnames = []
        cmd.iterate(f"{obj_name} and chain {ch}", "stored.resnames.append(resn)", quiet=1)
        resnames = set(stored.resnames)
        if resnames & aa_set:
            protein_chains.append(ch)
        else:
            ligand_chains.append(ch)

    return protein_chains, ligand_chains


def render_candidate(candidate, output_dir):
    """Render a single candidate's holo structure."""
    uid = candidate["uid"]
    gene = candidate["gene"]
    label = candidate["label"]
    pdb_path = candidate["pdb"]

    if not os.path.exists(pdb_path):
        print(f"  WARNING: {pdb_path} not found, skipping {gene}")
        return None

    print(f"  Rendering {gene} ({uid})...")
    cmd.delete("all")
    setup_common()

    obj = "complex"
    cmd.load(pdb_path, obj)

    protein_chains, ligand_chains = identify_chains(obj)
    if not protein_chains:
        print(f"    WARNING: no protein chains found in {gene}")
        return None

    prot_sel = " or ".join(f"chain {ch}" for ch in protein_chains)
    cmd.select("protein", f"{obj} and ({prot_sel})")

    # Protein: gray semi-transparent cartoon
    cmd.hide("everything", obj)
    cmd.show("cartoon", "protein")
    cmd.set("cartoon_transparency", 0.2, "protein")
    cmd.color("gray80", "protein")

    # Ligands: colored sticks + spheres
    for i, ch in enumerate(ligand_chains):
        lig_sel = f"{obj} and chain {ch}"
        n_atoms = cmd.count_atoms(lig_sel)
        if n_atoms == 0:
            continue

        color = CHAIN_COLORS.get(ch, "gray")
        cmd.show("sticks", lig_sel)
        cmd.show("spheres", lig_sel)
        cmd.color(color, f"({lig_sel}) and elem C")
        cmd.color("tv_red", f"({lig_sel}) and elem O")
        cmd.color("tv_blue", f"({lig_sel}) and elem N")
        cmd.color("orange", f"({lig_sel}) and elem P")
        cmd.color("brightorange", f"({lig_sel}) and elem MG")
        cmd.set("stick_radius", 0.18, lig_sel)
        cmd.set("sphere_scale", 0.25, lig_sel)

        # Metal ions as larger spheres
        stored.elems = []
        cmd.iterate(lig_sel, "stored.elems.append(elem)", quiet=1)
        if n_atoms <= 2 or any(e in ("MG", "ZN", "MN", "CA", "FE")
                               for e in set(stored.elems)):
            cmd.set("sphere_scale", 0.6, lig_sel)

    # Show pocket-lining residues as thin lines for context
    if ligand_chains:
        lig_all = " or ".join(f"chain {ch}" for ch in ligand_chains)
        cmd.select("near_lig", f"protein within 5.0 of ({obj} and ({lig_all}))")
        cmd.show("sticks", "near_lig and sidechain")
        cmd.color("gray60", "near_lig and elem C")
        cmd.set("stick_radius", 0.1, "near_lig")

    # Zoom to binding site
    if ligand_chains:
        lig_all = " or ".join(f"chain {ch}" for ch in ligand_chains)
        cmd.select("focus", f"({obj} and ({lig_all})) or near_lig")
        cmd.zoom("focus", buffer=6)
    else:
        cmd.zoom(obj, buffer=5)

    cmd.deselect()

    # Render
    out_path = os.path.join(output_dir, f"struct_{gene}_{uid}_holo.png")
    cmd.ray(WIDTH, HEIGHT)
    cmd.png(out_path, dpi=DPI)
    print(f"    Saved: {out_path}")
    return out_path


def make_composite(image_paths, output_path, ncols=4):
    """Combine individual PNGs into a grid using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  PIL not available, skipping composite grid")
        return

    images = []
    labels = []
    for cand, path in zip(CANDIDATES, image_paths):
        if path and os.path.exists(path):
            images.append(Image.open(path))
            labels.append(f"{cand['gene']} ({cand['uid'].split('_')[0]})")
        else:
            images.append(None)
            labels.append("")

    if not any(images):
        return

    nrows = (len(images) + ncols - 1) // ncols
    cell_w, cell_h = WIDTH, HEIGHT
    label_h = 60
    grid = Image.new("RGB", (ncols * cell_w, nrows * (cell_h + label_h)),
                     color=(255, 255, 255))

    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, (img, label) in enumerate(zip(images, labels)):
        col = idx % ncols
        row = idx // ncols
        x = col * cell_w
        y = row * (cell_h + label_h)

        if img:
            img_resized = img.resize((cell_w, cell_h), Image.LANCZOS)
            grid.paste(img_resized, (x, y + label_h))

        draw.text((x + cell_w // 2, y + label_h // 2), label,
                  fill=(0, 0, 0), font=font, anchor="mm")

    grid.save(output_path, dpi=(DPI, DPI))
    print(f"\nComposite saved: {output_path}")


def main():
    os.makedirs(ANALYSIS, exist_ok=True)

    print("Rendering holo structures for top candidates...")
    print(f"{'='*60}")

    rendered_paths = []
    for cand in CANDIDATES:
        path = render_candidate(cand, ANALYSIS)
        rendered_paths.append(path)

    composite_path = os.path.join(ANALYSIS, "struct_candidates_grid.png")
    make_composite(rendered_paths, composite_path)

    print(f"\n{'='*60}")
    print(f"Done. {sum(1 for p in rendered_paths if p)} of {len(CANDIDATES)} rendered.")


if __name__ == "__main__":
    main()
