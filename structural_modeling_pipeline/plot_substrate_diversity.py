#!/usr/bin/env python3
"""
Plot native substrate diversity across synthetase enzyme families.

Compares native substrate molecular weights to the nylon-6,6 target
monomers (adipic acid MW=146.14, HMDA MW=116.20) to identify which
enzyme families have the closest substrate chemistry match.

Outputs:
    analysis_results/substrate_diversity.png

Usage:
    python plot_substrate_diversity.py
"""

import csv
import json
import os
import re
import time
import urllib.request
import urllib.parse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(BASE_DIR, "analysis_results")
CACHE_PATH = os.path.join(ANALYSIS, "substrate_smiles_cache.json")

TARGET_SUBSTRATES = {
    "adipic acid": {"smiles": "OC(=O)CCCCC(=O)O", "mw": 146.14, "role": "acid"},
    "HMDA": {"smiles": "NCCCCCCN", "mw": 116.20, "role": "amine"},
}

FAMILY_DISPLAY = {
    "FhuF": "NIS synthetase",
    "ATP-grasp_4": "ATP-grasp",
    "ATP_grasp_subdomain_1": "ATP-grasp",
    "ATP-grasp": "ATP-grasp",
    "Mur_ligase_C": "Mur ligase",
    "Mur_ligase_M": "Mur ligase",
    "AMP-binding": "ANL superfamily",
    "AMP-binding_C": "ANL superfamily",
    "Pantoate_ligase": "Pantoate ligase",
    "Gln_synth/guanido_kin_cat_dom": "Gln synthetase fold",
    "GSP_synth": "GSP synthetase",
    "CHAP": "Amidase/CHAP",
    "Cyanophycin_syn": "Cyanophycin syn.",
    "PylC-like_N": "PylC family",
    "CapK_Type1_Caps_Biosynth": "ANL superfamily",
}

FAMILY_COLORS = {
    "NIS synthetase": "#e74c3c",
    "ATP-grasp": "#3498db",
    "ANL superfamily": "#9b59b6",
    "Pantoate ligase": "#2ecc71",
    "Mur ligase": "#f39c12",
    "Gln synthetase fold": "#1abc9c",
    "GSP synthetase": "#e67e22",
    "Amidase/CHAP": "#34495e",
    "Cyanophycin synthetase": "#c0392b",
    "PylC family": "#2980b9",
    "Other": "#95a5a6",
}


def load_smiles_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_smiles_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


NAME_NORMALIZATION = {
    "N(6)-acetyl-N(6)-hydroxy-L-lysine": "N6-acetyl-N6-hydroxylysine",
    "N(2)-citryl-N(6)-acetyl-N(6)-hydroxy-L-lysine": "citryl-acetylhydroxylysine",
    "(R)-pantoate": "pantoic acid",
    "(indol-3-yl)acetate": "indole-3-acetic acid",
    "(S)-2,3-diaminopropanoate": "2,3-diaminopropionic acid",
    "(3R)-3-methyl-D-ornithine": "3-methylornithine",
    "L-anticapsin": "anticapsin",
    "N(8)-citryl-spermidine": "citrylspermidine",
    "gamma-L-glutamyl-L-cysteine": "gamma-glutamylcysteine",
    "3-[[[(2R,3R)-3-carboxyoxiran-2-yl]carbonyl]amino]-L-alanine":
        "epoxycyclohexanone-alanine",
    "[L-4-(L-arginin-2-N-yl)aspartate](n)": "cyanophycin",
    "UDP-N-acetyl-alpha-D-muramate": "UDP-N-acetylmuramic acid",
    "UDP-N-acetyl-alpha-D-muramoyl-L-alanine": "UDP-MurNAc-L-alanine",
}

MANUAL_SMILES = {
    "N6-acetyl-N6-hydroxylysine": "CC(=O)N(O)[C@@H](CCCCN)C(=O)O",
    "citryl-acetylhydroxylysine": "OC(=O)CC(O)(CC(=O)O)C(=O)NCCCC[C@@H](NC(=O)C)C(=O)O",
    "anticapsin": "O=C1C=C[C@@H](O)[C@H]1C[C@@H](N)C(=O)O",
    "citrylspermidine": "OC(=O)CC(O)(CC(=O)O)C(=O)NCCCNCCCCN",
    "epoxycyclohexanone-alanine": "O=C1CC[C@@H]2O[C@@H]2C1C(=O)N[C@@H](CN)C(=O)O",
    "cyanophycin": "N=C(N)NCCCC(NC(=O)CC(N)C(=O)O)C(=O)O",
    "UDP-MurNAc-L-alanine": None,
}

MANUAL_MW = {
    "N6-acetyl-N6-hydroxylysine": 204.22,
    "citryl-acetylhydroxylysine": 396.35,
    "anticapsin": 185.18,
    "citrylspermidine": 337.37,
    "epoxycyclohexanone-alanine": 230.22,
    "cyanophycin": 289.29,
    "UDP-MurNAc-L-alanine": 750.45,
}


def pubchem_name_to_smiles(name):
    """Resolve a compound name to SMILES via PubChem REST API."""
    clean = re.sub(r"^\d+\s+", "", name.strip())
    clean = NAME_NORMALIZATION.get(clean, clean)

    if clean in MANUAL_MW:
        return {
            "smiles": MANUAL_SMILES.get(clean, "") or "",
            "mw": MANUAL_MW[clean],
        }

    encoded = urllib.parse.quote(clean)
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
           f"name/{encoded}/property/IsomericSMILES,CanonicalSMILES,MolecularWeight/JSON")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        smiles = (props.get("IsomericSMILES")
                  or props.get("SMILES")
                  or props.get("CanonicalSMILES")
                  or props.get("ConnectivitySMILES")
                  or "")
        return {
            "smiles": smiles,
            "mw": props.get("MolecularWeight", 0),
        }
    except Exception:
        return None


def resolve_substrates(substrate_names, cache):
    """Resolve a list of substrate names to SMILES and MW."""
    results = []
    for name in substrate_names:
        name = name.strip()
        if not name:
            continue
        stoich = 1
        m = re.match(r"^(\d+)\s+(.+)$", name)
        if m:
            stoich = int(m.group(1))
            name = m.group(2)

        if name in cache:
            entry = cache[name]
            if entry and (entry.get("smiles") or entry.get("mw")):
                for _ in range(stoich):
                    results.append({"name": name, **entry})
            continue

        info = pubchem_name_to_smiles(name)
        if info and (info.get("smiles") or info.get("mw")):
            if info.get("mw"):
                info["mw"] = float(info["mw"])
            cache[name] = info
            for _ in range(stoich):
                results.append({"name": name, **info})
        else:
            cache[name] = None
        time.sleep(0.3)

    return results


def count_functional_groups(smiles):
    """Count key functional groups using RDKit SMARTS patterns."""
    if not smiles:
        return {}

    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    def _count(smarts):
        pat = Chem.MolFromSmarts(smarts)
        return len(mol.GetSubstructMatches(pat)) if pat else 0

    counts = {}
    counts["carboxylate"] = _count("[CX3](=O)[OX1H0-,OX2H1]")
    counts["primary amine"] = _count("[NX3H2;!$([NX3H2]C=O)]")
    counts["amide"] = _count("[NX3][CX3](=O)")
    counts["hydroxyl"] = _count("[OX2H;!$([OX2H]C=O)]")
    counts["thiol"] = _count("[SX2H]")
    counts["aromatic"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    counts["phosphate"] = _count("[PX4]")
    counts["guanidinium"] = _count("[NX3]C(=[NX2])[NX3]")

    # Longest aliphatic carbon chain
    from rdkit.Chem import rdmolops
    try:
        chain_lengths = []
        for bond_path in Chem.FindAllPathsOfLengthN(mol, 20, useBonds=False):
            pass
        # Simpler: count total non-ring, non-aromatic carbons as proxy
        n_aliph_c = sum(1 for atom in mol.GetAtoms()
                        if atom.GetAtomicNum() == 6
                        and not atom.GetIsAromatic()
                        and not atom.IsInRing())
        counts["aliphatic C"] = n_aliph_c
    except Exception:
        counts["aliphatic C"] = 0

    return counts


FG_COLUMNS = [
    "carboxylate", "primary amine", "amide", "hydroxyl",
    "thiol", "aromatic", "guanidinium", "phosphate", "aliphatic C",
]


def _draw_bracket(ax, family_name, row_start, row_end, bracket_x, label_offset):
    """Draw a square bracket with family label to the left of the y-axis.
    For single-entry families, draw a short dash instead."""
    n_members = row_end - row_start
    top = float(row_start)
    bot = float(row_end - 1)
    mid = (top + bot) / 2
    fam_color = FAMILY_COLORS.get(family_name, "#333333")
    tick_w = 0.015

    if n_members == 1:
        top = mid - 0.3
        bot = mid + 0.3
        ax.plot([bracket_x, bracket_x], [top, bot],
                transform=ax.get_yaxis_transform(),
                color=fam_color, lw=1.5, clip_on=False)
        ax.plot([bracket_x, bracket_x + tick_w], [top, top],
                transform=ax.get_yaxis_transform(),
                color=fam_color, lw=1.5, clip_on=False)
        ax.plot([bracket_x, bracket_x + tick_w], [bot, bot],
                transform=ax.get_yaxis_transform(),
                color=fam_color, lw=1.5, clip_on=False)
    else:
        ax.plot([bracket_x, bracket_x], [top, bot],
                transform=ax.get_yaxis_transform(),
                color=fam_color, lw=1.5, clip_on=False)
        ax.plot([bracket_x, bracket_x + tick_w], [top, top],
                transform=ax.get_yaxis_transform(),
                color=fam_color, lw=1.5, clip_on=False)
        ax.plot([bracket_x, bracket_x + tick_w], [bot, bot],
                transform=ax.get_yaxis_transform(),
                color=fam_color, lw=1.5, clip_on=False)

    ax.text(bracket_x - 0.005, mid, family_name,
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8, fontweight="bold",
            color=fam_color, clip_on=False)


def main():
    cache = load_smiles_cache()

    with open(os.path.join(ANALYSIS, "enzyme_family_classification.csv")) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader
                if r["project"] == "synthetase" and r.get("native_substrates")]

    print(f"Synthetases with native substrates: {len(rows)}")

    enzymes = []
    for row in rows:
        uid = row["uniprot_id"]
        gene = row["enzyme_name"]
        family_raw = row["assigned_family"]
        family = FAMILY_DISPLAY.get(family_raw, "Other")
        substrate_names = [s.strip() for s in row["native_substrates"].split(";")]

        resolved = resolve_substrates(substrate_names, cache)
        if not resolved:
            print(f"  WARNING: no data for {uid} ({gene}): {substrate_names}")
            continue

        all_fg = {}
        for sub in resolved:
            if sub.get("smiles"):
                fg = count_functional_groups(sub["smiles"])
                for k, v in fg.items():
                    all_fg[k] = all_fg.get(k, 0) + v

        enzymes.append({
            "uid": uid,
            "gene": gene,
            "family": family,
            "substrates": resolved,
            "fg_combined": all_fg,
        })

    save_smiles_cache(cache)
    print(f"Resolved substrates for {len(enzymes)} enzymes")
    print(f"  With SMILES for FG analysis: "
          f"{sum(1 for e in enzymes if e['fg_combined'])}")

    family_order = [
        "NIS synthetase", "ATP-grasp", "Mur ligase", "ANL superfamily",
        "Pantoate ligase", "Gln synthetase fold", "GSP synthetase",
        "Cyanophycin synthetase", "PylC family", "Other",
    ]
    families_present = [f for f in family_order
                        if any(e["family"] == f for e in enzymes)]

    sorted_enzymes = []
    for fam in families_present:
        fam_enz = [e for e in enzymes if e["family"] == fam]
        sorted_enzymes.extend(fam_enz)

    # Target substrates FG profile
    target_fg = {}
    for tname, tinfo in TARGET_SUBSTRATES.items():
        fg = count_functional_groups(tinfo["smiles"])
        for k, v in fg.items():
            target_fg[k] = target_fg.get(k, 0) + v

    # Capitalize gene names (asbA -> AsbA, ddlB -> DdlB, etc.)
    def _capitalize_gene(g):
        if not g:
            return g
        return g[0].upper() + g[1:]

    SHORT_SUBSTRATE_NAMES = {
        "putrescine": "putrescine",
        "L-glutamate": "Glu",
        "spermidine": "spermidine",
        "citrate": "citrate",
        "N(8)-citryl-spermidine": "citryl-Spd",
        "glutathione": "GSH",
        "(S)-2,3-diaminopropanoate": "L-Dap",
        "fumarate": "fumarate",
        "N(6)-acetyl-N(6)-hydroxy-L-lysine": "AHL",
        "N(2)-citryl-N(6)-acetyl-N(6)-hydroxy-L-lysine": "citryl-AHL",
        "(indol-3-yl)acetate": "IAA",
        "L-lysine": "Lys",
        "beta-alanine": "β-Ala",
        "L-histidine": "His",
        "L-anticapsin": "anticapsin",
        "L-alanine": "Ala",
        "UDP-N-acetyl-alpha-D-muramoyl-L-alanine": "UDP-MurNAc-Ala",
        "D-glutamate": "D-Glu",
        "L-cysteine": "Cys",
        "(R)-pantoate": "pantoate",
        "3-[[[(2R,3R)-3-carboxyoxiran-2-yl]carbonyl]amino]-L-alanine": "Dap-epoxide",
        "L-valine": "Val",
        "UDP-N-acetyl-alpha-D-muramate": "UDP-MurNAc",
        "[L-4-(L-arginin-2-N-yl)aspartate](n)": "cyanophycin",
        "L-aspartate": "Asp",
        "gamma-L-glutamyl-L-cysteine": "γ-Glu-Cys",
        "glycine": "Gly",
        "D-alanine": "D-Ala",
        "(3R)-3-methyl-D-ornithine": "3-Me-D-Orn",
    }

    def _format_substrate_names(substrates):
        """Format substrate names for display below enzyme name."""
        names = []
        seen = set()
        for s in substrates:
            n = s.get("name", "")
            short = SHORT_SUBSTRATE_NAMES.get(n, n)
            if short not in seen:
                names.append(short)
                seen.add(short)
        if not names:
            return ""
        return " + ".join(names)

    # Build heatmap matrix
    row_labels = [_capitalize_gene(e['gene']) for e in sorted_enzymes]
    row_families = [e["family"] for e in sorted_enzymes]

    matrix = np.zeros((len(sorted_enzymes) + 1, len(FG_COLUMNS)))
    # First row: target
    for j, fg_name in enumerate(FG_COLUMNS):
        matrix[0, j] = target_fg.get(fg_name, 0)
    # Enzyme rows
    for i, enz in enumerate(sorted_enzymes):
        for j, fg_name in enumerate(FG_COLUMNS):
            matrix[i + 1, j] = enz["fg_combined"].get(fg_name, 0)

    # Use absolute counts for color mapping (global max across all cells)
    vmax = matrix.max()
    if vmax == 0:
        vmax = 1

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(left=0.33)

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=vmax)

    # Annotate cells with actual counts
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            if val > 0:
                color = "white" if (val / vmax) > 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    # Y-axis: target + enzyme names
    all_row_labels = ["TARGET\n(adipic acid + HMDA)"] + row_labels
    ax.set_yticks(range(len(all_row_labels)))
    ax.set_yticklabels(all_row_labels, fontsize=9)

    # X-axis: functional group names
    ax.set_xticks(range(len(FG_COLUMNS)))
    ax.set_xticklabels(FG_COLUMNS, rotation=45, ha="right", fontsize=10)

    # Highlight target row
    ax.axhline(y=0.5, color="black", linewidth=2)

    # Determine bracket position automatically based on longest enzyme name
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_x0 = 0
    for label in ax.get_yticklabels():
        if "TARGET" in label.get_text():
            continue
        bbox = label.get_window_extent(renderer=renderer)
        bbox_axes = bbox.transformed(ax.transAxes.inverted())
        max_label_x0 = max(max_label_x0, abs(bbox_axes.x0))
    bracket_x = -(max_label_x0 + 0.05)
    label_offset = 8
    prev_fam = None
    fam_start = 1
    for i, enz in enumerate(sorted_enzymes):
        fam = enz["family"]
        if fam != prev_fam:
            if prev_fam is not None:
                _draw_bracket(ax, prev_fam, fam_start, i + 1,
                              bracket_x, label_offset)
            fam_start = i + 1
            prev_fam = fam
    if prev_fam:
        _draw_bracket(ax, prev_fam, fam_start, len(sorted_enzymes) + 1,
                      bracket_x, label_offset)

    # No title — caption is in the LaTeX figure environment
    ax.set_xlabel("Functional group (combined across both substrates)", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, label="Count", pad=0.12)
    cbar.set_ticks(np.arange(0, int(vmax) + 1, max(1, int(np.ceil(vmax / 10)))))


    out_path = os.path.join(ANALYSIS, "substrate_diversity.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
