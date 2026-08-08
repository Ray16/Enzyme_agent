#!/usr/bin/env python3
"""
Auto-generate config.yaml for a structural modeling project.

Surveys AlphaFill across a sample of candidate enzymes to discover what
cofactors and metals they use, then writes a config.yaml with the
dominant cofactor group and co-occurring metals.

Usage:
    python generate_config.py <project_dir> \
        --substrates "hexamethylenediamine=[NH3+]CCCCCC[NH3+]" \
                     "adipic_acid=[O-]C(=O)CCCCC(=O)[O-]"

    python generate_config.py <project_dir> \
        --substrates "substrate_a=NCCCCCCN" "substrate_b=OC(=O)CCCCC(=O)O" \
        --reaction-name "Amide bond synthesis"
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import yaml

# Import from existing pipeline modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_analysis import fetch_alphafill, CRYSTALLIZATION_ARTIFACTS
from prepare_cofold_inputs import (
    COFACTOR_CCD_CODES, COFACTOR_PHYSIOLOGICAL_MAP, METAL_PHYSIOLOGICAL_MAP,
)

# Biologically relevant catalytic metal ions (CCD codes).
# These are kept when found in AlphaFill results alongside cofactors.
CATALYTIC_METAL_CCDS = {
    "MG", "MN", "ZN", "FE", "CA", "CO", "NI", "CU",
    "FE2", "MO", "W",  # less common but valid
}


def load_candidate_uids(project_dir):
    """Load candidate UniProt IDs from project directory."""
    uids = []

    # Source 1: structure_sequence_summary.csv
    csv_path = os.path.join(project_dir, "structure_sequence_summary.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                uid = row.get("uniprot_id", "").strip()
                if uid:
                    uids.append(uid)

    # Source 2: all_sequences.fasta headers
    if not uids:
        fasta_path = os.path.join(project_dir, "all_sequences.fasta")
        if os.path.exists(fasta_path):
            with open(fasta_path) as f:
                for line in f:
                    if line.startswith(">"):
                        uid = line[1:].strip().split()[0]
                        if uid:
                            uids.append(uid)

    return list(dict.fromkeys(uids))  # deduplicate preserving order


def survey_alphafill(uids, max_sample=10):
    """Query AlphaFill for a sample of candidates to discover ligands.

    Returns:
        ligand_counts: Counter of {ccd_code: n_candidates_with_it}
        per_candidate: dict of {uid: {ccd_code: n_contacts}}
    """
    ligand_counts = Counter()
    per_candidate = {}
    n_queried = 0
    n_with_data = 0

    sample = uids[:max_sample]
    print(f"  Surveying AlphaFill for {len(sample)} candidates...")

    for uid in sample:
        af_data = fetch_alphafill(uid)
        n_queried += 1

        if af_data and af_data.get("contacts"):
            n_with_data += 1
            contacts = af_data["contacts"]
            candidate_ligs = {}
            for ccd, residues in contacts.items():
                if ccd in CRYSTALLIZATION_ARTIFACTS:
                    continue
                candidate_ligs[ccd] = len(residues)
                ligand_counts[ccd] += 1
            per_candidate[uid] = candidate_ligs

        if n_queried % 5 == 0:
            print(f"    {n_queried}/{len(sample)} queried, "
                  f"{n_with_data} with data", flush=True)

    print(f"  AlphaFill survey: {n_with_data}/{n_queried} candidates "
          f"returned ligand data")
    return ligand_counts, per_candidate


def classify_ligands(ligand_counts):
    """Classify discovered ligands into cofactors, metals, and other.

    Uses COFACTOR_CCD_CODES and CATALYTIC_METAL_CCDS for classification.
    Does NOT hardcode specific cofactors — discovers from data.
    """
    cofactors = {}
    metals = {}
    other = {}

    for ccd, count in ligand_counts.items():
        if ccd in COFACTOR_CCD_CODES:
            cofactors[ccd] = count
        elif ccd in CATALYTIC_METAL_CCDS:
            metals[ccd] = count
        else:
            other[ccd] = count

    return cofactors, metals, other


def group_cofactors_by_physiological_form(cofactors):
    """Group cofactor variants by their physiological form.

    E.g., {ANP: 3, ADP: 5, ATP: 2} → {ATP: [ANP, ADP, ATP], count: 10}
    """
    groups = defaultdict(lambda: {"variants": [], "total_count": 0})

    for ccd, count in cofactors.items():
        physiological = COFACTOR_PHYSIOLOGICAL_MAP.get(ccd, ccd)
        groups[physiological]["variants"].append(ccd)
        groups[physiological]["total_count"] += count

    return dict(groups)


def build_cofactor_mappings(variants):
    """Build cofactor_mappings dict from variant list.

    Only includes non-identity mappings (e.g., ANP→ATP but not ATP→ATP).
    """
    mappings = {}
    for variant in variants:
        physiological = COFACTOR_PHYSIOLOGICAL_MAP.get(variant, variant)
        if variant != physiological:
            mappings[variant] = physiological
    return mappings


def build_metal_mappings(metal_ccds):
    """Build metal_mappings dict for observed metals."""
    mappings = {}
    for ccd in metal_ccds:
        physiological = METAL_PHYSIOLOGICAL_MAP.get(ccd, ccd)
        if ccd != physiological:
            mappings[ccd] = physiological
    return mappings


def parse_substrate_arg(substrate_str):
    """Parse 'name=SMILES' substrate argument."""
    if "=" in substrate_str:
        name, smiles = substrate_str.split("=", 1)
        return {"name": name.strip(), "smiles": smiles.strip()}
    else:
        # No name given, use a generic name
        return {"name": f"substrate", "smiles": substrate_str.strip()}


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate config.yaml by surveying AlphaFill")
    parser.add_argument("project_dir",
                        help="Project directory with candidate sequences")
    parser.add_argument("--substrates", nargs="+", required=True,
                        help="Substrate(s) as 'name=SMILES' (e.g., "
                             "'hexamethylenediamine=[NH3+]CCCCCC[NH3+]')")
    parser.add_argument("--reaction-name", default=None,
                        help="Reaction name (default: auto-generated)")
    parser.add_argument("--max-sample", type=int, default=10,
                        help="Max candidates to survey (default: 10)")
    parser.add_argument("--min-cofactor-freq", type=float, default=0.3,
                        help="Min fraction of candidates with cofactor to "
                             "include (default: 0.3)")
    parser.add_argument("--min-metal-freq", type=float, default=0.2,
                        help="Min fraction of candidates with metal to "
                             "include (default: 0.2)")
    parser.add_argument("-o", "--output", default="config.yaml",
                        help="Output config filename (default: config.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config but don't write file")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)

    # Load candidates
    uids = load_candidate_uids(project_dir)
    if not uids:
        sys.exit("ERROR: No candidate UniProt IDs found. Need "
                 "structure_sequence_summary.csv or all_sequences.fasta.")
    print(f"Found {len(uids)} candidate UniProt IDs")

    # Parse substrates
    substrates = [parse_substrate_arg(s) for s in args.substrates]
    for i, s in enumerate(substrates):
        if s["name"] == "substrate":
            s["name"] = f"substrate_{i+1}"
    print(f"Substrates: {[s['name'] for s in substrates]}")

    # Survey AlphaFill
    ligand_counts, per_candidate = survey_alphafill(uids, args.max_sample)
    n_surveyed = min(len(uids), args.max_sample)

    if not ligand_counts:
        print("\nWARNING: No ligands found in AlphaFill. Generating "
              "cofactor-free config (blind cofolding mode).")
        relevant_ligands = []
        catalytic_metals = []
        cofactor_mappings = {}
        metal_mappings = {}
    else:
        # Classify
        cofactors, metals, other = classify_ligands(ligand_counts)

        print(f"\nDiscovered ligands ({len(ligand_counts)} total):")
        if cofactors:
            print(f"  Cofactors: {dict(cofactors)}")
        if metals:
            print(f"  Metals: {dict(metals)}")
        if other:
            print(f"  Other: {dict(other)}")

        # Group cofactors by physiological form
        groups = group_cofactors_by_physiological_form(cofactors)

        # Select dominant cofactor group
        relevant_ligands = []
        cofactor_mappings = {}
        if groups:
            dominant = max(groups.items(), key=lambda x: x[1]["total_count"])
            dominant_form = dominant[0]
            dominant_info = dominant[1]
            all_variants = sorted(set(dominant_info["variants"]) | {dominant_form})

            # Frequency = fraction of surveyed candidates that have ANY variant
            n_with_cofactor = len(set(
                uid for uid, ligs in per_candidate.items()
                if any(l in all_variants for l in ligs)))
            freq = n_with_cofactor / n_surveyed if n_surveyed else 0
            if freq >= args.min_cofactor_freq:
                relevant_ligands = all_variants
                cofactor_mappings = build_cofactor_mappings(all_variants)
                print(f"\nDominant cofactor: {dominant_form} "
                      f"(variants: {all_variants}, "
                      f"freq: {freq:.0%})")
            else:
                print(f"\nNo cofactor exceeds frequency threshold "
                      f"({args.min_cofactor_freq:.0%}). "
                      f"Best: {dominant_form} at {freq:.0%}.")

            # Report other cofactor groups
            for form, info in sorted(groups.items()):
                if form != dominant_form:
                    f = info["total_count"] / n_surveyed
                    print(f"  Also found: {form} "
                          f"(variants: {info['variants']}, freq: {f:.0%})")

        # Select metals that co-occur with the cofactor.
        # Include both the primary metal AND known crystallographic
        # substitutes (e.g., if MG is found, also include MN since
        # MN commonly substitutes for MG in crystal structures).
        catalytic_metals = []
        metal_mappings = {}
        if metals:
            primary_metals = set()
            for ccd, count in sorted(metals.items(),
                                     key=lambda x: -x[1]):
                freq = count / n_surveyed
                if freq >= args.min_metal_freq:
                    physiological = METAL_PHYSIOLOGICAL_MAP.get(ccd, ccd)
                    primary_metals.add(physiological)
                    catalytic_metals.append(ccd)
                    print(f"  Metal: {ccd} (freq: {freq:.0%}, "
                          f"physiological: {physiological})")

            # Include known substitutes of primary metals
            for src, dst in METAL_PHYSIOLOGICAL_MAP.items():
                if dst in primary_metals and src not in catalytic_metals:
                    catalytic_metals.append(src)
                    print(f"  Metal: {src} (included as known "
                          f"substitute for {dst})")

            catalytic_metals = sorted(set(catalytic_metals))
            metal_mappings = build_metal_mappings(catalytic_metals)

    # Build config
    config = {
        "target_reaction": {
            "name": args.reaction_name or "auto-generated",
            "relevant_ligands": relevant_ligands,
            "catalytic_metals": catalytic_metals,
            "substrates": substrates,
        },
        "candidates": {
            "summary_csv": "structure_sequence_summary.csv",
            "structures_dirs": ["alphafold_structures", "pdb_structures"],
        },
    }

    # Add mappings only if non-empty
    if cofactor_mappings:
        config["target_reaction"]["cofactor_mappings"] = cofactor_mappings
    if metal_mappings:
        config["target_reaction"]["metal_mappings"] = metal_mappings

    # Output
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
    print(f"\n{'='*60}")
    print("Generated config.yaml:")
    print(f"{'='*60}")
    print(config_yaml)

    if not args.dry_run:
        out_path = os.path.join(project_dir, args.output)
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Written to {out_path}")
    else:
        print("(dry run — not written)")


if __name__ == "__main__":
    main()
