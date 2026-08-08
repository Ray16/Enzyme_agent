#!/usr/bin/env python3
"""
Pre-cofolding validation of Boltz-2 input YAML files.

Checks for structural and chemical sanity before submitting to Boltz-2,
catching constraint errors (e.g., Mg2+ anchored far from ATP) that would
waste GPU time and produce unphysical structures.

Usage:
    python validate_cofold_inputs.py <project_dir>
    python validate_cofold_inputs.py <project_dir> --strict
"""

import argparse
import glob
import os
import sys

import yaml

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
from prepare_cofold_inputs import COFACTOR_CCD_CODES, METAL_PHYSIOLOGICAL_MAP


# CCD codes for catalytic metals
METAL_CCDS = {
    "MG", "MN", "ZN", "FE", "CA", "CO", "NI", "CU",
    "FE2", "MO", "W",
}


def get_chain_info(boltz_yaml):
    """Extract chain IDs, types, and constraints from a Boltz YAML."""
    chains = {}
    for entry in boltz_yaml.get("sequences", []):
        if "protein" in entry:
            cid = entry["protein"]["id"]
            seq_len = len(entry["protein"].get("sequence", ""))
            chains[cid] = {"type": "protein", "seq_len": seq_len}
        elif "ligand" in entry:
            cid = entry["ligand"]["id"]
            ccd = entry["ligand"].get("ccd", "")
            smiles = entry["ligand"].get("smiles", "")
            if ccd:
                if ccd in COFACTOR_CCD_CODES:
                    ltype = "cofactor"
                elif ccd in METAL_CCDS:
                    ltype = "metal"
                else:
                    ltype = "ligand_ccd"
            else:
                ltype = "substrate"
            chains[cid] = {"type": ltype, "ccd": ccd, "smiles": smiles}

    constraints = {}
    for c in boltz_yaml.get("constraints", []):
        pocket = c.get("pocket", {})
        binder = pocket.get("binder")
        if binder:
            anchors = sorted(ct[1] for ct in pocket.get("contacts", []))
            constraints[binder] = anchors

    return chains, constraints


def validate_yaml(uid, boltz_yaml, project_dir=None):
    """Validate a single Boltz-2 cofolding YAML.

    Returns list of (severity, message) tuples.
    severity: "ERROR" (critical, should block), "WARNING" (non-critical)
    """
    issues = []
    chains, constraints = get_chain_info(boltz_yaml)

    # --- Check 1: Chain count ---
    n_chains = len(chains)
    if n_chains > 6:
        issues.append(("WARNING",
                        f"High chain count ({n_chains}). Boltz-2 may "
                        f"produce lower quality with many ligands."))

    # --- Check 2: Protein sequence length ---
    protein_chains = {k: v for k, v in chains.items() if v["type"] == "protein"}
    for cid, info in protein_chains.items():
        if info["seq_len"] > 1000:
            issues.append(("WARNING",
                            f"Large protein ({info['seq_len']} aa). "
                            f"May require >16 GB GPU memory."))

    # --- Check 3: Constraint residues within sequence bounds ---
    for cid, info in protein_chains.items():
        seq_len = info["seq_len"]
        for binder_id, anchors in constraints.items():
            for resnum in anchors:
                if resnum < 1 or resnum > seq_len:
                    issues.append(("ERROR",
                                    f"Constraint for chain {binder_id}: "
                                    f"residue {resnum} out of bounds "
                                    f"(protein has {seq_len} residues)."))

    # --- Check 4: Metal-cofactor overlap ---
    cofactor_chains = {k: v for k, v in chains.items()
                       if v["type"] == "cofactor"}
    metal_chains = {k: v for k, v in chains.items()
                    if v["type"] == "metal"}

    if cofactor_chains and metal_chains:
        for cof_id, cof_info in cofactor_chains.items():
            cof_anchors = set(constraints.get(cof_id, []))
            if not cof_anchors:
                continue

            for met_id, met_info in metal_chains.items():
                met_anchors = set(constraints.get(met_id, []))
                if not met_anchors:
                    continue

                overlap = cof_anchors & met_anchors
                if not overlap:
                    issues.append(("ERROR",
                                    f"Metal {met_info['ccd']} (chain "
                                    f"{met_id}) anchor residues "
                                    f"{sorted(met_anchors)} have ZERO "
                                    f"overlap with cofactor "
                                    f"{cof_info['ccd']} (chain {cof_id}) "
                                    f"anchors {sorted(cof_anchors)}. "
                                    f"Metal will be placed far from "
                                    f"cofactor — unphysical."))

    # --- Check 5: Substrate SMILES validity ---
    if HAS_RDKIT:
        for cid, info in chains.items():
            if info["type"] == "substrate" and info.get("smiles"):
                mol = Chem.MolFromSmiles(info["smiles"])
                if mol is None:
                    issues.append(("ERROR",
                                    f"Invalid SMILES for chain {cid}: "
                                    f"'{info['smiles']}'"))

    # --- Check 6: Cofactor without constraints ---
    for cof_id, cof_info in cofactor_chains.items():
        if cof_id not in constraints:
            issues.append(("WARNING",
                            f"Cofactor {cof_info['ccd']} (chain {cof_id}) "
                            f"has no pocket constraint. It will dock "
                            f"blindly — may not find the active site."))

    # --- Check 7: Metal without constraints ---
    for met_id, met_info in metal_chains.items():
        if met_id not in constraints:
            issues.append(("WARNING",
                            f"Metal {met_info['ccd']} (chain {met_id}) "
                            f"has no pocket constraint. It will dock "
                            f"blindly."))

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate Boltz-2 cofolding input YAML files")
    parser.add_argument("project_dir",
                        help="Project directory with cofold_input/")
    parser.add_argument("--input-dir", default="cofold_input",
                        help="Input YAML directory (default: cofold_input)")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors (non-zero exit)")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    input_dir = os.path.join(project_dir, args.input_dir)

    if not os.path.isdir(input_dir):
        sys.exit(f"ERROR: {input_dir} not found.")

    yaml_files = sorted(glob.glob(os.path.join(input_dir, "*.yaml")))
    if not yaml_files:
        sys.exit(f"ERROR: No YAML files found in {input_dir}")

    print(f"Validating {len(yaml_files)} cofolding inputs in {input_dir}")
    print(f"{'='*60}")

    total_errors = 0
    total_warnings = 0
    n_passed = 0

    for yaml_path in yaml_files:
        uid = os.path.splitext(os.path.basename(yaml_path))[0]

        with open(yaml_path) as f:
            boltz_yaml = yaml.safe_load(f)

        issues = validate_yaml(uid, boltz_yaml, project_dir)

        errors = [i for i in issues if i[0] == "ERROR"]
        warnings = [i for i in issues if i[0] == "WARNING"]
        total_errors += len(errors)
        total_warnings += len(warnings)

        if errors:
            print(f"  FAIL  {uid}")
            for severity, msg in issues:
                print(f"    {severity}: {msg}")
        elif warnings:
            print(f"  WARN  {uid}")
            for severity, msg in issues:
                print(f"    {severity}: {msg}")
        else:
            n_passed += 1

    print(f"\n{'='*60}")
    print(f"Results: {n_passed}/{len(yaml_files)} passed, "
          f"{total_errors} errors, {total_warnings} warnings")

    if total_errors > 0:
        print("\nCRITICAL: Fix errors before running Boltz-2.")
        sys.exit(1)
    elif args.strict and total_warnings > 0:
        print("\nStrict mode: warnings treated as errors.")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
