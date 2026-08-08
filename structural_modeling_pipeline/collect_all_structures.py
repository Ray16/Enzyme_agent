#!/usr/bin/env python3
"""
Collect AlphaFold apo structures and best cofolded structures for all
amide synthetase and hydrolase candidates into a unified output directory.

Output structure:
  all_structures/
    amide_synthetases_candidates/
      alphafold_apo_structure/    {enzyme}_{uid}.pdb
      cofolded_structure/         {enzyme}_{uid}.pdb
    amide_hydrolase_candidates/
      alphafold_apo_structure/    {enzyme}_{uid}.pdb
      cofolded_structure/         {enzyme}_{uid}.pdb
"""

import csv
import os
import re
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ── source directories ──────────────────────────────────────────────
SYNTHETASE_DIR = BASE / "amide_synthetase"
HYDROLASE_DIR  = BASE / "amide_hydrolase"

# ── output directories ──────────────────────────────────────────────
OUT = BASE / "all_structures"
SYN_APO   = OUT / "amide_synthetases_candidates" / "alphafold_apo_structure"
SYN_COFOLD = OUT / "amide_synthetases_candidates" / "cofolded_structure"
HYD_APO   = OUT / "amide_hydrolase_candidates"   / "alphafold_apo_structure"
HYD_COFOLD = OUT / "amide_hydrolase_candidates"   / "cofolded_structure"

for d in (SYN_APO, SYN_COFOLD, HYD_APO, HYD_COFOLD):
    d.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    """Make a filename-safe enzyme name."""
    # Remove 'status=active' tags and UPI prefixes that add noise
    name = re.sub(r'\s*status=\w+', '', name).strip()
    # Replace spaces / special chars with underscores
    name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name).strip('_')
    return name if name else "unknown"


# ── 1. Build synthetase enzyme-name map from structure_sequence_summary.csv ──
def load_synthetase_names() -> dict:
    """Returns {uniprot_id: gene_name}"""
    mapping = {}
    csv_path = SYNTHETASE_DIR / "structure_sequence_summary.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"].strip()
            gene = row["gene_name"].strip()
            if uid and gene:
                mapping[uid] = gene
    return mapping


# ── 2. Build hydrolase enzyme-name map from cofolding_metrics.csv ──
def load_hydrolase_names() -> dict:
    """Returns {uniprot_id: enzyme_name}"""
    mapping = {}
    csv_path = BASE / "analysis_results" / "cofolding_metrics.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["project"].strip() != "hydrolase":
                continue
            uid = row["uid"].strip()
            name = sanitize(row["enzyme_name"].strip())
            if uid and name:
                mapping[uid] = name
    return mapping


# ── 3. Also add hydrolase names from structure_sequence_summary.csv  ──
#       (catches any enzymes that may not be in cofolding_metrics)
def load_hydrolase_ids() -> list:
    """Returns list of all hydrolase UniProt IDs."""
    ids = []
    csv_path = HYDROLASE_DIR / "structure_sequence_summary.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"].strip()
            if uid:
                ids.append(uid)
    return ids


def find_alphafold_pdb(struct_dir: Path, uid: str) -> Path | None:
    """Locate the AlphaFold apo PDB for a given UniProt ID."""
    pdb = struct_dir / f"AF-{uid}-F1-model_v4.pdb"
    return pdb if pdb.exists() else None


def find_best_cofold(cofold_dir: Path, uid: str) -> Path | None:
    """Locate the best cofolded PDB for a given UniProt ID."""
    # Standard path: cofold_output/{uid}/boltz_results_{uid}/predictions/{uid}/{uid}_best_model.pdb
    best = cofold_dir / uid / f"boltz_results_{uid}" / "predictions" / uid / f"{uid}_best_model.pdb"
    if best.exists():
        return best
    # Fallback: search for any best_model.pdb
    search_dir = cofold_dir / uid
    if search_dir.is_dir():
        for p in search_dir.rglob("*best_model.pdb"):
            return p
    return None


def collect(project_label: str,
            uid_to_name: dict,
            all_uids: list,
            af_struct_dir: Path,
            cofold_dir: Path,
            out_apo: Path,
            out_cofold: Path):
    """Copy structures for one project (synthetase or hydrolase)."""
    copied_apo = 0
    copied_cofold = 0
    missing_apo = []
    missing_cofold = []

    for uid in all_uids:
        enzyme = uid_to_name.get(uid, uid)  # fall back to uid if no name
        safe_name = f"{enzyme}_{uid}"

        # AlphaFold apo
        apo_src = find_alphafold_pdb(af_struct_dir, uid)
        if apo_src:
            shutil.copy2(apo_src, out_apo / f"{safe_name}.pdb")
            copied_apo += 1
        else:
            missing_apo.append(uid)

        # Best cofolded structure
        cofold_src = find_best_cofold(cofold_dir, uid)
        if cofold_src:
            shutil.copy2(cofold_src, out_cofold / f"{safe_name}.pdb")
            copied_cofold += 1
        else:
            missing_cofold.append(uid)

    print(f"\n{'='*60}")
    print(f"  {project_label}")
    print(f"{'='*60}")
    print(f"  Total candidates:       {len(all_uids)}")
    print(f"  AlphaFold apo copied:   {copied_apo}")
    print(f"  Cofolded best copied:   {copied_cofold}")
    if missing_apo:
        print(f"  Missing AlphaFold apo ({len(missing_apo)}):")
        for m in missing_apo:
            print(f"    - {m} ({uid_to_name.get(m, '?')})")
    if missing_cofold:
        print(f"  Missing cofolded best ({len(missing_cofold)}):")
        for m in missing_cofold:
            print(f"    - {m} ({uid_to_name.get(m, '?')})")


def main():
    # ── Synthetases ──────────────────────────────────────────────────
    syn_names = load_synthetase_names()
    syn_uids = list(syn_names.keys())
    collect(
        project_label="Amide Synthetases",
        uid_to_name=syn_names,
        all_uids=syn_uids,
        af_struct_dir=SYNTHETASE_DIR / "alphafold_structures",
        cofold_dir=SYNTHETASE_DIR / "cofold_output",
        out_apo=SYN_APO,
        out_cofold=SYN_COFOLD,
    )

    # ── Hydrolases ───────────────────────────────────────────────────
    hyd_names = load_hydrolase_names()
    hyd_uids = load_hydrolase_ids()
    # Make sure every uid has a name entry (fallback to uid itself)
    for uid in hyd_uids:
        if uid not in hyd_names:
            hyd_names[uid] = uid
    collect(
        project_label="Amide Hydrolases",
        uid_to_name=hyd_names,
        all_uids=hyd_uids,
        af_struct_dir=HYDROLASE_DIR / "alphafold_structures",
        cofold_dir=HYDROLASE_DIR / "cofold_output",
        out_apo=HYD_APO,
        out_cofold=HYD_COFOLD,
    )

    print(f"\n{'='*60}")
    print(f"  Output: {OUT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
