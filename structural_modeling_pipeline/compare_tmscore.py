#!/usr/bin/env python3
"""Compare AlphaFold vs Boltz-2 apo structures using TM-score.

For each hydrolase with both an AlphaFold and Boltz-2 apo structure,
runs TMalign and prints the TM-score, RMSD, and aligned length.
"""

import glob
import json
import os
import re
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMALIGN = "/home/rzhu/miniforge3/envs/palm/bin/TMalign"


def find_alphafold_pdb(uid, pocket_analysis):
    entry = pocket_analysis.get(uid)
    if not entry or not isinstance(entry, dict):
        return None
    path = entry.get("path")
    if path and os.path.exists(path):
        return path
    return None


def find_boltz_pdb(uid):
    pattern = os.path.join(
        BASE_DIR, "amide_hydrolase", "boltz_output", uid, "**", "*_model_0.pdb"
    )
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


def run_tmalign(pdb1, pdb2):
    try:
        r = subprocess.run(
            [TMALIGN, pdb1, pdb2],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    result = {}
    for line in r.stdout.split("\n"):
        if line.startswith("Aligned length="):
            m = re.search(r"Aligned length=\s*(\d+).*RMSD=\s*([\d.]+)", line)
            if m:
                result["aligned_length"] = int(m.group(1))
                result["rmsd"] = float(m.group(2))
        if "TM-score=" in line and "Chain_1" in line:
            m = re.search(r"TM-score=\s*([\d.]+)", line)
            if m:
                result["tmscore_1"] = float(m.group(1))
        if "TM-score=" in line and "Chain_2" in line:
            m = re.search(r"TM-score=\s*([\d.]+)", line)
            if m:
                result["tmscore_2"] = float(m.group(1))
    if "Warning" in r.stdout or "Cannot parse" in r.stdout:
        return None
    return result if result else None


def main():
    with open(os.path.join(BASE_DIR, "amide_hydrolase", "pocket_analysis.json")) as f:
        pocket_analysis = json.load(f)

    candidates = []
    for uid, entry in pocket_analysis.items():
        if not isinstance(entry, dict) or entry.get("role") == "reference":
            continue
        af_pdb = find_alphafold_pdb(uid, pocket_analysis)
        b2_pdb = find_boltz_pdb(uid)
        if af_pdb and b2_pdb:
            candidates.append((uid, af_pdb, b2_pdb))

    candidates.sort(key=lambda x: x[0])

    header = f"{'UID':<16} {'TM-score(AF)':<14} {'TM-score(B2)':<14} {'RMSD':>8} {'Aligned':>8}"
    print(header)
    print("-" * len(header))

    tmscores = []
    for uid, af_pdb, b2_pdb in candidates:
        result = run_tmalign(af_pdb, b2_pdb)
        if result:
            tm1 = result.get("tmscore_1", 0)
            tm2 = result.get("tmscore_2", 0)
            rmsd = result.get("rmsd", 0)
            aligned = result.get("aligned_length", 0)
            tmscores.append(tm1)
            print(f"{uid:<16} {tm1:<14.4f} {tm2:<14.4f} {rmsd:>8.2f} {aligned:>8d}")
        else:
            print(f"{uid:<16} {'FAILED':<14} {'':14} {'':>8} {'':>8}")

    if tmscores:
        print("-" * len(header))
        avg = sum(tmscores) / len(tmscores)
        mn = min(tmscores)
        mx = max(tmscores)
        print(f"{'Summary':<16} avg={avg:.4f}       min={mn:.4f}       max={mx:.4f}")
        print(f"Total pairs: {len(tmscores)}")


if __name__ == "__main__":
    main()
