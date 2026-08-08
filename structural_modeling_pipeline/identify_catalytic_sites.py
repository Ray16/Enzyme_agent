#!/usr/bin/env python3
"""
Identify catalytic residues for all enzyme candidates.

Two-phase approach:
  Phase A: InterPro/UniProt sequence-based identification (fast, high confidence)
  Phase B: FoldSeek structural homolog transfer with LLM validation (fills gaps)

This script runs both phases sequentially. Phase A identifies catalytic
residues from UniProt "Active site" annotations and InterPro homologs.
Phase B fills remaining NO-CAT-DATA entries using FoldSeek structural
homologs (TM >= 0.7), with LLM-based validation to filter biochemically
implausible residue transfers.

Usage:
    python identify_catalytic_sites.py amide_synthetase amide_hydrolase
    python identify_catalytic_sites.py amide_synthetase --skip-enhance
    python identify_catalytic_sites.py amide_hydrolase --skip-llm --workers 8
"""

import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Identify catalytic residues (InterPro/UniProt + FoldSeek/LLM)")
    parser.add_argument("projects", nargs="+",
                        help="Project directories to process")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Parallel workers for Phase A (default: cpu_count)")
    parser.add_argument("--min-tmscore", type=float, default=0.7,
                        help="Minimum TM-score for FoldSeek homologs (default: 0.7)")
    parser.add_argument("--skip-enhance", action="store_true",
                        help="Skip Phase B (FoldSeek enhancement)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM validation in Phase B (keep all mapped residues)")
    args = parser.parse_args()

    # Phase A: InterPro/UniProt validation
    print("=" * 60)
    print("Phase A: InterPro/UniProt catalytic site identification")
    print("=" * 60)
    cmd_a = [
        sys.executable, os.path.join(BASE_DIR, "validate_catalytic_sites.py"),
        *args.projects, "--workers", str(args.workers),
    ]
    result = subprocess.run(cmd_a)
    if result.returncode != 0:
        print("Phase A failed.", file=sys.stderr)
        sys.exit(1)

    # Phase B: FoldSeek enhancement
    if not args.skip_enhance:
        print("\n" + "=" * 60)
        print("Phase B: FoldSeek structural homolog enhancement")
        print("=" * 60)

        # Set environment to control LLM behavior
        env = os.environ.copy()
        if args.skip_llm:
            env["SKIP_LLM_VALIDATION"] = "1"

        cmd_b = [
            sys.executable, os.path.join(BASE_DIR, "enhance_catalytic_sites.py"),
            *args.projects,
        ]
        result = subprocess.run(cmd_b, env=env)
        if result.returncode != 0:
            print("Phase B failed.", file=sys.stderr)
            sys.exit(1)

    print("\nDone. Output: catalytic_site_validation.json in each project directory.")


if __name__ == "__main__":
    main()
