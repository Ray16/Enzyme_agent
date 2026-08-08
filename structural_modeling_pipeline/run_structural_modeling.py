#!/usr/bin/env python3
"""
Structural Modeling Agent — Single Entry Point

Runs the complete structural modeling pipeline from UniProt IDs + substrate
SMILES to scored cofolding results.

Usage:
    # New project:
    python run_structural_modeling.py \
        --project my_enzyme_family \
        --uniprots P31663 P0A6W9 P39641 \
        --substrates "substrate_a=[NH3+]CCCCCC[NH3+]" "substrate_b=[O-]C(=O)CCCCC(=O)[O-]" \
        --reaction-name "Amide bond synthesis"

    # Resume interrupted run:
    python run_structural_modeling.py --project my_enzyme_family --resume

    # From file:
    python run_structural_modeling.py \
        --project my_enzyme_family \
        --uniprots-file candidates.txt \
        --substrates "substrate_a=SMILES1"
"""

import argparse
import os
import subprocess
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, description, check=True):
    """Run a subprocess command with logging."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"\nERROR: {description} failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    return result


def checkpoint_exists(project_dir, filename):
    """Check if a checkpoint file/directory exists and is non-empty."""
    path = os.path.join(project_dir, filename)
    if os.path.isdir(path):
        return len(os.listdir(path)) > 0
    if os.path.isfile(path):
        return os.path.getsize(path) > 0
    return False


def ensure_pipeline_config(project_dir, project_name):
    """Ensure pipeline_config.yaml lists this project."""
    config_path = os.path.join(BASE_DIR, "pipeline_config.yaml")

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    projects = config.get("projects", [])
    rel_path = os.path.relpath(project_dir, BASE_DIR)

    # Check if already listed
    for p in projects:
        if p.get("path") == rel_path or os.path.abspath(
                os.path.join(BASE_DIR, p.get("path", ""))) == project_dir:
            return  # already listed

    projects.append({"path": rel_path, "name": project_name})
    config["projects"] = projects

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Updated pipeline_config.yaml with project '{project_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Structural Modeling Agent — Single Entry Point")
    parser.add_argument("--project", required=True,
                        help="Project directory name (created under current dir)")
    parser.add_argument("--uniprots", nargs="+",
                        help="UniProt IDs to analyze")
    parser.add_argument("--uniprots-file",
                        help="File with one UniProt ID per line")
    parser.add_argument("--substrates", nargs="+",
                        help="Substrates as 'name=SMILES' pairs")
    parser.add_argument("--reaction-name", default=None,
                        help="Reaction name for config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed phase")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip structure fetching (use existing files)")
    parser.add_argument("--skip-cofolding", action="store_true",
                        help="Skip Boltz-2 cofolding (use existing results)")
    parser.add_argument("--skip-catalytic-enhance", action="store_true",
                        help="Skip Phase 3B (FoldSeek catalytic-residue transfer)")
    parser.add_argument("--skip-catalytic-llm", action="store_true",
                        help="Skip LLM validation of Phase 3B residue transfers")
    parser.add_argument("--skip-md", action="store_true",
                        help="Skip MD simulation phases (Phase 6 runs by default)")
    parser.add_argument("--md-candidates", nargs="+", default=None,
                        help="Run MD only for specific UIDs (default: all)")
    parser.add_argument("--md-time-ns", type=float, default=1.0,
                        help="MD production time in nanoseconds (default: 1)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Download/processing threads (default: 8)")
    args = parser.parse_args()

    # Resolve project directory
    project_dir = os.path.join(BASE_DIR, args.project)
    if os.path.isabs(args.project):
        project_dir = args.project
    project_name = os.path.basename(project_dir)

    print(f"Structural Modeling Pipeline")
    print(f"Project: {project_name}")
    print(f"Directory: {project_dir}")

    # Validate inputs
    if not args.resume:
        if not args.substrates:
            sys.exit("ERROR: --substrates required (e.g., "
                     "'substrate_a=[NH3+]CCCCCC[NH3+]')")
        if not args.uniprots and not args.uniprots_file:
            if not checkpoint_exists(project_dir,
                                     "structure_sequence_summary.csv"):
                sys.exit("ERROR: Provide --uniprots, --uniprots-file, "
                         "or use --resume with existing project")

    # Load UniProt IDs
    uniprots = []
    if args.uniprots:
        uniprots = list(dict.fromkeys(args.uniprots))
    elif args.uniprots_file:
        with open(args.uniprots_file) as f:
            uniprots = list(dict.fromkeys(
                line.strip() for line in f if line.strip()))

    if uniprots:
        print(f"Candidates: {len(uniprots)} UniProt IDs")
    if args.substrates:
        print(f"Substrates: {args.substrates}")

    # ── Phase 0: Fetch structures ──
    if not args.skip_fetch and uniprots:
        if args.resume and checkpoint_exists(project_dir,
                                             "structure_sequence_summary.csv"):
            print("\n  Phase 0 (fetch): SKIPPED (checkpoint exists)")
        else:
            run_cmd([
                sys.executable,
                os.path.join(BASE_DIR, "fetch_structures.py"),
                project_dir,
                "--uniprots", *uniprots,
                "--workers", str(args.workers),
            ], "Phase 0: Fetching structures and sequences")

    # ── Phase 1: Generate config ──
    if args.substrates:
        if args.resume and checkpoint_exists(project_dir, "config.yaml"):
            print("\n  Phase 1 (config): SKIPPED (checkpoint exists)")
        else:
            cmd = [
                sys.executable,
                os.path.join(BASE_DIR, "generate_config.py"),
                project_dir,
                "--substrates", *args.substrates,
            ]
            if args.reaction_name:
                cmd.extend(["--reaction-name", args.reaction_name])
            run_cmd(cmd, "Phase 1: Auto-generating config.yaml")

    # Verify config exists
    if not checkpoint_exists(project_dir, "config.yaml"):
        sys.exit(f"ERROR: {project_dir}/config.yaml not found. "
                 f"Provide --substrates or create manually.")

    # ── Phase 2: Pocket analysis ──
    if args.resume and checkpoint_exists(project_dir, "pocket_analysis.json"):
        print("\n  Phase 2 (pocket analysis): SKIPPED (checkpoint exists)")
    else:
        run_cmd([
            sys.executable,
            os.path.join(BASE_DIR, "pocket_analysis.py"),
            project_dir,
        ], "Phase 2: Detecting binding pockets")

    # ── Phase 3: Catalytic site identification ──
    # Runs both phases: A (InterPro/UniProt annotations, ~20% coverage) and
    # B (FoldSeek structural transfer with LLM plausibility validation).
    # Phase B is what lifts catalytic-residue coverage to the ~70% range, so
    # calling validate_catalytic_sites.py directly here would silently produce
    # a far less annotated run than the documented manual pipeline.
    if args.resume and checkpoint_exists(project_dir,
                                         "catalytic_site_validation.json"):
        print("\n  Phase 3 (catalytic sites): SKIPPED (checkpoint exists)")
    else:
        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "identify_catalytic_sites.py"),
            project_dir,
            "--workers", str(args.workers),
        ]
        if args.skip_catalytic_enhance:
            cmd.append("--skip-enhance")
        if args.skip_catalytic_llm:
            cmd.append("--skip-llm")
        run_cmd(cmd, "Phase 3: Mapping catalytic residues (InterPro/UniProt + FoldSeek)")

    # ── Phase 4: Cofolding ──
    if args.skip_cofolding:
        print("\n  Phase 4 (cofolding): SKIPPED (--skip-cofolding)")
    elif args.resume and checkpoint_exists(project_dir, "cofold_output"):
        print("\n  Phase 4 (cofolding): SKIPPED (checkpoint exists)")
    else:
        run_cmd([
            sys.executable,
            os.path.join(BASE_DIR, "run_cofold.py"),
            project_dir,
        ], "Phase 4: Running Boltz-2 cofolding")

    # ── Phase 5: Analysis ──
    ensure_pipeline_config(project_dir, project_name)

    analysis_dir = os.path.join(BASE_DIR, "analysis_results")
    metrics_csv = os.path.join(analysis_dir, "cofolding_metrics.csv")

    if args.resume and os.path.exists(metrics_csv):
        print("\n  Phase 5 (analysis): SKIPPED (checkpoint exists)")
    else:
        run_cmd([
            sys.executable,
            os.path.join(BASE_DIR, "analyze_cofolding.py"),
        ], "Phase 5a: Scoring cofolding results")

        run_cmd([
            sys.executable,
            os.path.join(BASE_DIR, "check_clashes.py"),
        ], "Phase 5b: Checking steric clashes", check=False)

        run_cmd([
            sys.executable,
            os.path.join(BASE_DIR, "pocket_volume_analysis.py"),
        ], "Phase 5c: Analyzing pocket geometry", check=False)

    # ── Phase 6: MD Simulations ──
    if not args.skip_md:
        # Phase 6a: Prepare MD systems
        if args.resume and checkpoint_exists(project_dir, "md_prep"):
            print("\n  Phase 6a (MD prep): SKIPPED (checkpoint exists)")
        else:
            cmd = [
                sys.executable,
                os.path.join(BASE_DIR, "prepare_md_systems.py"),
                project_dir,
            ]
            if args.md_candidates:
                cmd.extend(["--candidates", *args.md_candidates])
            run_cmd(cmd, "Phase 6a: Preparing MD systems (AmberTools)")

        # Phase 6b: Run MD simulations
        if args.resume and checkpoint_exists(project_dir, "md_output"):
            print("\n  Phase 6b (MD run): SKIPPED (checkpoint exists)")
        else:
            cmd = [
                sys.executable,
                os.path.join(BASE_DIR, "run_md.py"),
                project_dir,
                "--sim-time-ns", str(args.md_time_ns),
            ]
            if args.md_candidates:
                cmd.extend(["--candidates", *args.md_candidates])
            run_cmd(cmd, "Phase 6b: Running MD simulations (OpenMM)")

        # Phase 6c: Analyze MD trajectories
        md_metrics = os.path.join(project_dir, "md_output",
                                  "md_stability_metrics.csv")
        if args.resume and os.path.exists(md_metrics):
            print("\n  Phase 6c (MD analysis): SKIPPED (checkpoint exists)")
        else:
            cmd = [
                sys.executable,
                os.path.join(BASE_DIR, "analyze_md.py"),
                project_dir,
            ]
            if args.md_candidates:
                cmd.extend(["--candidates", *args.md_candidates])
            run_cmd(cmd, "Phase 6c: Analyzing MD trajectories")

    # ── Done ──
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults:")
    if os.path.exists(metrics_csv):
        print(f"  Metrics:  {metrics_csv}")
    summary_txt = os.path.join(analysis_dir, "cofolding_summary.txt")
    if os.path.exists(summary_txt):
        print(f"  Summary:  {summary_txt}")
    print(f"  Project:  {project_dir}")
    print()


if __name__ == "__main__":
    main()
