"""Regenerate the confidence report for an existing run using the new mechanistic ranking pipeline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from enzyme_discovery_agent import config
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.output.finalizer import RunFinalizer
from enzyme_discovery_agent.output.report_generator import (
    generate_two_pass_explorative,
    generate_two_pass_report,
)
from enzyme_discovery_agent.output.exploration_summary import write_exploration_summary
from enzyme_discovery_agent.output.csv_generator import generate_candidates_csv
from enzyme_discovery_agent.output.structure_downloader import download_candidate_structures
from enzyme_discovery_agent.output.structural_modeling_handoff import write_structural_modeling_handoff
from enzyme_discovery_agent.output.report_generator import is_error_text
from enzyme_discovery_agent.gates.ec_triage import _auto_evaluate_all, auto_screen_candidates


def main():
    parser = argparse.ArgumentParser(description="Resume report delivery for an existing run")
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top candidates for full profiles (default: 30)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--resume", action="store_true",
                        help="Generate only report/deliverables that are missing")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        sys.exit(1)

    all_evals_path = run_dir / "artifacts" / "candidates" / "all_evaluations.json"
    if not all_evals_path.exists():
        # Older runs may have screened candidates in their final iteration,
        # before the automatic evaluation transition was available. Recover
        # them deterministically rather than discarding the evidence-gathering
        # work or generating an empty report.
        print("No evaluation bundle found; recovering screened candidates...")
        auto_screen_candidates(run_dir, log_fn=print)
        _auto_evaluate_all(run_dir, log_fn=print)
        recovery_finalizer = RunFinalizer(run_dir, llm=None, log=print)
        recovered = recovery_finalizer._load_evaluations()
        if not recovered:
            print(f"Error: no evaluations could be recovered from {run_dir}")
            sys.exit(1)
        all_evals_path.parent.mkdir(parents=True, exist_ok=True)
        all_evals_path.write_text(json.dumps(recovered, indent=2, default=str))
        print(f"Recovered {len(recovered)} evaluations")

    with open(all_evals_path) as f:
        all_evals = json.load(f)
    print(f"Loaded {len(all_evals)} evaluations from {all_evals_path}")

    # Prefer the exact persisted query over the lossy directory-name suffix.
    query = run_dir.name.split("_", 3)[-1].replace("_", " ")
    events_path = run_dir / "events.jsonl"
    try:
        for line in events_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "run_start":
                query = event.get("query", query)
                break
    except (OSError, json.JSONDecodeError):
        pass
    print(f"Query: {query}")

    llm_client = LLMClient(model=args.model)
    token_usage = TokenUsage()

    def log(msg, **_):
        print(msg)

    finalizer = RunFinalizer(run_dir, llm_client, log)
    confidence_path = run_dir / "report_confidence.md"
    confidence_missing_or_empty = (
        not confidence_path.exists()
        or is_error_text(confidence_path.read_text())
    )
    if not args.resume or confidence_missing_or_empty:
        finalizer._run_finalization_stage(
            "confidence_report",
            lambda: generate_two_pass_report(
                run_dir, all_evals, query, llm_client, token_usage, log, top_n=args.top_n,
            ),
        )
    if confidence_path.exists():
        finalizer._run_finalization_stage(
            "citation_verification",
            lambda: finalizer._verify_confidence_report(confidence_path),
        )
        finalizer._run_finalization_stage(
            "structure_downloads",
            lambda: download_candidate_structures(run_dir, confidence_path.read_text(), run_dir / "artifacts", log),
        )
    explorative_path = run_dir / "report_explorative.md"
    explorative_missing_or_empty = (
        not explorative_path.exists()
        or is_error_text(explorative_path.read_text())
    )
    if not args.resume or explorative_missing_or_empty:
        finalizer._run_finalization_stage(
            "explorative_report",
            lambda: generate_two_pass_explorative(
                run_dir, all_evals, query, llm_client, token_usage, log, top_n=args.top_n,
            ),
        )
    finalizer._run_finalization_stage(
        "exploration_summary", lambda: write_exploration_summary(run_dir, log_fn=log),
    )
    finalizer._run_finalization_stage(
        "candidates_csv", lambda: generate_candidates_csv(run_dir, all_evals, log_fn=log),
    )
    finalizer._run_finalization_stage(
        "structural_modeling_handoff",
        lambda: write_structural_modeling_handoff(run_dir, all_evals, target={"query": query}),
    )

    usage = token_usage.to_dict()
    print(f"\nToken usage: {usage.get('total_tokens', 0):,} total tokens")
    print(f"Report written to: {run_dir / 'report_confidence.md'}")
    print(f"Ranking written to: {run_dir / 'pass1_ranking.json'}")


if __name__ == "__main__":
    main()
