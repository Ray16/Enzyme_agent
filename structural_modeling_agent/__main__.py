"""Run the structural-modeling agent without a discovery orchestrator."""
import argparse
import json
from pathlib import Path


def environment_diagnostics():
    """Check import availability without initializing providers or running jobs."""
    from importlib.util import find_spec
    modules = {"openai": "openai", "numpy": "numpy", "pandas": "pandas",
               "requests": "requests", "pymupdf": "fitz", "rdkit": "rdkit",
               "biopython": "Bio", "scipy": "scipy", "gemmi": "gemmi",
               "matplotlib": "matplotlib"}
    available = {name: find_spec(module) is not None for name, module in modules.items()}
    return {"available": available, "missing": [name for name, found in available.items() if not found],
            "note": "Checks imports only; provider credentials, reservations and scientific inputs are not verified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", type=Path, help="standalone task or schema-3.0 discovery handoff (2.0 leads are unverified)")
    source.add_argument("--resume", type=Path, help="existing modeling-agent run directory")
    source.add_argument("--check-environment", action="store_true", help="report missing package dependencies without running an agent")
    source.add_argument("--inspect-handoff", type=Path, help="inspect a discovery handoff and its unadmitted leads offline")
    source.add_argument("--inspect-run", type=Path, help="print this run's result locations offline, reconstructed from its checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-root", type=Path, help="root for new per-system runs; default ENV override or source-workspace results/ (installed package: ./results)")
    parser.add_argument("--system-id", help="optional safe system label; target identity hash still prevents mixing systems")
    parser.add_argument("--initialize-only", action="store_true", help="with --task, create the run and output index without calling a model or submitting compute")
    parser.add_argument("--model", help="shared provider client model identifier")
    parser.add_argument("--max-turns", type=int, default=30, help="decision-turn allowance for this invocation")
    parser.add_argument("--token-budget", type=int, default=100000)
    parser.add_argument("--runtime-budget-seconds", type=float, default=1800)
    parser.add_argument("--backend", type=Path, help="explicit provider adapter configuration; omitted means no submissions")
    parser.add_argument("--check-discovery-run", type=Path, help="explicit trusted discovery run to check snapshot freshness")
    parser.add_argument("--lead-id", help="show one investigation-only lead with --inspect-handoff")
    parser.add_argument("--offset", type=int, default=0, help="character offset for lead inspection")
    args = parser.parse_args()
    if args.output_dir is not None and args.results_root is not None:
        parser.error("choose --output-dir or --results-root")
    if args.resume and (args.output_dir is not None or args.results_root is not None or args.system_id is not None):
        parser.error("--resume keeps the saved system and output location")
    if args.initialize_only and args.task is None:
        parser.error("--initialize-only requires --task")
    if args.check_environment:
        result = environment_diagnostics()
        print(json.dumps(result, indent=2))
        return 2 if result["missing"] else 0
    if args.inspect_handoff:
        from .handoff_inspect import inspect_handoff
        print(json.dumps(inspect_handoff(args.inspect_handoff, args.lead_id, args.offset), indent=2))
        return 0
    if args.inspect_run:
        from .result_layout import read_run_index
        print(json.dumps(read_run_index(args.inspect_run), indent=2))
        return 0
    from .agent import ModelingAgent
    agent = ModelingAgent(task_path=args.task, run_dir=args.resume or args.output_dir,
                          model=args.model, backend_path=args.backend, resume=bool(args.resume),
                          discovery_run_path=args.check_discovery_run, results_root=args.results_root, system_id=args.system_id)
    if args.initialize_only:
        agent.state["status"] = "initialized"
        agent._save()
        print(json.dumps({"status": "initialized", "run_dir": str(agent.run_dir),
                          "results_index": str(agent.run_dir / "results_index.json"),
                          "model_called": False, "compute_submitted": False}, indent=2))
        return 0
    report = agent.run(max_turns=args.max_turns, token_budget=args.token_budget,
                       runtime_budget_seconds=args.runtime_budget_seconds)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
