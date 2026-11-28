"""CLI for the enzyme design agent."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from enzyme_discovery_agent import config

from .agent import EnzymeDesignAgent


def main() -> None:
    p = argparse.ArgumentParser(
        prog="enzyme-design-agent",
        description="Design an enzyme for a target chemistry: the agent chooses "
                    "between generative de novo design and saturation redesign, "
                    "specifies the motif or mutable set, runs it and reports.",
    )
    p.add_argument("query", nargs="?", help="The target chemistry, in plain words.")
    p.add_argument("--motif", default=None,
                   help="Notes on the catalytic motif. A starting point the agent "
                        "must still justify, not a specification it will trust.")
    p.add_argument("--scaffold", default=None,
                   help="Path to a holo structure, if one exists. Makes redesign "
                        "possible; does not decide the strategy.")
    p.add_argument("--handoff", default=None,
                   help="Path to a structured design_handoff.json from "
                        "structural_modeling_agent (carries catalytic profile, "
                        "soft-defect design targets, failure modes, and cofold evidence).")
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--output-dir", default=None,
                   help="Defaults to <package>/runs.")
    p.add_argument("--max-iterations", type=int, default=40)
    p.add_argument("--token-budget", type=int, default=None,
                   help="Optional total-token budget before forcing finalization.")
    p.add_argument("--runtime-budget-seconds", type=float, default=None,
                   help="Optional wall-clock budget in seconds before forcing finalization.")
    p.add_argument("--llm-timeout", type=int, default=config.LLM_TIMEOUT)
    p.add_argument("--list-tools", action="store_true",
                   help="Print the design toolkit and exit.")
    args = p.parse_args()

    if args.list_tools:
        from .bootstrap import design_tool_names

        for name in design_tool_names():
            print(name)
        return

    if args.handoff and not Path(args.handoff).expanduser().is_file():
        p.error(f"--handoff is not a file: {args.handoff}")
    if not args.query and not args.handoff:
        p.error("query or --handoff is required")
    if args.scaffold and not Path(args.scaffold).expanduser().is_file():
        p.error(f"--scaffold is not a file: {args.scaffold}")

    query = args.query or f"Design enzyme from handoff {Path(args.handoff).name}"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = EnzymeDesignAgent(
        model=args.model,
        output_dir=args.output_dir or Path(__file__).parent / "runs",
        max_iterations=args.max_iterations,
        llm_timeout=args.llm_timeout,
        query=query,
        token_budget=args.token_budget,
        runtime_budget_seconds=args.runtime_budget_seconds,
    )
    report = agent.run(query=query, motif_hint=args.motif,
                       scaffold_path=args.scaffold, handoff_path=args.handoff)
    log = logging.getLogger("enzyme_agent")
    log.info("\n%s\nDESIGN REPORT\n%s\n%s", "=" * 70, "=" * 70, report)


if __name__ == "__main__":
    main()
