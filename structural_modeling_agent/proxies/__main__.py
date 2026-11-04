"""CLI for catalog inspection and execution of reviewed proxy plans."""
import argparse
import json
from pathlib import Path

from enzyme_discovery_agent.structural_proxy_planning import load_proxy_catalog
from .workflow import run_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--record", type=Path, help="One resolved candidate modeling record")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.catalog:
        print(json.dumps(load_proxy_catalog(), indent=2))
        return
    if not all((args.plan, args.record, args.predictions, args.output)):
        parser.error("supply --plan, --record, --predictions and --output")
    result = run_plan(args.predictions, json.loads(args.record.read_text()), json.loads(args.plan.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from enzyme_discovery_agent.state_store import atomic_write_json
    atomic_write_json(args.output, result, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
