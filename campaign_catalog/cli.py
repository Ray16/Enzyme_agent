import argparse
import json
from .catalog import build, verify, record_feedback


def main():
    parser = argparse.ArgumentParser(description="Index explicitly supplied runs without moving their artifacts.")
    parser.add_argument("campaign_dir")
    parser.add_argument("--verify", action="store_true", help="Verify the published snapshot and audit chain")
    parser.add_argument("--feedback", help="Append a source-linked observation, then refresh the catalog")
    args = parser.parse_args()
    if args.feedback:
        print(json.dumps(record_feedback(args.campaign_dir, args.feedback), indent=2))
    print(json.dumps(verify(args.campaign_dir) if args.verify else build(args.campaign_dir), indent=2))
