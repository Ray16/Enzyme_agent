"""Offline operational comparison; no benchmark answers enter runtime context."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, discovery_status, family_label
from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.run_layout import run_file


def summarize(run_dir):
    root = Path(run_dir)
    usage = json.loads(run_file(root, 'token_usage.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    candidates = load_candidate_registry(root, strict=True).get('candidates', {})
    statuses = Counter(discovery_status(entry) for entry in candidates.values())
    evaluated = {acc: entry for acc, entry in candidates.items() if discovery_status(entry) in {'evaluated', 'ranked'}}
    events = read_events(root)
    decisions = [call for call in usage.get('calls', []) if isinstance(call.get('iteration'), int)]
    inputs = [call.get('prompt_tokens', 0) for call in decisions]
    attempts = [e for e in events if e.get('type') == 'candidates_evaluated']
    return {'run_dir': str(root.resolve()), 'status': manifest.get('status'), 'model': usage.get('model'),
            'tokens': usage.get('total_tokens'), 'accounting_complete': usage.get('accounting_complete', False),
            'model_calls': len(usage.get('calls', [])), 'controller_calls': len(decisions),
            'max_controller_input_tokens': max(inputs, default=0),
            'median_controller_input_tokens': median(inputs) if inputs else None,
            'registered_records': len(candidates), 'status_counts': dict(statuses),
            'evaluated_records': len(evaluated),
            'evaluated_family_groups': len({family_label(entry.get('metadata') or {}, acc) for acc, entry in evaluated.items()}),
            'evaluation_batches': len(attempts),
            'operational_only': True,
            'scientific_quality': 'Requires independent target-activity evidence and a held-out evaluation; throughput alone is insufficient.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.run_dirs], indent=2))


if __name__ == '__main__':
    main()
