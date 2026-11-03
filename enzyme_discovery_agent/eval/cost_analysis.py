"""Summarize recorded run cost and search yield without contacting services."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from enzyme_discovery_agent.eval.metrics import _load_events


def analyze_run(run_dir: str | Path) -> dict:
    events = _load_events(Path(run_dir))
    models = defaultdict(lambda: {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
    tools = defaultdict(lambda: {'calls': 0, 'failures': 0, 'elapsed_seconds_sum': 0.0,
                                 'new_accessions': 0})
    times = []
    for event in events:
        try:
            times.append(datetime.fromisoformat(event['ts']).timestamp())
        except (ValueError, TypeError, KeyError):
            pass
        if event.get('type') == 'llm_call':
            key = f"{event.get('backend') or 'unknown'}:{event.get('model') or 'unknown'}"
            row = models[key]
            row['calls'] += 1
            row['prompt_tokens'] += event.get('prompt_tokens') or 0
            row['completion_tokens'] += event.get('completion_tokens') or 0
        elif event.get('type') == 'tool_call':
            row = tools[str(event.get('tool') or 'unknown')]
            row['calls'] += 1
            row['failures'] += event.get('status') == 'error'
            row['elapsed_seconds_sum'] += event.get('elapsed_s') or 0
            row['new_accessions'] += (event.get('novelty') or {}).get('new_accessions') or 0
    return {
        'event_span_seconds': round(max(times) - min(times), 2) if len(times) >= 2 else None,
        'models': dict(sorted(models.items())),
        'tools': dict(sorted(tools.items())),
        'deduplicated_calls': sum(event.get('type') == 'tool_dedup' for event in events),
        'unmetered_llm_calls': sum(event.get('type') == 'llm_usage_unavailable' for event in events),
        'limitations': ['Event span includes gaps between resumed invocations.',
            'Tool elapsed times overlap under concurrency and are not additive wall time.',
            'New accessions are discovery counts, not supported or independent evidence.',
            'Missing historical events and usage cannot be reconstructed.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_run(args.run_dir), indent=2))


if __name__ == '__main__':
    main()
