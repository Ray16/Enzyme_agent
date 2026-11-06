"""Offline replay of saved review packets; never a production context source.

Run with --run PATH (repeatable) and --output PATH. This measures representation
size and exact reconstruction, not model comprehension or discovery quality.
"""

import argparse
import hashlib
import json
from pathlib import Path

from enzyme_discovery_agent.request_efficiency import factor_evidence, expand_evidence, serialized


def audit_run(root):
    batches = []
    for path in sorted((Path(root) / 'artifacts/target_reviews').glob('batch_*.json')):
        raw = path.read_bytes()
        data = json.loads(raw)
        candidates = [{key: value for key, value in row.items() if key not in
                       {'record_sha256', 'eligibility_sha256', 'target_sha256'}}
                      for row in data['packet']['candidates']]
        packed = factor_evidence(candidates)
        restored = expand_evidence(packed)
        if restored != candidates:
            raise ValueError(f'Evidence round trip failed: {path}')
        before = len(serialized({'candidates': candidates}))
        after = len(serialized(packed))
        if after > before:
            raise ValueError(f'Projection grew: {path}')
        batches.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'candidates': len(candidates), 'before_characters': before,
                        'after_characters': after, 'round_trip_equal': True})
    if not batches:
        raise ValueError(f'No saved target-review packets: {root}')
    before = sum(row['before_characters'] for row in batches)
    after = sum(row['after_characters'] for row in batches)
    return {'run': str(root), 'batches': batches, 'before_characters': before,
            'after_characters': after, 'reduction_fraction': 1 - after / before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = {'measurement': 'compact_json_characters_in_candidate_message',
              'limitations': ['Not provider token counts or cost.',
                              'Exact reconstruction does not establish unchanged model decisions.',
                              'Saved runs are development data, not held-out discovery evaluation.'],
              'runs': [audit_run(root) for root in args.run]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'batches'}
                      for row in result['runs']], indent=2))


if __name__ == '__main__':
    main()
