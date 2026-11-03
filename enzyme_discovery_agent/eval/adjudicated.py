"""Score externally curated cases and outputs; this module never runs discovery.

The input case file is an evaluation oracle and must remain outside runtime
context. Curators must document case independence and evidence adjudication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_case(case: dict, output: dict) -> dict:
    """Report retrieval, eligibility and ranking against explicit adjudications.

    Unjudged entries are counted separately and are excluded from precision's
    denominator. Ranking is supplied as an ordered list of exact identifiers.
    """
    labels = case['adjudications']
    if not isinstance(labels, dict) or not labels or any(v not in {'positive', 'negative'} for v in labels.values()):
        raise ValueError('adjudications must map identifiers to positive or negative')
    if case.get('id') != output.get('id'):
        raise ValueError('case and output identifiers differ')
    positives = {k for k, v in labels.items() if v == 'positive'}
    stages = {}
    for stage in ('retrieved', 'resolved', 'evaluated', 'eligible', 'ranked'):
        values = output.get(stage, [])
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            raise ValueError(f'{stage} must be a list of identifiers')
        selected = set(values)
        tp = len(selected & positives)
        fp = len({x for x in selected if labels.get(x) == 'negative'})
        stages[stage] = {'count': len(selected), 'true_positives': tp, 'false_positives': fp,
            'unjudged': len(selected - set(labels)),
            'recall': tp / len(positives) if positives else None,
            'judged_precision': tp / (tp + fp) if tp + fp else None}
    ranked = output.get('ranked', [])
    best_rank = next((i for i, key in enumerate(ranked, 1) if key in positives), None)
    reviews = output.get('claim_reviews', [])
    if not isinstance(reviews, list) or any(
        not isinstance(row, dict) or row.get('verdict') not in {'supported', 'unsupported', 'uncertain'}
        for row in reviews
    ):
        raise ValueError('claim_reviews must be a list of adjudicated verdict objects')
    review_counts = {verdict: sum(row['verdict'] == verdict for row in reviews)
                     for verdict in ('supported', 'unsupported', 'uncertain')}
    return {'id': case['id'], 'stages': stages, 'best_positive_rank': best_rank,
            'abstained': bool(output.get('abstained', False)),
            'claim_review_counts': review_counts,
            'limitations': ['Scores depend on the independently curated adjudications.',
                'Unjudged entries are not assumed negative.',
                'Claim reviews are reported as supplied, not verified by this scorer.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path, help='Independent curated oracle JSON list')
    parser.add_argument('outputs', type=Path, help='Recorded output JSON list')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    outputs = json.loads(args.outputs.read_text())
    by_id = {row['id']: row for row in outputs}
    if len(by_id) != len(outputs) or len({row['id'] for row in cases}) != len(cases) or set(by_id) != {row['id'] for row in cases}:
        parser.error('cases and outputs must contain the same unique IDs')
    print(json.dumps([score_case(case, by_id[case['id']]) for case in cases], indent=2))


if __name__ == '__main__':
    main()
