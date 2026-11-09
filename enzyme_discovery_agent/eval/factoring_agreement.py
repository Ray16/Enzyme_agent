"""Scoring for the review-representation replay. Frozen before any arm was run.

The question is narrow: does presenting the same evidence in a shared form change
the reviewer's decisions? Representation equality is already proven offline, so a
difference here is a behavioral effect, not an information loss.

Pre-registered rule
-------------------
Two arms receive byte-identical baseline input; their disagreement is this
comparison's noise floor. The factored arm passes a field only if it disagrees
with baseline no more often than the replicate does. A field where the replicate
disagrees more than factored is reported as inconclusive for that field rather
than as evidence of improvement: agreement cannot distinguish a better decision
from a luckier sample.

No significance test is reported. Twenty-four candidates cannot separate a small
behavioral effect from sampling variation, and quoting a p-value from a sample
this size would overstate what the replay can settle. A pass here is a
precondition for enabling factoring, never a demonstration of discovery benefit.
"""

import argparse
import json
from pathlib import Path

FIELDS = ('decision', 'evidence_level', 'role_bases', 'constraint_bases')


def assessments(message):
    """Index one arm's structured assessments by accession."""
    result = {}
    for call in message.get('tool_calls') or []:
        function = call.get('function') or {}
        if function.get('name') != 'record_target_assessment':
            continue
        arguments = function.get('arguments')
        try:
            value = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and isinstance(value.get('accession'), str):
            result[value['accession']] = value
    return result


def comparable(value):
    """Project one assessment onto the fields the rule scores."""
    roles = value.get('target_roles')
    constraints = value.get('constraints')
    return {
        'decision': value.get('decision'),
        'evidence_level': value.get('evidence_level'),
        'role_bases': sorted((role.get('participant'), role.get('basis'))
                             for role in roles if isinstance(role, dict)) if isinstance(roles, list) else None,
        'constraint_bases': sorted((item.get('basis'), item.get('severity'))
                                   for item in constraints
                                   if isinstance(item, dict)) if isinstance(constraints, list) else None,
    }


def score(comparisons):
    counts = {field: {'replicate_differs': 0, 'factored_differs': 0} for field in FIELDS}
    shared, missing, contract_errors = 0, [], {}
    for pair in comparisons:
        responses = pair.get('responses') or {}
        arms = {mode: assessments((responses.get(mode) or {}).get('message') or {})
                for mode in ('baseline', 'baseline_replicate', 'factored')}
        for mode, body in (responses or {}).items():
            if body.get('contract_errors'):
                contract_errors.setdefault(mode, []).extend(body['contract_errors'])
        common = set.intersection(*(set(arm) for arm in arms.values())) if all(arms.values()) else set()
        for mode, arm in arms.items():
            for accession in set(arm) - common:
                missing.append({'accession': accession, 'present_only_in': mode})
        for accession in sorted(common):
            shared += 1
            views = {mode: comparable(arm[accession]) for mode, arm in arms.items()}
            for field in FIELDS:
                if views['baseline_replicate'][field] != views['baseline'][field]:
                    counts[field]['replicate_differs'] += 1
                if views['factored'][field] != views['baseline'][field]:
                    counts[field]['factored_differs'] += 1
    verdicts = {}
    for field, count in counts.items():
        if count['factored_differs'] <= count['replicate_differs']:
            verdicts[field] = 'within_noise' if count['replicate_differs'] else 'identical'
        else:
            verdicts[field] = 'exceeds_noise'
    return {'candidates_compared': shared, 'counts': counts, 'verdicts': verdicts,
            'unmatched_assessments': missing, 'contract_errors': contract_errors,
            'passes_preregistered_rule': bool(shared) and not missing and not contract_errors
                                         and all(v != 'exceeds_noise' for v in verdicts.values()),
            'scope': ('Behavioral agreement on saved evidence packets. Not a discovery-benefit '
                      'measurement and not powered to exclude a small effect.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    report = {'model': data.get('model'), 'source': str(args.comparison), **score(data['comparisons'])}
    text = json.dumps(report, indent=1)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
