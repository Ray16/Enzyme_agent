"""Explicit, bounded live comparison of review representations on saved evidence.

This development replay does not run discovery or modify the source campaign.
Decision agreement is diagnostic, not proof of scientific noninferiority.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from enzyme_discovery_agent import config
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models import target_review
from enzyme_discovery_agent.request_efficiency import expand_evidence, serialized
from enzyme_discovery_agent.state_store import atomic_write_json


class Captured(Exception):
    pass


def capture_request(packet, factored):
    """Exercise the real prompt builder, stopping before any execution or writes."""
    request = {}

    class CaptureClient:
        def call(self, **kwargs):
            request.update(kwargs)
            raise Captured()

    with patch.object(config, 'TARGET_REVIEW_FACTOR_EVIDENCE', factored), \
         patch.object(target_review, 'review_queue', return_value=copy.deepcopy(packet['candidates'])), \
         patch.object(target_review, 'target_contract', return_value=copy.deepcopy(packet['target'])):
        try:
            target_review.run_review(Path('.'), CaptureClient(), limit=len(packet['candidates']))
        except Captured:
            pass
    if not request:
        raise ValueError('No review request captured')
    return request


def response_contract_errors(message, request):
    """Check structured-output validity separately from scientific agreement."""
    import jsonschema
    candidates = expand_evidence(json.loads(request['messages'][-1]['content']))
    expected = {row['accession']: row['source_id'] for row in candidates}
    participants = {row['id'] for row in json.loads(request['messages'][1]['content'])['target_inputs']}
    schema = request['tools'][0]['function']['parameters']
    errors, seen = [], set()
    for call in message.get('tool_calls') or []:
        try:
            function = call['function']
            if function['name'] != 'record_target_assessment':
                raise ValueError('Unpermitted tool')
            result = function['arguments']
            result = json.loads(result) if isinstance(result, str) else result
            jsonschema.validate(result, schema)
            accession = result['accession']
            if accession not in expected or accession in seen:
                raise ValueError('Unknown or duplicate accession')
            seen.add(accession)
            if result['source_refs'] != [expected[accession]]:
                raise ValueError('Incorrect source_refs')
            roles = result['target_roles']
            if len(roles) != len(participants) or {role['participant'] for role in roles} != participants:
                raise ValueError('Incorrect participant coverage')
            if result['decision'] in {'prioritize', 'investigate'} and (
                    result['evidence_level'] == 'speculative' or not result['strengths']):
                raise ValueError('Unsupported positive decision')
            if result['decision'] == 'exclude' and not any(
                    item['basis'] == 'observed_mismatch' for item in result['constraints']):
                raise ValueError('Exclusion without observed mismatch')
        except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as exc:
            errors.append(str(exc))
    if seen != set(expected):
        errors.append('Missing candidate assessments')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', action='append', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-total-tokens', type=int, default=100000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory to preserve previous usage and results')
    if args.max_total_tokens <= 0:
        raise ValueError('Token allowance must be positive')
    args.output_dir.mkdir(parents=True)
    client = LLMClient(model=args.model)
    usage = TokenUsage(model=args.model)
    client.attach_usage(usage, args.output_dir)
    client.budget_check = lambda: usage.total_tokens >= args.max_total_tokens
    comparisons = []
    for index, path in enumerate(args.batch):
        raw = path.read_bytes()
        packet = json.loads(raw)['packet']
        requests = {mode: capture_request(packet, mode == 'factored') for mode in ('baseline', 'factored')}
        views = {mode: expand_evidence(json.loads(request['messages'][-1]['content']))
                 for mode, request in requests.items()}
        if views['baseline'] != views['factored']:
            raise ValueError('Representations do not contain exactly the same evidence')
        pair = {'source_batch': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'round_trip_equal': True, 'responses': {}}
        comparisons.append(pair)
        # Alternate order to avoid always warming the common prefix for one arm.
        for mode in (('baseline', 'factored') if index % 2 == 0 else ('factored', 'baseline')):
            request = requests[mode]
            # Conservative preflight guard, not reported as measured token usage.
            # Reserve output plus one token per UTF-8 byte of the supplied payload.
            reserve = len(serialized(request).encode()) + 8192
            if usage.total_tokens + reserve > args.max_total_tokens:
                raise RuntimeError('Remaining replay allowance cannot conservatively fit the next request')
            response = client.call(**{**request, 'usage_label': f'replay_{index}_{mode}',
                                      'max_tokens': 8192, 'max_retries': 1})
            if response is None:
                raise RuntimeError('Live provider unavailable; completed requests remain in output directory')
            message = response.choices[0].message.model_dump()
            pair['responses'][mode] = {'request': request,
                                      'message': message,
                                      'contract_errors': response_contract_errors(message, request),
                                      'usage': usage.calls[-1] if usage.calls else None}
            atomic_write_json(args.output_dir/'comparison.json', {
                'model': args.model, 'comparisons': comparisons,
                'scope': 'Development replay, not held-out discovery evaluation',
                'usage': usage.to_dict()})
            print(json.dumps({'batch': index, 'mode': mode, 'tokens_so_far': usage.total_tokens}), flush=True)


if __name__ == '__main__':
    main()
