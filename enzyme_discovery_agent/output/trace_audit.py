"""Derived audit of recorded decisions; no retrospective scientific conclusions."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from enzyme_discovery_agent.run_layout import run_file


def verified_json(root, reference):
    if not reference:
        return None
    try:
        path = (root/reference['path']).resolve()
        path.relative_to(root.resolve())
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference['sha256']:
            return None
        return json.loads(raw)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def enrich_trace(root, events, cases):
    """Join checksummed source artifacts and exact request IDs, leaving events intact."""
    root = Path(root)
    path = run_file(root, 'token_usage.json')
    usage = json.loads(path.read_text()) if path.exists() else {}
    calls = {c['request_id']: c for c in usage.get('calls', []) if c.get('request_id')}
    claims = Counter(((c.get('model_provenance') or {}).get('request') or {}).get('request_id') for c in cases)
    details = {}
    prior_actions = {}
    prior_candidates = {}
    assessments_by_accession = {}
    for event in events:
        if event['type'] == 'target_assessment_recorded':
            record = event['record']
            sources = record.get('sources') or {}
            names = [v.get('protein_name') for v in sources.values() if isinstance(v, dict) and v.get('protein_name')]
            assessments_by_accession.setdefault(record.get('accession'), []).append({
                'event_id': event['id'], 'recorded_at': event.get('occurred_at'),
                'accession': record.get('accession'), 'protein_name': names[0] if names else None,
                'decision': record.get('decision'), 'evidence_level': record.get('evidence_level')})
    for case in cases:
        issues = []
        plan = verified_json(root, case.get('source'))
        actions = [case['action']]
        if plan is not None:
            actions = [{'tool': plan.get('tool_name'), 'arguments': plan.get('arguments') or {}}]
            actions += [{'tool': a.get('tool_name'), 'arguments': a.get('arguments') or {}}
                        for a in plan.get('additional_actions') or []]
        else:
            issues.append('Complete plan source unavailable or changed; batch completeness cannot be verified.')
        observation = case.get('observation') or {}
        result = verified_json(root, observation.get('result_source'))
        outcomes = []
        if result is not None:
            data = result.get('data')
            batched = isinstance(data, list) and bool(data) and all(
                isinstance(row, dict) and 'tool_name' in row and 'arguments' in row for row in data)
            rows = data if batched else [{'tool_name': case['action']['tool'],
                'arguments': case['action']['arguments'], **result}]
            for row in rows:
                facts = row.get('data')
                outcomes.append({'tool': row.get('tool_name'), 'arguments': row.get('arguments'),
                    'status': row.get('status'), 'summary': row.get('summary'),
                    'returned_records': len(facts) if isinstance(facts, list) else None,
                    'candidate_facts': {k: facts[k] for k in ('accession','protein_name','catalytic_activities',
                        'function_comments','critical_warnings') if k in facts} if isinstance(facts, dict) else None})
        elif observation:
            issues.append('Result source unavailable or changed; inspect the original observation record.')
        if case.get('revises') and not case.get('belief_update'):
            issues.append('Revision link exists but no explicit belief-update field was recorded; read the rationale.')
        if observation and not any(case.get(k) for k in ('followup_decisions','concluded_reviews','outcome_interpretations')):
            issues.append('No linked interpretation or follow-up recorded yet; this may be the currently active decision.')
        request_id = ((case.get('model_provenance') or {}).get('request') or {}).get('request_id')
        call = calls.get(request_id)
        cost = ({k: call.get(k) for k in ('request_id','backend','model','prompt_tokens','completion_tokens',
                                        'total_tokens','cache_read_tokens','cache_creation_tokens')} if call else None)
        if cost:
            cost['decisions_sharing_request'] = claims[request_id]
            cost['scope'] = 'Whole model request; includes all decisions emitted by that request. Do not sum shared requests twice.'
        accessions = set()
        repeated_actions = []
        for action in actions:
            signature = json.dumps(action, sort_keys=True)
            if signature in prior_actions:
                repeated_actions.append({'action': action, 'earlier_decision': prior_actions[signature]})
            args = action.get('arguments') or {}
            if isinstance(args, dict):
                if isinstance(args.get('accession'), str):
                    accessions.add(args['accession'])
                if isinstance(args.get('accessions'), list):
                    accessions.update(a for a in args['accessions'] if isinstance(a, str))
            prior_actions[signature] = case['decision_id']
        revisits = [{'accession': accession, 'previous_decisions': prior_candidates[accession]}
                    for accession in sorted(accessions) if accession in prior_candidates]
        for accession in accessions:
            prior_candidates[accession] = prior_candidates.get(accession, []) + [case['decision_id']]
        if repeated_actions:
            issues.append('Identical action arguments appeared earlier. Inspect whether new evidence justifies repeating the action.')
        detail = {'actions': actions, 'action_count': len(actions), 'outcomes': outcomes,
                  'request_usage': cost, 'issues': issues,
                  'repeated_actions': repeated_actions, 'candidate_revisits': revisits,
                  'candidate_assessments': [assessment for accession in sorted(accessions)
                      for assessment in assessments_by_accession.get(accession, [])],
                  'provenance': 'Derived from checksummed saved plan/result artifacts and exact ledger request IDs; original journal unchanged.'}
        case['audit'] = detail
        details['prediction_registered:'+case['decision_id']] = detail
    review_requests = set()
    for event in events:
        if event['type'] not in {'target_assessment_recorded', 'target_review_batch_recorded'}:
            continue
        record = event['record']
        recorded_usage = (record.get('request_usage') or
                          ((record.get('model_provenance') or {}).get('request')) or {})
        request_id = recorded_usage.get('request_id')
        call = calls.get(request_id)
        if call:
            review_requests.add(request_id)
        sources = record.get('sources') or {}
        names = [v.get('protein_name') for v in sources.values() if isinstance(v, dict) and v.get('protein_name')]
        details[event['id']] = {'candidate_name': names[0] if names else None,
                               'accession': record.get('accession'),
                               'request_usage': dict(call) if call else None,
                               'usage_scope': 'Whole review request; shared across its candidate assessments and batch record.'}
    linked = {((case.get('audit') or {}).get('request_usage') or {}).get('request_id') for case in cases}
    linked.discard(None)
    summary = {'registered_decisions': len(cases),
               'planned_actions': sum(c['audit']['action_count'] for c in cases),
               'decisions_with_usage': sum(c['audit']['request_usage'] is not None for c in cases),
               'unique_linked_requests': len(linked),
               'linked_request_tokens': sum(calls[key].get('total_tokens', 0) for key in linked),
               'target_review_linked_requests': len(review_requests),
               'target_review_linked_tokens': sum(calls[key].get('total_tokens', 0) for key in review_requests),
               'all_linked_request_tokens': sum(calls[key].get('total_tokens', 0) for key in linked | review_requests),
               'run_tokens': usage.get('total_tokens'),
               'assessment_decisions': dict(Counter(e['record'].get('decision') for e in events
                                                    if e['type']=='target_assessment_recorded')),
               'scope': 'Linked request costs are a subset of run usage, not all reasoning cost. Retrieval success is not scientific confirmation.'}
    return details, summary
