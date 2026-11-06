"""Synthetic scoped-evidence tests; answer keys never enter production context."""
import argparse
import hashlib
import json
from pathlib import Path

ACTIONS = ['resolve_partner', 'resolve_identity', 'verify_controls', 'test_mechanism',
           'measure_turnover', 'repeat_retrieval', 'compare_kinetics', 'inspect_primary_source']
SUITE_VERSION = 2


def cases():
    rows = [
        ('c01', 'Target: A1 + B1 -> P11, using synthetic protein S1.',
         'Purified S1 converts A0 + B1 to P01 in a controlled assay. A1 was not tested.',
         {'B1_acceptance_with_A0': 'supported', 'complete_requested_reaction': 'unresolved'}, ['resolve_partner']),
        ('c02', 'Target: A1 + B1 -> P11, using synthetic protein S1.',
         'Purified S1 converts A1 + B1 to independently identified P11. Both substrates are required; no-enzyme and inactive-protein controls show no product.',
         {'B1_acceptance_with_A1': 'supported', 'complete_requested_reaction': 'supported'}, ['compare_kinetics']),
        ('c03', 'Evaluate target activity of synthetic sequence S1.',
         'The controlled target-reaction assay used homolog S2 and confirmed formation of the specified product above no-enzyme controls. S1 and S2 are distinct sequences; S1 has not been tested.',
         {'activity_of_S2': 'supported', 'S1_catalyzes_the_target_reaction': 'unresolved'}, ['resolve_identity', 'measure_turnover']),
        ('c04', 'Evaluate target activity of synthetic sequence S1.',
         'The controlled target-reaction assay used a preparation called S2 and confirmed formation of the specified product above no-enzyme controls. Independent sequence and construct verification establish that S2 is exactly S1 under a different label.',
         {'activity_of_S2': 'supported', 'S1_catalyzes_the_target_reaction': 'supported'}, ['compare_kinetics']),
        ('c05', 'Target: join two separate molecules of NCCCC(=O)O through a C-N bond.',
         'The observed native reaction converts one molecule of NCCCC(=O)O into O=C1CCCN1 plus water. No intermolecular product has been measured.',
         {'native_reaction_joins_two_molecules': 'contradicted', 'S1_catalyzes_requested_intermolecular_reaction': 'unresolved'}, ['test_mechanism', 'measure_turnover']),
        ('c06', 'Assess whether synthetic protein S1 catalyzes the target reaction.',
         'The literature service returned an HTTP access error for every request. No experiment or successful source retrieval is available.',
         {'target_activity': 'unresolved', 'S1_is_inactive_on_target': 'unresolved'}, ['repeat_retrieval']),
        ('c07', 'Assess S1 target activity at pH 7 and whether inactivity is established under all possible conditions.',
         'At pH 7, the target-product signal is below the validated detection limit in replicated assays. A native-substrate positive control confirms active protein. Other conditions were not tested.',
         {'detectable_target_activity_at_pH7': 'contradicted', 'inactivity_under_all_conditions': 'unresolved'}, ['inspect_primary_source', 'test_mechanism']),
        ('c08', 'Assess target-reaction evidence from a predicted complex.',
         'A predicted pose places the proposed reacting atoms 3 angstroms apart. No turnover experiment was performed; activation and proton-transfer requirements remain unresolved.',
         {'predicted_atoms_are_nearby': 'supported', 'S1_catalyzes_the_target_reaction': 'unresolved'}, ['measure_turnover', 'test_mechanism']),
        ('c09', 'An earlier interpretation attributed product formation to S1. Reassess using the new control.',
         'The original assay reported product. A newly measured matched no-enzyme control produces the same amount of product. The current assay cannot distinguish an enzyme contribution from background.',
         {'assay_distinguishes_enzyme_from_background': 'contradicted', 'S1_target_activity': 'unresolved'}, ['verify_controls']),
        ('c10', 'Target: A1 + B1 -> P11 using S1; compare the B1 rate with the native reference.',
         'With native acid A0, B1 produces the expected product at 3 percent of the native-reference rate under identical conditions. Product identity and controls are sound. A1 was never tested.',
         {'some_turnover_with_A0_and_B1': 'supported', 'B1_rate_at_least_half_reference': 'contradicted', 'complete_requested_reaction': 'unresolved'}, ['resolve_partner', 'compare_kinetics']),
        ('c11', 'Assess a proposed design intervention for synthetic S1.',
         'S1 accepts B1 with A0. Acceptance of target A1 is unmeasured. A model proposes enlarging the donor pocket; no variant has been generated or tested and the affected residues may also position the catalytic group.',
         {'intervention_improves_target_activity': 'unresolved', 'proposed_variant_preserves_native_function': 'unresolved'}, ['test_mechanism', 'measure_turnover']),
    ]
    return [{'id': identity, 'public': {'target': target, 'sources': [{'id': 'S01', 'text': observation}],
              'claims': [{'id': 'K'+str(i), 'statement': claim.replace('_', ' ')}
                         for i, claim in enumerate(expected, 1)], 'available_actions': ACTIONS},
             'expected': {'K'+str(i): verdict for i, verdict in enumerate(expected.values(), 1)},
             'acceptable_actions': actions}
            for identity, target, observation, expected, actions in rows]


def public_case(case):
    return {'id': case['id'], **case['public']}


def grade(case, response):
    if not isinstance(response, dict):
        response = {}
    claims = response.get('claims')
    claims = claims if isinstance(claims, list) else []
    identifiers = [c.get('id') for c in claims if isinstance(c, dict)]
    exact = len(identifiers) == len(claims) == len(case['expected']) and set(identifiers) == set(case['expected'])
    checks = {'exact_claim_coverage': exact,
              'verdicts': exact and all(c.get('verdict') == case['expected'][c['id']] for c in claims),
              'references': bool(claims) and all(isinstance(c, dict) and c.get('evidence_refs') == ['S01'] for c in claims),
              'rationales_present': bool(claims) and all(isinstance(c, dict) and isinstance(c.get('reason'), str)
                                                       and c['reason'].strip() for c in claims),
              'next_action_in_menu': response.get('next_action') in ACTIONS}
    return {'id': case['id'], 'passed': all(checks.values()), 'checks': checks,
            'action_matches_reference': response.get('next_action') in case['acceptable_actions'],
            'limitation': 'Scoped verdicts and action admissibility only. Reference-action agreement is diagnostic, not a correctness gate: action optimality needs defined costs and objectives. Prose entailment requires independent review.'}


def run(output, model, budget):
    from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
    from enzyme_discovery_agent.state_store import atomic_write_json
    output.mkdir(parents=True, exist_ok=False)
    suite = cases()
    atomic_write_json(output/'oracle.json', suite)
    client = LLMClient(model=model)
    usage = TokenUsage(model=model)
    client.attach_usage(usage, output)
    client.budget_check = lambda: usage.total_tokens >= budget
    schema = {'type': 'object', 'properties': {
        'claims': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'id': {'type': 'string'}, 'verdict': {'type': 'string', 'enum': ['supported','contradicted','unresolved']},
            'evidence_refs': {'type': 'array', 'items': {'type': 'string'}}, 'reason': {'type': 'string'}},
            'required': ['id','verdict','evidence_refs','reason']}},
        'next_action': {'type': 'string', 'enum': ACTIONS}}, 'required': ['claims','next_action']}
    tool = {'type': 'function', 'function': {'name': 'record_assessment',
        'description': 'Assess the supplied claims and choose the next useful action.', 'parameters': schema}}
    results = []
    for case in suite:
        row = {'id': case['id'], 'status': 'not_run', 'passed': False}
        results.append(row)
        if usage.total_tokens + 10000 > budget:
            row['status'] = 'budget_deferred'
            continue
        public = public_case(case)
        messages = [{'role': 'system', 'content': 'Evaluate chemical claims from supplied synthetic observations. '
            'Use only the supplied sources. Assess each claim at its stated scope. '
            'Call record_assessment exactly once, with one concise reason and source reference for every claim.'},
            {'role': 'user', 'content': json.dumps(public)}]
        atomic_write_json(output/(case['id']+'_request.json'), {'messages': messages, 'tools': [tool]})
        response = client.call(messages=messages, tools=[tool], max_tokens=2500, max_retries=1,
                               usage_label=case['id'], timeout=120)
        if response is None:
            row['status'] = 'provider_unavailable'
        else:
            message = response.choices[0].message.model_dump()
            atomic_write_json(output/(case['id']+'_response.json'), message)
            calls = message.get('tool_calls') or []
            try:
                if len(calls) != 1 or calls[0]['function']['name'] != 'record_assessment':
                    raise ValueError('Exactly one assessment call required')
                value = calls[0]['function']['arguments']
                value = json.loads(value) if isinstance(value, str) else value
                row.update(status='graded', **grade(case, value))
            except (ValueError, TypeError, KeyError) as exc:
                row.update(status='invalid_output', error=str(exc))
        atomic_write_json(output/'partial_results.json', results)
        print(json.dumps({'case': case['id'], 'status': row['status'], 'passed': row['passed'],
                          'tokens': usage.total_tokens}), flush=True)
    report = {'suite_version': SUITE_VERSION, 'model': model, 'budget': budget, 'tokens': usage.total_tokens, 'cases': results,
              'passed': sum(row['passed'] for row in results), 'total': len(suite),
              'suite_sha256': hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest(),
              'scope': 'Synthetic development component baseline. Not autonomous discovery, full chemical reasoning or held-out generalization.'}
    atomic_write_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--token-budget', type=int, default=100000)
    args = parser.parse_args()
    if args.token_budget < 10000:
        parser.error('Require at least one bounded request allowance')
    report = run(args.output, args.model, args.token_budget)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))
    return 0 if report['passed'] == report['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
