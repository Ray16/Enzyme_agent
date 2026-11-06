import copy
import json

from enzyme_discovery_agent.eval.chemical_reasoning import cases, public_case, grade


def answer(case):
    return {'claims': [{'id': key, 'verdict': verdict, 'evidence_refs': ['S01'],
                        'reason': 'Synthetic grader fixture; not model performance.'}
                       for key, verdict in case['expected'].items()],
            'next_action': case['acceptable_actions'][0]}


def test_public_observations_exclude_oracles_and_include_exact_claim_ids():
    for case in cases():
        public = public_case(case)
        assert 'expected' not in public and 'acceptable_actions' not in public
        assert {c['id'] for c in public['claims']} == set(case['expected'])
        assert all('verdict' not in c for c in public['claims'])


def test_grader_rejects_omissions_duplicates_extra_claims_and_wrong_sources():
    case = cases()[0]
    good = answer(case)
    assert grade(case, good)['passed']
    for claims in [good['claims'][:1], good['claims']*2,
                   good['claims']+[{'id': 'unknown', 'verdict': 'supported'}]]:
        assert not grade(case, {**good, 'claims': claims})['passed']
    bad = copy.deepcopy(good); bad['claims'][0]['evidence_refs'] = ['invented']
    assert not grade(case, bad)['passed']
    assert not grade(case, None)['passed']


def test_blanket_abstention_and_positive_promotions_cannot_pass_the_suite():
    for uniform in ('unresolved', 'supported', 'contradicted'):
        results = []
        for case in cases():
            response = answer(case)
            for claim in response['claims']:
                claim['verdict'] = uniform
            results.append(grade(case, response)['passed'])
        assert not all(results)


def test_paired_evidence_changes_require_different_verdicts():
    suite = cases()
    assert suite[0]['expected']['K2'] != suite[1]['expected']['K2']
    assert suite[2]['expected']['K2'] != suite[3]['expected']['K2']
    # No provider is called by fixture generation or grading.
    assert all(grade(case, answer(case))['passed'] for case in suite)
