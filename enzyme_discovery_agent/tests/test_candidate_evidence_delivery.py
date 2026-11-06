"""Sequential evidence delivery through production projections and saved recall."""
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from enzyme_discovery_agent.context import build_model_context, compact_for_llm
from enzyme_discovery_agent.evidence_projection import preview_evidence
from enzyme_discovery_agent.models.candidate_evidence import candidate_record
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.models import research_decisions as decisions, target_review as review
from enzyme_discovery_agent.models.research_memory import recall_memory
from enzyme_discovery_agent.tests.test_scientific_trace import proposal
from enzyme_discovery_agent.tests.test_target_review_gate import setup, assessment
from enzyme_discovery_agent.tools.reasoning import get_candidate_evidence


def context(root, result):
    identifier = decisions.start_decision(root, proposal(root), {'synthetic_retrieval'})
    decisions.finish_decision(root, identifier, result)
    packet = decisions.decision_context(root)
    packet['review_due'] = True
    with patch('enzyme_discovery_agent.context.build_research_state', return_value={'adaptive_reasoning': packet}):
        messages = build_model_context([{'role': 'system', 'content': 'Respect source scope.'},
                                       {'role': 'user', 'content': 'Assess the target.'}], root)
    block = next(m['content'] for m in messages if m.get('content', '').startswith('CURRENT DECISION EVIDENCE'))
    return packet, messages, json.loads(block.split('\n', 1)[1])


def test_complete_legacy_reactions_survive_current_views_and_later_updates(tmp_path):
    setup(tmp_path)
    reactions = ['Synthetic native step ' + str(i) for i in range(5)]
    update_candidate_registry(tmp_path, 'SYNTH1', None, {
        'catalytic_activities': reactions[:3],
        'evaluation_data': {'catalytic_activities': reactions, 'literature_evidence': ['Old observation']},
        'literature_evidence': ['New control observation']})
    metadata = load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    before = copy.deepcopy(metadata)
    assert candidate_record(metadata)['catalytic_activities'] == reactions
    assert candidate_record(metadata)['literature_evidence'] == ['New control observation']
    assert metadata == before
    assert review.scientific_record(metadata)['catalytic_activities'] == reactions
    assert get_candidate_evidence('SYNTH1', tmp_path)['data']['catalytic_activities'] == reactions
    packet, _, _ = context(tmp_path, get_candidate_evidence('SYNTH1', tmp_path))
    assert packet['previous_test']['result_examples'][0]['catalytic_activities'] == reactions
    for changed in ([], ['A replacement native reaction']):
        assert candidate_record({**metadata, 'catalytic_activities': changed})['catalytic_activities'] == changed
    changed = copy.deepcopy(metadata)
    changed['evaluation_data']['catalytic_activities'][-1] = 'Contrasting late reaction'
    assert review.digest(review.scientific_record(changed)) != review.digest(review.scientific_record(metadata))


def test_prior_target_judgment_is_recalled_and_invalidated_after_evidence_change(tmp_path):
    packet = setup(tmp_path)
    saved = review.save_assessment(tmp_path, packet, assessment('investigate'))
    recalled = recall_memory(tmp_path, 'reasoning', query='SYNTH1')['records']
    receipt = next(r['assessment_receipt'] for r in recalled if r['kind'] == 'target_assessment_recorded')
    assert receipt['is_current'] and receipt['artifact_verified']
    assert receipt['target_roles'] == saved['target_roles']
    assert receipt['revisit_when'] == saved['revisit_when']
    evidence = get_candidate_evidence('SYNTH1', tmp_path)
    assert compact_for_llm(evidence)['data']['assessment_receipt']['is_current']
    _, messages, _ = context(tmp_path, evidence)
    assert saved['revisit_when'] in json.dumps(messages)
    update_candidate_registry(tmp_path, 'SYNTH1', None, {
        'literature_evidence': [{'observation': 'A matched inactive control gives the same product signal.'}]})
    stale = get_candidate_evidence('SYNTH1', tmp_path)['data']['assessment_receipt']
    assert stale['artifact_verified'] and not stale['is_current'] and not stale['record_matches']
    assert stale['decision'] == 'investigate'  # History is retained, not rewritten.
    assert review.review_queue(tmp_path)[0]['accession'] == 'SYNTH1'
    packet = {'target': review.target_contract(tmp_path), 'target_inputs': review.target_inputs(review.target_contract(tmp_path)),
              'candidates': [{**review.review_queue(tmp_path)[0], 'source_id': 'R1'}]}
    corrected = {**assessment('defer'), 'reason': 'The new control leaves enzyme-dependent turnover unresolved.',
                 'evidence_level': 'speculative', 'strengths': []}
    review.save_assessment(tmp_path, packet, corrected)
    history = [r['assessment_receipt'] for r in recall_memory(tmp_path, 'reasoning', query='SYNTH1')['records']
               if r['kind'] == 'target_assessment_recorded']
    assert history[0]['decision'] == 'defer' and history[0]['is_current']
    assert not history[1]['is_current'] and not history[1]['is_latest']


def test_batched_passages_memory_and_failures_survive_the_actual_context(tmp_path):
    passage = 'The inactive control produced the same signal; target turnover remains unresolved.'
    rows = [
        {'tool_name': 'read_primary_paper', 'status': 'success', 'data': {
            'text_available': True, 'paper': {'title': 'Synthetic controls'}, 'passage': passage,
            'source_path': 'artifacts/fulltext/synthetic.txt', 'text_sha256': 'source_digest'}},
        {'tool_name': 'get_research_memory', 'status': 'success', 'data': {
            'records': [{'accession': 'SYNTH1', 'decision': 'defer'}], 'total': 2, 'next_offset': 1}},
        {'tool_name': 'search_records', 'status': 'error', 'summary': 'Source unavailable', 'data': None}]
    packet, messages, brief = context(tmp_path, {'status': 'success', 'data': rows})
    assert len(packet['previous_test']['result_views']) == 3
    assert passage in json.dumps(messages)
    assert brief['last_result_views'][1]['data']['next_offset'] == 1
    assert brief['last_result_views'][2]['source_status'] == 'error'
    source = brief['full_result_source']
    raw = (tmp_path/source['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256']
    assert json.loads(raw)['data'] == rows


@pytest.mark.parametrize('position', [0, 3, 9])
def test_contradictory_observation_survives_reordering_when_context_fits(tmp_path, position):
    facts = ['Native annotation ' + str(i) for i in range(10)]
    facts[position] = 'DECISIVE_CONTROL: the product signal is background.'
    _, messages, brief = context(tmp_path, {'status': 'success', 'data': {
        'accession': 'SYNTH1', 'protein_name': 'Synthetic catalyst', 'function_comments': facts}})
    assert 'DECISIVE_CONTROL' in json.dumps(messages)
    assert brief['last_lookup_records'][0]['function_comments'] == facts


def test_context_pressure_reports_omissions_and_provides_inspectable_complete_evidence(tmp_path):
    from enzyme_discovery_agent import config
    from enzyme_discovery_agent.tools.candidate_inputs import inspect_discovery_source
    facts = ['Source context ' * 200 for _ in range(50)] + ['DECISIVE_CONTROL: signal is background.']
    with patch.object(config, 'MODEL_CONTEXT_MAX_CHARS', 12000):
        _, messages, brief = context(tmp_path, {'status': 'success', 'data': {
            'accession': 'SYNTH1', 'protein_name': 'Synthetic catalyst', 'function_comments': facts}})
    assert len(json.dumps(messages, ensure_ascii=False)) <= 12000
    assert brief['omission_count'] > 0
    assert any(r.get('omitted_items') for r in brief['preview_omissions'])
    source = brief['full_result_source']['path']
    retrieved = inspect_discovery_source(source, find='DECISIVE_CONTROL', run_dir=tmp_path)
    assert retrieved['status'] == 'success' and 'DECISIVE_CONTROL' in retrieved['data']['passage']
    assert 'otherwise keep' in json.dumps(messages)


def test_preview_never_cuts_molecular_strings_or_source_identifiers():
    source = {'target_smiles': 'C'*500, 'accession': 'SYNTH_'+'X'*500,
              'source_path': 'artifacts/'+'a'*500, 'facts': ['a', 'b', 'c', 'd']}
    view, omissions = preview_evidence(source, text_limit=20, list_limit=2)
    assert all(view[k] == source[k] for k in ('target_smiles', 'accession', 'source_path'))
    assert omissions == [{'path': '/facts', 'total_items': 4, 'omitted_items': 2}]


def test_review_usage_is_bound_to_its_response_and_saved_assessments(tmp_path):
    from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
    setup(tmp_path)
    client = LLMClient(model='claude-code:synthetic-model')
    client.attach_usage(TokenUsage(), tmp_path)
    payload = {'tool_calls': [{'function': {'name': 'record_target_assessment', 'arguments': json.dumps(assessment())}}]}
    message = SimpleNamespace(tool_calls=payload['tool_calls'], content=None, model_dump=lambda: payload)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12))
    with patch.object(client, '_call_backend', return_value=response):
        output = review.run_review(tmp_path, client)
    assert output['assessed'] == ['SYNTH1']
    ledger = response._enzyme_request_usage
    batch = json.loads(next((tmp_path/'artifacts/target_reviews').glob('batch_*.json')).read_text())
    assert batch['request_usage']['request_id'] == ledger['request_id']
    saved = load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']['target_assessment']
    assert saved['model_provenance']['request']['request_id'] == ledger['request_id']


def test_source_only_correction_changes_controller_evidence_signal(tmp_path):
    setup(tmp_path)
    before = decisions._signal(decisions.observations(tmp_path)[0])
    update_candidate_registry(tmp_path, 'SYNTH1', None, {
        'function_comments': ['New matched control changes the interpretation without changing the native equation.']})
    assert decisions._signal(decisions.observations(tmp_path)[0]) != before
