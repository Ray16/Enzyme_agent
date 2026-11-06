import copy
import hashlib
import json

from enzyme_discovery_agent.output.trace_audit import enrich_trace


def saved(root, name, value):
    path = root/name
    path.write_text(json.dumps(value))
    return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def case(root, name='one'):
    source = saved(root, name+'.json', {'tool_name': 'read_record', 'arguments': {'id': 'SYNTH_A'},
        'additional_actions': [{'tool_name': 'search_records', 'arguments': {'query': 'synthetic'}}]})
    result = saved(root, name+'_result.json', {'status': 'success', 'data': [
        {'tool_name': 'read_record', 'arguments': {'id': 'SYNTH_A'}, 'status': 'success',
         'data': {'accession': 'SYNTH_A', 'protein_name': 'Synthetic catalyst',
                  'function_comments': ['Native observation; target activity remains unknown.'],
                  'sequence': 'OMITTED_FROM_DISPLAY'}},
        {'tool_name': 'search_records', 'arguments': {'query': 'synthetic'}, 'status': 'success', 'data': []}]})
    return {'decision_id': name, 'source': source,
            'action': {'tool': 'read_record', 'arguments': {'id': 'SYNTH_A'}},
            'observation': {'result_source': result}, 'revises': 'prior',
            'model_provenance': {'request': {'request_id': 'shared'}}}


def test_verified_batches_empty_results_and_shared_request_costs_are_visible(tmp_path):
    cases = [case(tmp_path), case(tmp_path, 'two')]
    (tmp_path/'token_usage.json').write_text(json.dumps({'total_tokens': 500, 'calls': [
        {'request_id': 'shared', 'prompt_tokens': 80, 'completion_tokens': 20, 'total_tokens': 100}]}))
    events = [{'id': 'original', 'record': {}, 'type': 'unknown'}]
    before = copy.deepcopy(events)
    details, summary = enrich_trace(tmp_path, events, cases)
    assert events == before
    assert summary['planned_actions'] == 4
    assert summary['linked_request_tokens'] == 100
    assert summary['unique_linked_requests'] == 1
    audit = details['prediction_registered:one']
    assert audit['request_usage']['decisions_sharing_request'] == 2
    assert audit['outcomes'][1]['returned_records'] == 0
    assert 'sequence' not in audit['outcomes'][0]['candidate_facts']
    assert any('belief-update' in issue for issue in audit['issues'])
    assert details['prediction_registered:two']['repeated_actions']


def test_changed_artifact_is_not_used_to_rewrite_a_recorded_plan(tmp_path):
    item = case(tmp_path)
    (tmp_path/'one.json').write_text('{"tool_name":"changed"}')
    (tmp_path/'one_result.json').write_text('{"status":"changed"}')
    details, summary = enrich_trace(tmp_path, [], [item])
    audit = details['prediction_registered:one']
    assert audit['actions'] == [item['action']]
    assert audit['outcomes'] == []
    assert audit['request_usage'] is None
    assert summary['decisions_with_usage'] == 0
    assert any('changed' in issue for issue in audit['issues'])


def test_candidate_name_is_derived_from_its_recorded_source_not_model_memory(tmp_path):
    event = {'id': 'assessment', 'type': 'target_assessment_recorded', 'record': {
        'accession': 'SYNTH_A', 'decision': 'defer',
        'sources': {'R1': {'protein_name': 'Synthetic catalyst'}}}}
    details, summary = enrich_trace(tmp_path, [event], [])
    assert details['assessment']['candidate_name'] == 'Synthetic catalyst'
    assert summary['assessment_decisions'] == {'defer': 1}


def test_assessment_links_use_explicit_inspected_accessions(tmp_path):
    inspected = case(tmp_path)
    inspected['source'] = saved(tmp_path, 'one.json', {'tool_name': 'read_record',
        'arguments': {'accession': 'SYNTH_A'}})
    other = case(tmp_path, 'other')
    event = {'id': 'assessment', 'type': 'target_assessment_recorded', 'occurred_at': 'recorded_time',
        'record': {'accession': 'SYNTH_A', 'decision': 'defer', 'sources': {'R1': {'protein_name': 'Synthetic catalyst'}}}}
    details, _ = enrich_trace(tmp_path, [event], [inspected, other])
    assert details['prediction_registered:one']['candidate_assessments'][0]['event_id'] == 'assessment'
    assert details['prediction_registered:other']['candidate_assessments'] == []


def test_review_costs_join_exact_requests_without_double_counting_candidate_records(tmp_path):
    (tmp_path/'token_usage.json').write_text(json.dumps({'total_tokens': 150, 'calls': [
        {'request_id': 'review', 'total_tokens': 100}, {'request_id': 'other', 'total_tokens': 50}]}))
    events = [{'id': str(i), 'type': 'target_assessment_recorded', 'record': {
        'accession': 'SYNTH_'+str(i), 'model_provenance': {'request': {'request_id': 'review'}}}}
        for i in range(2)]
    events.append({'id': 'batch', 'type': 'target_review_batch_recorded',
                   'record': {'request_usage': {'request_id': 'review'}}})
    details, summary = enrich_trace(tmp_path, events, [])
    assert summary['target_review_linked_requests'] == 1
    assert summary['target_review_linked_tokens'] == summary['all_linked_request_tokens'] == 100
    assert details['batch']['request_usage']['request_id'] == 'review'
