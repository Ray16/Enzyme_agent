"""Request compression must preserve evidence and observable accounting."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

from enzyme_discovery_agent.request_efficiency import (
    expand_evidence, factor_evidence, request_metrics, serialized,
)


def evidence():
    return {'observation': 'Synthetic source passage with explicit uncertainty. ' * 40,
            'verdict': 'insufficient_evidence', 'native_activity': None,
            'occurrences': ['first', 'first'], 'source': {'locator': 'section 2'}}


def test_factoring_is_lossless_and_never_mutates_or_merges_occurrences():
    fact = evidence()
    candidates = [{'accession': 'SYNTH_A', 'source_id': 'R1', 'facts': [fact, fact]},
                  {'accession': 'SYNTH_B', 'source_id': 'R2', 'facts': [fact], 'missing': None}]
    before = copy.deepcopy(candidates)
    request = factor_evidence(candidates)
    assert request['shared_evidence']
    assert len(serialized(request)) < len(serialized({'candidates': candidates}))
    assert expand_evidence(request) == candidates == before
    assert request == factor_evidence(candidates)
    expanded = expand_evidence(request)
    expanded[0]['facts'][0]['verdict'] = 'changed'
    assert expanded[0]['facts'][1]['verdict'] == 'insufficient_evidence'
    assert candidates == before


def test_factoring_is_local_and_preserves_conflicting_facts():
    fact = evidence()
    conflict = {**fact, 'verdict': 'contradicted'}
    candidates = [{'facts': [fact, fact, conflict], 'accession': 'SYNTH_A'}]
    assert expand_evidence(factor_evidence(candidates)) == candidates
    assert factor_evidence([{'facts': [conflict]}]) == {'candidates': [{'facts': [conflict]}]}


def test_lists_of_references_are_not_encoded_as_tables_of_literal_references():
    candidates = [{'facts': [evidence()] * 8}]
    packed = factor_evidence(candidates)
    assert expand_evidence(packed) == candidates


def test_factoring_falls_back_for_small_payloads_and_reserved_keys():
    for candidates in ([{'value': 1}] * 3,
                       [{'facts': [evidence()] * 3, '$evidence_ref': 'external'}]):
        assert factor_evidence(candidates) == {'candidates': candidates}


def test_table_encoding_preserves_nulls_missing_fields_and_column_associations():
    records = [{'long_observation_field': f'Observation {i}',
                'long_uncertainty_field': None, 'long_verdict_field': 'unknown',
                'long_occurrence_field': i} for i in range(30)]
    candidates = [{'accession': 'SYNTH_A', 'claims': records,
                   'heterogeneous': [{'a': None}, {}, {'b': None}]}]
    packed = factor_evidence(candidates)
    assert '$evidence_table' in serialized(packed)
    assert expand_evidence(packed) == candidates
    assert len(serialized(packed)) < len(serialized({'candidates': candidates}))


def test_profile_contains_sizes_and_hashes_without_source_content():
    messages = [{'role': 'system', 'content': 'private instruction'},
                {'role': 'user', 'content': 'private evidence'}]
    tools = [{'type': 'function', 'function': {'name': 'synthetic_tool'}}]
    metrics = request_metrics(messages, tools)
    assert metrics['messages_characters'] == len(serialized(messages))
    assert metrics['tools_characters'] == len(serialized(tools))
    assert 'private' not in json.dumps(metrics)
    changed = request_metrics(messages + [{'role': 'tool', 'content': 'new fact'}], tools)
    assert changed['message_components'][:2] == metrics['message_components']
    assert changed['tools_sha256'] == metrics['tools_sha256']


def test_client_profile_links_to_provider_usage_without_changing_payload(tmp_path):
    from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
    from enzyme_discovery_agent.run_layout import run_file
    client = LLMClient(model='claude-code:synthetic-model')
    usage = TokenUsage()
    client.attach_usage(usage, tmp_path)
    messages = [{'role': 'user', 'content': 'Synthetic request'}]
    response = SimpleNamespace(choices=[], usage=SimpleNamespace(
        prompt_tokens=10, completion_tokens=2, total_tokens=12))
    with patch.object(client, '_call_backend', return_value=response) as backend:
        assert client.call(messages, usage_label='synthetic') is response
    assert backend.call_args.args[1] is messages
    events = [json.loads(line) for line in run_file(tmp_path, 'events.jsonl').read_text().splitlines()]
    profile = next(e for e in events if e['type'] == 'llm_request_profile')
    call = next(e for e in events if e['type'] == 'llm_call')
    assert call['request_profile_id'] == profile['request_profile_id']
    assert usage.total_tokens == 12


def test_review_factoring_only_changes_request_not_saved_packet(tmp_path, monkeypatch):
    from enzyme_discovery_agent import config
    from enzyme_discovery_agent.models import target_review as review
    from enzyme_discovery_agent.tests.test_target_review_gate import setup, assessment, _StubLLM
    packet = setup(tmp_path)
    packet['candidates'][0]['record']['repeated_evidence'] = [evidence()] * 3
    # Queue projection alone is varied: the canonical record hashes stay unchanged.
    monkeypatch.setattr(review, 'review_queue', lambda *a, **kw: copy.deepcopy(packet['candidates']))
    monkeypatch.setattr(config, 'TARGET_REVIEW_FACTOR_EVIDENCE', True)
    llm = _StubLLM(assessment())
    original_call = llm.call
    captured = []
    def call(**kwargs):
        captured.append(json.loads(kwargs['messages'][-1]['content']))
        return original_call(**kwargs)
    llm.call = call
    result = review.run_review(tmp_path, llm)
    assert result['assessed'] == ['SYNTH1']
    batch = json.loads(next((tmp_path/'artifacts/target_reviews').glob('batch_*.json')).read_text())
    expected = [{k: v for k, v in row.items() if k not in
                 {'record_sha256', 'eligibility_sha256', 'target_sha256'}}
                for row in batch['packet']['candidates']]
    assert captured[0]['shared_evidence']
    assert expand_evidence(captured[0]) == expected
    assert batch['packet']['candidates'] == packet['candidates']


def test_live_replay_capture_uses_same_prompt_and_evidence_without_writes(tmp_path, monkeypatch):
    from enzyme_discovery_agent.eval.request_efficiency_live import capture_request
    packet = {'target': {'substrate_smiles': 'CCO.CC=O'},
              'candidates': [{'accession': 'SYNTH_A', 'source_id': 'R1',
                              'record': {'facts': [evidence()] * 8}}]}
    before = copy.deepcopy(packet)
    monkeypatch.chdir(tmp_path)
    baseline = capture_request(packet, False)
    factored = capture_request(packet, True)
    assert baseline['tools'] == factored['tools']
    assert baseline['messages'][:2] == factored['messages'][:2]
    assert expand_evidence(json.loads(baseline['messages'][-1]['content'])) == expand_evidence(
        json.loads(factored['messages'][-1]['content']))
    assert packet == before
    assert list(tmp_path.iterdir()) == []


def test_live_replay_reports_decorated_identifiers_and_unsupported_positive_decisions():
    from enzyme_discovery_agent.eval.request_efficiency_live import capture_request, response_contract_errors
    from enzyme_discovery_agent.tests.test_target_review_gate import assessment
    request = capture_request({'target': {'substrate_smiles': 'CCO.CC=O'},
                               'candidates': [{'accession': 'SYNTH1', 'source_id': 'R1', 'record': {}}]}, False)
    def message(result):
        return {'tool_calls': [{'function': {'name': 'record_target_assessment',
                                            'arguments': json.dumps(result)}}]}
    assert response_contract_errors(message(assessment()), request) == []
    invalid = assessment(); invalid['accession'] += ' (Synthetic catalyst)'
    assert 'Unknown or duplicate accession' in response_contract_errors(message(invalid), request)
    invalid = assessment(); invalid['evidence_level'] = 'speculative'
    assert 'Unsupported positive decision' in response_contract_errors(message(invalid), request)
