import copy
import json

import pytest

from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.tests.test_scientific_trace import proposal


@pytest.mark.parametrize('batched', [False, True])
def test_candidate_chemistry_survives_saved_observation_and_next_model_context(tmp_path, monkeypatch, batched):
    from enzyme_discovery_agent.context import build_model_context
    facts = {'accession': 'SYNTH_A', 'protein_name': 'Synthetic oxidation catalyst',
             'catalytic_activities': ['Ethanol + oxygen produces aldehyde and peroxide.'],
             'function_comments': ['Observed acceptance of partner B; requested partner A remains untested.'],
             'native_substrates': [{'smiles': 'CCO'}, {'smiles': 'O=O'}],
             'critical_warnings': ['Native turnover does not establish the complete requested reaction.'],
             'sequence': 'SHOULD_NOT_BE_IN_DECISION_PREVIEW'}
    identifier = decisions.start_decision(tmp_path, proposal(tmp_path), {'synthetic_retrieval'})
    payload = ([{'tool_name': 'get_candidate_evidence', 'arguments': {'accession': 'SYNTH_A'},
                 'status': 'success', 'data': facts}] if batched else facts)
    decisions.finish_decision(tmp_path, identifier, {'status': 'success', 'data': payload})
    packet = decisions.decision_context(tmp_path)
    examples = packet['previous_test']['result_examples']
    assert len(examples) == 1
    assert examples[0]['function_comments'] == facts['function_comments']
    assert examples[0]['native_substrates'] == facts['native_substrates']
    assert 'sequence' not in examples[0]
    if batched:
        assert examples[0]['source_status'] == 'success'
    packet['review_due'] = True
    monkeypatch.setattr('enzyme_discovery_agent.context.build_research_state',
                        lambda root: {'adaptive_reasoning': copy.deepcopy(packet)})
    context = build_model_context([{'role': 'system', 'content': 'Respect evidence scope.'},
                                   {'role': 'user', 'content': 'Assess the synthetic target.'}], tmp_path)
    sent = json.dumps(context)
    assert facts['function_comments'][0] in sent
    assert facts['critical_warnings'][0] in sent
    assert 'SYNTH_A' in sent and 'CCO' in sent and 'O=O' in sent
    assert 'SHOULD_NOT_BE_IN_DECISION_PREVIEW' not in sent


def test_multiple_candidate_objects_and_search_lists_survive_the_same_batch(tmp_path):
    identifier = decisions.start_decision(tmp_path, proposal(tmp_path), {'synthetic_retrieval'})
    decisions.finish_decision(tmp_path, identifier, {'status': 'success', 'data': [
        {'tool_name': 'get_candidate_evidence', 'data': {'accession': 'SYNTH_A', 'function_comments': ['Observation A.']}},
        {'tool_name': 'get_candidate_evidence', 'data': {'accession': 'SYNTH_B', 'function_comments': ['Contrasting observation B.']}},
        {'tool_name': 'search_records', 'data': [{'accession': 'SYNTH_C', 'protein_name': 'Synthetic catalyst'}]}]})
    rows = decisions.decision_context(tmp_path)['previous_test']['result_examples']
    assert [row['accession'] for row in rows] == ['SYNTH_A', 'SYNTH_B', 'SYNTH_C']
    assert rows[1]['function_comments'] == ['Contrasting observation B.']


@pytest.mark.parametrize('fresh', [False, True])
def test_prior_assessments_remain_visible_after_skipped_or_partial_batch(tmp_path, monkeypatch, fresh):
    from enzyme_discovery_agent.tools.research_decisions import assess_next_batch
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
    from enzyme_discovery_agent.context import build_model_context
    update_candidate_registry(tmp_path, 'SYNTH_OLD', 'evaluated', {
        'protein_name': 'Previously evaluated catalyst', 'catalytic_activities': ['Recorded native transformation.']})
    update_candidate_registry(tmp_path, 'SYNTH_PENDING', 'evidence_pending', {
        'protein_name': 'Unresolved catalyst', 'evaluation_data': {'claim_validation': {'blocking_claims': ['native_reaction']}}})
    selected = ['SYNTH_OLD', 'SYNTH_PENDING']
    if fresh:
        selected.extend(['SYNTH_NEW', 'SYNTH_WAIT'])
        update_candidate_registry(tmp_path, 'SYNTH_NEW', 'screened', {'protein_name': 'New catalyst'})
        update_candidate_registry(tmp_path, 'SYNTH_WAIT', 'screened', {'protein_name': 'Waiting catalyst'})
    def evaluate(root, **kwargs):
        assert kwargs['fresh_only'] is True
        kwargs['stats'].update(attempted=int(fresh), supported=int(fresh), evidence_pending=0,
                               rejected=0, failed=0, accessions_attempted=['SYNTH_NEW'] if fresh else [])
        if fresh:
            update_candidate_registry(root, 'SYNTH_NEW', 'evaluated', {'catalytic_activities': ['New native observation.']})
    monkeypatch.setattr('enzyme_discovery_agent.gates.ec_triage._auto_evaluate_all', evaluate)
    result = assess_next_batch(tmp_path, accessions=selected)
    assert result['status'] == ('success' if fresh else 'skipped')
    assert result['data']['attempted'] == int(fresh)
    existing = result['data']['existing_assessments']
    assert {row['accession'] for row in existing} == ({'SYNTH_OLD', 'SYNTH_PENDING', 'SYNTH_WAIT'}
                                                    if fresh else {'SYNTH_OLD', 'SYNTH_PENDING'})
    if fresh:
        assert next(row for row in existing if row['accession']=='SYNTH_WAIT')['status'] == 'screened'
    assert all(row['protein_name'] and row['work_status'] == 'not_evaluated_in_this_batch' for row in existing)
    assert next(row for row in existing if row['accession']=='SYNTH_PENDING')['missing_claims'] == ['native_reaction']
    if not fresh:
        assert result['data']['new_evidence'] is False
    identifier = decisions.start_decision(tmp_path, proposal(tmp_path), {'synthetic_retrieval'})
    decisions.finish_decision(tmp_path, identifier, result)
    packet = decisions.decision_context(tmp_path)
    assert packet['previous_test']['assessment']['existing_assessments'] == existing
    packet['review_due'] = True
    monkeypatch.setattr('enzyme_discovery_agent.context.build_research_state',
                        lambda root: {'adaptive_reasoning': copy.deepcopy(packet)})
    context = build_model_context([{'role':'system','content':'Keep evidence scoped.'},
                                   {'role':'user','content':'Review the saved work.'}], tmp_path)
    sent = json.dumps(context)
    assert 'Previously evaluated catalyst' in sent and 'Unresolved catalyst' in sent
    assert 'not_evaluated_in_this_batch' in sent
    if not fresh:
        assert '"new_evidence": false' in '\n'.join(m.get('content','') for m in context)
