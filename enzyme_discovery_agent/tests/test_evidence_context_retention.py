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
