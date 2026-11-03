"""One hypothesis must be able to retrieve every scope it judged relevant.

A plan that names several candidate scopes and can act on only one leaves the rest
untested while still paying the full deliberation cost, so breadth has to be a
property of a single registered prediction rather than of repeated ones.
"""
import json

import pytest

from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry


def evidence(root, count=20):
    for i in range(count):
        update_candidate_registry(root, f'SYNTH_{i}', 'evaluated', {
            'screening_family_id': 'SYNTH_FOLD', 'claim_validation': {'promotion_eligible': True},
            'substrate_similarity': {'accommodation': {
                'unaccommodated_targets': ['SYNTH_ROLE'],
                'all_target_substrates_accommodated': False}}})


def plan(root, **updates):
    context = decisions.decision_context(root)
    data = {'hypothesis': 'Several annotation scopes may each contain a compatible catalyst.',
            'alternative': 'Every scope may share the same disqualifying native direction.',
            'scope': 'search_strategy',
            'prediction': 'At least one scope returns members with the required reaction direction.',
            'contrasting_prediction': 'All scopes return only the opposite direction.',
            'why_this_test': 'It separates scope choice from direction compatibility.',
            'revision_trigger': 'Revisit if a scope returns a compatible member.',
            'evidence_refs': [next((row['id'] for row in context['observations']
                                    if row['kind'] != 'user_target'), context['observations'][0]['id'])],
            'tool_name': 'search_uniprot', 'arguments': {'query': 'first scope'}}
    data.update(updates)
    return data


ALLOWED = {'search_uniprot', 'search_modelseed_reactions', 'search_interpro', 'set_candidate_focus'}


def test_one_hypothesis_may_retrieve_several_scopes(tmp_path):
    evidence(tmp_path)
    proposal = plan(tmp_path, additional_actions=[
        {'tool_name': 'search_modelseed_reactions', 'arguments': {'query': 'second scope'}},
        {'tool_name': 'search_interpro', 'arguments': {'query': 'third scope'}}])
    validated = decisions.validate_plan(tmp_path, proposal, ALLOWED)
    actions = decisions.planned_actions(validated)
    assert [a['tool_name'] for a in actions] == [
        'search_uniprot', 'search_modelseed_reactions', 'search_interpro']
    assert [a['arguments']['query'] for a in actions] == ['first scope', 'second scope', 'third scope']


def test_registered_plan_persists_every_action(tmp_path):
    evidence(tmp_path)
    proposal = plan(tmp_path, additional_actions=[
        {'tool_name': 'search_interpro', 'arguments': {'query': 'second scope'}}])
    identifier = decisions.start_decision(tmp_path, proposal, ALLOWED)
    stored = json.loads((decisions._root(tmp_path) / identifier / 'plan.json').read_text())
    assert len(decisions.planned_actions(stored)) == 2


def test_priority_changes_cannot_share_a_hypothesis(tmp_path):
    """A single prediction must never record two separate verdicts."""
    evidence(tmp_path)
    with pytest.raises(ValueError, match='on its own'):
        decisions.validate_plan(tmp_path, plan(
            tmp_path, tool_name='set_candidate_focus', arguments={'accession': 'SYNTH_1'},
            additional_actions=[{'tool_name': 'search_uniprot', 'arguments': {'query': 'x'}}]), ALLOWED)
    with pytest.raises(ValueError, match='repeatable retrieval'):
        decisions.validate_plan(tmp_path, plan(tmp_path, additional_actions=[
            {'tool_name': 'set_candidate_focus', 'arguments': {'accession': 'SYNTH_1'}}]), ALLOWED)


def test_unavailable_actions_and_overlong_fanouts_are_refused(tmp_path):
    evidence(tmp_path)
    with pytest.raises(ValueError, match='not an available'):
        decisions.validate_plan(tmp_path, plan(tmp_path, additional_actions=[
            {'tool_name': 'search_kegg_enzyme', 'arguments': {'query': 'x'}}]), ALLOWED)
    with pytest.raises(ValueError, match='at most'):
        decisions.validate_plan(tmp_path, plan(tmp_path, additional_actions=[
            {'tool_name': 'search_interpro', 'arguments': {'query': f'q{i}'}} for i in range(8)]), ALLOWED)
    with pytest.raises(ValueError, match='tool_name'):
        decisions.validate_plan(tmp_path, plan(tmp_path, additional_actions=['search_interpro']), ALLOWED)


def test_duplicate_actions_collapse_and_absent_fanout_is_unchanged(tmp_path):
    evidence(tmp_path)
    duplicated = decisions.validate_plan(tmp_path, plan(tmp_path, additional_actions=[
        {'tool_name': 'search_uniprot', 'arguments': {'query': 'first scope'}},
        {'tool_name': 'search_interpro', 'arguments': {'query': 'second scope'}}]), ALLOWED)
    assert len(decisions.planned_actions(duplicated)) == 2
    single = decisions.validate_plan(tmp_path, plan(tmp_path), ALLOWED)
    assert 'additional_actions' not in single
    assert len(decisions.planned_actions(single)) == 1


def test_controller_executes_every_planned_action(tmp_path):
    """The schema change is inert unless the controller runs the whole fan-out."""
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from enzyme_discovery_agent.agent import EnzymeAgent
    evidence(tmp_path)
    agent = EnzymeAgent.__new__(EnzymeAgent)
    agent.run_dir, agent.tools, agent.llm = tmp_path, [], SimpleNamespace()
    agent._decision_action_tools = ALLOWED
    agent._require_research_decision = True
    agent._search_history, agent._search_queries, agent._search_log = [], [], []
    agent.log, agent._log_event = Mock(), Mock()
    proposal = plan(tmp_path, additional_actions=[
        {'tool_name': 'search_modelseed_reactions', 'arguments': {'query': 'second scope'}},
        {'tool_name': 'search_interpro', 'arguments': {'query': 'third scope'}}])

    performed = []

    def run(name, args, injected_args=None):
        performed.append((name, dict(args)))
        return {'status': 'success', 'summary': f'{name} returned hits', 'data': {'hits': [name]}}

    with patch('enzyme_discovery_agent.agent.execute_tool', side_effect=run):
        result = agent._execute_single_tool('plan_research_step', proposal, 1)

    assert [name for name, _ in performed] == [
        'search_uniprot', 'search_modelseed_reactions', 'search_interpro']
    assert [args['query'] for _, args in performed] == ['first scope', 'second scope', 'third scope']
    # All three outcomes must reach the record the prediction is judged against.
    assert result['status'] == 'success'
    assert len(result['data']) == 3
    assert result['research_decision'] and result['outcome_evidence'].startswith('evidence:')


def test_one_failed_retrieval_does_not_discard_the_others(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from enzyme_discovery_agent.agent import EnzymeAgent
    evidence(tmp_path)
    agent = EnzymeAgent.__new__(EnzymeAgent)
    agent.run_dir, agent.tools, agent.llm = tmp_path, [], SimpleNamespace()
    agent._decision_action_tools = ALLOWED
    agent._require_research_decision = True
    agent._search_history, agent._search_queries, agent._search_log = [], [], []
    agent.log, agent._log_event = Mock(), Mock()
    proposal = plan(tmp_path, additional_actions=[
        {'tool_name': 'search_interpro', 'arguments': {'query': 'second scope'}}])

    def run(name, args, injected_args=None):
        if name == 'search_uniprot':
            raise RuntimeError('upstream timeout')
        return {'status': 'success', 'summary': 'hits', 'data': {'hits': [name]}}

    with patch('enzyme_discovery_agent.agent.execute_tool', side_effect=run):
        result = agent._execute_single_tool('plan_research_step', proposal, 1)

    assert result['status'] == 'success'
    statuses = [row['status'] for row in result['data']]
    assert statuses == ['error', 'success']
