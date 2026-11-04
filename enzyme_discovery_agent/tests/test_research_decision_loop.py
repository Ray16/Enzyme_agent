"""Evidence-linked decisions must change execution, not just produce prose."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry, discovery_status
from enzyme_discovery_agent.models.family_decisions import record_family_decisions
from enzyme_discovery_agent.models.rolling import fresh_evaluation_backlog
from enzyme_discovery_agent import config as _config


@pytest.fixture(autouse=True)
def _enable_adaptive_reasoning(monkeypatch):
    # This module exercises the adaptive deliberation loop, which is no longer the
    # default (direct discovery is now the default; see config.ADAPTIVE_REASONING).
    # Pin it on so these tests exercise the behaviour they are written for.
    monkeypatch.setattr(_config, "ADAPTIVE_REASONING", True)


def evidence(root, count=20, mismatch=True):
    for i in range(count):
        update_candidate_registry(root, f'SYNTH_{i}', 'evaluated', {
            'screening_family_id': 'SYNTH_FOLD', 'claim_validation': {'promotion_eligible': True},
            'substrate_similarity': {'accommodation': {
                'unaccommodated_targets': ['SYNTH_ROLE'] if mismatch else [],
                'all_target_substrates_accommodated': not mismatch}}})


def plan(root, **updates):
    context = decisions.decision_context(root)
    data = {'hypothesis': 'The retrieved family may accommodate every target input.',
            'alternative': 'Its native chemistry may leave a required input role unsupported.',
            'scope': 'search_strategy', 'prediction': 'Focused evidence will establish a counterpart for the missing role.',
            'contrasting_prediction': 'The evidence will retain a missing role despite shared reaction direction.',
            'why_this_test': 'It separates native reaction similarity from compatibility with all inputs.',
            'revision_trigger': 'Revisit the strategy if a compatible member or contradictory primary record appears.',
            'evidence_refs': [next((row['id'] for row in context['observations'] if row['kind'] != 'user_target'), context['observations'][0]['id'])],
            'tool_name': 'inspect_synthetic_evidence', 'arguments': {}}
    data.update(updates)
    return data


def test_new_native_mismatch_interrupts_but_repeated_variants_do_not_manufacture_new_signal(tmp_path):
    evidence(tmp_path)
    assert decisions.review_due(tmp_path)
    before = decisions._signal(decisions.observations(tmp_path)[0])
    evidence(tmp_path, count=25)
    assert decisions._signal(decisions.observations(tmp_path)[0]) == before
    other = tmp_path / 'compatible'
    evidence(other, mismatch=False)
    assert not decisions.review_due(other)


def test_predictions_registered_before_action_and_revision_cites_actual_outcome(tmp_path):
    evidence(tmp_path)
    proposal = plan(tmp_path)
    identifier = decisions.start_decision(tmp_path, proposal, {'inspect_synthetic_evidence'})
    folder = decisions._root(tmp_path) / identifier
    assert json.loads((folder/'plan.json').read_text())['prediction'] == proposal['prediction']
    assert not (folder/'outcome.json').exists()
    outcome = decisions.finish_decision(tmp_path, identifier, {'status': 'error', 'summary': 'synthetic transport unavailable'})
    assert decisions.review_due(tmp_path)
    with pytest.raises(ValueError, match='previous'):
        decisions.validate_plan(tmp_path, plan(tmp_path), {'inspect_synthetic_evidence'})
    revised = plan(tmp_path, revises=identifier, evidence_refs=[outcome], belief_update='suspend')
    assert decisions.validate_plan(tmp_path, revised, {'inspect_synthetic_evidence'})
    with pytest.raises(ValueError, match='biological'):
        decisions.validate_plan(tmp_path, {**revised, 'scope': 'native_function', 'belief_update': 'revise'}, {'inspect_synthetic_evidence'})


def test_forged_or_mutated_evidence_and_unavailable_actions_are_rejected(tmp_path):
    proposal = plan(tmp_path)
    with pytest.raises(ValueError, match='available'):
        decisions.validate_plan(tmp_path, proposal, set())
    digest = proposal['evidence_refs'][0].split(':')[1]
    (decisions._root(tmp_path)/'evidence'/f'{digest}.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        decisions.validate_plan(tmp_path, proposal, {'inspect_synthetic_evidence'})


def test_deferred_work_is_retained_and_explicitly_reopenable(tmp_path):
    update_candidate_registry(tmp_path, 'SYNTH_A', 'screened', {'screening_family_id': 'SYNTH_FOLD'})
    assert fresh_evaluation_backlog(tmp_path) == ['SYNTH_A']
    record_family_decisions(tmp_path, defer=['SYNTH_FOLD'], reasons={'SYNTH_FOLD': 'Unresolved substrate counterpart; investigate another family first.'})
    assert fresh_evaluation_backlog(tmp_path) == []
    assert discovery_status(load_candidate_registry(tmp_path)['candidates']['SYNTH_A']) == 'screened'
    record_family_decisions(tmp_path, expand=['SYNTH_FOLD'], reasons={'SYNTH_FOLD': 'New evidence warrants another test.'})
    assert fresh_evaluation_backlog(tmp_path) == ['SYNTH_A']


def test_annotation_scope_changes_work_without_stopping_other_members_of_broad_fold(tmp_path):
    for acc, gene in [('SYNTH_A', 'alpha'), ('SYNTH_B', 'beta')]:
        update_candidate_registry(tmp_path, acc, 'screened', {
            'screening_family_id': 'SYNTH_BROAD_FOLD',
            'screening_annotation': {'gene_names': [gene], 'ec_numbers': ['1.2.3.4']}})
    record_family_decisions(tmp_path, defer=['annotation:alpha|1.2.3'],
                            reasons={'annotation:alpha|1.2.3': 'Unresolved substrate role.'})
    assert fresh_evaluation_backlog(tmp_path) == ['SYNTH_B']
    assert set(load_candidate_registry(tmp_path)['candidates']) == {'SYNTH_A', 'SYNTH_B'}


def test_controller_executes_selected_action_after_prediction_is_saved(tmp_path):
    from enzyme_discovery_agent.agent import EnzymeAgent
    agent = EnzymeAgent.__new__(EnzymeAgent)
    agent.run_dir, agent.tools, agent.llm = tmp_path, [], SimpleNamespace()
    agent._decision_action_tools = {'inspect_synthetic_evidence'}
    agent._require_research_decision = True
    agent._search_history, agent._search_queries, agent._search_log = [], [], []
    agent.log, agent._log_event = Mock(), Mock()
    proposal = plan(tmp_path)
    def inspect(name, args, injected_args=None):
        saved = list(decisions._root(tmp_path).glob('*/plan.json'))
        assert len(saved) == 1 and not saved[0].with_name('outcome.json').exists()
        return {'status': 'success', 'summary': 'Synthetic observation saved.', 'data': {'value': 1}}
    with patch('enzyme_discovery_agent.agent.execute_tool', side_effect=inspect) as execute:
        result = agent._execute_single_tool('plan_research_step', proposal, 1)
    execute.assert_called_once()
    assert result['research_decision']
    assert result['outcome_evidence'].startswith('evidence:')


def test_rejected_proposal_returns_current_citation_and_revision_ids(tmp_path):
    from enzyme_discovery_agent.agent import EnzymeAgent
    agent=EnzymeAgent.__new__(EnzymeAgent)
    agent.run_dir,agent.tools,agent.llm=tmp_path,[],SimpleNamespace()
    agent._decision_action_tools={'inspect_synthetic_evidence'}
    agent._require_research_decision=True
    agent._search_history,agent._search_queries,agent._search_log=[],[],[]
    agent.log,agent._log_event=Mock(),Mock()
    rejected=agent._execute_single_tool('plan_research_step',plan(tmp_path,evidence_refs=['evidence:'+'0'*64]),1)
    assert rejected['decision_executed'] is False
    repair=rejected['repair_context']
    valid=repair['available_evidence_refs'][0]['id']
    with patch('enzyme_discovery_agent.agent.execute_tool',return_value={'status':'success','summary':'Synthetic evidence retrieved','data':{'observed':True}}):
        executed=agent._execute_single_tool('plan_research_step',plan(tmp_path,evidence_refs=[valid]),2)
    assert executed['research_decision']
    rejected=agent._execute_single_tool('plan_research_step',plan(tmp_path,evidence_refs=[valid]),3)
    assert rejected['repair_context']['revises']==executed['research_decision']
    assert rejected['repair_context']['previous_outcome_evidence_id']==executed['outcome_evidence']


def test_whole_turn_reviews_contradiction_before_draining_backlog_or_expanding(tmp_path):
    from openai.types.chat import ChatCompletion
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.tool_registry import get_tool_info
    import enzyme_discovery_agent.tools.research_decisions  # register controller tools
    import enzyme_discovery_agent.tools.family_reactions  # register the chosen action
    tools = [get_tool_info(name)['schema'] for name in ('plan_research_step', 'get_decision_context', 'set_family_focus')]
    with patch('enzyme_discovery_agent.agent.register_builtin_tools'), \
         patch('enzyme_discovery_agent.agent.get_all_tools', return_value=tools), \
         patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning', return_value={}), \
         patch('enzyme_discovery_agent.models.recovery.next_recovery_action', return_value=None):
        agent = EnzymeAgent(output_dir=tmp_path, token_budget=100)
        evidence(agent.run_dir)
        update_candidate_registry(agent.run_dir, 'SYNTH_PENDING', 'screened', {'screening_family_id': 'SYNTH_FOLD'})
        calls = []
        def respond(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) > 1:
                assert fresh_evaluation_backlog(agent.run_dir) == []
                agent.llm.resource_stop_reason = 'Synthetic adaptive test complete'
                return None
            proposal = plan(agent.run_dir, tool_name='set_family_focus', arguments={
                'defer': ['SYNTH_FOLD'], 'reasons': {'SYNTH_FOLD': 'Missing native counterpart; defer pending work without rejecting activity.'}})
            return ChatCompletion.model_validate({'id':'synthetic', 'object':'chat.completion', 'created':0, 'model':'synthetic',
                'choices':[{'index':0, 'finish_reason':'tool_calls', 'message':{'role':'assistant', 'content':None,
                'tool_calls':[{'id':'synthetic_call', 'type':'function', 'function':{
                    'name':'plan_research_step', 'arguments':json.dumps(proposal)}}]}}],
                'usage':{'prompt_tokens':1, 'completion_tokens':1, 'total_tokens':2}})
        with patch.object(agent.llm, 'call', side_effect=respond), \
             patch('enzyme_discovery_agent.gates.ec_triage._auto_evaluate_all') as evaluate, \
             patch('enzyme_discovery_agent.tools.family_reactions.interpro_name', return_value='Synthetic family'), \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot', return_value=('checkpoint', 20)):
            agent.run('synthetic reaction', substrate_smiles='CCO')
        evaluate.assert_not_called()
        assert len(calls) == 2
        assert list(decisions._root(agent.run_dir).glob('*/outcome.json'))


def test_live_eval_serializes_native_subscription_response():
    from enzyme_discovery_agent.claude_subscription import Response, Choice, Message, Usage
    from enzyme_discovery_agent.eval.adaptive_evals import response_record
    response = Response([Choice(Message(None, [{'id': 'synthetic', 'function': {'name': 'plan_research_step', 'arguments': '{}'}}]), 'tool_calls')], Usage(10, 5))
    record = response_record(response)
    assert record['message']['tool_calls'][0]['function']['name'] == 'plan_research_step'
    assert record['usage']['prompt_tokens'] == 10


def test_revision_can_compare_old_and_current_evidence_but_cannot_reuse_stale_as_current(tmp_path):
    evidence(tmp_path)
    initial = plan(tmp_path)
    identifier = decisions.start_decision(tmp_path, initial, {'inspect_synthetic_evidence'})
    outcome = decisions.finish_decision(tmp_path, identifier, {'status': 'success', 'summary': 'Synthetic first test', 'data': {'observed': True}})
    accession = next(row['source']['accession'] for row in decisions.decision_context(tmp_path)['observations'] if row['id'] == initial['evidence_refs'][0])
    update_candidate_registry(tmp_path, accession, None, {'substrate_similarity': {'accommodation': {
        'unaccommodated_targets': [], 'all_target_substrates_accommodated': True}}})
    context = decisions.decision_context(tmp_path)
    current = next(row['id'] for row in context['observations'] if row.get('source', {}).get('accession') == accession)
    revised = plan(tmp_path, revises=identifier, belief_update='revise',
                   evidence_refs=[outcome, current, initial['evidence_refs'][0]])
    assert decisions.validate_plan(tmp_path, revised, {'inspect_synthetic_evidence'})
    with pytest.raises(ValueError, match='historically'):
        decisions.validate_plan(tmp_path, {**revised, 'evidence_refs': [outcome, initial['evidence_refs'][0]]}, {'inspect_synthetic_evidence'})


def test_nested_native_mcp_name_maps_only_to_an_exposed_tool(tmp_path):
    proposal = plan(tmp_path, tool_name='mcp__discovery__inspect_synthetic_evidence')
    normalized = decisions.validate_plan(tmp_path, proposal, {'inspect_synthetic_evidence'})
    assert normalized['tool_name'] == 'inspect_synthetic_evidence'
    identifier = decisions.start_decision(tmp_path, proposal, {'inspect_synthetic_evidence'})
    saved = json.loads((decisions._root(tmp_path)/identifier/'plan.json').read_text())
    assert saved['requested_tool_name'] == proposal['tool_name']
    with pytest.raises(ValueError, match='available'):
        decisions.validate_plan(tmp_path, proposal, {'different_tool'})
    with pytest.raises(ValueError, match='available'):
        decisions.validate_plan(tmp_path, {**proposal, 'tool_name':'mcp__other__inspect_synthetic_evidence'}, {'inspect_synthetic_evidence'})


def test_scoped_batch_reports_what_it_actually_sampled(tmp_path):
    from enzyme_discovery_agent.tools.research_decisions import assess_next_batch
    for acc, gene in [('SYNTH_A', 'alpha'), ('SYNTH_B', 'beta')]:
        update_candidate_registry(tmp_path, acc, 'screened', {'gene_names': [gene], 'ec_numbers': ['1.2.3.4']})
    def evaluate(root, accession, **kwargs):
        update_candidate_registry(root, accession, 'evaluated')
        return {'status': 'success', 'data': {'accession': accession}}
    with patch('enzyme_discovery_agent.tools.reasoning.evaluate_candidate', side_effect=evaluate):
        result = assess_next_batch(tmp_path, family_scopes=['annotation:alpha|1.2.3'])
    assert result['count'] == 1
    assert result['data']['actual_sample'][0]['accession'] == 'SYNTH_A'
    assert discovery_status(load_candidate_registry(tmp_path)['candidates']['SYNTH_B']) == 'screened'
    assert 'did not sample' in result['data']['scope_limit']


def test_recorded_decision_reference_is_auditable_outcome_alias(tmp_path):
    initial = plan(tmp_path)
    identifier = decisions.start_decision(tmp_path, initial, {'inspect_synthetic_evidence'})
    outcome = decisions.finish_decision(tmp_path, identifier, {'status':'success','summary':'Synthetic observation','data':{'value':1}})
    revised = plan(tmp_path, revises=identifier, evidence_refs=[outcome, identifier])
    canonical = decisions.validate_plan(tmp_path, revised, {'inspect_synthetic_evidence'})
    assert canonical['evidence_refs'] == [outcome]


def test_unspecified_assessment_scope_does_not_silently_test_old_queue(tmp_path):
    from enzyme_discovery_agent.tools.research_decisions import assess_next_batch
    update_candidate_registry(tmp_path, 'SYNTH_A', 'screened')
    result = assess_next_batch(tmp_path, limit=1)
    assert result['status'] == 'error'
    assert 'scope_mode=ready_queue' in result['summary']
    assert discovery_status(load_candidate_registry(tmp_path)['candidates']['SYNTH_A']) == 'screened'


def test_read_only_inspection_is_available_before_a_research_plan(tmp_path):
    from enzyme_discovery_agent.agent import EnzymeAgent
    agent = EnzymeAgent.__new__(EnzymeAgent)
    agent.run_dir, agent.llm, agent.tools = tmp_path, SimpleNamespace(), []
    agent._require_research_decision = True
    agent._search_history, agent._search_queries, agent._search_log = [], [], []
    agent.log, agent._log_event = Mock(), Mock()
    with patch('enzyme_discovery_agent.agent.execute_tool', return_value={'status':'success','data':{'question':'inspect current evidence'}}) as read:
        result = agent._execute_single_tool('get_exploration_state', {'section':'hypotheses'}, 1)
    read.assert_called_once()
    assert result['status'] == 'success'
    assert getattr(agent, '_invalid_research_decisions', 0) == 0


def test_realistic_context_pressure_keeps_decision_evidence_and_recent_inspection(tmp_path):
    from enzyme_discovery_agent.context import build_model_context
    from enzyme_discovery_agent.steering import submit
    from enzyme_discovery_agent import config
    (tmp_path/'manifest.json').write_text('{}')
    evidence(tmp_path)
    initial = plan(tmp_path)
    identifier = decisions.start_decision(tmp_path, initial, {'inspect_synthetic_evidence'})
    outcome = decisions.finish_decision(tmp_path, identifier, {'status':'success','summary':'Synthetic test outcome','data':{'value':1}})
    submit(tmp_path, 'instruction', 'Keep the user constraints. '*60)
    from enzyme_discovery_agent.steering import consume
    consume(tmp_path)
    messages = [{'role':'system','content':'S'*5977}, {'role':'user','content':'Q'*2209},
        {'role':'assistant','content':None,'tool_calls':[{'id':'inspection','function':{'name':'get_exploration_state','arguments':'{"section":"families"}'}}]},
        {'role':'tool','tool_call_id':'inspection','content':'INSPECTED_SCOPE '+('detail '*3000)},
        {'role':'user','content':'ADAPTIVE RESEARCH DECISION '+('instruction '*150)}]
    result = build_model_context(messages, tmp_path)
    encoded = json.dumps(result,ensure_ascii=False)
    assert len(encoded) <= config.MODEL_CONTEXT_MAX_CHARS
    assert outcome in encoded and identifier in encoded
    assert initial['evidence_refs'][0] in encoded
    assert 'INSPECTED_SCOPE' in encoded
    assert 'ADAPTIVE RESEARCH DECISION' in encoded
    assert 'Keep the user constraints.' in encoded
    assert any(message.get('role') == 'assistant' and message.get('tool_calls') for message in result)
    assert any(message.get('tool_call_id') == 'inspection' for message in result)


def test_large_mandatory_prefix_keeps_a_bounded_retrievable_decision_packet(tmp_path):
    from enzyme_discovery_agent.context import build_model_context
    from enzyme_discovery_agent import config
    evidence(tmp_path)
    initial=plan(tmp_path)
    identifier=decisions.start_decision(tmp_path,initial,{'inspect_synthetic_evidence'})
    outcome=decisions.finish_decision(tmp_path,identifier,{'status':'success','summary':'Synthetic observation remains uncertain.',
        'data':[{'title':'Synthetic source with a long description '*30,'doi':'synthetic:retrieved'} for _ in range(3)]})
    raw='EXPLICIT_SAVED_INSPECTION '+('retrieved details '*1000)
    messages=[{'role':'system','content':'S'*10000},{'role':'user','content':'Q'*2200},
        {'role':'assistant','tool_calls':[{'id':'large','function':{'name':'get_exploration_state','arguments':'{"section":"families"}'}}]},
        {'role':'tool','tool_call_id':'large','content':raw}]
    original=json.dumps(messages)
    projected=build_model_context(messages,tmp_path)
    encoded=json.dumps(projected,ensure_ascii=False)
    assert len(encoded)<=config.MODEL_CONTEXT_MAX_CHARS
    assert identifier in encoded and outcome in encoded
    assert 'EXPLICIT_SAVED_INSPECTION' in encoded
    snapshots=list((tmp_path/'artifacts/evidence/inspection_memory').glob('*.json'))
    assert snapshots and json.loads(snapshots[0].read_text())['result_text']==raw
    assert json.dumps(messages)==original


def test_historical_inspection_keeps_observation_ahead_of_long_plan(tmp_path):
    from enzyme_discovery_agent.context import build_model_context
    evidence(tmp_path)
    messages = [{'role':'system','content':'system'}, {'role':'user','content':'target'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'read','function':{
            'name':'get_decision_context','arguments':'{"decision_id":"11111111111111111111111111111111"}'}}]},
        {'role':'tool','tool_call_id':'read','content':json.dumps({'status':'success','data':{
            'plan':{'hypothesis':'long explanation '*500}, 'result':{'data':{'observation':'ACTUAL_OBSERVATION'}}}})}]
    result = build_model_context(messages, tmp_path)
    assert 'ACTUAL_OBSERVATION' in json.dumps(result)


def test_strategic_interface_retains_all_action_contracts():
    from enzyme_discovery_agent.context import strategic_tool_definitions
    from enzyme_discovery_agent.tool_registry import get_tool_info
    import enzyme_discovery_agent.tools.research_decisions
    tools = [get_tool_info(name)['schema'] for name in ('plan_research_step','get_decision_context','assess_next_batch')]
    selected = strategic_tool_definitions(tools)
    assert len(selected) == 1 and selected[0]['function']['name'] == 'plan_research_step'
    properties = selected[0]['function']['parameters']['properties']
    assert set(properties['tool_name']['enum']) == {'get_decision_context','assess_next_batch'}
    # The arguments contract is carried once, by each action's own definition in the
    # payload the model receives, rather than copied a second time into the planner.
    assert 'anyOf' not in properties['arguments']
    from enzyme_discovery_agent.context import stable_tool_definitions
    sent = {tool['function']['name']: tool['function'].get('parameters') or {}
            for tool in stable_tool_definitions(tools)}
    for name in properties['tool_name']['enum']:
        assert (sent[name].get('properties') or {}), f'{name} is offered with no schema sent'


def test_unreviewed_source_is_not_a_biological_contradiction(tmp_path):
    update_candidate_registry(tmp_path,'SYNTH_A','evidence_pending',{'claim_validation':{
        'blocking_claims':['native_reaction'], 'claims':[{'claim_type':'reviewed_status','verdict':'contradicted'}]}})
    rows,_=decisions.observations(tmp_path)
    native=next(row for row in rows if row['kind']=='native_analogue_evidence')
    assert native['facts']['contradicted_native_claims']==[]
    assert native['facts']['missing_required_claims']==['native_reaction']


def test_planned_inspection_preserves_strategic_checkpoint(tmp_path):
    proposal=plan(tmp_path,tool_name='get_exploration_state',scope='retrieval',arguments={'section':'hypotheses'})
    identifier=decisions.start_decision(tmp_path,proposal,{'get_exploration_state'})
    decisions.finish_decision(tmp_path,identifier,{'status':'success','data':{'hypotheses':[]}})
    assert decisions.inspection_completed(tmp_path)


def test_valid_serialized_arguments_and_cited_outcome_keep_same_decision(tmp_path):
    initial=plan(tmp_path)
    identifier=decisions.start_decision(tmp_path,initial,{'inspect_synthetic_evidence'})
    outcome=decisions.finish_decision(tmp_path,identifier,{'status':'success','data':{'value':1}})
    next_plan=plan(tmp_path,arguments='{"query":"synthetic"}',evidence_refs=json.dumps([identifier]))
    canonical=decisions.validate_plan(tmp_path,next_plan,{'inspect_synthetic_evidence'})
    assert canonical['arguments']=={'query':'synthetic'}
    assert canonical['evidence_refs']==[outcome]
    assert canonical['revises']==identifier
    assert canonical['revision_link_inferred_from_citation'] is True


def test_already_evaluated_records_do_not_look_untested(tmp_path):
    from enzyme_discovery_agent.tools.research_decisions import assess_next_batch
    update_candidate_registry(tmp_path,'SYNTH_A','evaluated')
    result=assess_next_batch(tmp_path,accessions=['SYNTH_A'])
    assert result['status']=='skipped' and result['data']['new_evidence'] is False
    assert result['data']['existing_assessments'][0]['status']=='evaluated'


def test_review_can_conclude_and_reopens_only_when_scientific_evidence_changes(tmp_path):
    evidence(tmp_path)
    initial=plan(tmp_path)
    identifier=decisions.start_decision(tmp_path,initial,{'inspect_synthetic_evidence'})
    outcome=decisions.finish_decision(tmp_path,identifier,{'status':'unavailable','summary':'Synthetic transport unavailable'})
    assert decisions.review_due(tmp_path)
    result=decisions.conclude_review(tmp_path,
        hypothesis=initial['hypothesis'],alternative=initial['alternative'],
        conclusion='Access failed; functional capability remains unresolved. Evaluate other queued representatives.',
        evidence_refs=[outcome], prediction=initial['prediction'],contrasting_prediction=initial['contrasting_prediction'],
        revision_trigger='Reconsider when a new functional record or primary text is obtained.',
        scope='retrieval',belief_update='suspend',revises=identifier)
    assert result['status']=='success' and not decisions.review_due(tmp_path)
    assert decisions.decision_summary(tmp_path)['concluded_reviews']==1
    update_candidate_registry(tmp_path,'SYNTH_NEW','evaluated',{
        'screening_family_id':'SYNTH_OTHER_FOLD','substrate_similarity':{'accommodation':{
            'unaccommodated_targets':[], 'all_target_substrates_accommodated':True}}})
    assert decisions.review_due(tmp_path)


def test_strategy_interface_keeps_review_closure_outside_nested_action(tmp_path):
    from enzyme_discovery_agent.context import strategic_tool_definitions
    from enzyme_discovery_agent.tool_registry import get_tool_info
    tools=[get_tool_info(name)['schema'] for name in ('plan_research_step','conclude_research_review','assess_next_batch')]
    result=strategic_tool_definitions(tools)
    assert [tool['function']['name'] for tool in result]==['plan_research_step','conclude_research_review']
    assert 'conclude_research_review' not in result[0]['function']['parameters']['properties']['tool_name']['enum']


def test_concluded_review_resumes_backlog_without_a_model_turn_per_candidate(tmp_path):
    from openai.types.chat import ChatCompletion
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.tool_registry import get_tool_info
    from enzyme_discovery_agent.bootstrap import register_builtin_tools
    register_builtin_tools()
    tools=[get_tool_info(name)['schema'] for name in ('conclude_research_review','plan_research_step')]
    with patch('enzyme_discovery_agent.agent.register_builtin_tools'), \
         patch('enzyme_discovery_agent.agent.get_all_tools',return_value=tools), \
         patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}), \
         patch('enzyme_discovery_agent.models.recovery.next_recovery_action',return_value=None):
        agent=EnzymeAgent(output_dir=tmp_path,token_budget=100)
        evidence(agent.run_dir)
        for i in range(25):
            update_candidate_registry(agent.run_dir,f'SYNTH_PENDING_{i}','screened',{'screening_family_id':'SYNTH_FOLD'})
        performed=[]
        calls=[]
        def evaluate(run_dir,accession,**kwargs):
            performed.append(accession)
            update_candidate_registry(run_dir,accession,'evaluated',{
                'claim_validation':{'promotion_eligible':True},
                'substrate_similarity':{'accommodation':{
                    'unaccommodated_targets':['SYNTH_ROLE'],'all_target_substrates_accommodated':False}}})
            return {'status':'success','data':{'accession':accession,'claim_validation':{'promotion_eligible':True}}}
        def respond(*args,**kwargs):
            calls.append(kwargs)
            if len(calls)>1:
                assert len(performed)==25
                agent.llm.resource_stop_reason='Synthetic closure test complete'
                return None
            context=decisions.decision_context(agent.run_dir)
            reference=next(row['id'] for row in context['observations'] if row['kind']=='native_analogue_evidence')
            proposal={'hypothesis':'The native counterpart may not support the required input role.',
                'alternative':'The enzyme could still accept the input through unmeasured promiscuity.',
                'conclusion':'Native analogy remains uncertain. Assess the remaining representatives before deeper model work.',
                'evidence_refs':[reference],'prediction':'Other representatives may retain the native mismatch.',
                'contrasting_prediction':'A representative may show both native input roles.',
                'revision_trigger':'Reopen when a qualitatively different compatibility observation arrives.',
                'scope':'target_compatibility','belief_update':'suspend'}
            return ChatCompletion.model_validate({'id':'synthetic','object':'chat.completion','created':0,'model':'synthetic',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                    'tool_calls':[{'id':'closure','type':'function','function':{
                        'name':'conclude_research_review','arguments':json.dumps(proposal)}}]}}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
        for gate in (agent._literature_harvest_gate,agent._mechanism_family_gate,agent._family_survey_gate,
                     agent._systematic_gate,agent._gate_pipeline.gates[3]):
            gate.should_fire=Mock(return_value=False)
        with patch.object(agent.llm,'call',side_effect=respond), \
             patch('enzyme_discovery_agent.tools.reasoning.evaluate_candidate',side_effect=evaluate), \
             patch('enzyme_discovery_agent.gates.ec_triage._extract_target_params',return_value={}), \
             patch('enzyme_discovery_agent.gates.ec_triage.auto_screen_candidates'), \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',45)):
            agent.run('synthetic reaction',substrate_smiles='CCO')
        assert len(performed)==25 and len(calls)==2
        assert decisions.decision_summary(agent.run_dir)['concluded_reviews']==1


def test_paper_identifiers_survive_later_noninformative_actions(tmp_path):
    first=plan(tmp_path,tool_name='search_pubmed')
    identifier=decisions.start_decision(tmp_path,first,{'search_pubmed'})
    outcome=decisions.finish_decision(tmp_path,identifier,{'status':'success','data':[{
        'pmid':'SYNTHETIC_PMID','title':'Synthetic primary evidence','doi':'synthetic:doi'}]})
    next_plan=plan(tmp_path,revises=identifier,evidence_refs=[outcome])
    next_id=decisions.start_decision(tmp_path,next_plan,{'inspect_synthetic_evidence'})
    decisions.finish_decision(tmp_path,next_id,{'status':'skipped','summary':'No new evidence'})
    context=decisions.decision_context(tmp_path)
    assert context['recent_paper_findings'][0]['papers'][0]['pmid']=='SYNTHETIC_PMID'


def test_review_action_budget_forces_uncertain_conclusion_and_resets_afterward(tmp_path,monkeypatch):
    from enzyme_discovery_agent import config
    monkeypatch.setattr(config,'RESEARCH_REVIEW_MAX_ACTIONS',2)
    evidence(tmp_path)
    prior=None
    for i in range(2):
        arguments=plan(tmp_path)
        if prior:
            arguments.update(revises=prior[0],evidence_refs=[prior[1]])
        identifier=decisions.start_decision(tmp_path,arguments,{'inspect_synthetic_evidence'})
        outcome=decisions.finish_decision(tmp_path,identifier,{'status':'unavailable','summary':'Synthetic unavailable retrieval'})
        prior=(identifier,outcome)
    limits=decisions.review_limits(tmp_path)
    assert limits['conclusion_required'] and limits['action_count']==2
    assert not decisions.review_action_allowed(tmp_path,'inspect_synthetic_evidence')
    assert decisions.review_action_allowed(tmp_path,'conclude_research_review')
    with pytest.raises(ValueError,match='exhausted'):
        decisions.start_decision(tmp_path,plan(tmp_path,revises=prior[0],evidence_refs=[prior[1]]),{'inspect_synthetic_evidence'})
    proposal=plan(tmp_path)
    decisions.conclude_review(tmp_path,hypothesis=proposal['hypothesis'],alternative=proposal['alternative'],
        conclusion='Access remains unavailable; retain functional uncertainty and resume queued evaluation.',
        evidence_refs=[prior[1]],prediction=proposal['prediction'],contrasting_prediction=proposal['contrasting_prediction'],
        revision_trigger=proposal['revision_trigger'],scope='retrieval',belief_update='suspend',revises=prior[0])
    assert not decisions.review_due(tmp_path)
    assert decisions.review_limits(tmp_path)['action_count']==0


def test_inspection_allowance_cannot_be_reset_by_a_planned_inspection(tmp_path,monkeypatch):
    from enzyme_discovery_agent import config
    monkeypatch.setattr(config,'RESEARCH_REVIEW_MAX_INSPECTIONS',1)
    decisions.record_inspection(tmp_path)
    assert not decisions.review_action_allowed(tmp_path,'get_exploration_state')
    proposal=plan(tmp_path,tool_name='get_exploration_state')
    with pytest.raises(ValueError,match='exhausted'):
        decisions.start_decision(tmp_path,proposal,{'get_exploration_state'})
    assert decisions.review_action_allowed(tmp_path,'read_primary_paper')
