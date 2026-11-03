import csv
import json
import pytest
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry,load_candidate_registry
from enzyme_discovery_agent.models import target_review as review
from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates


def setup(root,annotated=True):
    (root/'target_context.json').write_text(json.dumps({'substrate_smiles':'CCO.CC=O','target_reasoning':{'reaction_direction':'redox'}}))
    update_candidate_registry(root,'SYNTH1','evaluated' if annotated else 'evidence_pending',{
        'protein_name':'Synthetic catalyst','catalytic_activities':['Synthetic native reaction'] if annotated else [],
        'claim_validation':{'promotion_eligible':annotated,'claims':[{
            'claim_type':kind,'subject':'SYNTH1','verdict':'supported'} for kind in ['accession_identity','record_active']]}})
    row=review.review_queue(root)[0];row['source_id']='R1'
    return {'target':review.target_contract(root),'target_inputs':review.target_inputs(review.target_contract(root)),'candidates':[row]}


def assessment(decision='investigate'):
    return {'accession':'SYNTH1','decision':decision,'evidence_level':'mechanistic_transfer',
        'reason':'Candidate-specific evidence supports a conditional test of the requested chemistry.',
        'mechanism_and_step':'The supplied evidence supports the declared reacting-center transformation.',
        'target_roles':[{'participant':p,'role':'Mapped reacting partner','basis':'hypothesis'} for p in ['T1','T2']],
        'source_refs':['R1'],'strengths':['Source-linked catalytic machinery.'],'constraints':[],
        'unknowns':['Target activity is unmeasured.'],'revisit_when':'A discriminating structural or functional observation is available.'}


def test_annotation_only_rows_never_fall_back_into_an_empty_shortlist(tmp_path):
    setup(tmp_path)
    assert refresh_ranked_candidates(tmp_path)==[]
    assert list(csv.DictReader((tmp_path/'ranked_candidates.csv').open()))==[]
    assert len(list(csv.DictReader((tmp_path/'candidate_assessment_queue.csv').open())))==1


@pytest.mark.parametrize('annotated',[True,False])
def test_source_bound_target_hypothesis_is_eligible_independently_of_native_annotation(tmp_path,annotated):
    packet=setup(tmp_path,annotated)
    review.save_assessment(tmp_path,packet,assessment('prioritize'))
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    assert review.eligible(tmp_path,'SYNTH1',meta)
    assert len(refresh_ranked_candidates(tmp_path))==1
    update_candidate_registry(tmp_path,'SYNTH1',None,{'catalytic_activities':['Changed scientific evidence']})
    assert refresh_ranked_candidates(tmp_path)==[]


def test_source_refs_all_partners_and_actual_exclusion_basis_are_required(tmp_path):
    packet=setup(tmp_path)
    bad=assessment();bad['target_roles']=bad['target_roles'][:1]
    with pytest.raises(ValueError,match='every'):review.save_assessment(tmp_path,packet,bad)
    bad=assessment();bad['target_roles'].append(dict(bad['target_roles'][0]))
    with pytest.raises(ValueError,match='exactly once'):review.save_assessment(tmp_path,packet,bad)
    bad=assessment();bad['source_refs']=['unrelated']
    with pytest.raises(ValueError,match='exact'):review.save_assessment(tmp_path,packet,bad)
    bad=assessment('exclude');bad['constraints']=[{'finding':'Missing annotation','basis':'missing_evidence'}]
    with pytest.raises(ValueError,match='Missing evidence'):review.save_assessment(tmp_path,packet,bad)
    bad=assessment();bad['evidence_level']='speculative'
    with pytest.raises(ValueError,match='speculation'):review.save_assessment(tmp_path,packet,bad)


def test_target_roles_and_priority_reasoning_are_saved_in_the_verified_trace(tmp_path):
    from enzyme_discovery_agent.export_reasoning_trace import export_trace
    packet=setup(tmp_path)
    saved=review.save_assessment(tmp_path,packet,assessment())
    trace=export_trace(tmp_path)
    assert trace['source_files_intact'] and trace['hash_chain_valid']
    event=next(e for e in trace['events'] if e['type']=='target_assessment_recorded')
    assert event['capture_mode']=='live'
    assert event['record']['target_roles']==saved['target_roles']
    assert event['record']['mechanism_and_step']==saved['mechanism_and_step']
    assert saved['reason'] in (tmp_path/'reasoning/trace.md').read_text()


def test_pending_search_interpretation_does_not_starve_candidate_assessment(tmp_path,monkeypatch):
    from enzyme_discovery_agent import config
    monkeypatch.setattr(config,'AUTO_TARGET_REVIEW',True)
    from unittest.mock import patch
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.models import research_decisions as decisions
    from enzyme_discovery_agent.tests.test_review_protocol_recovery import pending
    with patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}):
        agent=EnzymeAgent(output_dir=tmp_path,token_budget=1000)
        pending(agent.run_dir)
        def assess(*args,**kwargs):
            assert decisions._read(decisions._root(agent.run_dir)/'state.json',{})['awaiting_interpretation']
            agent.llm.resource_stop_reason='Synthetic target assessment scheduling verified'
            return {'assessed':[],'errors':[],'unassessed':[]}
        with patch.object(review,'review_queue',return_value=[{'accession':'SYNTH_REVIEW'}]), \
             patch.object(review,'run_review',side_effect=assess) as executed, \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',0)):
            agent.run('synthetic transformation',substrate_smiles='CCO')
    executed.assert_called_once()
    assert decisions._read(decisions._root(agent.run_dir)/'state.json',{})['awaiting_interpretation']


def test_partial_target_batch_returns_to_assessment_before_gathering_more_native_records(tmp_path,monkeypatch):
    from enzyme_discovery_agent import config
    monkeypatch.setattr(config,'AUTO_TARGET_REVIEW',True)
    from unittest.mock import Mock,patch
    from enzyme_discovery_agent.agent import EnzymeAgent
    with patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}):
        agent=EnzymeAgent(output_dir=tmp_path,token_budget=1000)
        update_candidate_registry(agent.run_dir,'SYNTH_NEW','screened',{'protein_name':'Synthetic catalyst'})
        gate=agent._gate_pipeline.gates[2]
        gate.should_fire=Mock(return_value=True)
        gate.execute=Mock(side_effect=AssertionError('Partial target batch fell through to native evaluation'))
        batches=[]
        def assess(*args,**kwargs):
            batches.append(True)
            if len(batches)==2:agent.llm.resource_stop_reason='Synthetic partial-batch scheduling verified'
            return {'assessed':[],'errors':['Synthetic malformed assessment'],'unassessed':['SYNTH_READY']}
        with patch.object(review,'review_queue',return_value=[{'accession':'SYNTH_READY'}]), \
             patch.object(review,'run_review',side_effect=assess), \
             patch('enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status',return_value={'required':False}), \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',0)):
            agent.run('synthetic transformation',substrate_smiles='CCO')
    assert len(batches)==2
    gate.execute.assert_not_called()


def test_target_review_diversifies_before_repeating_annotation_rich_family(tmp_path):
    setup(tmp_path)
    entry=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    for acc,family,gene in [('A_FIRST','FOLD_A','shared'),('A_SECOND','FOLD_A','shared'),
                            ('Z_OTHER','FOLD_Z','different')]:
        update_candidate_registry(tmp_path,acc,'evaluated',{**entry,'interpro_family_id':family,'gene_names':[gene],
            'claim_validation':{'promotion_eligible':True,'claims':[{'claim_type':kind,'subject':acc,'verdict':'supported'}
                for kind in ['accession_identity','record_active']]}})
    update_candidate_registry(tmp_path,'SYNTH1','rejected',{'reason':'Synthetic fixture excluded from scheduling.'})
    assert [row['accession'] for row in review.review_queue(tmp_path,limit=2)]==['A_FIRST','Z_OTHER']


def test_target_reviewer_receives_selected_reaction_and_substrate_context():
    record={'native_substrate_reaction_scope':{'reaction_index':1,'reaction':'SELECTED_REACTION',
              'native_substrates':[{'name':'source partner','smiles':'CCO','occurrence_id':'native:0:0'}]},
            'substrate_similarity':{'property_comparisons':[{'target_smiles':'CCCO','native_smiles':'CCO',
                'mw_similarity':.75,'functional_group_similarity':1.0}]}}
    projected=review.compact_record(review.scientific_record(record))
    assert projected['native_substrate_reaction_scope']['reaction']=='SELECTED_REACTION'
    assert projected['substrate_context']['comparisons']
    assert 'not measured binding' in projected['substrate_context']['scope']


def test_a_transfer_hypothesis_ranks_but_is_not_in_the_strict_shortlist(tmp_path):
    """Non-native engineering: the requested reaction is undocumented by definition.

    Requiring an observation on the requested substrates to rank left the list empty
    however strong the mechanistic case was, so transfer hypotheses now rank, while the
    prioritize-only view stays available for the stricter question.
    """
    packet=setup(tmp_path);review.save_assessment(tmp_path,packet,assessment())
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    assert review.eligible(tmp_path,'SYNTH1',meta,purpose='modeling')
    assert review.eligible(tmp_path,'SYNTH1',meta)
    assert not review.eligible(tmp_path,'SYNTH1',meta,purpose='shortlist')
    assert len(refresh_ranked_candidates(tmp_path))==1


def test_a_speculative_assessment_still_never_ranks(tmp_path):
    """Leniency about documentation is not leniency about having any case at all."""
    packet=setup(tmp_path)
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    saved=dict(meta.get('target_assessment') or {})
    speculative={**assessment(),'evidence_level':'speculative'}
    with pytest.raises(ValueError):
        review.save_assessment(tmp_path,packet,speculative)
    update_candidate_registry(tmp_path,'SYNTH1',None,{'target_assessment':{**saved,'evidence_level':'speculative'}})
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    assert not review.eligible(tmp_path,'SYNTH1',meta)
    assert not review.eligible(tmp_path,'SYNTH1',meta,purpose='modeling')


def test_major_observed_conflict_is_not_hidden_by_a_positive_transfer_label(tmp_path):
    packet=setup(tmp_path);result=assessment('prioritize')
    result['constraints']=[{'finding':'Observed failure of a necessary modeled condition','basis':'observed_mismatch','severity':'major'}]
    saved=review.save_assessment(tmp_path,packet,result)
    assert saved['decision']=='investigate' and saved['requested_decision']=='prioritize'


def test_rolling_modeling_waits_for_target_review_instead_of_native_annotations(tmp_path):
    from enzyme_discovery_agent.models.modeling_portfolio import select_modeling_portfolio
    packet=setup(tmp_path)
    assert not select_modeling_portfolio(tmp_path,max_candidates=2)
    review.save_assessment(tmp_path,packet,assessment())
    assert [r['accession'] for r in select_modeling_portfolio(tmp_path,max_candidates=2)]==['SYNTH1']


def test_handoff_preserves_unassessed_native_entries_as_leads(tmp_path):
    from enzyme_discovery_agent.output.structural_modeling_handoff import write_structural_modeling_handoff
    setup(tmp_path)
    path=write_structural_modeling_handoff(tmp_path,[{'status':'success','data':{'accession':'SYNTH1'}}],{'substrate_smiles':'CCO.CC=O'})
    document=json.loads(path.read_text())
    assert not document['candidates'] and document['unresolved_leads'][0]['accession']=='SYNTH1'


class _Msg:
    def __init__(self,tool_calls):self.tool_calls=tool_calls;self.content=''
    def model_dump(self):return {'role':'assistant','content':None,'tool_calls':self.tool_calls}


class _Choice:
    def __init__(self,message):self.message=message


class _Response:
    def __init__(self,message):self.choices=[_Choice(message)]


class _StubLLM:
    """Returns one tool call per request; records how many requests were made."""
    def __init__(self,arguments):self.arguments=arguments;self.calls=0
    def call(self,**kwargs):
        self.calls+=1
        return _Response(_Msg([{'id':'toolu_1','type':'function','function':{
            'name':'record_target_assessment','arguments':json.dumps(self.arguments)}}]))


UNPARSED={'__unparsedToolInput':{'raw':'{"accession": "SYNTH1", "decision": "inv','len':2048}}


def test_provider_mangled_tool_input_is_not_recorded_as_a_scientific_verdict(tmp_path):
    setup(tmp_path)
    llm=_StubLLM(UNPARSED)
    result=review.run_review(tmp_path,llm,limit=4)
    assert result['assessed']==[] and result['unassessed']==['SYNTH1']
    assert result['errors']==[] and result['provider_faults']
    attempt=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']['target_review_attempt']
    assert attempt['fault']=='provider' and attempt['provider_attempts']==1
    # The candidate must come back to the queue: no judgement about it was recorded.
    assert [row['accession'] for row in review.review_queue(tmp_path)]==['SYNTH1']


def test_a_persistent_provider_fault_stops_requeueing_after_a_bounded_number_of_tries(tmp_path):
    setup(tmp_path)
    llm=_StubLLM(UNPARSED)
    for _ in range(review.PROVIDER_FAULT_ATTEMPTS+2):
        if review.review_queue(tmp_path):review.run_review(tmp_path,llm,limit=4)
    assert llm.calls==review.PROVIDER_FAULT_ATTEMPTS
    assert review.review_queue(tmp_path)==[]


def test_validation_failure_gets_one_correction_with_feedback_then_stops(tmp_path):
    setup(tmp_path)
    bad={**assessment(),'source_refs':['unrelated']}
    llm=_StubLLM(bad)
    result=review.run_review(tmp_path,llm,limit=4)
    assert result['errors'] and result['provider_faults']==[]
    attempt=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']['target_review_attempt']
    assert attempt['fault']=='validation'
    row,=review.review_queue(tmp_path)
    assert row['repair_feedback']['errors']==result['errors']
    assert row['repair_feedback']['previous_response']==bad
    assert refresh_ranked_candidates(tmp_path)==[]
    review.run_review(tmp_path,llm,limit=4)
    assert llm.calls==2
    assert review.review_queue(tmp_path)==[]


def test_correction_admits_only_a_valid_assessment_and_preserves_failed_attempt(tmp_path):
    from enzyme_discovery_agent.export_reasoning_trace import export_trace
    setup(tmp_path)
    bad={**assessment(),'target_roles':assessment()['target_roles'][:1]}
    llm=_StubLLM(bad)
    review.run_review(tmp_path,llm)
    assert refresh_ranked_candidates(tmp_path)==[]
    llm.arguments=assessment()
    assert review.run_review(tmp_path,llm)['assessed']==['SYNTH1']
    assert len(refresh_ranked_candidates(tmp_path))==1
    trace=export_trace(tmp_path)
    assert trace['source_files_intact'] and trace['hash_chain_valid']
    assert [e['type'] for e in trace['events']]==[
        'target_review_batch_recorded','target_assessment_recorded','target_review_batch_recorded']
    first=trace['events'][0]['record']
    assert first['attempts']['SYNTH1']['response']==bad
    path=tmp_path/first['source']['path']
    path.write_text('{}')
    assert not export_trace(tmp_path,reconstruct_missing=False)['source_files_intact']


def test_changed_evidence_resets_the_shared_attempt_budget(tmp_path):
    setup(tmp_path)
    llm=_StubLLM(UNPARSED)
    review.run_review(tmp_path,llm)
    llm.arguments={**assessment(),'source_refs':['unrelated']}
    review.run_review(tmp_path,llm)
    assert review.review_queue(tmp_path)==[]
    update_candidate_registry(tmp_path,'SYNTH1',None,{'function_comments':['New source evidence']})
    row,=review.review_queue(tmp_path)
    assert 'repair_feedback' not in row
    review.run_review(tmp_path,llm)
    attempt=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']['target_review_attempt']
    assert attempt['attempts']==1 and review.review_queue(tmp_path)


def test_resume_can_correct_legacy_validation_failures_once(tmp_path):
    packet=setup(tmp_path)
    row=packet['candidates'][0]
    update_candidate_registry(tmp_path,'SYNTH1',None,{'target_review_attempt':{
        'target_sha256':row['target_sha256'],'record_sha256':row['record_sha256'],
        'fault':'scientific','errors':['Account for every target participant exactly once']}})
    assert review.review_queue(tmp_path)[0]['repair_feedback']['errors']
    result=review.run_review(tmp_path,_StubLLM({**assessment(),'source_refs':['unrelated']}))
    assert result['unassessed']==['SYNTH1']
    assert review.review_queue(tmp_path)==[]


@pytest.mark.parametrize('stop', ['budget_exhausted','resource_stop_reason'])
def test_resource_deferral_preserves_the_remaining_correction_on_resume(tmp_path,stop):
    from types import SimpleNamespace
    from enzyme_discovery_agent.export_reasoning_trace import export_trace
    setup(tmp_path)
    review.run_review(tmp_path,_StubLLM(UNPARSED))
    before=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']['target_review_attempt']
    llm=SimpleNamespace(call=lambda **kwargs:None,**{stop:True if stop=='budget_exhausted' else 'Capacity unavailable'})
    for _ in range(3):
        result=review.run_review(tmp_path,llm)
        assert result['deferred_for_resources'] and not result['provider_faults']
    after=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']['target_review_attempt']
    assert before==after
    assert review.run_review(tmp_path,_StubLLM(assessment()))['assessed']==['SYNTH1']
    trace=export_trace(tmp_path)
    assert trace['source_files_intact']
    deferred=[e for e in trace['events'] if e['type']=='target_review_batch_recorded'
              and e['record']['result'].get('deferred_for_resources')]
    assert len(deferred)==3 and all(not e['record']['attempts'] for e in deferred)


def test_mixed_batch_failure_does_not_blame_omitted_candidates(tmp_path):
    setup(tmp_path)
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    update_candidate_registry(tmp_path,'SYNTH2','evaluated',{**meta,'claim_validation':{'claims':[
        {'claim_type':kind,'subject':'SYNTH2','verdict':'supported'} for kind in ['accession_identity','record_active']]}})
    review.run_review(tmp_path,_StubLLM({**assessment(),'source_refs':['unrelated']}))
    candidates=load_candidate_registry(tmp_path)['candidates']
    assert candidates['SYNTH1']['metadata']['target_review_attempt']['fault']=='validation'
    omitted=candidates['SYNTH2']['metadata']['target_review_attempt']
    assert omitted['fault']=='omitted'
    assert omitted['errors']==['No valid assessment returned for this candidate']


@pytest.mark.parametrize('arguments',[[],{'accession':[]},{**assessment(),'constraints':[None]}])
def test_malformed_assessments_remain_unresolved_without_crashing(tmp_path,arguments):
    setup(tmp_path)
    result=review.run_review(tmp_path,_StubLLM(arguments))
    assert result['unassessed']==['SYNTH1'] and result['errors']
    assert refresh_ranked_candidates(tmp_path)==[]


def test_fresh_candidates_are_reviewed_before_repairs(tmp_path):
    setup(tmp_path)
    review.run_review(tmp_path,_StubLLM({**assessment(),'source_refs':['unrelated']}))
    update_candidate_registry(tmp_path,'ZZ_FRESH','evaluated',{
        'protein_name':'Another synthetic catalyst','claim_validation':{'claims':[
            {'claim_type':kind,'subject':'ZZ_FRESH','verdict':'supported'} for kind in ['accession_identity','record_active']]}})
    assert review.review_queue(tmp_path,limit=1)[0]['accession']=='ZZ_FRESH'


@pytest.mark.parametrize('tranche_exhausted',[False,True])
@pytest.mark.parametrize('invalid_decisions',[0,2])
def test_discovered_candidates_progress_during_pending_review_with_bounded_tranches(tmp_path,tranche_exhausted,invalid_decisions):
    from unittest.mock import Mock,patch
    from enzyme_discovery_agent import config
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.gates import GateResult
    from enzyme_discovery_agent.models import research_decisions as decisions
    from enzyme_discovery_agent.tests.test_review_protocol_recovery import pending
    with patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}):
        agent=EnzymeAgent(output_dir=tmp_path,token_budget=1000)
        agent._invalid_research_decisions=invalid_decisions
        pending(agent.run_dir)
        state_path=decisions._root(agent.run_dir)/'state.json'
        state=json.loads(state_path.read_text());state['initial_sources_reviewed']=True
        state_path.write_text(json.dumps(state))
        update_candidate_registry(agent.run_dir,'SYNTH_NEW','discovered',{'protein_name':'Synthetic new catalyst'})
        gate=agent._gate_pipeline.gates[2]
        def evaluate(*args,**kwargs):
            assert json.loads(state_path.read_text())['awaiting_interpretation']
            agent.llm.resource_stop_reason='Synthetic candidate scheduling verified'
            return GateResult(intercept=True)
        gate.should_fire=Mock(return_value=True)
        gate.execute=Mock(side_effect=evaluate)
        def model_turn(*args,**kwargs):
            assert tranche_exhausted,'A pending review starved newly discovered candidates'
            agent.llm.resource_stop_reason='Synthetic reflection boundary verified'
            return None
        if tranche_exhausted:agent._automatic_evidence_batches=config.MAX_AUTOMATED_BATCHES_BEFORE_REFLECTION
        with patch.object(agent.llm,'call',side_effect=model_turn), \
             patch.object(review,'review_queue',return_value=[]), \
             patch('enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status',return_value={'required':False}), \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',0)):
            agent.run('synthetic transformation',substrate_smiles='CCO')
        assert gate.execute.call_count==int(not tranche_exhausted)
        assert json.loads(state_path.read_text())['awaiting_interpretation']


def test_modeling_evidence_arriving_later_does_not_void_the_review(tmp_path):
    """Modeling runs after the review and writes onto the same record.

    Hashing that field into the eligibility check meant an assessment survived only
    until its candidate was modelled, which silently emptied the decision list.
    """
    packet=setup(tmp_path);review.save_assessment(tmp_path,packet,assessment())
    assert len(refresh_ranked_candidates(tmp_path))==1
    update_candidate_registry(tmp_path,'SYNTH1',None,{'modeling_validation_evidence':{
        'accession':'SYNTH1','status':'incomplete','review_errors':['pose validator identity is missing']}})
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    admitted,reason=review.eligibility_verdict(tmp_path,'SYNTH1',meta)
    assert admitted,reason
    assert len(refresh_ranked_candidates(tmp_path))==1


def test_discovery_side_changes_still_invalidate_the_review_and_say_so(tmp_path):
    """Narrowing the digest must not stop real evidence changes from forcing a re-review."""
    packet=setup(tmp_path);review.save_assessment(tmp_path,packet,assessment())
    update_candidate_registry(tmp_path,'SYNTH1',None,{'catalytic_activities':['A different native reaction']})
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    admitted,reason=review.eligibility_verdict(tmp_path,'SYNTH1',meta)
    assert not admitted
    assert reason=='candidate record changed since the review'
    assert refresh_ranked_candidates(tmp_path)==[]


def test_every_refusal_names_itself(tmp_path):
    """The filter once emptied the shortlist with nothing recording why."""
    packet=setup(tmp_path)
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    assert review.eligibility_verdict(tmp_path,'SYNTH1',meta)[1]=='no target assessment recorded'
    review.save_assessment(tmp_path,packet,{**assessment('defer'),'evidence_level':'speculative',
                                            'strengths':[],'constraints':[]})
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    assert review.eligibility_verdict(tmp_path,'SYNTH1',meta)[1]=="review decision is 'defer'"


def test_a_budget_stop_snapshot_keeps_the_candidates_the_live_ranking_admitted(tmp_path):
    """The run ends on its allowance, and the checkpoint rewrites the decision list.

    A candidate resting on a registered hypothesis is still 'evidence_pending', so a
    status-only filter dropped every one of them at the checkpoint -- overwriting a
    populated ranked_candidates.csv with an empty one at the very last step.
    """
    from enzyme_discovery_agent.output.finalizer import RunFinalizer
    packet=setup(tmp_path);review.save_assessment(tmp_path,packet,assessment('prioritize'))
    update_candidate_registry(tmp_path,'SYNTH1','evidence_pending',{})
    assert len(refresh_ranked_candidates(tmp_path))==1
    before=list(csv.DictReader((tmp_path/'ranked_candidates.csv').open()))
    assert [r['accession'] for r in before]==['SYNTH1']

    RunFinalizer(tmp_path,llm=None,log=lambda *a,**k:None).snapshot(
        query='ligate the target partners',reason='token allowance reached')
    after=list(csv.DictReader((tmp_path/'ranked_candidates.csv').open()))
    assert [r['accession'] for r in after]==['SYNTH1'],'the checkpoint emptied the decision list'
