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
    bad=assessment();bad['source_refs']=['unrelated']
    with pytest.raises(ValueError,match='exact'):review.save_assessment(tmp_path,packet,bad)
    bad=assessment('exclude');bad['constraints']=[{'finding':'Missing annotation','basis':'missing_evidence'}]
    with pytest.raises(ValueError,match='Missing evidence'):review.save_assessment(tmp_path,packet,bad)
    bad=assessment();bad['evidence_level']='speculative'
    with pytest.raises(ValueError,match='speculation'):review.save_assessment(tmp_path,packet,bad)


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


def test_a_validation_failure_is_the_reviewers_answer_and_is_not_retried(tmp_path):
    setup(tmp_path)
    bad={**assessment(),'source_refs':['unrelated']}
    llm=_StubLLM(bad)
    result=review.run_review(tmp_path,llm,limit=4)
    assert result['errors'] and result['provider_faults']==[]
    attempt=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']['target_review_attempt']
    assert attempt['fault']=='scientific'
    assert review.review_queue(tmp_path)==[]


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
