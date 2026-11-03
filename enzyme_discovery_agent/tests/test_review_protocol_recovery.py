"""Synthetic regression cases for outcome interpretation and offered tool contracts."""
import json

import pytest

from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.context import strategic_tool_definitions
from enzyme_discovery_agent.tool_registry import get_tool_info
import enzyme_discovery_agent.tools.research_decisions


def pending(root):
    update_candidate_registry(root,'SYNTH_PENDING','evidence_pending',{'protein_name':'Synthetic catalyst'})
    reference=next(row['id'] for row in decisions.decision_context(root)['observations']
                   if row['kind']=='native_analogue_evidence')
    proposal={'hypothesis':'The retrieved record may resolve the catalytic role.',
        'alternative':'The record may leave the catalytic role unresolved.',
        'prediction':'The record will provide a candidate-linked functional source.',
        'contrasting_prediction':'The result will contain no functional source.',
        'why_this_test':'Resolve whether further investigation is justified.',
        'revision_trigger':'Revisit when a source resolves the catalytic role.',
        'scope':'retrieval','tool_name':'read_synthetic_record','arguments':{},'evidence_refs':[reference]}
    identifier=decisions.start_decision(root,proposal,{'read_synthetic_record'})
    outcome=decisions.finish_decision(root,identifier,{'status':'partial','summary':'The lookup was deduplicated.',
        'deduplicated':True,'data':None})
    return identifier,outcome


@pytest.mark.parametrize('action_limit', [1, 4])
def test_interpretation_releases_pending_outcome_without_another_action(tmp_path, monkeypatch, action_limit):
    from enzyme_discovery_agent import config
    monkeypatch.setattr(config, 'RESEARCH_REVIEW_MAX_ACTIONS', action_limit)
    identifier,outcome=pending(tmp_path)
    before=decisions.review_limits(tmp_path)
    result=decisions.interpret_outcome(tmp_path,decision_id=identifier,evidence_refs=[outcome],
        interpretation='No new evidence was retrieved; the functional hypothesis remains unresolved.',belief_update='suspend')
    assert result['status']=='success'
    state=json.loads((decisions._root(tmp_path)/'state.json').read_text())
    assert not state['awaiting_interpretation']
    assert state['last_decision']==identifier
    assert decisions.review_limits(tmp_path)==before
    assert len(list(decisions._root(tmp_path).glob('*/plan.json')))==1
    journal=[json.loads(line) for line in (tmp_path/'reasoning/trace.jsonl').read_text().splitlines()]
    assert journal[-1]['type']=='outcome_interpreted'


def test_interpretation_rejects_wrong_outcome_and_modified_result(tmp_path):
    identifier,outcome=pending(tmp_path)
    args={'decision_id':identifier,'evidence_refs':['evidence:'+'0'*64],
        'interpretation':'The returned observation leaves both explanations unresolved.','belief_update':'suspend'}
    with pytest.raises(ValueError):decisions.interpret_outcome(tmp_path,**args)
    (decisions._root(tmp_path)/identifier/'result.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        decisions.interpret_outcome(tmp_path,**{**args,'evidence_refs':[outcome]})
    assert decisions.decision_context(tmp_path)['awaiting_interpretation']


def test_review_interface_only_offers_direct_controls_and_permitted_inspections():
    names=['plan_research_step','conclude_research_review','interpret_research_outcome','get_decision_context','assess_next_batch']
    tools=[get_tool_info(name)['schema'] for name in names]
    offered=strategic_tool_definitions(tools,include_inspections=True)
    assert {tool['function']['name'] for tool in offered}==set(names)-{'assess_next_batch'}
    planner=offered[0]['function']['parameters']['properties']
    assert 'assess_next_batch' in planner['tool_name']['enum']
    assert 'interpret_research_outcome' not in planner['tool_name']['enum']


def test_review_preserves_a_full_conclusion_beyond_the_brief_test_rationale_limit(tmp_path):
    identifier,outcome=pending(tmp_path)
    conclusion='The synthetic observation leaves native activity unresolved. '*40
    assert decisions.BRIEF_FIELD_LIMIT<len(conclusion)<decisions.REVIEW_CONCLUSION_LIMIT
    result=decisions.conclude_review(tmp_path,hypothesis='The source might establish a catalytic role.',
        alternative='The source may leave it unknown.',conclusion=conclusion,evidence_refs=[outcome],
        prediction='A functional source is retrieved.',contrasting_prediction='No functional source is retrieved.',
        revision_trigger='Reopen after new functional evidence.',scope='retrieval',revises=identifier)
    assert result['status']=='success'
    assert result['data']['conclusion']==conclusion


def test_review_schema_declares_the_real_length_limit_and_reports_the_public_field(tmp_path):
    from enzyme_discovery_agent.tools.research_decisions import conclude_research_review
    tool=get_tool_info('conclude_research_review')['schema']
    assert tool['function']['parameters']['properties']['conclusion']['maxLength']==decisions.REVIEW_CONCLUSION_LIMIT
    result=conclude_research_review(tmp_path,hypothesis='Synthetic source supports a test.',
        alternative='The source leaves it unknown.',conclusion='x'*(decisions.REVIEW_CONCLUSION_LIMIT+1),
        evidence_refs=[],prediction='A functional source is retrieved.',contrasting_prediction='No source is retrieved.',
        revision_trigger='Reopen on new evidence.',scope='retrieval')
    assert 'conclusion exceeds' in result['summary']


def test_stale_citation_diagnostic_exposes_current_candidate_facts_without_closing_review(tmp_path):
    pending(tmp_path)
    old=next(row['id'] for row in decisions.decision_context(tmp_path)['observations']
             if row['facts'].get('accession')=='SYNTH_PENDING')
    update_candidate_registry(tmp_path,'SYNTH_PENDING',None,{'new_source':'new synthetic observation'})
    repair=decisions.proposal_repair_context(tmp_path,proposal={'evidence_refs':[old]})
    changed,=repair['changed_candidate_records']
    assert changed['previous_evidence_id']==old and changed['accession']=='SYNTH_PENDING'
    assert changed['current_candidate_evidence'][0]['id']!=old
    assert decisions.decision_context(tmp_path)['awaiting_interpretation']


def test_work_priority_does_not_invalidate_unchanged_native_facts(tmp_path):
    identifier,outcome=pending(tmp_path)
    row=next(row for row in decisions.decision_context(tmp_path)['observations']
             if row['facts'].get('accession')=='SYNTH_PENDING')
    plan=json.loads((decisions._root(tmp_path)/identifier/'plan.json').read_text())
    proposal={**plan,'revises':identifier,'evidence_refs':[row['id'],outcome]}
    update_candidate_registry(tmp_path,'SYNTH_PENDING',None,{'priority_assessment':{
        'decision':'investigate','reason':'Synthetic work allocation only.'},'modeling_review_status':'not_returned'})
    checked=decisions.validate_plan(tmp_path,proposal,{'read_synthetic_record'})
    assert checked['unchanged_cited_fact_scopes'][0]['accession']=='SYNTH_PENDING'
    update_candidate_registry(tmp_path,'SYNTH_PENDING',None,{'catalytic_activities':['A different synthetic native reaction.']})
    with pytest.raises(ValueError,match='Changed candidate evidence'):
        decisions.validate_plan(tmp_path,proposal,{'read_synthetic_record'})


def test_work_priority_cannot_reopen_review_as_new_biological_evidence(tmp_path):
    for accession in ['SYNTH_A','SYNTH_B']:
        update_candidate_registry(tmp_path,accession,'evidence_pending',{'protein_name':'Same synthetic family'})
    before=decisions._signal(decisions.observations(tmp_path)[0])
    update_candidate_registry(tmp_path,'SYNTH_A',None,{'priority_assessment':{
        'decision':'investigate','reason':'Synthetic work allocation only.'}})
    assert decisions._signal(decisions.observations(tmp_path)[0])==before


def test_review_closure_needs_observations_and_citations_not_another_prospective_test(tmp_path):
    identifier,outcome=pending(tmp_path)
    result=decisions.conclude_review(tmp_path,conclusion='The saved result leaves the native role unknown; further sources are needed.',
        evidence_refs=[outcome],revises=identifier)
    assert result['status']=='success' and result['data']['review_only']
    assert not result['data']['new_prospective_test_registered']
    assert result['data']['hypothesis']==''
    assert len(list(decisions._root(tmp_path).glob('*/plan.json')))==1
    with pytest.raises(ValueError,match='Only a review closure'):
        decisions.validate_plan(tmp_path,{'tool_name':'read_synthetic_record'},
                                {'read_synthetic_record'},prospective=False)


def test_registered_review_control_accepts_its_minimal_public_contract(tmp_path):
    from enzyme_discovery_agent.tool_registry import execute_tool
    identifier,outcome=pending(tmp_path)
    result=execute_tool('conclude_research_review',{
        'conclusion':'The source added no new functional evidence; the hypothesis remains unresolved.',
        'evidence_refs':[outcome]},injected_args={'run_dir':str(tmp_path)})
    assert result['status']=='success'
    assert result['data']['revises']==identifier


def test_recent_primary_sources_are_not_hidden_by_source_filename_order(tmp_path):
    import os
    folder=tmp_path/'artifacts/literature';folder.mkdir(parents=True)
    old=folder/'z_old.json';new=folder/'a_new.json'
    for path,title,doi in [(old,'Older synthetic source','10.test/old'),(new,'New synthetic source','10.test/new')]:
        path.write_text(json.dumps({'result':{'data':[{'title':title,'doi':doi}]}}))
    os.utime(old,ns=(100,100));os.utime(new,ns=(200,200))
    assert decisions._literature_observations(tmp_path)[0]['facts']['doi']=='10.test/new'
