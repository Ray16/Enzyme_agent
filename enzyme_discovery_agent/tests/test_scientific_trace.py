"""Scientific trace provenance, causal order, recovery and tamper detection."""
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import pytest

from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.models.reasoning_trace import read_scientific_trace,append_scientific_trace
from enzyme_discovery_agent.export_reasoning_trace import export_trace


def proposal(root):
    evidence=decisions.decision_context(root)['observations'][0]['id']
    return {'hypothesis':'A synthetic route may support the target.','alternative':'Only an adjacent transformation may be supported.',
        'scope':'retrieval','prediction':'A record will establish the required transformation.',
        'contrasting_prediction':'The record will describe adjacent chemistry.',
        'why_this_test':'Inspect the relevant functional record.','revision_trigger':'Change priorities if native function differs.',
        'tool_name':'synthetic_retrieval','arguments':{},'evidence_refs':[evidence]}


def test_prediction_is_durable_before_action_result_and_export_is_source_linked(tmp_path):
    plan=proposal(tmp_path)
    identifier=decisions.start_decision(tmp_path,plan,{'synthetic_retrieval'})
    journal=read_scientific_trace(tmp_path)
    assert len(journal)==1 and journal[0]['type']=='prediction_registered'
    assert journal[0]['capture_mode']=='live'
    assert not (decisions._root(tmp_path)/identifier/'result.json').exists()
    outcome=decisions.finish_decision(tmp_path,identifier,{'status':'unavailable','summary':'Synthetic source unavailable'})
    decisions.conclude_review(tmp_path,hypothesis=plan['hypothesis'],alternative=plan['alternative'],
        conclusion='Function remains unknown; assess other candidates.',evidence_refs=[outcome],
        prediction=plan['prediction'],contrasting_prediction=plan['contrasting_prediction'],
        revision_trigger=plan['revision_trigger'],scope='retrieval',belief_update='suspend',revises=identifier)
    result=export_trace(tmp_path)
    assert result['hash_chain_valid'] and result['source_files_intact']
    assert [event['type'] for event in result['events']]==['prediction_registered','observation_recorded','review_concluded']
    assert result['examples'][0]['prospective_capture_verified']
    assert result['examples'][0]['concluded_reviews']
    assert 'Function remains unknown' in (tmp_path/'reasoning/trace.md').read_text()
    before=(tmp_path/'reasoning/trace.jsonl').read_bytes()
    export_trace(tmp_path)
    assert (tmp_path/'reasoning/trace.jsonl').read_bytes()==before


def test_recovered_history_is_never_claimed_as_prospective_capture(tmp_path):
    with patch('enzyme_discovery_agent.models.reasoning_trace.capture_scientific_record'):
        identifier=decisions.start_decision(tmp_path,proposal(tmp_path),{'synthetic_retrieval'})
        decisions.finish_decision(tmp_path,identifier,{'status':'success','data':{'observed':True}})
    result=export_trace(tmp_path)
    assert all(event['capture_mode']=='reconstructed_from_saved_artifact' for event in result['events'])
    assert not result['examples'][0]['prospective_capture_verified']


def test_journal_corruption_is_rejected_without_overwriting_original(tmp_path):
    identifier=decisions.start_decision(tmp_path,proposal(tmp_path),{'synthetic_retrieval'})
    path=tmp_path/'reasoning/trace.jsonl'
    damaged=path.read_text().replace('prediction_registered','prediction_corrupted',1)
    path.write_text(damaged)
    with pytest.raises(ValueError,match='Invalid scientific trace'):
        read_scientific_trace(tmp_path)
    with pytest.raises(ValueError,match='Invalid scientific trace'):
        append_scientific_trace(tmp_path,'other',{},record_id=identifier,occurred_at='synthetic')
    assert path.read_text()==damaged


def test_changed_source_is_reported_without_rewriting_saved_trace(tmp_path):
    identifier=decisions.start_decision(tmp_path,proposal(tmp_path),{'synthetic_retrieval'})
    plan_path=decisions._root(tmp_path)/identifier/'plan.json'
    plan=json.loads(plan_path.read_text());plan['prediction']='Changed after capture'
    plan_path.write_text(json.dumps(plan))
    result=export_trace(tmp_path)
    assert not result['source_files_intact']
    assert result['examples'][0]['prediction']!='Changed after capture'


def test_parallel_trace_appends_have_unbroken_sequence(tmp_path):
    def append(index):
        append_scientific_trace(tmp_path,'synthetic',{'index':index},record_id=str(index),occurred_at='synthetic')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append,range(12)))
    events=read_scientific_trace(tmp_path)
    assert len(events)==12
    assert len({event['id'] for event in events})==12


def test_export_surfaces_controller_turns_and_family_state_decisions(tmp_path):
    # The unified trace must include the two reasoning dimensions that live
    # outside the scientific-trace journal, so one reader shows the whole picture.
    plan=proposal(tmp_path)
    identifier=decisions.start_decision(tmp_path,plan,{'synthetic_retrieval'})
    decisions.finish_decision(tmp_path,identifier,{'status':'unavailable','summary':'none'})
    from enzyme_discovery_agent.models.event_log import append_event
    append_event(tmp_path,'assistant_decision',iteration=1,tools=['search_uniprot'],
                 reasoning='Two folds remain; broaden the stronger one.',reasoning_source='thinking',
                 reasoning_chars=41)
    (tmp_path/'agent_state').mkdir(exist_ok=True)
    (tmp_path/'agent_state'/'family_learning.json').write_text(json.dumps({'families':{
        'ipr:IPR000001':{'name':'Synthetic ligase fold','state':'deprioritized',
            'state_basis':{'basis':'all_assessed_members_deprioritized','members_assessed':3,
                           'from_agent_decision':False,'agent_reason':None}}}}))
    result=export_trace(tmp_path)
    assert result['controller_decisions'][0]['reasoning'].startswith('Two folds remain')
    assert result['controller_decisions'][0]['reasoning_source']=='thinking'
    assert result['family_state_decisions'][0]['state']=='deprioritized'
    text=(tmp_path/'reasoning/trace.md').read_text()
    assert 'Controller turn decisions' in text and 'Two folds remain' in text
    assert 'Family-state decisions' in text and 'all_assessed_members_deprioritized' in text
