"""Cache reuse, assessment publication and source-scoped alias regressions."""
import copy
import csv
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.tests.test_target_review_gate import setup, assessment


def test_assessment_schemas_and_common_context_are_stable_across_batches(tmp_path):
    from enzyme_discovery_agent.models import target_review as review
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
    setup(tmp_path)
    meta=load_candidate_registry(tmp_path)['candidates']['SYNTH1']['metadata']
    meta=copy.deepcopy(meta)
    for claim in meta['claim_validation']['claims']:claim['subject']='SYNTH2'
    update_candidate_registry(tmp_path,'SYNTH2','evaluated',copy.deepcopy(meta))
    calls=[]
    class Client:
        def call(self,**kwargs):
            calls.append(kwargs)
            row=json.loads(kwargs['messages'][-1]['content'])['candidates'][0]
            value={**assessment(),'accession':row['accession'],'source_refs':[row['source_id']]}
            from enzyme_discovery_agent.claude_subscription import Message
            message=Message(None,[{'id':'test','function':{
                'name':'record_target_assessment','arguments':json.dumps(value)}}])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client=Client()
    assert review.run_review(tmp_path,client,accessions=['SYNTH1'])['assessed']==['SYNTH1']
    assert review.run_review(tmp_path,client,accessions=['SYNTH2'])['assessed']==['SYNTH2']
    assert calls[0]['tools']==calls[1]['tools']
    assert calls[0]['messages'][:2]==calls[1]['messages'][:2]
    assert calls[0]['messages'][-1]!=calls[1]['messages'][-1]
    assert 'enum' not in calls[0]['tools'][0]['function']['parameters']['properties']['accession']


def test_csv_uses_current_assessment_even_when_native_projection_omits_it(tmp_path):
    from enzyme_discovery_agent.models import target_review as review
    from enzyme_discovery_agent.output.csv_generator import generate_candidates_csv
    packet=setup(tmp_path)
    review.save_assessment(tmp_path,packet,assessment('prioritize'))
    generate_candidates_csv(tmp_path,[{'data':{'accession':'SYNTH1','protein_name':'Synthetic catalyst'}}],
        ranking_entries=[{'accession':'SYNTH1','rank':1,'tier':1}])
    row,=list(csv.DictReader((tmp_path/'ranked_candidates.csv').open()))
    assert row['target_assessment_decision']=='prioritize'
    assert row['target_activity_evidence']=='mechanistic_transfer'


def test_history_appends_between_compaction_boundaries(tmp_path,monkeypatch):
    from enzyme_discovery_agent import config
    from enzyme_discovery_agent.context import build_model_context
    monkeypatch.setattr(config,'MODEL_CONTEXT_COMPACT_INTERVAL',3)
    with patch('enzyme_discovery_agent.context.build_research_state',return_value={}):
        messages=[{'role':'system','content':'constraints'},{'role':'user','content':'target'}]
        messages.extend([{'role':'assistant','content':'first'},{'role':'user','content':'observation'}])
        first=build_model_context(messages,tmp_path)
        messages.extend([{'role':'assistant','content':'second'},{'role':'user','content':'next'}])
        second=build_model_context(messages,tmp_path)
        messages.extend([{'role':'assistant','content':'third'},{'role':'user','content':'more'}])
        third=build_model_context(messages,tmp_path)
        assert second[:len(first)-1]==first[:-1]
        assert third[:len(second)-1]==second[:-1]
        original=copy.deepcopy(messages)
        messages.extend([{'role':'assistant','content':'fourth'},{'role':'user','content':'latest'}])
        compacted=build_model_context(messages,tmp_path)
        assert 'first' not in [m.get('content') for m in compacted]
        assert messages[:len(original)]==original


def test_rewritten_target_invalidates_history_window(tmp_path):
    from enzyme_discovery_agent.context import build_model_context
    with patch('enzyme_discovery_agent.context.build_research_state',return_value={}):
        messages=[{'role':'system','content':'rules'},{'role':'user','content':'old target'}]
        build_model_context(messages,tmp_path)
        messages[1]['content']='new target'
        result=build_model_context(messages,tmp_path)
        assert result[1]['content']=='new target'


def test_species_prefixed_alias_uses_a_source_scoped_fallback(tmp_path):
    from enzyme_discovery_agent.models.literature_leads import register_literature_lead
    from enzyme_discovery_agent.tools.identity_resolution import resolve_literature_lead
    from enzyme_discovery_agent.tests.test_identity_resolution import entry
    path=tmp_path/'artifacts/paper.txt';path.parent.mkdir()
    path.write_text('Synthetic transferase (SsZyxQ) from Synthetic species was characterized.')
    lead=register_literature_lead(tmp_path,{'name':'SsZyxQ'},{'doi':'10.test/alias'},path)
    def search(query,**kwargs):
        return [entry()] if 'gene_exact:"ZyxQ"' in query else []
    with patch('enzyme_discovery_agent.tools.fulltext._uniprot_search',side_effect=search) as lookup:
        result=resolve_literature_lead(lead,run_dir=tmp_path)
    assert result['data']['resolved_accession']=='P12345'
    assert result['data']['activity_validated'] is False
    assert lookup.call_count==2
    assert all('organism_name:"Synthetic species"' in c.args[0] for c in lookup.call_args_list)


def test_reverse_source_association_and_ambiguous_alias_stay_scoped():
    from enzyme_discovery_agent.tools.identity_resolution import source_associations, select_sequence
    from enzyme_discovery_agent.tests.test_identity_resolution import entry
    sources=[{'path':'synthetic','sha256':'synthetic','source_id':'synthetic',
              'text':'Synthetic species transferase (SsZyxQ) was characterized.'}]
    assert source_associations('SsZyxQ',sources)[0]['organism']=='Synthetic species'
    selected,result=select_sequence([entry(),entry('P54321','MACD')],
        'SsZyxQ','Synthetic species',aliases=['ZyxQ'])
    assert selected is None and result['status']=='ambiguous'


def test_explicit_source_taxon_supersedes_an_unsourced_registry_hint(tmp_path):
    from enzyme_discovery_agent.models.literature_leads import register_literature_lead
    from enzyme_discovery_agent.tools.identity_resolution import resolve_literature_lead
    from enzyme_discovery_agent.tests.test_identity_resolution import entry
    path=tmp_path/'artifacts/paper.txt';path.parent.mkdir()
    path.write_text('ZyxQ from Synthetic species was characterized.')
    lead=register_literature_lead(tmp_path,{'name':'ZyxQ'},{'doi':'10.test/hint'},path)
    update_candidate_registry(tmp_path,lead,None,{'organism':'Different species'})
    with patch('enzyme_discovery_agent.tools.fulltext._uniprot_search',return_value=[entry()]):
        result=resolve_literature_lead(lead,run_dir=tmp_path)
    assert result['data']['resolved_accession']=='P12345'
    assert result['data']['superseded_organism_hint']=='Different species'


def test_successful_subscription_auth_is_cached_and_settings_invalidate_it(tmp_path,monkeypatch):
    from enzyme_discovery_agent import claude_subscription as transport
    from enzyme_discovery_agent.tests.test_claude_subscription import stream
    settings=tmp_path/'settings.json';settings.write_text('{}')
    monkeypatch.setenv('ENZYME_CLAUDE_SETTINGS',str(settings))
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        if 'auth' in argv:
            return SimpleNamespace(returncode=0,stdout=json.dumps({
                'loggedIn':True,'authMethod':'claude.ai','apiProvider':'firstParty'}))
        return SimpleNamespace(returncode=0,stdout=stream({'type':'text','text':'ok'},subtype='success'))
    with patch.object(transport.subprocess,'run',side_effect=run):
        for _ in range(2):transport.call_claude([],[], 'claude-sonnet-5',30,None)
        assert sum('auth' in c for c in calls)==1
        settings.write_text('{"changed":true}')
        transport.call_claude([],[], 'claude-sonnet-5',30,None)
    assert sum('auth' in c for c in calls)==2


def test_failed_auth_never_enters_the_success_cache(tmp_path,monkeypatch):
    from enzyme_discovery_agent import claude_subscription as transport
    settings=tmp_path/'settings.json';settings.write_text('{}')
    monkeypatch.setenv('ENZYME_CLAUDE_SETTINGS',str(settings))
    with patch.object(transport.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='{}')) as run:
        for _ in range(2):
            with pytest.raises(transport.ClaudeTransportError,match='authentication'):
                transport.call_claude([],[],'claude-sonnet-5',30,None)
    assert run.call_count==2


def test_review_tranche_yields_to_a_model_decision_before_native_gathering(tmp_path,monkeypatch):
    from unittest.mock import Mock
    from enzyme_discovery_agent import config
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.models import target_review as review
    monkeypatch.setattr(config,'AUTO_TARGET_REVIEW',True)
    monkeypatch.setattr(config,'TARGET_REVIEW_BATCHES_BEFORE_DECISION',1)
    order=[]
    with patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}):
        agent=EnzymeAgent(output_dir=tmp_path,token_budget=1000)
        update_candidate_registry(agent.run_dir,'SYNTH_NEW','screened',{'protein_name':'Synthetic catalyst'})
        gate=agent._gate_pipeline.gates[2];gate.should_fire=Mock(return_value=True)
        gate.execute=Mock(side_effect=AssertionError('A review tranche must yield to discovery strategy'))
        def assess(*args,**kwargs):
            order.append('review')
            return {'assessed':[],'errors':[],'unassessed':[]}
        def decide(**kwargs):
            order.append('model');agent.llm.resource_stop_reason='Synthetic decision reached'
            return None
        with patch.object(review,'review_queue',return_value=[{'accession':'SYNTH_READY'}]), \
             patch.object(review,'run_review',side_effect=assess),patch.object(agent.llm,'call',side_effect=decide), \
             patch('enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status',return_value={'required':False}), \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',0)):
            agent.run('Synthetic transformation',substrate_smiles='CCO')
        gate.execute.assert_not_called()
    assert order==['review','model']


@pytest.mark.parametrize('adaptive',['0','1'])
def test_default_assessment_queue_is_enabled_in_both_modes(adaptive,monkeypatch):
    import os,subprocess,sys
    env=os.environ.copy();env['ENZYME_AGENT_ADAPTIVE_REASONING']=adaptive
    env.pop('ENZYME_AGENT_AUTO_TARGET_REVIEW',None)
    output=subprocess.check_output([sys.executable,'-c',
        'from enzyme_discovery_agent import config;print(config.AUTO_TARGET_REVIEW)'],env=env,text=True)
    assert output.strip()=='True'


def test_disabled_review_does_not_keep_an_old_review_scheduling_debt(tmp_path,monkeypatch):
    from unittest.mock import Mock
    from enzyme_discovery_agent import config
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.gates import GateResult
    monkeypatch.setattr(config,'AUTO_TARGET_REVIEW',False)
    monkeypatch.setattr(config,'ADAPTIVE_REASONING',False)
    with patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}):
        agent=EnzymeAgent(output_dir=tmp_path,token_budget=1000)
        agent._automatic_review_batches=1
        update_candidate_registry(agent.run_dir,'SYNTH_NEW','screened',{'protein_name':'Synthetic catalyst'})
        gate=agent._gate_pipeline.gates[2];gate.should_fire=Mock(return_value=True)
        def native(*args):
            agent.llm.resource_stop_reason='Synthetic native queue serviced'
            return GateResult(intercept=True)
        gate.execute=Mock(side_effect=native)
        with patch('enzyme_discovery_agent.models.literature_bootstrap.bootstrap_status',return_value={'required':False}), \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',0)):
            agent.run('Synthetic transformation',substrate_smiles='CCO')
        gate.execute.assert_called_once()
