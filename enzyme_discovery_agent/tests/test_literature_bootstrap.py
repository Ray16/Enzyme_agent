"""Evidence retrieval must precede cold hypothesis/family expansion, with bounded outages."""
import json
from types import SimpleNamespace
from unittest.mock import Mock,patch
import pytest

from enzyme_discovery_agent.models.literature_bootstrap import bootstrap_status
from enzyme_discovery_agent.models.event_log import append_event


def test_cold_search_is_required_and_outages_are_bounded_without_becoming_absence(tmp_path):
    tools={'search_pubmed','search_uniprot'}
    assert bootstrap_status(tmp_path,tools)['required']
    append_event(tmp_path,'tool_call',tool='search_pubmed',status='error',count=0)
    assert bootstrap_status(tmp_path,tools)['required']
    append_event(tmp_path,'tool_call',tool='search_europepmc',status='error',count=0)
    result=bootstrap_status(tmp_path,tools)
    assert not result['required'] and result['status']=='degraded_no_sources'


def test_retrieved_sources_enable_family_work_without_demanding_repeated_reading(tmp_path):
    append_event(tmp_path,'tool_call',tool='search_pubmed',status='success',count=2)
    result=bootstrap_status(tmp_path,{'search_pubmed'})
    assert not result['required'] and result['status']=='retrieved_sources'
    assert not bootstrap_status(tmp_path,{'search_pubmed'},mechanisms_derived=True)['required']


def test_explicit_same_campaign_derivation_survives_gate_state_reconstruction(tmp_path):
    (tmp_path/'mechanism_families.json').write_text(json.dumps([{'name':'Synthetic saved hypothesis'}]))
    assert bootstrap_status(tmp_path, {'search_pubmed'})['status']=='already_started'
    cold=tmp_path/'cold'
    cold.mkdir()
    assert bootstrap_status(cold, {'search_pubmed'})['required']


@pytest.mark.parametrize('review_due', [False, True])
def test_controller_retrieves_before_firing_mechanism_gate(tmp_path, review_due):
    from openai.types.chat import ChatCompletion
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.gates import GateResult
    source_tool={'type':'function','function':{'name':'search_pubmed','parameters':{
        'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}}
    other_tool={'type':'function','function':{'name':'search_uniprot','parameters':{
        'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}}
    with patch('enzyme_discovery_agent.agent.register_builtin_tools'), \
         patch('enzyme_discovery_agent.agent.get_all_tools',return_value=[source_tool,other_tool]), \
         patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}), \
         patch('enzyme_discovery_agent.models.research_decisions.review_due',
               side_effect=lambda root, **kw: review_due and not bootstrap_status(root, {'search_pubmed'})['required']), \
         patch('enzyme_discovery_agent.models.recovery.next_recovery_action',return_value=None):
        agent=EnzymeAgent(output_dir=tmp_path,token_budget=100)
        selected=[]
        def respond(*args,**kwargs):
            # Schemas are now stable across turns so the provider cache prefix survives;
            # the reconnaissance restriction is carried by the permitted-tool directive and
            # enforced before execution instead of by deleting schemas.
            names=[tool['function']['name'] for tool in kwargs['tools']]
            assert 'search_pubmed' in names
            assert agent._offered_direct_tools=={'search_pubmed'}
            assert 'PERMITTED THIS TURN: search_pubmed' in kwargs['messages'][-1]['content']
            selected.append(['search_pubmed'])
            return ChatCompletion.model_validate({'id':'synthetic','object':'chat.completion','created':0,'model':'synthetic',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                    'tool_calls':[{'id':'source','type':'function','function':{
                        'name':'search_pubmed','arguments':json.dumps({'query':'synthetic target chemistry'})}}]}}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
        def should_fire(state,*args):
            assert state.accumulated_papers[0]['title']=='Synthetic primary evidence'
            return True
        def mechanism(state,*args):
            agent.llm.resource_stop_reason='Synthetic evidence precondition verified'
            return GateResult(intercept=True,state_updates={'mechanism_families_done':True})
        agent._mechanism_family_gate.should_fire=should_fire
        agent._mechanism_family_gate.execute=mechanism
        for gate in (agent._literature_harvest_gate,agent._family_survey_gate,agent._systematic_gate,
                     agent._gate_pipeline.gates[2],agent._gate_pipeline.gates[3]):
            gate.should_fire=Mock(return_value=False)
        with patch.object(agent.llm,'call',side_effect=respond), \
             patch('enzyme_discovery_agent.agent.execute_tool',return_value={'status':'success','count':1,
                   'summary':'Retrieved one primary source','data':[{'title':'Synthetic primary evidence','abstract':'Synthetic retrieved chemistry.'}]}), \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',0)):
            agent.run('synthetic transformation',substrate_smiles='CCO')
        assert len(selected)==1


def test_actual_retrieved_passages_reach_mechanism_derivation(tmp_path):
    from enzyme_discovery_agent.tools import mechanism_families as mechanisms
    paper={'title':'Synthetic newly reported catalyst','abstract':'Distinct observed native chemistry from the source.',
           'doi':'synthetic:primary'}
    seen=[]
    def derive(query,*args,**kwargs):
        seen.append(query)
        return [{'name':'Synthetic candidate mechanism'}]
    with patch.object(mechanisms.config,'ENABLE_MECHANISM_FAMILY_CACHE',False), \
         patch.object(mechanisms,'_derive_once',side_effect=derive), \
         patch.object(mechanisms,'family_passes_constraints',return_value=(True,'')):
        result=mechanisms.derive_mechanism_families('synthetic transformation','CCO',SimpleNamespace(active_label='synthetic'),
            max_families=1,min_families=1,max_attempts=1,reflect=False,retrieved_papers=[paper])
    assert result and paper['title'] in seen[0] and paper['abstract'] in seen[0]
