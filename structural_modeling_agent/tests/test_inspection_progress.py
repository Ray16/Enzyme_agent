"""Read history survives compaction and unchanged rereads cannot consume a run."""
import json
from types import SimpleNamespace

from test_modeling_agent import make_agent,Message


def test_saved_inspection_history_distinguishes_a_new_page_and_preserves_state(tmp_path):
    agent=make_agent(tmp_path)
    for i in range(4):agent._record_inspection('read_evidence',{'path':'synthetic','offset':0},{'text':'same source'},'synthetic-artifact')
    assert agent.state['repeated_inspections']==3
    assert 'completed_inspections' in agent._context()[-1]['content']
    history=next(iter(agent.state['inspection_history'].values()))
    assert history['reads']==4
    agent._record_inspection('read_evidence',{'path':'synthetic','offset':100},{'text':'new passage'},'new-artifact')
    assert agent.state['repeated_inspections']==0
    assert len(agent.state['inspection_history'])==2


def test_schema_copy_read_through_another_artifact_is_not_new_information(tmp_path):
    agent=make_agent(tmp_path)
    schema={'assembly':{'expected':'source-supported chain inventory'}}
    agent._record_inspection('inspect_validation_contract',{'section':'setup'},schema,'first-artifact')
    for index in range(3):
        agent._record_inspection('read_evidence',{'path':f'copy-{index}'},
            {'text':json.dumps(schema),'offset':0,'next_offset':None},f'artifact-{index}')
    assert agent.state['repeated_inspections']==3
    assert len(agent.state['inspection_history'])==1


def test_production_mode_retains_complete_contract_and_hides_nonsetup_tools(tmp_path):
    class Client:
        def call(self,**kwargs):
            text=json.dumps(kwargs['messages'])
            assert 'validation_plan' in text and 'source_record' in text and 'scientific_reasoning' in text
            names={tool['function']['name'] for tool in kwargs['tools']}
            assert 'inspect_validation_contract' not in names
            assert 'submit_cofold' not in names and 'check_cofold' not in names
            assert 'prepare_bioemu2' not in names and 'propose_model' in names
            return SimpleNamespace(choices=[SimpleNamespace(message=Message([('finish_task',{
                'outcome':'inconclusive','rationale':'Synthetic task lacks independent setup sources.'})]))],
                usage=SimpleNamespace(prompt_tokens=20,completion_tokens=10))
    agent=make_agent(tmp_path,client=Client());agent.task['workflow_mode']='production_cofold';agent.state['task']=agent.task
    assert agent.run(max_turns=1,token_budget=100)['status']=='inconclusive'


def test_existing_prediction_review_exposes_validation_after_a_proposal(tmp_path):
    class Client:
        def call(self,**kwargs):
            names={tool['function']['name'] for tool in kwargs['tools']}
            assert 'validate_predictions' in names
            assert 'submit_cofold' not in names and 'render_inputs' not in names
            return SimpleNamespace(choices=[SimpleNamespace(message=Message([('finish_task',{
                'outcome':'inconclusive','rationale':'Synthetic saved models need source transfer review.'})]))],
                usage=SimpleNamespace(prompt_tokens=20,completion_tokens=10))
    agent=make_agent(tmp_path,client=Client());agent.task['workflow_mode']='production_cofold'
    agent.task['candidates'][0]['evidence']['existing_cofolds']=[{'source':'synthetic-pool'}]
    agent.state['task']=agent.task;agent.state['reuse_review_required']=True
    agent.state['candidates']['SYNTHETIC1']['revisions']=[{'revision':1,'execution':{'input':'synthetic'}}]
    assert agent.run(max_turns=1,token_budget=100)['status']=='inconclusive'


def test_live_loop_offers_new_actions_instead_of_repeating_the_same_inspection(tmp_path):
    class Client:
        def __init__(self):self.turn=0;self.last_tools=None
        def call(self,**kwargs):
            self.turn+=1;self.last_tools={tool['function']['name'] for tool in kwargs['tools']}
            calls=([('inspect_candidate',{'accession':'SYNTHETIC1'})] if self.turn<5 else
                   [('finish_task',{'outcome':'inconclusive','rationale':'The supplied sources cannot resolve the required assembly.'})])
            return SimpleNamespace(choices=[SimpleNamespace(message=Message(calls))],
                usage=SimpleNamespace(prompt_tokens=20,completion_tokens=10))
    client=Client();agent=make_agent(tmp_path,client=client)
    result=agent.run(max_turns=6,token_budget=1000)
    assert client.turn==5
    assert 'inspect_candidate' not in client.last_tools
    assert 'propose_model' in client.last_tools and 'retrieve_reference' in client.last_tools
    assert result['status']=='inconclusive'
    assert result['usage']['total_tokens']==150


def test_the_candidate_record_survives_compaction_during_the_setup_stage(tmp_path):
    """Without this the model rereads the record every turn and never proposes."""
    from structural_modeling_agent.agent import CONTEXT_ALLOWANCE
    agent=make_agent(tmp_path)
    assert agent._before_proposal()
    record={'candidate':{'accession':'SYNTH1','evidence':{'note':'distinctive-setup-evidence'}}}
    agent._record_inspection('inspect_candidate',{},record,'synthetic-artifact')
    assert agent.state['setup_anchor']
    context=agent._context()
    system=context[0]['content']
    assert 'Candidate record under setup' in system and 'distinctive-setup-evidence' in system
    # The anchor buys its own room; it must not evict conversational history.
    assert len(json.dumps(context))<=CONTEXT_ALLOWANCE+len(agent.state['setup_anchor'])


def test_a_record_too_large_to_carry_whole_is_not_anchored_as_another_preview(tmp_path):
    from structural_modeling_agent.agent import SETUP_ANCHOR_LIMIT
    agent=make_agent(tmp_path)
    agent._record_inspection('inspect_candidate',{},{'bulk':'x'*(SETUP_ANCHOR_LIMIT+1)},'synthetic-artifact')
    assert agent.state['setup_anchor'] is None
    assert 'Candidate record under setup' not in agent._context()[0]['content']


def test_the_setup_anchor_is_dropped_once_a_proposal_exists(tmp_path):
    agent=make_agent(tmp_path)
    agent._record_inspection('inspect_candidate',{},{'candidate':{'note':'setup-only-evidence'}},'synthetic-artifact')
    assert 'setup-only-evidence' in agent._context()[0]['content']
    accession=next(iter(agent.state['candidates']))
    agent.state['candidates'][accession]['revisions'].append({'revision':1})
    assert not agent._before_proposal()
    assert 'setup-only-evidence' not in agent._context()[0]['content']
