"""Retrieved sources must inform a recorded decision before broad automation."""
import json
from unittest.mock import Mock,patch

import pytest

from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.models.reasoning_trace import read_scientific_trace
from enzyme_discovery_agent.export_reasoning_trace import export_trace


def source(root):
    path=root/'artifacts/literature/search_synthetic_001.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'result':{'status':'success','data':[{
        'title':'Synthetic precursor production differs from the requested bond-forming step',
        'doi':'synthetic:source', 'abstract':'The source describes production of a building block. '
        'It does not report the subsequent coupling of two independent inputs.'}]}}))
    return path


def proposal(root):
    packet=decisions.decision_context(root)
    paper=next(row for row in packet['observations'] if row['kind']=='retrieved_literature_evidence')
    return {'hypothesis':'The retrieved record may concern precursor production rather than the requested step.',
        'alternative':'A complementary source may report the direct transformation.',
        'scope':'retrieval','prediction':'The next source will identify the actual bond-changing participants.',
        'contrasting_prediction':'It will describe only building-block production.',
        'why_this_test':'Distinguish synthesis of inputs from catalysis of the requested reaction.',
        'revision_trigger':'Change the search route when retrieved passages remain on precursor production.',
        'evidence_refs':[paper['id']], 'tool_name':'search_pubmed','arguments':{'query':'synthetic direct transformation'}}


def test_initial_sources_are_citable_and_their_integrity_is_checked(tmp_path):
    path=source(tmp_path)
    assert decisions.review_due(tmp_path)
    plan=proposal(tmp_path)
    identifier=decisions.start_decision(tmp_path,plan,{'search_pubmed'})
    assert read_scientific_trace(tmp_path)[0]['type']=='prediction_registered'
    assert not (decisions._root(tmp_path)/identifier/'result.json').exists()
    exported=export_trace(tmp_path)
    assert any(row['path']==str(path.relative_to(tmp_path)) for row in exported['source_checks'])
    path.write_text('{}')
    with pytest.raises(ValueError,match='source artifact changed'):
        decisions.validate_plan(tmp_path,plan,{'search_pubmed'})
    assert not export_trace(tmp_path)['source_files_intact']


def test_review_closes_and_resumes_work_without_closing_scientific_uncertainty(tmp_path):
    source(tmp_path)
    plan=proposal(tmp_path)
    identifier=decisions.start_decision(tmp_path,plan,{'search_pubmed'})
    outcome=decisions.finish_decision(tmp_path,identifier,{'status':'empty','summary':'No direct transformation found in this route.'})
    decisions.conclude_review(tmp_path,hypothesis=plan['hypothesis'],alternative=plan['alternative'],
        conclusion='The initial source is indirect. Resume bounded work while retaining complementary routes.',
        evidence_refs=[outcome],prediction=plan['prediction'],contrasting_prediction=plan['contrasting_prediction'],
        revision_trigger=plan['revision_trigger'],scope='search_strategy',revises=identifier)
    assert not decisions.review_due(tmp_path)
    assert json.loads((decisions._root(tmp_path)/'state.json').read_text())['initial_sources_reviewed']


def test_controller_saves_a_prediction_before_early_automation(tmp_path):
    from enzyme_discovery_agent.agent import EnzymeAgent
    from enzyme_discovery_agent.tool_registry import get_tool_info
    import enzyme_discovery_agent.tools.research_decisions
    from openai.types.chat import ChatCompletion
    schemas=[get_tool_info(name)['schema'] for name in ('plan_research_step','get_decision_context')]
    schemas.append({'type':'function','function':{'name':'search_pubmed','parameters':{'type':'object','properties':{'query':{'type':'string'}}}}})
    with patch('enzyme_discovery_agent.agent.register_builtin_tools'), \
         patch('enzyme_discovery_agent.agent.get_all_tools',return_value=schemas), \
         patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning',return_value={}), \
         patch('enzyme_discovery_agent.models.recovery.next_recovery_action',return_value=None):
        agent=EnzymeAgent(output_dir=tmp_path,token_budget=100)
        source(agent.run_dir)
        calls=[]
        def respond(*args,**kwargs):
            calls.append(kwargs)
            if len(calls)>1:
                agent.llm.resource_stop_reason='Synthetic planning boundary verified'
                return None
            return ChatCompletion.model_validate({'id':'synthetic','object':'chat.completion','created':0,'model':'synthetic',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':[
                    {'id':'plan','type':'function','function':{'name':'plan_research_step','arguments':json.dumps(proposal(agent.run_dir))}}]}}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
        def execute(*args,**kwargs):
            assert read_scientific_trace(agent.run_dir)[0]['type']=='prediction_registered'
            return {'status':'success','summary':'Synthetic new source retrieved','data':[]}
        for gate in (agent._literature_harvest_gate,agent._mechanism_family_gate,agent._family_survey_gate):
            gate.should_fire=Mock(side_effect=AssertionError('Automation ran before initial source review'))
        with patch.object(agent.llm,'call',side_effect=respond), \
             patch('enzyme_discovery_agent.agent.execute_tool',side_effect=execute) as performed, \
             patch('enzyme_discovery_agent.output.finalizer.RunFinalizer.snapshot',return_value=('checkpoint',0)):
            agent.run('synthetic transformation',substrate_smiles='CCO')
        performed.assert_called_once()
        assert [event['type'] for event in read_scientific_trace(agent.run_dir)]==['prediction_registered','observation_recorded']
