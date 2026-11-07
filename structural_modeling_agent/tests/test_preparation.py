import copy
import json
import os
import sys
from pathlib import Path

import pytest
from test_sampling_setup import declaration
from test_modeling_agent import make_agent
from structural_modeling_agent.sampling.preparation import STAGES, evaluate, inspect_pdb, run_operator
from structural_modeling_agent.validation import fingerprint, file_digest


def study(setup, path):
    return dict(schema_version='preparation-study-1.0', setup_id='s', question='Synthetic study', variants=[
        dict(id='a', rationale='Synthetic alternative', setup_sha256=fingerprint(setup), stages={name:
             dict(status='passed', interpretation='Synthetic measurement', criteria={'finite': True}, measurements={'finite': True},
                  artifacts=[dict(path=str(path), sha256=file_digest(path), role='synthetic evidence')], next_action='') for name in STAGES})])


def test_changed_upstream_evidence_invalidates_downstream(tmp_path):
    setup=declaration(tmp_path)
    output=tmp_path/'output.json'; output.write_text('{"finite":true}')
    plan=study(setup,output)
    args=({'s':{'setup':setup}},'AG',{'reaction':'synthetic target transformation'},lambda p:Path(p))
    assert evaluate(plan,*args)['status']=='ready_for_sampling_review'
    output.write_text('{}')
    result=evaluate(plan,*args)
    assert result['status']=='needs_preparation'
    assert not any(s['ready'] for s in result['variants'][0]['stages'])


def test_alternative_failure_and_missing_measurements_prevent_completion(tmp_path):
    setup=declaration(tmp_path); output=tmp_path/'output.json'; output.write_text('{}')
    plan=study(setup,output); plan['variants'].append(copy.deepcopy(plan['variants'][0])); plan['variants'][1]['id']='b'
    plan['variants'][1]['stages']['equilibration']['status']='failed'
    plan['variants'][1]['stages']['equilibration']['next_action']='Investigate coordination instability'
    result=evaluate(plan,{'s':{'setup':setup}},'AG',{'reaction':'synthetic target transformation'},lambda p:Path(p))
    assert result['variants'][0]['ready']
    assert not result['variants'][1]['ready']
    assert result['status']=='needs_preparation'
    plan['variants'][0]['stages']['pilot_review']['measurements']={}
    assert not evaluate(plan,{'s':{'setup':setup}},'AG',{'reaction':'synthetic target transformation'},lambda p:Path(p))['variants'][0]['ready']


def test_agent_preparation_objective_requires_validated_artifacts(tmp_path):
    agent=make_agent(tmp_path,objective='prepare_sampling'); setup=declaration(tmp_path)
    saved=agent.dispatch('declare_sampling_setup',dict(accession='SYNTHETIC1',setup=setup))
    output=agent.run_dir/'artifacts'/'measured.json'; output.parent.mkdir(exist_ok=True); output.write_text('{}')
    plan=study(setup,output); plan['setup_id']=saved['id']
    assert agent.completion_errors()
    result=agent.dispatch('declare_preparation_study',dict(accession='SYNTHETIC1',study=plan))
    assert result['status']=='ready_for_sampling_review'
    assert not agent.completion_errors()
    output.write_text('changed')
    assert agent.completion_errors()
    assert Path(result['artifact']).is_file()


def test_operator_preserves_failure_and_checks_actual_outputs(tmp_path,monkeypatch):
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<2: pytest.skip('Requires two available CPU affinity slots')
    monkeypatch.setattr(os,'getloadavg',lambda:(0,0,0))
    original=os.sched_getaffinity(0)
    result=run_operator(tmp_path/'ok',sys.executable,"from pathlib import Path\nPath('out.json').write_text('{}')\n",[],['out.json'],5,Path)
    assert result['status']=='operator_complete_requires_review'
    assert os.sched_getaffinity(0)==original
    result=run_operator(tmp_path/'missing',sys.executable,'pass',[],['out.json'],5,Path)
    assert result['status']=='operator_failed'
    result=run_operator(tmp_path/'timeout',sys.executable,'import time; time.sleep(10)',[],['out.json'],1,Path)
    assert result['status']=='operator_failed'
    assert 'Operator timed out' in result['errors']
    assert os.sched_getaffinity(0)==original
    with pytest.raises(ValueError):
        run_operator(tmp_path/'escape',sys.executable,'pass',[],['../outside'],5,Path)


def test_preparation_context_keeps_task_instructions_and_evidence(tmp_path):
    agent=make_agent(tmp_path,objective='prepare_sampling')
    agent.task['preparation_instructions']='Review a synthetic bound microstate.'
    context=json.dumps(agent._context())
    assert 'Review a synthetic bound microstate.' in context
    assert 'preparation-study-1.0' in context


def test_operator_rejects_changed_input_before_creating_attempt(tmp_path):
    source=tmp_path/'input';source.write_text('old')
    item=dict(path=str(source),sha256=file_digest(source));source.write_text('new')
    with pytest.raises(ValueError,match='input changed'):
        run_operator(tmp_path/'attempt',sys.executable,'pass',[item],['out'],5,Path)
    assert not (tmp_path/'attempt').exists()


def test_operator_exposes_verified_inputs_to_the_script(tmp_path,monkeypatch):
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<2: pytest.skip('Requires two available CPU affinity slots')
    monkeypatch.setattr(os,'getloadavg',lambda:(0,0,0))
    source=tmp_path/'evidence.pdb';source.write_text('HETATM bound bytes\n')
    item=dict(path=str(source),sha256=file_digest(source))
    code=("from pathlib import Path\n"
          "Path('out.txt').write_text(Path('inputs/evidence.pdb').read_text())\n")
    result=run_operator(tmp_path/'bound',sys.executable,code,[item],['out.txt'],5,Path)
    assert result['status']=='operator_complete_requires_review',result['errors']
    assert (tmp_path/'bound'/'out.txt').read_text()=='HETATM bound bytes\n'
    assert json.loads((tmp_path/'bound'/'inputs.json').read_text())['evidence.pdb']['sha256']==item['sha256']
    named=dict(item,name='renamed.pdb')
    code=("from pathlib import Path\n"
          "Path('out.txt').write_text(Path('inputs/renamed.pdb').read_text())\n")
    assert run_operator(tmp_path/'named',sys.executable,code,[named],['out.txt'],5,Path)['status']=='operator_complete_requires_review'


def test_operator_rejects_colliding_input_names_and_reserved_outputs(tmp_path):
    source=tmp_path/'evidence.pdb';source.write_text('x')
    item=dict(path=str(source),sha256=file_digest(source))
    with pytest.raises(ValueError,match='unique simple local file name'):
        run_operator(tmp_path/'dup',sys.executable,'pass',[item,dict(item)],['out'],5,Path)
    with pytest.raises(ValueError,match='unique simple local file name'):
        run_operator(tmp_path/'slash',sys.executable,'pass',[dict(item,name='a/b')],['out'],5,Path)
    for bad in ('inputs/out.txt','inputs.json','request.json'):
        with pytest.raises(ValueError,match='safe relative path'):
            run_operator(tmp_path/'reserved',sys.executable,'pass',[],[bad],5,Path)
    assert not (tmp_path/'dup').exists() and not (tmp_path/'reserved').exists()
