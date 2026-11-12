"""Actual controller-loop coverage with synthetic, non-computing adapters."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_modeling_agent import make_agent,Message
from structural_modeling_agent.agent import ModelingAgent
from structural_modeling_agent.validation import file_digest,fingerprint
from structural_modeling_agent.barrier.workflow import TOOLS


def prepared(agent,mode='embedded'):
    root=agent.run_dir/'artifacts'/'synthetic_preparation';root.mkdir(parents=True)
    inputs=[]
    for name in ('system.prmtop','frame.npz','reaction.json'):
        path=root/name;path.write_text('Synthetic fixture; not molecular data');inputs.append(path)
    outputs=[]
    names=('input.json','region.json','positions.npy') if mode=='embedded' else ('cluster_candidate.xyz','cluster_candidate.meta.json','cluster_candidate.map.json')
    for name in names:
        path=root/name;path.write_text('Synthetic fixture');outputs.append(path)
    manifest=root/'preparation.json';manifest.write_text(json.dumps(dict(
        status='inputs_prepared_requires_chemical_review',mode=mode,output_dir=str(root),
        input_sha256={str(p):file_digest(p) for p in inputs},output_sha256={str(p):file_digest(p) for p in outputs})))
    return manifest,inputs


def plan_args(manifest,method='dft'):
    return dict(accession='SYNTHETIC1',preparation_path=str(manifest),label='synthetic_v1',method=method,
        rationale='Synthetic workflow contract test; no scientific claim or GPU execution.',
        setup_review=dict(sequence_mapping_reviewed=True,molecular_graph_reviewed=True,protonation_reviewed=True,
            metal_shell_reviewed=True,parameters_reviewed=True,conditional_state='Synthetic test only',
            sources=[dict(path=str(manifest),sha256=file_digest(manifest))]),
        settings=dict(images=3,max_steps=2,wall_seconds=30,route='concerted'))


def fake_backend(tmp_path):
    adapter=tmp_path/'fake_barrier_adapter.py'
    adapter.write_text('''import json,sys,hashlib
from pathlib import Path
request=json.load(sys.stdin)
plan=json.loads(Path(request['plan_path']).read_text())
root=Path(plan['work_dir'])
with (root/'adapter_calls.jsonl').open('a') as handle:handle.write(json.dumps(request['operation'])+'\\n')
reply=dict(idempotency_key=request['idempotency_key'],provider='synthetic_reserved')
if request['operation']=='submit':reply['status']='queued'
else:
    profile=root/'synthetic_profile.json';profile.write_text('Synthetic profile, not a scientific calculation')
    result=dict(status='calculation_complete',calculation_status='converged_profile',
        plan_sha256=request['plan_sha256'],quantity=plan['quantity'],mode=plan['mode'],method=plan['method'],dE_act_kcal=12.0,
        output_sha256={str(profile):hashlib.sha256(profile.read_bytes()).hexdigest()},validated_activation_free_energy=False)
    output=Path(plan['result_path']);output.write_text(json.dumps(result))
    reply.update(status='complete',result_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
print(json.dumps(reply))
''')
    return dict(provider='synthetic_reserved',argv=[sys.executable,str(adapter)],supported_methods=['dft','uma','uma-xtb'],
        resource_policy=dict(reservation_managed=True,max_jobs=1,max_wall_seconds=60))


def assessment(identifier):
    return dict(accession='SYNTHETIC1',study_id=identifier,interpretation='Synthetic adapter contract exercised; no enzyme inference.',
        limitations=['Synthetic calculation output; this is an interface test'],answers_question=True,next_action='Stop the synthetic test')


def test_saved_plan_submission_status_review_and_tamper_detection(tmp_path):
    agent=make_agent(tmp_path,objective='evaluate_barriers');agent.barrier_backend=fake_backend(tmp_path)
    manifest,_=prepared(agent);saved=agent.dispatch('plan_barrier_calculation',plan_args(manifest))
    identity=dict(accession='SYNTHETIC1',study_id=saved['study_id'])
    assert agent.dispatch('submit_barrier_calculation',identity)['status']=='queued'
    assert agent.dispatch('submit_barrier_calculation',identity)['status']=='already_requested'
    assert agent.dispatch('check_barrier_calculation',identity)['status']=='complete'
    assert agent.completion_errors()
    reviewed=agent.dispatch('record_barrier_assessment',assessment(saved['study_id']))
    assert not reviewed['validated_activation_free_energy'] and not agent.completion_errors()
    record=agent.state['candidates']['SYNTHETIC1']['barrier_studies'][saved['study_id']]
    calls=Path(record['plan']['work_dir'],'adapter_calls.jsonl').read_text().splitlines()
    assert calls==['"submit"','"status"']
    profile=Path(record['plan']['work_dir'],'synthetic_profile.json');profile.write_text('changed')
    assert agent.completion_errors()


@pytest.mark.parametrize('mode,method',[('embedded','dft'),('embedded','uma-xtb'),('cluster','uma')])
def test_agent_plans_both_models_with_explicit_method(tmp_path,mode,method):
    agent=make_agent(tmp_path,objective='evaluate_barriers');manifest,_=prepared(agent,mode)
    result=agent.dispatch('plan_barrier_calculation',plan_args(manifest,method))
    assert result['mode']==mode and result['method']==method
    assert not Path(result['plan_path']).with_name('worker.json').exists()


def test_wrong_method_missing_review_and_missing_backend_do_not_submit(tmp_path):
    agent=make_agent(tmp_path,objective='evaluate_barriers');manifest,_=prepared(agent)
    args=plan_args(manifest,'uma')
    with pytest.raises(ValueError,match='does not match'):agent.dispatch('plan_barrier_calculation',args)
    args=plan_args(manifest);args['setup_review']['parameters_reviewed']=False
    with pytest.raises(ValueError,match='unresolved'):agent.dispatch('plan_barrier_calculation',args)
    saved=agent.dispatch('plan_barrier_calculation',plan_args(manifest))
    with pytest.raises(ValueError,match='No barrier backend'):
        agent.dispatch('submit_barrier_calculation',dict(accession='SYNTHETIC1',study_id=saved['study_id']))


def test_uncertain_submission_is_reconciled_without_duplicate_launch(tmp_path,monkeypatch):
    agent=make_agent(tmp_path,objective='evaluate_barriers');agent.barrier_backend=fake_backend(tmp_path)
    manifest,_=prepared(agent);saved=agent.dispatch('plan_barrier_calculation',plan_args(manifest))
    identity=dict(accession='SYNTHETIC1',study_id=saved['study_id']);calls=[]
    def timeout(*args,**kwargs):
        calls.append(args);raise subprocess.TimeoutExpired(args[0],1)
    with monkeypatch.context() as context:
        context.setattr('structural_modeling_agent.barrier.workflow.subprocess.run',timeout)
        with pytest.raises(subprocess.TimeoutExpired):agent.dispatch('submit_barrier_calculation',identity)
        assert agent.dispatch('submit_barrier_calculation',identity)['status']=='already_requested'
    assert len(calls)==1
    assert agent.dispatch('check_barrier_calculation',identity)['status']=='complete'


@pytest.mark.parametrize('objective,production',[('prepare_sampling',False),('prepare_sampling',True),('prepare_inputs',True),('evaluate_barriers',True)])
def test_barrier_tools_are_offered_in_the_real_controller_loop(tmp_path,objective,production):
    agent=make_agent(tmp_path,objective=objective)
    if production:
        agent.task['workflow_mode']='production_cofold';agent.state['task_sha256']=fingerprint(agent.task)
    captured=[]
    class Client:
        def call(self,**kwargs):
            captured.append({tool['function']['name'] for tool in kwargs['tools']})
            assert 'Select embedded DFT/MM' in kwargs['messages'][0]['content']
            assert 'isolated cluster/UMA' in kwargs['messages'][0]['content']
            return SimpleNamespace(choices=[SimpleNamespace(message=Message([]))],
                usage=SimpleNamespace(prompt_tokens=10,completion_tokens=10))
    agent.llm=Client();agent.run(max_turns=1)
    assert captured and TOOLS<=captured[0]


def test_forced_preparation_progress_still_allows_barrier_mutations_and_status(tmp_path):
    agent=make_agent(tmp_path,objective='prepare_sampling');agent.state['preparation_read_turns']=2;captured=[]
    class Client:
        def call(self,**kwargs):
            captured.append({tool['function']['name'] for tool in kwargs['tools']})
            return SimpleNamespace(choices=[SimpleNamespace(message=Message([]))],
                usage=SimpleNamespace(prompt_tokens=10,completion_tokens=10))
    agent.llm=Client();agent.run(max_turns=1)
    assert {'prepare_barrier_inputs','plan_barrier_calculation','submit_barrier_calculation',
            'check_barrier_calculation','record_barrier_assessment'}<=captured[0]


def test_scripted_agent_uses_the_entire_barrier_lifecycle(tmp_path,monkeypatch):
    agent=make_agent(tmp_path,objective='evaluate_barriers');agent.barrier_backend=fake_backend(tmp_path)
    manifest,inputs=prepared(agent)
    def fake_operator(*args,**kwargs):
        return dict(status='operator_complete_requires_review',artifacts=[dict(
            path=str(manifest),sha256=file_digest(manifest),role='preparation.json')])
    monkeypatch.setattr('structural_modeling_agent.sampling.preparation.run_operator',fake_operator)
    observed=[]
    class Client:
        def call(self,**kwargs):
            step=len(observed)
            current=agent.state['candidates']['SYNTHETIC1'].get('barrier_studies',{})
            identifier=next(iter(current),None)
            if step==0:name,args='inspect_barrier_contract',{}
            elif step==1:name,args='prepare_barrier_inputs',dict(accession='SYNTHETIC1',prmtop=str(inputs[0]),
                coordinates=str(inputs[1]),spec_path=str(inputs[2]),python=sys.executable,mode='embedded',seconds=5)
            elif step==2:name,args='plan_barrier_calculation',plan_args(manifest)
            elif step==3:name,args='submit_barrier_calculation',dict(accession='SYNTHETIC1',study_id=identifier)
            elif step==4:name,args='check_barrier_calculation',dict(accession='SYNTHETIC1',study_id=identifier)
            elif step==5:name,args='inspect_barrier_result',dict(path=current[identifier]['result']['path'])
            elif step==6:name,args='record_barrier_assessment',assessment(identifier)
            else:name,args='finish_task',dict(outcome='complete',rationale='Synthetic lifecycle contract verified')
            assert name in {tool['function']['name'] for tool in kwargs['tools']}
            observed.append(name)
            return SimpleNamespace(choices=[SimpleNamespace(message=Message([(name,args)]))],
                usage=SimpleNamespace(prompt_tokens=10,completion_tokens=10))
    agent.llm=Client();report=agent.run(max_turns=8)
    assert report['status']=='complete',report['completion_gaps']
    assert set(observed)>=TOOLS|{'finish_task'}
    assert report['candidates'][0]['barrier_studies']
    index=json.loads((agent.run_dir/'results_index.json').read_text())
    assert index['candidates']['SYNTHETIC1']['barrier_studies']
    assert report['objective']=='evaluate_barriers'


def test_resume_reconciles_pending_barrier_before_asking_model(tmp_path):
    agent=make_agent(tmp_path,objective='evaluate_barriers');agent.barrier_backend=fake_backend(tmp_path)
    agent.state['barrier_backend']=agent.barrier_backend
    manifest,_=prepared(agent);saved=agent.dispatch('plan_barrier_calculation',plan_args(manifest))
    identity=dict(accession='SYNTHETIC1',study_id=saved['study_id'])
    agent.dispatch('submit_barrier_calculation',identity)
    agent.state['status']='waiting_for_backend';agent._save()
    class Client:
        def call(self,**kwargs):
            record=resumed.state['candidates']['SYNTHETIC1']['barrier_studies'][saved['study_id']]
            assert record['status']=='complete'
            return SimpleNamespace(choices=[SimpleNamespace(message=Message([
                ('record_barrier_assessment',assessment(saved['study_id'])),
                ('finish_task',dict(outcome='complete',rationale='Synthetic resumed study reviewed'))]))],
                usage=SimpleNamespace(prompt_tokens=10,completion_tokens=10))
    resumed=ModelingAgent(run_dir=agent.run_dir,resume=True,llm_client=Client())
    report=resumed.run(max_turns=1)
    assert report['status']=='complete'
    root=Path(resumed.state['candidates']['SYNTHETIC1']['barrier_studies'][saved['study_id']]['plan']['work_dir'])
    assert (root/'adapter_calls.jsonl').read_text().splitlines()==['"submit"','"status"']


def test_input_changes_and_resource_budget_block_submission(tmp_path):
    agent=make_agent(tmp_path,objective='evaluate_barriers');agent.barrier_backend=fake_backend(tmp_path)
    manifest,inputs=prepared(agent);saved=agent.dispatch('plan_barrier_calculation',plan_args(manifest))
    identity=dict(accession='SYNTHETIC1',study_id=saved['study_id'])
    agent.barrier_backend['resource_policy']['max_wall_seconds']=1
    with pytest.raises(ValueError,match='wall budget'):agent.dispatch('submit_barrier_calculation',identity)
    agent.barrier_backend['resource_policy']['max_wall_seconds']=60
    inputs[0].write_text('changed topology')
    with pytest.raises(ValueError,match='changed'):agent.dispatch('submit_barrier_calculation',identity)


def test_primary_cofold_requirement_is_preserved_for_optional_barrier_compute(tmp_path):
    agent=make_agent(tmp_path);agent.barrier_backend=fake_backend(tmp_path)
    agent.task['workflow_mode']='production_cofold';manifest,_=prepared(agent)
    saved=agent.dispatch('plan_barrier_calculation',plan_args(manifest))
    with pytest.raises(ValueError,match='propose a modeling decision'):
        agent.dispatch('submit_barrier_calculation',dict(accession='SYNTHETIC1',study_id=saved['study_id']))


def test_wrong_method_receipt_is_not_accepted_as_completed(tmp_path):
    agent=make_agent(tmp_path,objective='evaluate_barriers');agent.barrier_backend=fake_backend(tmp_path)
    manifest,_=prepared(agent);saved=agent.dispatch('plan_barrier_calculation',plan_args(manifest))
    identity=dict(accession='SYNTHETIC1',study_id=saved['study_id'])
    adapter=Path(agent.barrier_backend['argv'][1])
    adapter.write_text(adapter.read_text().replace("method=plan['method']","method='uma'"))
    agent.dispatch('submit_barrier_calculation',identity)
    with pytest.raises(ValueError,match='differs from the plan'):
        agent.dispatch('check_barrier_calculation',identity)
    study=agent.state['candidates']['SYNTHETIC1']['barrier_studies'][saved['study_id']]
    assert study['job']['status']=='unknown' and 'result' not in study
    assert agent.completion_errors()


def test_local_submission_leaves_two_cpus_for_later_preparation(tmp_path,monkeypatch):
    import os
    from structural_modeling_agent.sampling.preparation import run_operator
    from structural_modeling_agent.barrier import provider
    original=set(os.sched_getaffinity(0))
    if len(original)<2:pytest.skip('Controller/preparation resource test needs two CPUs')
    control=sorted(original)[:2];others=[i for i in range(os.cpu_count()) if i not in control]
    agent=make_agent(tmp_path,objective='evaluate_barriers')
    agent.barrier_backend=dict(provider='local_reserved',host='localhost',supported_methods=['dft'],
        python=sys.executable,gpu_reserve=sys.executable,controller_cpus=control+[others[0]],cpus=[others[1]],
        threads=1,gpu=0,minimum_gpu_memory_GiB=24,minimum_host_memory_GiB=32,queue_wall_seconds=30,
        resource_policy=dict(reservation_managed=True,max_jobs=1,max_wall_seconds=30))
    manifest,_=prepared(agent);saved=agent.dispatch('plan_barrier_calculation',plan_args(manifest))
    def reply(*args,**kwargs):
        request=json.loads(kwargs['input'])
        return SimpleNamespace(stdout=json.dumps(dict(idempotency_key=request['idempotency_key'],status='queued')))
    try:
        with monkeypatch.context() as context:
            context.setattr(provider,'verify_cpu_capacity',lambda _:None)
            context.setattr('structural_modeling_agent.barrier.workflow.subprocess.run',reply)
            agent.dispatch('submit_barrier_calculation',dict(accession='SYNTHETIC1',study_id=saved['study_id']))
        assert set(os.sched_getaffinity(0))==set(control)
        result=run_operator(tmp_path/'next_preparation',sys.executable,
            'from pathlib import Path\nPath("next.json").write_text("{}")\n',[],['next.json'],5,Path)
        assert result['status']=='operator_complete_requires_review'
        assert result['request']['worker_cpu'] in control
    finally:os.sched_setaffinity(0,original)
