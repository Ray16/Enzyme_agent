import json
from pathlib import Path
import signal
import sys
import os
from types import ModuleType,SimpleNamespace

import pytest

pytest.importorskip('ase')
from structural_modeling_agent.barrier import provider,runtime


@pytest.mark.parametrize('mode,method',[('cluster','uma'),('embedded','dft'),('embedded','uma-xtb')])
def test_typed_runtime_dispatch_preserves_method_budget_and_output_identity(tmp_path,monkeypatch,mode,method):
    prepared=tmp_path/'prepared';prepared.mkdir()
    names=('cluster_candidate.meta.json','cluster_candidate.xyz','cluster_candidate.map.json') if mode=='cluster' else ('input.json','region.json','positions.npy')
    for name in names:(prepared/name).write_text('Synthetic operator fixture')
    producer,hashes=provider.freeze_producer(tmp_path/'producer')
    work=tmp_path/'work';work.mkdir()
    plan=dict(mode=mode,method=method,prepared_directory=str(prepared),work_dir=str(work),
        result_path=str(work/'result.json'),quantity='constrained_electronic_energy_gap',
        input_sha256={str(p):provider.digest(p) for p in prepared.iterdir()},
        producer_directory=str(producer),producer_sha256=hashes,
        settings=dict(images=3,max_steps=2,wall_seconds=7,route='concerted',model='synthetic-model',basis='synthetic-basis',xc='synthetic-xc'))
    path=work/'plan.json';provider.write_json(path,plan);calls=[]
    if mode=='cluster':
        from structural_modeling_agent.barrier import scan
        def fake_scan(meta,**kwargs):
            calls.append(kwargs)
            result=dict(all_optimizations_converged=True,mechanism_gate=dict(ok=True),
                        i_reactant=0,i_ts=1,i_product=2,dE_act_kcal=12.,scan_curve=[{},{},{}])
            (Path(meta).parent/'synthetic_result.json').write_text(json.dumps(result))
            return result
        monkeypatch.setattr(scan,'scan_barrier',fake_scan)
    else:
        from structural_modeling_agent.barrier import path as paths,stationary
        def fake_explore(snapshot,out,**kwargs):
            calls.append(kwargs);out=Path(out);out.mkdir()
            images=[]
            for i in range(3):
                pos=out/f'point{i}.npy';pos.write_text('synthetic positions');images.append(dict(positions=str(pos)))
            result=dict(all_optimizations_converged=True,images=images)
            (out/'result.json').write_text(json.dumps(result));return result
        def fake_summary(source,out):
            result=dict(has_interior_maximum=True,constrained_path_gap_kcal=12.)
            Path(out).write_text(json.dumps(result));return result
        monkeypatch.setattr(paths,'explore',fake_explore);monkeypatch.setattr(stationary,'summarize',fake_summary)
    result=runtime.calculate(path,provider.digest(path))
    assert result['calculation_status']=='converged_profile' and result['dE_act_kcal']==12
    assert result['plan_sha256']==provider.digest(path)
    assert result['method']==method and not result['validated_activation_free_energy']
    assert calls[0]['images']==3 and calls[0]['max_steps']==2 and calls[0]['wall_seconds']==7
    if mode=='embedded':assert calls[0]['engine']==method and calls[0]['model']=='synthetic-model'
    for name,h in plan['input_sha256'].items():assert provider.digest(name)==h
    for name,h in result['output_sha256'].items():assert provider.digest(name)==h
    with pytest.raises(FileExistsError):runtime.calculate(path,provider.digest(path))


def test_real_cpu_worker_preflights_executes_and_reuses_exact_outputs(tmp_path,monkeypatch):
    from structural_modeling_agent.barrier import worker
    producer,_=provider.freeze_producer(tmp_path/'producer')
    cpus=sorted(os.sched_getaffinity(0));preflight=tmp_path/'preflight.json';output=tmp_path/'output.json'
    monkeypatch.setattr(os,'getloadavg',lambda:(0,0,0))
    for key in worker.THREAD_VARS+worker.RUNTIME_THREAD_VARS:monkeypatch.setenv(key,'1')
    cfg=dict(device='cpu',cpus=cpus,threads=1,cwd=str(producer),producer_directory=str(producer),
        preflight_argv=[sys.executable,'-c',f'from pathlib import Path; Path({str(preflight)!r}).write_text(\'{{"passed":true}}\')'],
        preflight_result=str(preflight),tasks=[dict(id='cpu_fixture',
            argv=[sys.executable,'-c',f'from pathlib import Path; Path({str(output)!r}).write_text(\'{{"passed":true}}\')'],
            outputs=[str(output)],required_fields=dict(passed=True))])
    manifest=tmp_path/'worker.json';provider.write_json(manifest,cfg)
    original={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        worker.run(manifest);worker.run(manifest)
    finally:
        for s,handler in original.items():signal.signal(s,handler)
    state=json.loads((tmp_path/'tasks/cpu_fixture/state.json').read_text())
    assert state['status']=='complete' and state['attempts']==1
    assert state['owner']['cpus']==cpus and state['producer_sha256']
    assert not (tmp_path/'tasks/cpu_fixture/claim').exists()
    assert len(list((tmp_path/'tasks/cpu_fixture').glob('attempt*.log')))==1


def test_failed_cpu_preflight_does_not_claim_a_scientific_task(tmp_path,monkeypatch):
    from structural_modeling_agent.barrier import worker
    producer,_=provider.freeze_producer(tmp_path/'producer');preflight=tmp_path/'preflight.json'
    monkeypatch.setattr(os,'getloadavg',lambda:(0,0,0))
    for key in worker.THREAD_VARS+worker.RUNTIME_THREAD_VARS:monkeypatch.setenv(key,'1')
    manifest=tmp_path/'worker.json';provider.write_json(manifest,dict(device='cpu',cpus=sorted(os.sched_getaffinity(0)),
        threads=1,producer_directory=str(producer),cwd=str(producer),preflight_result=str(preflight),
        preflight_argv=[sys.executable,'-c',f'from pathlib import Path; Path({str(preflight)!r}).write_text(\'{{"passed":false}}\')'],
        tasks=[dict(id='must_not_run',argv=[sys.executable,'-c','raise RuntimeError("should not execute")'],outputs=[])]))
    with pytest.raises(RuntimeError,match='before claiming'):worker.run(manifest)
    assert not (tmp_path/'tasks').exists()


def test_checkpoint_preflight_and_changed_weight_rejection_use_no_gpu(tmp_path,monkeypatch):
    producer,hashes=provider.freeze_producer(tmp_path/'producer')
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_text('synthetic cached weights')
    source=tmp_path/'input';source.write_text('synthetic')
    root=tmp_path/'job';root.mkdir()
    plan=dict(mode='cluster',method='uma',work_dir=str(root),result_path=str(root/'result.json'),
        producer_directory=str(producer),producer_sha256=hashes,input_sha256={str(source):provider.digest(source)},
        settings=dict(model='synthetic-model',images=3,max_steps=1,wall_seconds=5))
    path=root/'plan.json';provider.write_json(path,plan)
    torch=ModuleType('torch');torch.cuda=SimpleNamespace(is_available=lambda:True,
        get_device_properties=lambda _:SimpleNamespace(name='synthetic-device',total_memory=64*1024**3))
    fairchem=ModuleType('fairchem');fairchem.__path__=[]
    core=ModuleType('fairchem.core');core.__path__=[]
    resolved=[]
    def resolve(name):
        resolved.append(name);return SimpleNamespace(path=str(checkpoint),atom_refs={'synthetic':1},form_elem_refs=None)
    core.pretrained_mlip=SimpleNamespace(resolve_checkpoint=resolve);fairchem.core=core
    monkeypatch.setitem(sys.modules,'torch',torch);monkeypatch.setitem(sys.modules,'fairchem',fairchem)
    monkeypatch.setitem(sys.modules,'fairchem.core',core)
    report=runtime.preflight(path,24,provider.digest(path))
    assert report['passed'] and resolved==['synthetic-model']
    assert report['checkpoint']['sha256']==provider.digest(checkpoint)
    checkpoint.write_text('changed cached weights')
    with pytest.raises(ValueError,match='checkpoint changed'):runtime.calculate(path,provider.digest(path))
    def missing(_):raise FileNotFoundError('Synthetic offline cache miss')
    core.pretrained_mlip.resolve_checkpoint=missing
    with pytest.raises(RuntimeError,match='offline cache miss'):runtime.preflight(path,24,provider.digest(path))
    assert not (root/'result.json').exists()
