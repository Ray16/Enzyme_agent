"""Provider ownership/provenance contracts without launching GPU processes."""
from pathlib import Path
import copy
import json
import os
import socket
import sys
from types import SimpleNamespace

import pytest

from structural_modeling_agent.barrier import provider


def config():
    cpus=sorted(os.sched_getaffinity(0));first=cpus[0]
    pool=[i for i in range(os.cpu_count()) if i!=first]
    return dict(provider='local_reserved',host='localhost',python=sys.executable,gpu_reserve=sys.executable,
        controller_cpus=[first,pool[0],pool[1]],cpus=[pool[2]],threads=1,gpu=0,minimum_gpu_memory_GiB=24,
        minimum_host_memory_GiB=32,
        queue_wall_seconds=60,resource_policy=dict(reservation_managed=True,max_jobs=1,max_wall_seconds=60))


def fixture_plan(tmp_path):
    source=tmp_path/'input.dat';source.write_text('synthetic')
    producer,hashes=provider.freeze_producer(tmp_path/'producer')
    root=tmp_path/'job';root.mkdir()
    plan=dict(input_sha256={str(source):provider.digest(source)},producer_directory=str(producer),
              producer_sha256=hashes,work_dir=str(root),result_path=str(root/'result.json'),
              settings=dict(wall_seconds=5))
    path=root/'plan.json';provider.write_json(path,plan)
    request=dict(idempotency_key='synthetic-key',plan_path=str(path),plan_sha256=provider.digest(path),config=config())
    return plan,request


def test_config_requires_disjoint_bounded_resource_sets():
    settings=config();assert provider.validate_config(settings)
    broken=copy.deepcopy(settings);broken['cpus']=broken['controller_cpus'][:1]
    with pytest.raises(ValueError,match='overlap'):provider.validate_config(broken)
    broken=copy.deepcopy(settings);broken['threads']=2
    with pytest.raises(ValueError,match='threads'):provider.validate_config(broken)
    broken=copy.deepcopy(settings);broken['environment']={'CUDA_VISIBLE_DEVICES':'0'}
    with pytest.raises(ValueError,match='override'):provider.validate_config(broken)


def test_frozen_canonical_producer_has_real_hashes_and_detects_changes(tmp_path):
    from structural_modeling_agent.barrier.worker import producer_hashes
    plan,request=fixture_plan(tmp_path)
    hashes=producer_hashes(plan['producer_directory'])
    assert 'structural_modeling_agent/barrier/embedded.py' in hashes
    assert provider._checked_plan(request)==plan
    path=Path(plan['producer_directory'])/'structural_modeling_agent/barrier/embedded.py'
    path.write_text(path.read_text()+'\n# changed\n')
    with pytest.raises(ValueError,match='producer changed'):provider._checked_plan(request)
    with pytest.raises(ValueError,match='no barrier'):producer_hashes(tmp_path/'missing')


def test_unrecorded_launch_claim_is_uncertain_and_never_reclaimed(tmp_path,monkeypatch):
    plan,request=fixture_plan(tmp_path);root=Path(plan['work_dir']);(root/'launch_claim').mkdir()
    monkeypatch.setattr(provider,'validate_config',lambda _:None)
    monkeypatch.setattr(provider.subprocess,'Popen',lambda *a,**k:pytest.fail('Must not relaunch an unresolved claim'))
    assert provider.submit(request,plan)['status']=='unknown'
    assert (root/'launch_claim').exists()


def test_completed_receipt_requires_real_worker_output_hash(tmp_path):
    plan,request=fixture_plan(tmp_path);root=Path(plan['work_dir'])
    provider.write_json(root/'adapter_receipt.json',dict(idempotency_key='synthetic-key'))
    state=root/'tasks/calculation/state.json';state.parent.mkdir(parents=True)
    result=Path(plan['result_path']);result.write_text('{}')
    provider.write_json(state,dict(status='complete',output_sha256={str(result):provider.digest(result)}))
    assert provider.status(request,plan)['status']=='complete'
    result.write_text('changed')
    with pytest.raises(ValueError,match='output changed'):provider.status(request,plan)


def test_submission_builds_reserved_worker_and_pins_plan_identity(tmp_path,monkeypatch):
    plan,request=fixture_plan(tmp_path);root=Path(plan['work_dir']);captured=[]
    monkeypatch.setattr(os,'getloadavg',lambda:(0,0,0))
    monkeypatch.setattr(provider,'verify_cpu_capacity',lambda _:None)
    controller=set(request['config']['controller_cpus'][:-1])
    monkeypatch.setattr(os,'sched_getaffinity',lambda _: controller)
    def launch(argv,**kwargs):
        captured.append((argv,kwargs));return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(provider.subprocess,'Popen',launch)
    assert provider.submit(request,plan)['status']=='queued'
    worker=json.loads((root/'worker.json').read_text())
    assert worker['cpus']==request['config']['cpus']
    assert '--expected-plan-sha256' in worker['tasks'][0]['argv']
    assert request['plan_sha256'] in worker['tasks'][0]['argv']
    assert worker['preflight_argv'] and worker['producer_directory']==plan['producer_directory']
    assert '--minimum-host-memory-GiB' in worker['preflight_argv']
    assert captured[0][0][2]==str(request['config']['controller_cpus'][-1])
    assert captured[0][1]['env']['HF_HUB_OFFLINE']=='1'
    assert 'CUDA_VISIBLE_DEVICES' not in captured[0][1]['env'] or os.environ.get('CUDA_VISIBLE_DEVICES')==captured[0][1]['env']['CUDA_VISIBLE_DEVICES']
    assert provider.submit(request,plan)['status']=='queued'
    assert len(captured)==1


def test_runtime_rejects_changed_plan_before_any_calculator_import(tmp_path):
    from structural_modeling_agent.barrier.runtime import checked_plan
    plan,request=fixture_plan(tmp_path)
    assert checked_plan(request['plan_path'],request['plan_sha256'])==plan
    path=Path(request['plan_path']);path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='after submission'):checked_plan(path,request['plan_sha256'])
