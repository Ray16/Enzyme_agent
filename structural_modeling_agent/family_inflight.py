"""Run-independent exact-system claims, including protection for legacy workers."""
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

import yaml
from campaign_catalog.lookup import physical_system_key
from enzyme_discovery_agent.state_store import atomic_write_json,state_lock


_QUEUES={}


def _live(owner):
    try:
        if owner['host']==socket.gethostname():
            path=Path('/proc/'+str(owner['pid'])+'/stat')
            if not path.exists():return False
            fields=path.read_text().rsplit(')',1)[1].split()
            return fields[19]==owner['process_start_ticks'] and fields[0] not in {'Z','X'}
        code="import json,sys;from pathlib import Path;r=json.load(sys.stdin);p=Path('/proc/'+str(r['pid'])+'/stat');s=p.read_text().rsplit(')',1)[1].split() if p.exists() else None;print(json.dumps(bool(s and s[19]==r['process_start_ticks'] and s[0] not in ['Z','X'])))"
        command=shlex.join([owner.get('python','python3'),'-c',code])
        response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',owner['host'],command],
            input=json.dumps(owner),text=True,capture_output=True,timeout=12,check=True)
        return json.loads(response.stdout)
    except (OSError,ValueError,KeyError,subprocess.SubprocessError):return None


def _native_pending(run,job_id):
    root=Path(run)/'family_cofold'
    for path in root.glob('workers/*/provider/backend_jobs/'+job_id+'/receipt.json'):
        try:receipt=json.loads(path.read_text())
        except (OSError,ValueError):return True
        if receipt.get('status') in {'queued','running','unknown'}:return True
        if receipt.get('status')=='complete' and not (root/'jobs'/job_id/'complete.json').exists():return True
    return False


def _legacy_busy(parent,key):
    for path in Path(parent).glob('*/family_cofold/claims/*/owner.json'):
        try:
            owner=json.loads(path.read_text())
            if owner.get('host')==socket.gethostname() and owner.get('pid')==os.getpid():continue
            run=path.parents[3];queue=run/'family_cofold/queue.json'
            stamp=(queue.stat().st_mtime_ns,queue.stat().st_size)
            if _QUEUES.get(str(queue),(None,None))[0]!=stamp:
                _QUEUES[str(queue)]=(stamp,{j['id']:j for j in json.loads(queue.read_text())['items']})
            job_id=owner.get('job_id') or path.parent.name
            job=_QUEUES[str(queue)][1].get(job_id)
            if not job:continue
            document=yaml.safe_load(Path(job['input_yaml']).read_text())
            if physical_system_key(document)!=key:continue
            if _native_pending(run,job_id) or _live(owner) is not False:return True
        except (OSError,ValueError,KeyError,TypeError):
            # Missing evidence cannot authorize replacing potentially active work.
            return True
    return False


def _unclaimed_native_busy(run,job,key):
    for path in run.parent.glob('*/family_cofold/workers/*/provider/backend_jobs/*/request.json'):
        source_run=path.parents[6]
        if source_run.resolve()==run.resolve() and path.parent.name==job['id']:continue
        try:
            if not _native_pending(source_run,path.parent.name):continue
            request=json.loads(path.read_text());document=yaml.safe_load(Path(request['input_yaml']).read_text())
            if physical_system_key(document)==key:return True
        except (OSError,ValueError,KeyError,TypeError):return True
    return False


def acquire(run,job,worker_id):
    run=Path(run).resolve();key=physical_system_key(yaml.safe_load(Path(job['input_yaml']).read_text()))
    root=run.parent/'.family_inflight';root.mkdir(exist_ok=True);claim=root/key
    with state_lock(root/'dispatch'):
        if _legacy_busy(run.parent,key) or _unclaimed_native_busy(run,job,key):return None
        if claim.exists():
            path=claim/'owner.json'
            if not path.exists():return None
            owner=json.loads(path.read_text())
            live=_live(owner)
            same_job=Path(owner['run_dir']).resolve()==run and owner['job_id']==job['id']
            if live is not False or (_native_pending(owner['run_dir'],owner['job_id']) and not same_job):return None
            archive=root/'archive';archive.mkdir(exist_ok=True)
            os.rename(claim,archive/(key+'_'+str(time.time_ns())))
        staged=root/'staging'/(key+'_'+str(os.getpid())+'_'+str(time.time_ns()))
        staged.mkdir(parents=True)
        atomic_write_json(staged/'owner.json',{'run_dir':str(run),'discovery_run_id':run.name,'job_id':job['id'],
            'worker_id':worker_id,'host':socket.gethostname(),'pid':os.getpid(),
            'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            'python':sys.executable,'input_yaml':job['input_yaml'],'physical_system_key':key})
        os.rename(staged,claim)
        return claim


def release(claim):
    if claim is None:return
    claim=Path(claim);owner=json.loads((claim/'owner.json').read_text())
    if owner['host']!=socket.gethostname() or owner['pid']!=os.getpid():
        raise RuntimeError('Cannot release another system owner')
    # Retire the populated claim atomically. Interruption cannot leave an empty
    # public claim that every subsequent worker must conservatively preserve.
    with state_lock(claim.parent/'dispatch'):
        archive=claim.parent/'archive';archive.mkdir(exist_ok=True)
        os.rename(claim,archive/(claim.name+'_released_'+str(time.time_ns())))
