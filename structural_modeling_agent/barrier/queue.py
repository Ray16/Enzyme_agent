"""Bounded local continuation queue; delegates GPU ownership to gpu_reserve.

Only predeclared task manifests are launched. Dependency failures remain visible;
no chemistry is invented and no global locks are removed.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from .jobs import atomic_json,sha256
from .worker import THREAD_VARS


def available_cpus(requested):
    requested=set(requested)
    if not requested<=set(range(os.cpu_count())) or max(os.getloadavg())+len(requested)>os.cpu_count():return False
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if b'python' not in p.read_bytes().split(b'\x00')[0]:continue
            affinity=set(os.sched_getaffinity(int(p.parent.name)))
            if len(affinity)<os.cpu_count() and affinity&requested:return False
        except OSError:pass
    return True


def complete(manifest):
    root=Path(manifest).parent;cfg=json.loads(Path(manifest).read_text())
    for task in cfg['tasks']:
        p=root/'tasks'/task['id']/'state.json'
        if not p.exists():return False
        d=json.loads(p.read_text())
        if d.get('status')!='complete':return False
        if any(not Path(f).is_file() or sha256(f)!=d.get('output_sha256',{}).get(f) for f in task['outputs']):return False
    return True


def run(path):
    path=Path(path).resolve();cfg=json.loads(path.read_text());root=path.parent;start=time.monotonic();active={};finished=set()
    record=dict(pid=os.getpid(),host=socket.gethostname(),queue_sha256=sha256(path),started_at=time.time(),jobs={})
    atomic_json(root/'queue_status.json',record)
    while time.monotonic()-start<cfg.get('max_wall_seconds',21600):
        for name,proc in list(active.items()):
            if proc.poll() is not None:
                record['jobs'][name]['returncode']=proc.returncode
                record['jobs'][name]['status']='finished' if proc.returncode==0 else 'launch_or_worker_failed'
                job=next(j for j in cfg['jobs'] if j['id']==name)
                worker=json.loads(Path(job['manifest']).read_text())
                states=[Path(job['manifest']).parent/'tasks'/t['id']/'state.json' for t in worker['tasks']]
                preflight=Path(worker['preflight_result']) if worker.get('preflight_result') else None
                preflight_failed=preflight is not None and preflight.exists() and json.loads(preflight.read_text()).get('passed') is False
                contention=any(p.exists() and json.loads(p.read_text()).get('status')=='contention_stopped' for p in states)
                # A lost atomic reservation or transient preflight refusal is
                # retried without consuming a scientific task attempt.
                if preflight_failed:
                    record['jobs'][name]['status']='preflight_failed';finished.add(name)
                elif contention or (proc.returncode!=0 and not any(s.exists() for s in states)):
                    record['jobs'][name]['status']='preflight_or_reservation_retry'
                else:finished.add(name)
                del active[name]
        for job in cfg['jobs']:
            name=job['id']
            if name in active or name in finished:continue
            if any(not Path(f).exists() for f in job.get('required_inputs',[])):continue
            if not all(complete(f) for f in job.get('depends_on',[])):continue
            manifest=Path(job['manifest']);worker=json.loads(manifest.read_text())
            if complete(manifest):finished.add(name);continue
            if not available_cpus(worker['cpus']):continue
            free=subprocess.run([cfg['gpu_reserve'],'list','--hosts',socket.gethostname()],text=True,capture_output=True,timeout=20)
            if free.returncode or f"{socket.gethostname()} {worker['gpu']}" not in free.stdout.splitlines():continue
            env=[v+'='+str(worker['threads']) for v in THREAD_VARS]
            argv=[cfg['gpu_reserve'],'run',str(worker['gpu']),'--label',name,'--','taskset','-c',','.join(map(str,worker['cpus'])),'env',*env,cfg['python'],'-u','-m','enzyme_barrier.worker',str(manifest)]
            with (root/(name+'.queue.log')).open('a') as log:
                proc=subprocess.Popen(argv,cwd=worker['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;record['jobs'][name]=dict(pid=proc.pid,argv=argv,status='launched',manifest_sha256=sha256(manifest),started_at=time.time())
            print(json.dumps(record['jobs'][name]),flush=True)
        atomic_json(root/'queue_status.json',record)
        if len(finished)==len(cfg['jobs']):break
        time.sleep(10)
    record.update(finished_at=time.time(),pending=[j['id'] for j in cfg['jobs'] if j['id'] not in finished and j['id'] not in active])
    atomic_json(root/'queue_status.json',record)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');a=p.parse_args(argv);run(a.config)


if __name__=='__main__':main()
