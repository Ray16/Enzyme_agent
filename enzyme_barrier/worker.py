"""Bounded, resource-checked tasks on one gate-reserved GPU or assigned CPU set.

Invoke only through gpu_reserve run. Claims and outputs are immutable per task
identity; a changed command/input starts a new task. No automatic stale reclaim.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from .jobs import atomic_json,sha256


THREAD_VARS=('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


def gpu_processes(index):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[1]) for row in rows.splitlines() if row.split(',')[0].strip()==uuid}


def descendants(pid):
    found={pid}
    for _ in range(5):
        before=len(found)
        for f in Path('/proc').glob('[0-9]*/stat'):
            try:
                # comm may contain spaces; parse fields after the closing parenthesis.
                fields=f.read_text().rsplit(')',1)[1].split()
                if int(fields[1]) in found:found.add(int(f.parent.name))
            except (OSError,ValueError,IndexError):pass
        if len(found)==before:break
    return found


def run(manifest):
    path=Path(manifest).resolve();cfg=json.loads(path.read_text());root=path.parent
    gpu=None if cfg.get('device')=='cpu' else int(cfg['gpu'])
    cpus=set(cfg['cpus']);threads=int(cfg.get('threads',len(cpus)))
    if not cpus or threads>len(cpus):raise ValueError('Invalid CPU/thread allocation')
    if set(os.sched_getaffinity(0))!=cpus:raise ValueError('Worker must inherit its exact assigned CPU set')
    if gpu is not None and os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('GPU visibility does not match gate assignment')
    if max(os.getloadavg())+len(cpus)>os.cpu_count():raise RuntimeError('CPU capacity unavailable')
    for v in THREAD_VARS:
        if os.environ.get(v)!=str(threads):raise ValueError(f'Missing thread limit {v}')
    occupied=gpu_processes(gpu) if gpu is not None else set()
    if occupied:raise RuntimeError(f'GPU occupied before task: {occupied}')
    owner=dict(host=socket.gethostname(),pid=os.getpid(),process_start=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
               gpu=gpu,cpus=sorted(cpus),threads=threads,manifest_sha256=sha256(path),started_at=time.time(),
               source_sha256={str(f.resolve()):sha256(f) for f in Path(__file__).parent.glob('*.py')},
               thread_limits={v:os.environ.get(v) for v in THREAD_VARS})
    owner_file=root/(f'worker_gpu{gpu}.json' if gpu is not None else f'worker_{path.stem}.json')
    atomic_json(owner_file,owner)
    failed=0;stop=[False];child=[None]
    def terminate(signum,frame):
        stop[0]=True
        if child[0] is not None and child[0].poll() is None:os.killpg(child[0].pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    for task in cfg['tasks']:
        if stop[0] or failed>=3:break
        tid=task['id'];directory=root/'tasks'/tid;directory.mkdir(parents=True,exist_ok=True)
        producer_root=Path(cfg.get('producer_directory',Path(__file__).resolve().parent.parent))
        producer_hashes={str(f.relative_to(producer_root)):sha256(f) for f in (producer_root/'enzyme_barrier').glob('*.py')}
        identity=__import__('hashlib').sha256(json.dumps(dict(task=task,producer=producer_hashes),sort_keys=True).encode()).hexdigest()
        state=directory/'state.json'
        if state.exists():
            previous=json.loads(state.read_text())
            if previous.get('identity')!=identity:raise ValueError(f'Task identity changed: {tid}')
            if previous.get('status')=='complete':
                if all(Path(f).is_file() and sha256(f)==previous.get('output_sha256',{}).get(f) for f in task['outputs']):continue
                raise ValueError(f'Completed output changed or is missing: {tid}; start a versioned task')
            if previous.get('attempts',0)>=task.get('max_attempts',2):continue
        else:previous={}
        claim=directory/'claim'
        try:claim.mkdir()
        except FileExistsError:raise RuntimeError(f'Unresolved task claim {claim}; inspect owner before recovery')
        atomic_json(claim/'owner.json',owner)
        attempt=previous.get('attempts',0)+1
        record=dict(identity=identity,status='running',attempts=attempt,argv=task['argv'],owner=owner,started_at=time.time(),producer_sha256=producer_hashes)
        atomic_json(state,record)
        try:
            with (directory/f'attempt{attempt}.log').open('x') as log:
                child_env=os.environ.copy()
                if cfg.get('producer_directory'):child_env['PYTHONPATH']=str(producer_root)
                proc=subprocess.Popen(task['argv'],cwd=str(producer_root) if cfg.get('producer_directory') else cfg['cwd'],env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child[0]=proc;foreign=None
                while proc.poll() is None:
                    time.sleep(5)
                    if time.time()-record['started_at']>task.get('max_wall_seconds',14400):
                        os.killpg(proc.pid,signal.SIGTERM);record['stop_reason']='task wall budget reached';break
                    try:foreign=(gpu_processes(gpu)-descendants(os.getpid())) if gpu is not None else set()
                    except Exception as exc:foreign={f'occupancy_check_failed:{exc}'}
                    if foreign:
                        os.killpg(proc.pid,signal.SIGTERM);stop[0]=True
                        record['stop_reason']=f'GPU contention/uncertainty: {foreign}';break
                try:rc=proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
            record.update(returncode=rc,finished_at=time.time())
            valid=rc==0 and all(Path(f).is_file() for f in task['outputs'])
            for f in task['outputs']:
                if valid and f.endswith('.json'):
                    d=json.loads(Path(f).read_text())
                    if d.get('passed') is False:valid=False
                    for key,expected in task.get('required_fields',{}).items():
                        if d.get(key)!=expected:valid=False
            record['status']='complete' if valid else ('contention_stopped' if foreign else 'failed')
            if foreign:record['attempts']=attempt-1
            if valid:record['output_sha256']={f:sha256(f) for f in task['outputs']};failed=0
            else:failed+=1
            atomic_json(state,record)
        finally:
            # Release only after the child has stopped writing.
            if child[0] is None or child[0].poll() is not None:
                (claim/'owner.json').unlink();claim.rmdir()
        print(json.dumps(record),flush=True)
        if record['status']!='complete' and task.get('stop_on_failure',True):break
    owner.update(finished_at=time.time(),stopped=stop[0]);atomic_json(owner_file,owner)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');a=p.parse_args(argv);run(a.manifest)


if __name__=='__main__':main()
