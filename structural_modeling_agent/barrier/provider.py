"""Local reservation-aware adapter using the existing queue and worker.

Only trusted deployment configuration supplies devices, CPU sets and interpreter.
The agent supplies a typed, hashed scientific plan, never a shell command.
The JSON stdin/stdout protocol also permits explicit external scheduler adapters.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def freeze_producer(destination):
    """Freeze numerical sources so updates cannot alter an active/resumed job."""
    root=Path(destination).resolve();package=root/'structural_modeling_agent'/'barrier'
    package.mkdir(parents=True,exist_ok=False)
    (package.parent/'__init__.py').write_text('"""Frozen structural modeling calculation sources."""\n')
    source=Path(__file__).resolve().parent
    for file in source.glob('*.py'):shutil.copy2(file,package/file.name)
    shutil.copytree(source/'specs',package/'specs')
    hashes={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    return root,hashes


def validate_config(config):
    if config.get('provider')!='local_reserved':raise ValueError('Expected local_reserved provider')
    if config.get('host') not in {'localhost',socket.gethostname(),socket.gethostname().split('.')[0]}:
        raise ValueError('Local barrier backend host must match this node; use an explicit remote adapter otherwise')
    for name in ('python','gpu_reserve'):
        path=Path(config.get(name,''))
        if not path.is_absolute() or not path.is_file():raise ValueError(f'Explicit installed barrier {name} required')
    worker=config.get('cpus');controller=config.get('controller_cpus')
    for name,values in [('cpus',worker),('controller_cpus',controller)]:
        if not isinstance(values,list) or not values or len(set(values))!=len(values) or any(type(i) is not int or not 0<=i<os.cpu_count() for i in values):
            raise ValueError(f'Explicit valid {name} required')
    if set(worker)&set(controller):raise ValueError('Controller and worker CPU sets overlap')
    if len(controller)<3:raise ValueError('Allocate two agent/preparation CPUs and a separate queue CPU')
    if config.get('resource_policy',{}).get('max_jobs')!=1:
        raise ValueError('Local backend owns one declared GPU; set max_jobs=1 or use an explicit fleet adapter')
    if type(config.get('threads')) is not int or not 1<=config['threads']<=len(worker):
        raise ValueError('Worker native threads exceed the assigned CPU set')
    if type(config.get('gpu')) is not int or config['gpu']<0:raise ValueError('Explicit GPU index required')
    if type(config.get('queue_wall_seconds')) is not int or config['queue_wall_seconds']<1:
        raise ValueError('Explicit bounded queue wall allowance required')
    for field in ('minimum_gpu_memory_GiB','minimum_host_memory_GiB'):
        if type(config.get(field)) not in (int,float) or not 0<config[field]<10000:
            raise ValueError(f'Explicit positive {field} required')
    if config.get('resource_policy',{}).get('reservation_managed') is not True:
        raise ValueError('Reservation-managed resource policy required')
    for key in config.get('environment',{}):
        if key not in {'OPENMM_PLUGIN_DIR','BARRIER_AMBERHOME','BARRIER_XTB_COMMAND','HF_HOME','TMPDIR'}:
            raise ValueError(f'Unsupported runtime environment override: {key}')
        if not isinstance(config['environment'][key],str):raise ValueError('Environment values must be strings')
    return config


def verify_cpu_capacity(config):
    """Check declared CPU ownership before pinning/launching; never steal CPUs."""
    wanted=set(config['controller_cpus'])|set(config['cpus'])
    if max(os.getloadavg())+len(wanted)>os.cpu_count():raise RuntimeError('Insufficient verified CPU capacity')
    memory={line.split(':')[0]:line.split(':')[1].strip().split()[0] for line in Path('/proc/meminfo').read_text().splitlines()}
    if int(memory['MemAvailable'])*1024 < config['minimum_host_memory_GiB']*1024**3:
        raise RuntimeError('Insufficient available host memory for the configured barrier runtime')
    ancestors={os.getpid()};pid=os.getpid()
    while pid>1:
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError,IndexError):break
        if pid in ancestors:break
        ancestors.add(pid)
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        pid=int(path.parent.name)
        if pid in ancestors:continue
        try:
            if b'python' not in path.read_bytes().split(b'\0')[0]:continue
            affinity=set(os.sched_getaffinity(pid))
            if len(affinity)<os.cpu_count() and affinity&wanted:
                raise RuntimeError(f'Configured barrier CPU set overlaps a live Python process: {pid}')
        except (ProcessLookupError,FileNotFoundError):continue
        except PermissionError as exc:raise RuntimeError('Cannot verify live CPU ownership') from exc


def process_matches(owner):
    """PID plus start time, never age, determines whether our local owner lives."""
    if owner.get('host') not in {socket.gethostname(),socket.gethostname().split('.')[0]}:return None
    try:
        text=Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')',1)[1].split()
        if text[0]=='Z':return False
        return text[19]==str(owner['process_start'])
    except FileNotFoundError:return False
    except (OSError,KeyError,IndexError):return None


def _checked_plan(request):
    path=Path(request['plan_path']).resolve(strict=True)
    if digest(path)!=request['plan_sha256']:raise ValueError('Plan changed before provider submission')
    plan=json.loads(path.read_text())
    for name,h in plan['input_sha256'].items():
        if digest(name)!=h:raise ValueError(f'Input changed: {name}')
    producer=Path(plan['producer_directory'])
    for name,h in plan['producer_sha256'].items():
        if digest(producer/name)!=h:raise ValueError('Frozen numerical producer changed')
    return plan


def status(request,plan):
    root=Path(plan['work_dir']);key=request['idempotency_key'];receipt=root/'adapter_receipt.json'
    base=dict(idempotency_key=key,provider='local_reserved')
    if not receipt.exists():
        return dict(base,status='unknown' if (root/'launch_claim').exists() else 'not_found')
    owner=json.loads(receipt.read_text())
    if owner['idempotency_key']!=key:raise ValueError('Existing launch belongs to a different request')
    state=root/'tasks'/'calculation'/'state.json'
    if state.exists():
        task=json.loads(state.read_text());value=task['status']
        if value=='complete':
            for file,h in task.get('output_sha256',{}).items():
                if digest(file)!=h:raise ValueError('Completed worker output changed')
            output=Path(plan['result_path'])
            if not task.get('output_sha256') or str(output) not in task['output_sha256']:
                raise ValueError('Worker did not fingerprint the requested result')
            return dict(base,status='complete',result_sha256=digest(output),worker_state=str(state))
        if value=='failed':return dict(base,status='failed',worker_state=str(state))
        alive=process_matches(task.get('owner',{}))
        if alive is not False:
            return dict(base,status='running' if alive is True and value=='running' else 'unknown',worker_state=str(state))
    preflight=root/'preflight.json'
    if preflight.exists() and json.loads(preflight.read_text()).get('passed') is False:
        return dict(base,status='failed',reason='Runtime preflight failed',preflight=str(preflight))
    alive=process_matches(owner)
    return dict(base,status='queued' if alive else 'failed' if alive is False else 'unknown',
                reason=None if alive else 'Queue owner stopped; preserve partial outputs and inspect logs')


def submit(request,plan):
    config=validate_config(request['config']);root=Path(plan['work_dir'])
    if (root/'adapter_receipt.json').exists() or (root/'launch_claim').exists():return status(request,plan)
    if set(os.sched_getaffinity(0))!=set(config['controller_cpus'][:-1]):
        raise ValueError('Adapter must inherit exactly its configured controller CPU set')
    verify_cpu_capacity(config)
    # A launch claim is never reclaimed automatically, including after a crash
    # between Popen and recording its PID. Status then reports uncertainty.
    claim=root/'launch_claim'
    try:claim.mkdir()
    except FileExistsError:return status(request,plan)
    write_json(claim/'request.json',dict(idempotency_key=request['idempotency_key'],plan_sha256=request['plan_sha256']))
    producer=Path(plan['producer_directory']);python=config['python']
    runtime=[python,'-u','-m','structural_modeling_agent.barrier.runtime',
             '--expected-plan-sha256',request['plan_sha256']]
    manifest=dict(gpu=config['gpu'],cpus=config['cpus'],threads=config['threads'],
        cwd=str(producer),producer_directory=str(producer),
        preflight_argv=runtime+['--preflight',request['plan_path'],'--minimum-gpu-memory-GiB',str(config['minimum_gpu_memory_GiB']),
                                '--minimum-host-memory-GiB',str(config['minimum_host_memory_GiB'])],
        preflight_result=str(root/'preflight.json'),
        tasks=[dict(id='calculation',argv=runtime+[request['plan_path']],outputs=[plan['result_path']],
                    required_fields=dict(status='calculation_complete'),
                    max_wall_seconds=plan['settings']['wall_seconds']+120,max_attempts=1)])
    manifest_path=root/'worker.json';write_json(manifest_path,manifest)
    queue=dict(gpu_reserve=config['gpu_reserve'],python=python,max_wall_seconds=config['queue_wall_seconds'],
        jobs=[dict(id='barrier-'+request['plan_sha256'][:12],manifest=str(manifest_path),required_inputs=list(plan['input_sha256']))])
    queue_path=root/'queue.json';write_json(queue_path,queue)
    env=os.environ.copy();env.update(config.get('environment',{}));env['PYTHONPATH']=str(producer)
    # Cache installation is a separate preparation operation, not a download
    # silently performed while holding a scientific GPU reservation.
    env['HF_HUB_OFFLINE']='1'
    for name in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[name]='1'
    argv=['taskset','-c',str(config['controller_cpus'][-1]),python,'-u','-m',
          'structural_modeling_agent.barrier.queue',str(queue_path)]
    with (root/'queue.log').open('xb') as log:
        process=subprocess.Popen(argv,cwd=producer,env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    started=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    receipt=dict(idempotency_key=request['idempotency_key'],pid=process.pid,host=socket.gethostname(),
        process_start=started,argv=argv,plan_sha256=request['plan_sha256'],started_at=time.time(),
        worker_manifest_sha256=digest(manifest_path),queue_manifest_sha256=digest(queue_path))
    write_json(root/'adapter_receipt.json',receipt)
    return dict(idempotency_key=request['idempotency_key'],status='queued',provider='local_reserved')


def main():
    request=json.load(sys.stdin);plan=_checked_plan(request)
    if request['operation']=='submit':reply=submit(request,plan)
    elif request['operation']=='status':reply=status(request,plan)
    else:raise ValueError('Unknown barrier adapter operation')
    print(json.dumps(reply,allow_nan=False))


if __name__=='__main__':main()
