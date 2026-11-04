"""Coordinate fixed-GPU workers through the shared reservation gate only."""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from .evaluate import verified_paths
from .worker import atomic_json

REMOTE_LAUNCH = r'''
import json,os,subprocess,sys,time
from pathlib import Path
c=json.load(sys.stdin)
if c.get('defer_if_foreign_gpu_processes'):
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
 if p.returncode:raise RuntimeError('Cannot verify resident GPU processes')
 for line in p.stdout.splitlines():
  if not line.strip():continue
  try:
   pid=int(line.strip()); cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  except (ValueError,OSError):raise RuntimeError('Cannot verify a resident GPU process')
  if not cmd or str(Path(c['manifest']).parent).encode() not in cmd:
   raise RuntimeError('Foreign GPU workload present on node; admission deferred')
used=set()
for p in Path('/proc').glob('[0-9]*'):
 try:
  if not p.joinpath('cmdline').read_bytes():continue
  a=os.sched_getaffinity(int(p.name))
  if len(a)<os.cpu_count()/2:used.update(a)
 except (OSError,ProcessLookupError):pass
available=sorted(os.sched_getaffinity(0)-used)
slots=c['cpu_slots']
if len(available)<slots or os.getloadavg()[0]+slots>os.cpu_count():
 raise RuntimeError('Insufficient verified CPU capacity')
mem_available_kib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
if mem_available_kib/1024<c.get('min_available_ram_mib',0):
 raise RuntimeError('Insufficient verified host RAM')
cpus=available[-slots:]
env=dict(os.environ)
for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
env['LD_LIBRARY_PATH']=c['library_path']
env['TOKENIZERS_PARALLELISM']='false'
argv=['taskset','-c',','.join(map(str,cpus)),c['gate'],'run',str(c['gpu']),'--label',c['label'],'--',c['python'],'-u','-m','enzyme_design_agent.saturation.worker',c['manifest'],'--cache',c['cache']]
if c.get('max_sequence_length'):
 argv.extend(['--max-sequence-length',str(c['max_sequence_length'])])
with open(c['log'],'a') as log:
 p=subprocess.Popen(argv,cwd=c['repo'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':p.pid,'start_ticks':Path(f'/proc/{p.pid}/stat').read_text().split()[21], 'cpu_affinity':cpus,'argv':argv,'started':time.time(),'host':c['host'],'gpu':c['gpu'],'log':c['log']}
print(json.dumps(record),flush=True)
'''
REMOTE_ALIVE = r'''
import json,sys
from pathlib import Path
records=json.load(sys.stdin);out=[]
for r in records:
 try:
  fields=Path('/proc/'+str(r['pid'])+'/stat').read_text().split()
  live=fields[21]==r['start_ticks'] and fields[2]!='Z'
 except OSError:live=False
 out.append(live)
print(json.dumps(out))
'''


def remote(host,python,code,payload):
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join([python,'-c',code])]
    p=subprocess.run(cmd,input=json.dumps(payload),text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout)


def ordered_free_slots(output,hosts):
    slots=[line.split() for line in output.splitlines()]
    slots=[slot for slot in slots if len(slot)==2 and slot[0] in hosts and slot[1].isdigit()]
    return sorted(slots,key=lambda slot:hosts[slot[0]].get('priority',100))


def guard_allows(out, host, max_age=120):
    """Fail closed if the independent resource monitor is stale or reports risk."""
    try:
        s=json.loads((out/'resource_guard_state.json').read_text())
        if s['host'] != os.uname().nodename or time.time()-s['updated']>max_age:return False
        fields=Path('/proc',str(s['pid']),'stat').read_text().split()
        if fields[21]!=s['start_ticks'] or fields[2]=='Z':return False
        observation=s['host_status'][host]
        return observation['ok'] is True and time.time()-observation['checked']<=max_age
    except (OSError,ValueError,KeyError,TypeError):return False


def external_partition_claims_live(out,claimed):
    """Recognize only claims held by this run's live external partition controller."""
    if not claimed:return False
    try:
        control=json.loads((out/'aurora_partition_state.json').read_text())
        if control['phase'] not in {'reserving','preparing','submitting','queued','running','packing','transferring','waiting_for_transfer_auth','promoting','cancelling','recovering'}:return False
        if control['host']!=os.uname().nodename:return False
        stat=Path('/proc',str(control['pid']),'stat').read_text().split()
        if stat[21]!=control['start_ticks'] or stat[2]=='Z':return False
        for ident in claimed:
            owner=json.loads((out/'claims'/ident/'owner.json').read_text())
            if owner.get('kind')!='aurora_partition' or owner.get('assignment_id')!=control['assignment_id']:
                return False
            if owner.get('pid')!=control['pid'] or owner.get('start_ticks')!=control['start_ticks'] or owner.get('host')!=control['host']:
                return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def coordinate(config_path):
    cfg=json.loads(Path(config_path).read_text());mp=Path(cfg['manifest']);m=json.loads(mp.read_text());out=Path(m['output_dir'])
    (out/'fleet_logs').mkdir(exist_ok=True)
    state_path=out/'fleet_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'workers':[],'disabled_slots':[]}
    jobs_by_id = {j['id']: j for j in m['jobs']}
    for host in cfg['hosts']:
        pre=json.loads(Path(cfg['hosts'][host]['preflight']).read_text())
        if not pre.get('ok'):raise RuntimeError(f'{host}: preflight did not pass')
    lock=out/'coordinator.claim'
    lock.mkdir()  # Never steal a previous coordinator's ownership.
    atomic_json(lock/'owner.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21]})
    try:
        while True:
            active=[]
            for host,setting in cfg['hosts'].items():
                prior=[r for r in state['workers'] if r['host']==host and r.get('active')]
                if not prior:continue
                try:liveness=remote(host,setting['python'],REMOTE_ALIVE,prior)
                except Exception as e:
                    print('Host ownership unresolved',host,str(e),flush=True);active.extend(prior);continue
                for r,live in zip(prior,liveness):
                    r['active']=live
                    if live:active.append(r)
                    else:
                        r['exited']=time.time()
                        text=Path(r['log']).read_text() if Path(r['log']).exists() else ''
                        if 'Traceback' in text or 'circuit_break' in text or 'CIRCUIT_BREAK' in text:
                            slot=f"{host}:{r['gpu']}"
                            if slot not in state['disabled_slots']:state['disabled_slots'].append(slot)
            completed=[];ready=[];pending=[];terminal=[];claimed=[]
            for job in m['jobs']:
                ident=job['id']
                if verified_paths(out,job,m['n_samples']) is not None:completed.append(ident);continue
                ap=out/'attempts'/(ident+'.json')
                attempt=json.loads(ap.read_text()) if ap.exists() else {}
                if attempt.get('count',0)>=3:terminal.append(ident);continue
                if (out/'claims'/ident).exists():claimed.append(ident);continue
                if (out/'inputs'/(ident+'.yaml')).exists() and attempt.get('retry_after',0)<=time.time():ready.append(ident)
                else:pending.append(ident)
            state.update(updated=time.time(),completed=len(completed),ready=len(ready),pending=len(pending),
                         terminal_failures=terminal,claimed=len(claimed),expected=len(m['jobs']))
            if state.get('status')=='waiting_for_aurora_partition' and (ready or active):
                state.pop('status',None)
            atomic_json(state_path,state)
            print('PROGRESS',json.dumps({k:state[k] for k in ('completed','ready','pending','claimed','expected')}),'active',len(active),flush=True)
            if len(completed)==len(m['jobs']):
                state['status']='cofolding_complete';atomic_json(state_path,state);return
            if ready:
                try:
                    result=subprocess.run([cfg['gate'],'list','--hosts',','.join(cfg['hosts'])],text=True,capture_output=True,timeout=60)
                except (subprocess.TimeoutExpired,OSError) as e:
                    print('Availability observation deferred',type(e).__name__,str(e),flush=True)
                    time.sleep(30)
                    continue
                if result.returncode:
                    print('Availability probe failed',result.stderr,flush=True)
                else:
                    free=ordered_free_slots(result.stdout,cfg['hosts'])
                    launched=0
                    new_per_host = {}
                    for host,gpu in free:
                        slot=f'{host}:{gpu}'
                        if host not in cfg['hosts'] or slot in state['disabled_slots']:continue
                        if cfg.get('resource_guard_required') and not guard_allows(out,host):
                            print('Admission deferred: resource guard unavailable or reports risk',host,flush=True);continue
                        if any(r['host']==host and str(r['gpu'])==gpu for r in active):continue
                        if launched>=len(ready) or len(active)+launched>=cfg['max_workers']:break
                        setting=cfg['hosts'][host]
                        host_active = sum(r['host']==host for r in active) + new_per_host.get(host,0)
                        if host_active >= setting.get('max_workers',cfg['max_workers']):continue
                        eligible = [j for j in ready if len(jobs_by_id[j]['sequence']) <= setting.get('max_sequence_length',10**9)]
                        if len(eligible) <= new_per_host.get(host,0):continue
                        payload={**cfg,**setting,'host':host,'gpu':int(gpu),
                                 'log':str(out/'fleet_logs'/f'{host}_gpu{gpu}_{time.time_ns()}.log')}
                        try:record=remote(host,setting['python'],REMOTE_LAUNCH,payload)
                        except Exception as e:
                            print('Launch deferred',slot,str(e),flush=True);continue
                        record['active']=True;state['workers'].append(record);launched+=1
                        new_per_host[host] = new_per_host.get(host,0) + 1
                        atomic_json(state_path,state)
                        print('LAUNCHED',host,gpu,record['pid'],record['cpu_affinity'],flush=True)
                        time.sleep(3)
            elif not active:
                if terminal or claimed:
                    if not terminal and external_partition_claims_live(out,claimed):
                        state['status']='waiting_for_aurora_partition';atomic_json(state_path,state)
                        print('WAITING_FOR_LIVE_AURORA_PARTITION',len(claimed),'held variants',flush=True)
                        time.sleep(30)
                        continue
                    state['status']='requires_failure_or_claim_review';atomic_json(state_path,state);return
                producer=json.loads((out/'msa_launch.json').read_text())
                try:
                    fields=Path(f"/proc/{producer['pid']}/stat").read_text().split()
                    msa_live=fields[21]==producer['start_ticks'] and fields[2]!='Z'
                except OSError:msa_live=False
                if not msa_live and not (out/'msa_complete.json').exists():
                    state['status']='msa_producer_stopped';atomic_json(state_path,state);return
            if ready and not active and len(state['disabled_slots'])>=cfg['max_workers']:
                state['status']='workers_failed_preflight_or_runtime';atomic_json(state_path,state);return
            time.sleep(30)
    finally:
        (lock/'owner.json').unlink(missing_ok=True);lock.rmdir()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config');a=ap.parse_args();coordinate(a.config)
