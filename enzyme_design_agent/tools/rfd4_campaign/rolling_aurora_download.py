"""Bounded rolling download via Aurora control RPC when Globus transfer is unavailable."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,os,subprocess,time
from globus_compute_sdk import Client,Executor
from globus_compute_sdk.serialize import ComputeSerializer,AllCodeStrategies

def remote_inventory(root,job):
 import hashlib,json,subprocess,time
 from pathlib import Path
 root=Path(root);rows=[];timings={}
 for p in sorted((root/'analysis').glob('*.json')):rows.extend(json.loads(p.read_text()).get('rows',[]))
 for row in rows:
  f=Path(row['path'])
  if f.is_relative_to(root/'outputs') and f.exists() and not row.get('structure_sha256'):row['structure_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
 for p in (root/'outcomes').glob('*.json'):
  d=json.loads(p.read_text());timings[p.stem]=d.get('validation',{}).get('run_statistics',{})
 try:
  q=subprocess.run(['qstat','-x','-f','-F','json',job],capture_output=True,text=True,timeout=20);j=next(iter(json.loads(q.stdout).get('Jobs',{}).values()),{}) if q.returncode==0 else {};state=j.get('job_state');exit_status=j.get('Exit_status')
 except Exception:state=None;exit_status=None
 return dict(rows=rows,timings=timings,pbs_state=state,pbs_exit_status=exit_status,observed=time.time())

def remote_fetch(root,requests):
 import base64,gzip,hashlib
 from pathlib import Path
 root=Path(root).resolve();files=[]
 if len(requests)>3:raise ValueError('At most three files per control request')
 for row in requests:
  f=Path(row['path']).resolve()
  if not f.is_relative_to(root/'outputs') or not f.name.endswith('_generated.cif'):raise ValueError('Outside campaign output')
  raw=f.read_bytes();h=hashlib.sha256(raw).hexdigest()
  if h!=row['structure_sha256']:raise ValueError('Remote file changed')
  files.append(dict(relative=str(f.relative_to(root)),sha256=h,data=base64.b64encode(gzip.compress(raw,compresslevel=1)).decode()))
 return files

def save(p,d):
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class RPC:
 def __init__(self,cfg,root):self.cfg=cfg;self.root=root;self.client=Client(do_version_check=False);self.ex=None;self.variant=0
 def connect(self):
  m=self.client.get_endpoint_metadata(self.cfg['endpoint'])
  if 'aurora' not in (m.get('hostname') or '') or 'type: LocalProvider' not in (m.get('user_config_template') or ''):raise RuntimeError('Expected Aurora login control endpoint')
  cfg=({'worker_init':'export TMPDIR=/tmp\nexport OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'} if self.variant==0 else None)
  self.ex=Executor(endpoint_id=self.cfg['endpoint'],client=self.client,user_endpoint_config=cfg);self.ex.serializer=ComputeSerializer(strategy_code=AllCodeStrategies())
 def call(self,fn,*args):
  for attempt in range(2):
   if self.ex is None:self.connect()
   future=None
   try:
    future=self.ex.submit(fn,*args)
    try:return future.result(timeout=45)
    except TimeoutError:
     # Observe the accepted task rather than submitting a duplicate.
     for _ in range(20):
      try:return self.client.get_result(str(future.task_id))
      except Exception as e:
       if type(e).__name__!='TaskPending':raise
      time.sleep(3)
     raise RuntimeError('Control task remains unresolved; see receipt')
   except Exception as e:
    if 'ENDPOINT_NOT_ONLINE' not in str(e) or attempt:raise
    self.ex.shutdown(wait=False);self.ex=None;self.variant=1-self.variant
   finally:
    if future is not None:
     with (self.root/'download_rpc_receipts.jsonl').open('a') as f:f.write(json.dumps(dict(task_id=str(future.task_id),operation=fn.__name__,time=time.time()))+'\n')
 def close(self):
  if self.ex:self.ex.shutdown(wait=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('config',type=Path);a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['campaign_dir']);state=root/'rolling_download_status.json';rpc=RPC(cfg,root);started=time.monotonic();failures=0
 try:
  while time.monotonic()-started<cfg.get('max_seconds',10800):
   try:
    inv=rpc.call(remote_inventory,cfg['remote_root'],cfg['pbs_job']);save(root/'download_inventory.json',inv)
    missing=[]
    for r in inv['rows']:
     relative=Path(r['path']).relative_to(cfg['remote_root']);dest=root/'downloads'/relative
     if not dest.exists() or sha(dest)!=r.get('structure_sha256'):missing.append(r)
    save(state,dict(status='downloading',remote_structures=len(inv['rows']),remaining=len(missing),pbs_state=inv['pbs_state'],updated=time.time()))
    for i in range(0,len(missing),3):
     fetched=rpc.call(remote_fetch,cfg['remote_root'],missing[i:i+3])
     for item in fetched:
      rel=Path(item['relative'])
      if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe download path')
      raw=gzip.decompress(base64.b64decode(item['data']))
      if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Downloaded checksum mismatch')
      dest=root/'downloads'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
      if dest.exists() and sha(dest)!=item['sha256']:
       archive=root/'downloads/stale'/str(time.time_ns())/rel;archive.parent.mkdir(parents=True,exist_ok=True);dest.rename(archive)
      tmp=dest.with_name(dest.name+'.download.tmp');tmp.write_bytes(raw);tmp.replace(dest)
     # Publish analysis after each small download batch.
     q=subprocess.run([cfg['analysis_python'],str(Path(__file__).with_name('analyze_downloads.py')),str(a.config.resolve())],capture_output=True,text=True,timeout=120)
     if q.returncode:raise RuntimeError(q.stderr[-2000:])
     save(state,dict(status='downloading',remote_structures=len(inv['rows']),downloaded_this_cycle=min(i+3,len(missing)),remaining=max(0,len(missing)-i-3),pbs_state=inv['pbs_state'],updated=time.time()))
    if not missing:
     q=subprocess.run([cfg['analysis_python'],str(Path(__file__).with_name('analyze_downloads.py')),str(a.config.resolve())],capture_output=True,text=True,timeout=120)
     if q.returncode:raise RuntimeError(q.stderr[-2000:])
    failures=0
    done=inv['pbs_state']=='F'
    save(state,dict(status='complete' if done else 'watching',remote_structures=len(inv['rows']),verified_local=len(inv['rows']),expected=cfg['expected'],pbs_state=inv['pbs_state'],pbs_exit_status=inv['pbs_exit_status'],updated=time.time()))
    if done:return
   except Exception as e:
    failures+=1;save(state,dict(status='retry_cooldown',consecutive_errors=failures,error=repr(e),updated=time.time()))
    if failures>=3:raise
   time.sleep(cfg.get('interval_seconds',120))
  save(state,dict(status='bounded_monitor_time_limit',updated=time.time()))
 finally:rpc.close()
if __name__=='__main__':main()
