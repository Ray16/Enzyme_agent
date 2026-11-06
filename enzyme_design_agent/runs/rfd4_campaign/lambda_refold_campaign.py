"""Reservation-aware Lambda refolding queue, with AF3 and ESMFold2 kept separate."""
from pathlib import Path
import argparse,csv,hashlib,json,os,signal,socket,subprocess,sys,time,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def info(pid):
 fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return int(fields[1]),fields[19]
def descendant(pid,ancestor):
 for _ in range(60):
  if pid==ancestor:return True
  try:pid=info(pid)[0]
  except OSError:return False
  if pid<=1:return False
 return False
def command(args,**kw):return subprocess.run(args,text=True,capture_output=True,timeout=kw.pop('timeout',30),**kw)

def publish(root,m):
 rows=[]
 for p in (root/'complete').glob('*.json'):rows.extend(json.loads(p.read_text())['rows'])
 completed={r['id'] for r in rows};failed={p.stem for p in (root/'failed').glob('*.json')};by={}
 for method in sorted({j['oracle'] for j in m['jobs']}):
  group=[r for r in rows if r['oracle']==method];by[method]=dict(completed=len(group),AME_style_pass=sum(r['ame_gate']=='pass' for r in group),AME_style_fail=sum(r['ame_gate']=='fail' for r in group))
 save(root/'summary.json',dict(expected=len(m['jobs']),completed=len(completed),failed=len(failed),pending=len(m['jobs'])-len(completed)-len(failed),by_oracle=by,scope='Independent refolding, oracle-specific AME-style geometry gates; no activity claim or 50-pose ensemble ranking',updated=time.time()))
 fields=['id','oracle','variant','msa_unique_sequences','ame_gate','refolded_backbone_rmsd.rmsd','refolded_motif_allatom_rmsd.rmsd','distance_A','angle_deg','mean_protein_plddt','independent_model']
 tmp=root/'metrics.tmp'
 with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 tmp.replace(root/'metrics.csv')

def worker(root,batch):
 m=json.loads((root/'manifest.json').read_text());b=json.loads((batch/'batch.json').read_text());slots=sorted(os.sched_getaffinity(0));pid=os.getpid()
 if socket.gethostname().split('.')[0]!=m['host'] or len(slots)<4:raise RuntimeError('Wrong host or CPU assignment')
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if visible!=str(b['gpu']):raise RuntimeError('GPU gate did not assign exactly the requested accelerator')
 owner=dict(pid=pid,host=socket.gethostname(),start_ticks=info(pid)[1],gpu=b['gpu'],cpu_affinity=slots,started=time.time(),batch=b['id']);save(batch/'owner.json',owner)
 os.sched_setaffinity(0,{slots[0]});child=None
 def stop(sig,frame):
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 env=dict(os.environ)
 for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):env[k]='2'
 for k in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
 env['JAX_PLATFORMS']='cpu';method=b['oracle'];out=batch/'prediction'
 if method=='af3':
  argv=['bash',m['af3_wrapper'],'pred','--config','h100','--mode','exact','--input_dir',str(batch/'inputs'),'--output_dir',str(out),'--num_diffusion_samples','1','--norun_data_pipeline']
 else:
  argv=['bash',m['esm_wrapper'],'pred','--config','h100','--mode','exact','--variant','full_nomsa','--input',str(batch/'esm_inputs.json'),'--out_dir',str(out),'--num_loops','20','--num_sampling_steps','200','--num_diffusion_samples','1','--det','1','--seeds',str(m['seed'])]
 argv=['taskset','-c',','.join(map(str,slots[1:])),*argv];save(batch/'execution.json',dict(owner,status='starting',argv=argv,oracle=method,mode='exact'))
 try:
  for job in b['jobs']:
   if sha(root/'inputs'/f"{job['id']}.json")!=job['input_sha256']:raise RuntimeError('Frozen model input changed')
   if sha(Path(job['reference']))!=job['reference_sha256']:raise RuntimeError('Frozen scoring reference changed')
  with (batch/'predict.log').open('x') as log:
   began=time.monotonic();child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(batch/'execution.json',dict(owner,status='predicting',argv=argv,predictor_pid=child.pid,oracle=method,mode='exact'))
   while child.poll() is None:
    check=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
    if check.returncode:raise RuntimeError('Cannot verify accelerator occupancy')
    for line in check.stdout.splitlines():
     parts=[x.strip() for x in line.split(',')]
     if len(parts)!=2 or parts[0]!=b['gpu_uuid']:continue
     other=int(parts[1])
     if (Path('/proc')/str(other)).exists() and not descendant(other,pid):raise RuntimeError('Foreign process on assigned GPU; stop our predictor')
    time.sleep(10)
   rc=child.wait();child=None;wall=time.monotonic()-began
  text=(batch/'predict.log').read_text(errors='replace')
  if rc or 'mode=exact' not in text:raise RuntimeError('Exact-mode prediction failed rc='+str(rc))
  rows=[]
  for job in b['jobs']:
   ident=job['id']
   if method=='af3':model=out/ident/(ident+'_model.cif')
   else:
    models=list((out/'cif_all').glob(ident+'__full__s*_x0.cif'))
    if len(models)!=1:raise RuntimeError('Expected one full-model ESMFold2 output for '+ident)
    model=models[0]
   if not model.exists():raise RuntimeError('Missing expected refold '+str(model))
   metrics=batch/(ident+'.metrics.json');q=command([m['score_python'],str(root/'producer/score_independent_refold.py'),job['reference'],str(model),'--output',str(metrics)],env=env,timeout=120)
   if q.returncode:raise RuntimeError('Scoring failed: '+q.stderr[-1500:])
   row=json.loads(metrics.read_text());row.update(oracle=method,variant=job['variant'],msa_unique_sequences=job['msa_unique_sequences'],generation_source_id=job['source_id'],refold_input_sha256=job['input_sha256'],provider='Lambda H100',batch_process_wall_seconds=wall,batch_average_seconds_per_sample=wall/len(b['jobs']),startup_import_seconds=None,warmup_to_first_batch_seconds=None)
   destination=root/'independent_refolds'/f'{ident}.cif'
   if destination.exists():raise RuntimeError('Preserve prior independent refold '+ident)
   import shutil
   shutil.copy2(model,destination)
   if sha(destination)!=row['independent_model_sha256']:raise RuntimeError('Copied model checksum mismatch')
   row['independent_model']=str(destination);save(root/'independent_refolds'/f'{ident}.metrics.json',row);rows.append(row)
  save(root/'complete'/f"{b['id']}.json",dict(owner,rows=rows,process_wall_seconds=wall,finished=time.time()));save(batch/'execution.json',dict(owner,status='complete',oracle=method,process_wall_seconds=wall,completed=len(rows),finished=time.time()))
 except BaseException:
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  error=traceback.format_exc();save(batch/'execution.json',dict(owner,status='failed_preserved',oracle=method,error=error,finished=time.time()))
  for job in b['jobs']:save(root/'failed'/f"{job['id']}.json",dict(id=job['id'],batch=b['id'],oracle=method,error=error))
  raise
 finally:
  for job in b['jobs']:
   claim=root/'claims'/job['id'];(claim/'owner.json').unlink(missing_ok=True)
   try:claim.rmdir()
   except FileNotFoundError:pass

def occupied(exclude):
 out=set()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==exclude:continue
  try:
   if p.stat().st_uid<1000 or not (p/'cmdline').read_bytes():continue
   cpus=os.sched_getaffinity(int(p.name))
   if len(cpus)<os.cpu_count():out|=cpus
  except (OSError,PermissionError):pass
 return out

def serve(root):
 m=json.loads((root/'manifest.json').read_text());pid=os.getpid();controller_cpus=sorted(os.sched_getaffinity(0));save(root/'controller_owner.json',dict(pid=pid,host=socket.gethostname(),start_ticks=info(pid)[1],cpu_affinity=controller_cpus,started=time.time()));active={};began=time.monotonic();failed_batches=0
 if socket.gethostname().split('.')[0]!=m['host']:raise RuntimeError('Wrong controller host')
 while time.monotonic()-began<m['max_seconds']:
  for gpu,run in list(active.items()):
   if run['process'].poll() is None:continue
   rc=run['process'].wait();run['log'].close();b=run['batch'];owner=(b/'owner.json').exists()
   if owner and not (root/'complete'/(b.name+'.json')).exists():failed_batches+=1
   if not owner:
    for j in json.loads((b/'batch.json').read_text())['jobs']:
     claim=root/'claims'/j['id'];(claim/'owner.json').unlink(missing_ok=True)
     try:claim.rmdir()
     except FileNotFoundError:pass
   del active[gpu]
  publish(root,m)
  if failed_batches>=3:save(root/'controller_status.json',dict(status='stopped_three_failed_batches',updated=time.time()));return
  done={j['id'] for p in (root/'complete').glob('*.json') for j in json.loads(p.read_text())['rows']};failed={p.stem for p in (root/'failed').glob('*.json')};pending=[j for j in m['jobs'] if j['id'] not in done|failed and not (root/'claims'/j['id']).exists()]
  if not pending and not active:save(root/'controller_status.json',dict(status='finished',completed=len(done),failed=len(failed),updated=time.time()));return
  discovery=command([m['gpu_reserve'],'list','--hosts',m['host']],timeout=30)
  if discovery.returncode:raise RuntimeError('Shared GPU availability gate failed')
  available=[int(line.split()[1]) for line in discovery.stdout.splitlines() if line.startswith(m['host']+' ')]
  g=command(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader']);uuids={int(l.split(',')[0]):l.split(',')[1].strip() for l in g.stdout.splitlines()}
  for gpu in available:
   if len(active)>=m['max_workers'] or not pending:break
   taken=occupied(pid)|set(controller_cpus);free=sorted(set(m['permitted_cpus'])-taken)
   if len(free)<m['cpus_per_worker'] or max(os.getloadavg())+m['cpus_per_worker']>os.cpu_count():break
   validated_methods={json.loads(p.read_text())['rows'][0]['oracle'] for p in (root/'complete').glob('*.json')}
   testing_methods={json.loads((x['batch']/'batch.json').read_text())['oracle'] for x in active.values()}-validated_methods
   eligible=[j for j in pending if j['oracle'] not in testing_methods]
   if not eligible:break
   method=eligible[0]['oracle'];validated=method in validated_methods;count=m['batch_items'] if validated else 1
   jobs=[j for j in pending if j['oracle']==method][:count];ident=f'{method}_{time.time_ns()}';batch=root/'batches'/ident;batch.mkdir();(batch/'inputs').mkdir();claimed=[]
   for j in jobs:
    c=root/'claims'/j['id']
    try:c.mkdir()
    except FileExistsError:continue
    save(c/'owner.json',dict(controller_pid=pid,host=m['host'],batch=ident,started=time.time()));claimed.append(j)
   if not claimed:continue
   import shutil
   for j in claimed:shutil.copy2(root/'inputs'/f"{j['id']}.json",batch/'inputs'/f"{j['id']}.json")
   if method=='esmfold2':save(batch/'esm_inputs.json',[json.loads((root/'inputs'/f"{j['id']}.json").read_text()) for j in claimed])
   save(batch/'batch.json',dict(id=ident,oracle=method,gpu=gpu,gpu_uuid=uuids[gpu],jobs=claimed));cpus=free[:m['cpus_per_worker']];argv=['taskset','-c',','.join(map(str,cpus)),m['gpu_reserve'],'run',str(gpu),'--label','AS9_motif_refold_'+method,'--',sys.executable,'-u',str(Path(__file__).resolve()),str(root),'--batch',str(batch)];log=(batch/'launcher.log').open('x');proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT);active[gpu]=dict(process=proc,log=log,batch=batch);pending=[j for j in pending if j['id'] not in {x['id'] for x in claimed}]
  save(root/'controller_status.json',dict(status='predicting' if active else 'waiting_for_free_H100',active_gpus=sorted(active),pending=len(pending),completed=len(done),failed=len(failed),updated=time.time()));time.sleep(20)
 save(root/'controller_status.json',dict(status='bounded_wait_time_limit',active_gpus=sorted(active),updated=time.time()))
 # Existing gate-owned workers retain ownership and finish their current batch.

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--batch',type=Path);a=p.parse_args();root=a.root.resolve()
 if a.batch:worker(root,a.batch)
 else:serve(root)
if __name__=='__main__':main()
