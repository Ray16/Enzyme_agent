"""Independent AF3/OpenFold3 exact-mode refolding on scheduler-assigned Aurora XPUs."""
from pathlib import Path
import hashlib,json,os,signal,socket,subprocess,sys,time,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def find_ranked_model(prediction,inputfile):
 """The AF3 kit names its output directory and model from the JSON file stem."""
 stem=Path(inputfile).stem
 expected=Path(prediction)/stem/(stem+'_model.cif')
 if expected.is_file():return expected
 candidates=[p for p in Path(prediction).glob('*/*_model.cif') if not any(part.startswith('.') for part in p.relative_to(prediction).parts)]
 if len(candidates)!=1:raise RuntimeError('Expected one ranked model CIF; found '+str([str(p) for p in candidates]))
 return candidates[0]

def main():
 root=Path(__file__).resolve().parent;m=json.loads((root/'manifest.json').read_text());ranks=[os.environ[k] for k in ('PMI_RANK','PMIX_RANK','OMPI_COMM_WORLD_RANK') if k in os.environ]
 if not ranks or len(set(ranks))!=1 or not os.environ.get('PBS_JOBID') or not os.environ.get('ZE_AFFINITY_MASK'):raise RuntimeError('Missing scheduler/rank/XPU binding')
 rank=int(ranks[0]);size=m['ranks'];slots=sorted(os.sched_getaffinity(0));owner=dict(host=socket.gethostname(),rank=rank,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],pbs_job=os.environ['PBS_JOBID'],cpu_affinity=slots,device_mask=os.environ['ZE_AFFINITY_MASK'],started=time.time())
 if len(slots)<4 or max(os.getloadavg())+size*len(slots)>(os.cpu_count() or 0):raise RuntimeError('Insufficient CPU capacity')
 save(root/'workers'/f'rank{rank}.json',owner)
 deadline=time.monotonic()+60
 while True:
  peers=[json.loads(p.read_text()) for p in (root/'workers').glob('rank*.json')]
  peers=[p for p in peers if p['pbs_job']==owner['pbs_job']]
  if len(peers)==size:break
  if time.monotonic()>deadline:raise RuntimeError('Not all assigned workers reached resource preflight')
  time.sleep(1)
 for i,left in enumerate(peers):
  for right in peers[i+1:]:
   if left['host']==right['host'] and (set(left['cpu_affinity'])&set(right['cpu_affinity']) or left['device_mask']==right['device_mask']):raise RuntimeError('Worker CPU/XPU overlap')
 os.sched_setaffinity(0,{slots[0]})
 import torch
 if not torch.xpu.is_available() or torch.xpu.device_count()!=1:raise RuntimeError('Expected exactly one assigned Intel XPU')
 owner['device']=torch.xpu.get_device_name(0);owner['device_memory_bytes']=torch.xpu.get_device_properties(0).total_memory;save(root/'workers'/f'rank{rank}.json',owner)
 for name,digest in m['payload_sha256'].items():
  if sha(root/name)!=digest:raise RuntimeError('Frozen payload changed '+name)
 receipt=json.loads(Path(m['msa_receipt']).read_text());msas={x['id']:x for x in receipt['records']}
 if receipt['status']!='complete':raise RuntimeError('Fresh batch MSA search incomplete')
 failures=0;any_failed=False;child=None;claim=None
 def stop(sig,frame):
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 for job in m['jobs'][rank::size]:
  ident=job['id'];done=root/'complete'/f'{ident}.json'
  if done.exists():continue
  claim=root/'claims'/ident
  try:claim.mkdir()
  except FileExistsError:print('OWNED_REQUIRES_REVIEW',ident,flush=True);continue
  save(claim/'owner.json',owner);state=dict(id=ident,owner=owner,status='starting',started=time.time(),precision='AF3/OpenFold3 exact mode at the pinned stock numerical baseline',protein_length=job['length'],samples=m['samples_per_design'])
  try:
   msa=msas[ident]
   if msa['sequence_sha256']!=job['sequence_sha256'] or sha(Path(msa['a3m']))!=msa['a3m_sha256']:raise RuntimeError('Fresh sequence-specific MSA identity mismatch')
   inp=json.loads((root/'inputs'/f'{ident}.json').read_text());p=inp['sequences'][0]['protein']
   if p['sequence']!=job['sequence'] or p.get('templates') or 'bondedAtomPairs' in inp:raise RuntimeError('Unexpected sequence/templates/structural conditioning')
   p['unpairedMsa']=Path(msa['a3m']).read_text();p['pairedMsa']='';p['templates']=[]
   out=root/'results'/ident
   if out.exists():raise RuntimeError('Preserve prior partial output')
   out.mkdir();inputfile=out/'input.json';save(inputfile,inp);state.update(msa_depth=msa['depth'],msa_sha256=msa['a3m_sha256'],runtime_input_sha256=sha(inputfile))
   env=dict(os.environ)
   for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','NUMEXPR_NUM_THREADS'):env[k]='2'
   for k in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
   env.update(m['runtime_env']);env['PATH']=str(Path(env['AF3_TORCH_PY']).parent)+os.pathsep+env['PATH'];env.pop('AF3_TORCH_OPT',None)
   argv=['taskset','-c',','.join(map(str,slots[1:])),'bash',m['kit']+'/run.sh','pred','--config','pvc','--mode','exact','--json_path',str(inputfile),'--output_dir',str(out/'prediction'),'--num_diffusion_samples',str(m['samples_per_design']),'--norun_data_pipeline']
   state.update(status='predicting',argv=argv);save(root/'attempts'/f'{ident}.json',state);log=root/'logs'/f'{ident}.log';print('INDEPENDENT_AF3_START',ident,'exact',flush=True)
   with log.open('x') as stream:
    child=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);state['predictor_pid']=child.pid;save(root/'attempts'/f'{ident}.json',state);rc=child.wait();child=None
   text=log.read_text(errors='replace')
   if rc or '[af3-torch-opt] ACTIVE mode=exact' not in text or 'DONE ok=1 rc=0' not in text:raise RuntimeError('AF3 exact prediction did not complete cleanly: rc='+str(rc))
   model=find_ranked_model(out/'prediction',inputfile)
   scored=out/'metrics.json';q=subprocess.run([m['score_python'],str(root/'score_independent_refold.py'),str(root/'references'/f'{ident}.json'),str(model),'--output',str(scored)],env=env,capture_output=True,text=True,timeout=120)
   if q.returncode:raise RuntimeError('Independent scoring failed: '+q.stderr[-2000:])
   metrics=json.loads(scored.read_text());state.update(status='complete',metrics=metrics,model=str(model),model_sha256=sha(model),process_wall_seconds=time.time()-state['started'],startup_import_seconds=None,warmup_to_first_batch_seconds=None,kit_reports={str(p.relative_to(out)):json.loads(p.read_text()) for p in (out/'prediction/.af3_torch_opt_work').glob('*.json')});save(done,state);print('INDEPENDENT_AF3_COMPLETE',ident,metrics['ame_gate'],flush=True);failures=0
  except BaseException:
   failures+=1;any_failed=True;state.update(status='failed_preserved',error=traceback.format_exc());save(root/'failed'/f'{ident}.json',state);print('FAILED',ident,traceback.format_exc(),flush=True)
  finally:
   state['finished']=time.time();save(root/'attempts'/f'{ident}.json',state);(claim/'owner.json').unlink(missing_ok=True);claim.rmdir()
  if failures>=3:raise RuntimeError('Three consecutive failures; stop this worker')
 if any_failed:raise RuntimeError('This worker has preserved failed predictions or collection errors')

if __name__=='__main__':main()
