"""One CPU-pinned batch search against Aurora's existing ColabFold databases."""
from pathlib import Path
import csv,hashlib,json,os,signal,socket,subprocess,sys,time,traceback,shutil

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def read_a3m(text):
 rows=[];header=None;seq=[]
 for line in text.splitlines():
  if not line or line.startswith('#'):continue
  if line.startswith('>'):
   if header is not None:rows.append((header,''.join(seq)))
   header=line;seq=[]
  else:seq.append(line.strip())
 if header is not None:rows.append((header,''.join(seq)))
 return rows

def main():
 root=Path(__file__).resolve().parent;manifest=json.loads((root/'manifest.json').read_text());slots=sorted(os.sched_getaffinity(0));threads=manifest['threads']
 if not os.environ.get('PBS_JOBID') or len(slots)<threads+1:raise RuntimeError('Missing exclusive PBS allocation/CPU binding')
 if max(os.getloadavg())+threads+1>(os.cpu_count() or 0):raise RuntimeError('Insufficient CPU capacity')
 owner=dict(pid=os.getpid(),host=socket.gethostname(),start_ticks=Path('/proc/self/stat').read_text().split()[21],pbs_job=os.environ['PBS_JOBID'],cpu_affinity=slots,search_cpus=slots[1:threads+1],supervisor_cpu=slots[0],threads=threads,started=time.time())
 save(root/'workers/msa.json',owner);os.sched_setaffinity(0,{slots[0]})
 for name,h in manifest['payload_sha256'].items():
  if digest(root/name)!=h:raise RuntimeError('Payload changed '+name)
 for asset in manifest['required_assets']:
  if not Path(asset).exists():raise RuntimeError('Missing search asset '+asset)
 state=dict(status='starting',owner=owner,designs=len(manifest['jobs']),database_versions=manifest['database_versions']);save(root/'attempts/batch.json',state)
 env=dict(os.environ)
 for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT'):env[k]=str(threads)
 for k in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
 env['PATH']=str(Path(manifest['search_binary']).parent)+os.pathsep+env['PATH'];env['TMPDIR']=str(root/'tmp');(root/'tmp').mkdir(exist_ok=True)
 if manifest.get('ignore_precomputed_index'):env['MMSEQS_IGNORE_INDEX']='1'
 binary=manifest['search_binary'];help_result=subprocess.run([binary,'--help'],capture_output=True,text=True,env=env,timeout=30)
 if help_result.returncode or '--db-load-mode' not in help_result.stdout+help_result.stderr:raise RuntimeError('Unexpected colabfold_search interface')
 if (root/'search_output').exists():raise RuntimeError('Preserve prior partial search output before restarting')
 if manifest.get('resume_from'):
  prior=manifest['resume_from'];source=Path(prior['root']);search=source/'search_output'
  if digest(source/'sequences.fasta')!=digest(root/'sequences.fasta'):raise RuntimeError('Resume query batch differs')
  old=subprocess.run(['qstat','-x','-f','-F','json',prior['pbs_job']],capture_output=True,text=True,timeout=20)
  oldjob=next(iter(json.loads(old.stdout).get('Jobs',{}).values()),{}) if old.returncode==0 else {}
  if oldjob.get('job_state')!='F':raise RuntimeError('Prior MSA owner is not proven terminal')
  copied=[];(root/'search_output').mkdir()
  for prefix in prior['completed_db_prefixes']:
   if not (search/(prefix+'.dbtype')).exists():raise RuntimeError('Missing completed database marker '+prefix)
   for p in search.iterdir():
    if (p.name==prefix or p.name.startswith(prefix+'.')) and p.is_file():
     dest=root/'search_output'/p.name;shutil.copy2(p,dest)
     if digest(dest)!=digest(p):raise RuntimeError('Copied checkpoint checksum mismatch '+p.name)
     copied.append(dict(name=p.name,sha256=digest(dest),bytes=dest.stat().st_size))
  save(root/'resume_provenance.json',dict(source=prior,copied=copied,prior_state=oldjob.get('job_state'),partial_expansion_files='Not reused; preserved in original run'))
  # ColabFold unconditionally removes this directory at the end, even when its
  # initial search was skipped because the completed result database was reused.
  (root/'search_output/tmp').mkdir(exist_ok=True)
  print('REUSED_COMPLETED_MSA_STAGES',prior['completed_db_prefixes'],flush=True)
 cmd=['taskset','-c',','.join(map(str,owner['search_cpus'])),binary,'--threads',str(threads),'--gpu','0','--use-templates','0','--use-env','1','--db-load-mode','0',str(root/'sequences.fasta'),manifest['database_root'],str(root/'search_output')]
 state.update(status='searching',argv=cmd);save(root/'attempts/batch.json',state);print('LOCAL_BATCH_MSA_START',len(manifest['jobs']),'queries',threads,'CPU threads',flush=True);child=None
 def stop(sig,frame):
  if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
  raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  with (root/'logs/search.log').open('x') as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['search_pid']=child.pid;save(root/'attempts/batch.json',state);rc=child.wait()
  if rc:raise RuntimeError('Local MSA search returned '+str(rc))
  dest=Path(manifest['msa_output_dir']);dest.mkdir(parents=True,exist_ok=True);records=[]
  for i,job in enumerate(manifest['jobs']):
   possible=[root/'search_output'/f"{job['id']}.a3m",root/'search_output'/f'{i}.a3m'];p=next((p for p in possible if p.exists()),None)
   if p is None:raise RuntimeError('Missing searched alignment '+job['id'])
   text=p.read_text();seqs=read_a3m(text)
   if not seqs or seqs[0][1].replace('-','')!=job['sequence']:raise RuntimeError('Exact query mismatch '+job['id'])
   h=hashlib.sha1(job['sequence'].encode()).hexdigest()[:16];a3m=dest/(h+'.a3m');csvpath=dest/(h+'.csv')
   if a3m.exists() and a3m.read_text()!=text:raise RuntimeError('Preserve differing alignment '+h)
   a3m.write_text(text)
   with csvpath.open('w',newline='') as out:w=csv.writer(out);w.writerow(['key','sequence']);w.writerows([(-1,s) for _,s in seqs])
   records.append(dict(id=job['id'],sequence_sha256=job['sequence_sha256'],a3m=str(a3m),csv=str(csvpath),a3m_sha256=digest(a3m),csv_sha256=digest(csvpath),depth=len(seqs),searched=True))
  save(dest/'search_receipt.json',dict(status='complete',database_versions=manifest['database_versions'],records=records,started=owner['started'],finished=time.time(),protocol='One fresh 100-sequence local ColabFold batch; no native-MSA reuse'))
  state.update(status='complete',completed=len(records),records=records);save(root/'complete/batch.json',state);print('LOCAL_BATCH_MSA_COMPLETE',len(records),flush=True)
 except BaseException:
  state.update(status='failed_preserved',error=traceback.format_exc());save(root/'failed/batch.json',state);raise
 finally:state['finished']=time.time();save(root/'attempts/batch.json',state)

if __name__=='__main__':main()
