"""Consume a prepared RFD4 queue within one reserved, CPU-pinned GPU process tree."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,socket,subprocess,time,traceback
from pathlib import Path
from run_logging import summarize_run

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,d):
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(path)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('queue',type=Path);p.add_argument('--cwd',type=Path,required=True);a=p.parse_args()
 root=a.queue.resolve();m=json.loads((root/'manifest.json').read_text());wrapper=Path(m['jobs'][0]['argv'][2])
 if sha(wrapper)!=m['wrapper_sha256']:raise ValueError('Profile wrapper changed')
 if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Start within a gpu_reserve-owned process tree')
 for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
  if not os.environ.get(k):raise RuntimeError('Missing native thread cap '+k)
 env=dict(os.environ,RFD4_BATCH_SIZE=str(m['batch_size']));child=None;failures=0
 def stop(sig,frame):
  if child and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=8)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 for job in m['jobs']:
  claim=root/'claims'/job['id'];done=root/'complete'/(job['id']+'.json');failed=root/'failed'/(job['id']+'.json')
  if done.exists() or failed.exists():continue
  try:claim.mkdir()
  except FileExistsError:continue
  save(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),start_ticks=Path('/proc/self/stat').read_text().split()[21],started=time.time(),cpu_affinity=sorted(os.sched_getaffinity(0)),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
  try:
   if done.exists() or failed.exists():continue
   if sha(job['input'])!=job['input_sha256']:raise ValueError('Input changed')
   if Path(job['output']).exists():raise RuntimeError('Partial output exists; inspect before resuming')
   start=time.time();monotonic=time.monotonic()
   with (root/'logs'/(job['id']+'.log')).open('x') as log:
    child=subprocess.Popen(job['argv'],cwd=a.cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(claim/'predictor.json',dict(pid=child.pid,started=start,argv=job['argv']))
    print('GENERATION_START',job['id'],child.pid,flush=True);rc=child.wait();duration=time.monotonic()-monotonic;finished=time.time();child=None
   if rc:raise RuntimeError(f'Generator returned {rc}')
   files=sorted(Path(job['output']).rglob('*_generated.cif'))
   if len(files)!=job['expected']:raise RuntimeError(f'Expected {job["expected"]} outputs, found {len(files)}')
   compatibility_path=Path(job['output'])/'compatibility_manifest.json'
   compatibility=json.loads(compatibility_path.read_text()) if compatibility_path.exists() else {}
   statistics=summarize_run(
       process_started=start, entrypoint_started=compatibility.get('started'),
       sampling_started=None, sampling_finished=None, finished=compatibility.get('finished',finished),
       protein_lengths=[d['protein_CA_count'] for d in compatibility.get('designs',[])],
       requested_samples=job['expected'], completed_samples=len(files), batch_size=m['batch_size'])
   rows=[dict(path=str(f),sha256=sha(f),qc=dict(technical_pass=None,scope='independent chemistry and task-specific QC not run')) for f in files]
   save(done,dict(id=job['id'],expected=job['expected'],batch_size=m['batch_size'],replicates=len(files),generation_started=start,generation_finished=finished,generation_wall_seconds=duration,amortized_seconds_per_structure=duration/len(files),timing_scope='predictor subprocess wall time / outputs; batch throughput, not individual latency',run_statistics=statistics,rows=rows))
   failures=0;print('GENERATION_COMPLETE',job['id'],len(files),flush=True)
  except Exception:
   failures+=1;save(failed,dict(id=job['id'],error=traceback.format_exc(),finished=time.time()));print('GENERATION_FAILED',job['id'],flush=True)
  finally:
   (claim/'predictor.json').unlink(missing_ok=True);(claim/'owner.json').unlink(missing_ok=True);claim.rmdir()
  if failures>=3:raise RuntimeError('Consecutive failure circuit breaker')
if __name__=='__main__':main()
