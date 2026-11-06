"""Checksum-staged Aurora refolding control through the existing login endpoint."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,time
from globus_compute_sdk import Client,Executor
from globus_compute_sdk.serialize import ComputeSerializer,AllCodeStrategies

def preflight(cfg):
 import os,socket,subprocess,shutil
 from pathlib import Path
 if 'aurora-uan' not in socket.gethostname() or os.environ.get('PBS_JOBID'):raise RuntimeError('Expected Aurora login control host')
 assets=[cfg['python'],cfg['cache']+'/boltz2_conf.ckpt',cfg['cache']+'/mols', '/soft/tools/mpi_wrapper_utils/gpu_dev_compact.sh']
 base=Path('/lus/flare/projects/FRAME-IDP/rzhu')
 queues={}
 for q in ('debug','capacity'):
  try:
   r=subprocess.run(['qstat','-Qf',q],capture_output=True,text=True,timeout=8);queues[q]=dict(returncode=r.returncode,stdout=r.stdout[-7000:],stderr=r.stderr[-1000:])
  except subprocess.TimeoutExpired:queues[q]=dict(error='qstat timed out after 8 seconds')
 envroot=base/'Software/envs'
 runtimes={p.name:dict(python=(p/'bin/python').exists(),python_versions=[x.name for x in (p/'lib').glob('python*')]) for p in envroot.glob('*') if p.is_dir()}
 diagnostics={}
 for command in (['qstat','--version'],['getent','hosts','aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov']):
  try:
   rr=subprocess.run(command,capture_output=True,text=True,timeout=3);diagnostics[' '.join(command)]=dict(returncode=rr.returncode,stdout=rr.stdout[-1000:],stderr=rr.stderr[-500:])
  except subprocess.TimeoutExpired:diagnostics[' '.join(command)]='timeout'
 return dict(host=socket.gethostname(),assets={p:Path(p).exists() for p in assets},
  af3_environment={k:bool(os.environ.get(k)) for k in ('AF3_IMAGE','AF3_MODEL_DIR','AF3_JAX_CACHE_DIR')},
  env_directories=runtimes,software_directories=[p.name for p in (base/'Software').glob('*')],free_disk_bytes=shutil.disk_usage(base).free,diagnostics=diagnostics,
  queues=queues)

def stage(cfg,payload,manifest):
 import base64,gzip,hashlib,json,os,socket
 from pathlib import Path
 if 'aurora-uan' not in socket.gethostname() or os.environ.get('PBS_JOBID'):raise RuntimeError('Wrong staging host')
 root=Path(cfg['remote_root']);root.mkdir(parents=True,exist_ok=True)
 missing=[p for p in manifest.get('required_assets',[]) if not Path(p).exists()]
 if missing:raise RuntimeError('Missing runtime assets: '+str(missing))
 if (root/'submission.json').exists() or (root/'submission.claim').exists():raise RuntimeError('Inspect existing submission before staging')
 for name,item in payload.items():
  if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe stage name')
  data=gzip.decompress(base64.b64decode(item['data']));p=root/name
  if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Payload checksum mismatch')
  if p.exists() and p.read_bytes()!=data:raise RuntimeError('Existing staged file differs: '+name)
  p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():p.write_bytes(data)
 for d in ('logs','claims','workers','attempts','complete','failed','results'):(root/d).mkdir(exist_ok=True)
 text=json.dumps(manifest,indent=2)+'\n';p=root/'manifest.json'
 if p.exists() and p.read_text()!=text:raise RuntimeError('Existing manifest differs')
 p.write_text(text)
 return dict(status='staged',files=len(payload),root=str(root))

def runtime_inventory(cfg):
 import os,socket,subprocess,urllib.request
 from pathlib import Path
 base=Path('/lus/flare/projects/FRAME-IDP/rzhu/Software')
 kit=base/'uplifting-biomolecular-modeling/af3_torch'
 result=dict(host=socket.gethostname(),kit_exists=kit.exists(),files={})
 for p in [kit/'README.md',kit/'fold.sh',kit/'run_alphafold.py',base/'ubm_af3_runs/README.md']:
  if p.is_file():result['files'][str(p)]=p.read_text(errors='replace')[:12000]
 result['af3_runs_files']=[str(p.relative_to(base/'ubm_af3_runs')) for p in (base/'ubm_af3_runs').glob('*')][:35]
 result['msa_files']=[p.name for p in (base/'MSA').glob('*')][:35]
 for url in ('https://api.colabfold.com','https://api.colabfold.com/ticket/msa'):
  try:
   with urllib.request.urlopen(url,timeout=5) as response:result[url]=dict(status=response.status)
  except Exception as e:result[url]=repr(e)
 return result

def support_files(cfg):
 from pathlib import Path
 base=Path('/lus/flare/projects/FRAME-IDP/rzhu/Software');files={}
 paths=[base/'MSA/run_msa.sh',base/'MSA/run_6n0x_msa.pbs',base/'ubm_af3_runs/T_exact.log']
 paths+=list((base/'ubm_af3_runs/scripts').glob('*'))[:20]
 paths+=list((base/'ubm_af3_runs').glob('*.pbs'))+list((base/'ubm_af3_runs').glob('*.sh'))
 kit=base/'uplifting-biomolecular-modeling/af3_torch'
 paths += [kit/'configs/pvc.env',kit/'configs/aurora.env',kit/'run.sh']
 paths += list(Path('/home/ray16/miniforge3/envs/colabfold_msa/lib').glob('python*/site-packages/colabfold/mmseqs/search.py'))
 for p in paths:
  if p.is_file() and p.stat().st_size<100000:files[str(p)]=p.read_text(errors='replace')[:18000]
 return files

def submit(cfg):
 import hashlib,json,os,re,socket,subprocess,time
 from pathlib import Path
 if 'aurora-uan' not in socket.gethostname() or os.environ.get('PBS_JOBID'):raise RuntimeError('Wrong submission host')
 root=Path(cfg['remote_root']);marker=root/'submission.json'
 if marker.exists():return json.loads(marker.read_text())
 m=json.loads((root/'manifest.json').read_text())
 if m['provider']!='Aurora PBS' or len(m['jobs'])!=m['expected_designs']:raise RuntimeError('Manifest count/provider mismatch')
 for name,digest in m['payload_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Staged payload changed '+name)
 pbs=root/'campaign.pbs';body=pbs.read_text()
 for token in ('#PBS -A FRAME-IDP', '/soft/tools/mpi_wrapper_utils/gpu_dev_compact.sh', '--cpu-bind=depth'):
  if token not in body:raise RuntimeError('Missing resource invariant '+token)
 claim=root/'submission.claim';claim.mkdir()
 (claim/'owner.json').write_text(json.dumps(dict(pid=os.getpid(),host=socket.gethostname(),started=time.time())))
 try:r=subprocess.run(['qsub',str(pbs)],cwd=root,capture_output=True,text=True,timeout=20)
 except subprocess.TimeoutExpired:raise RuntimeError('Submission uncertain; inspect PBS before retrying')
 if r.returncode:
  (claim/'owner.json').unlink();claim.rmdir();raise RuntimeError(r.stderr)
 job=r.stdout.strip()
 if not re.fullmatch(r'\d+\.[^\s]+',job):raise RuntimeError('Ambiguous PBS submission result')
 rec=dict(status='submitted',job_id=job,submitted=time.time(),expected_designs=m['expected_designs'],samples_per_design=m['samples_per_design'])
 marker.write_text(json.dumps(rec,indent=2)+'\n');(claim/'owner.json').unlink();claim.rmdir();return rec

def status(cfg,job):
 import json,subprocess,time
 from pathlib import Path
 root=Path(cfg['remote_root']);q=subprocess.run(['qstat','-x','-f','-F','json',job],capture_output=True,text=True,timeout=20)
 pbs=json.loads(q.stdout) if q.returncode==0 else dict(error=q.stderr)
 return dict(pbs=pbs,workers={p.name:json.loads(p.read_text()) for p in (root/'workers').glob('*.json')},
  complete={p.name:json.loads(p.read_text()) for p in (root/'complete').glob('*.json')},
  failed={p.name:json.loads(p.read_text()) for p in (root/'failed').glob('*.json')},
  attempts={p.name:json.loads(p.read_text()) for p in (root/'attempts').glob('*.json')},
  logs={p.name:p.read_text(errors='replace')[-1300:] for p in (root/'logs').glob('*') if p.is_file()},
  pbs_error=(root/'pbs.err').read_text(errors='replace')[-2000:] if (root/'pbs.err').exists() else '',observed=time.time())

def rpc(cfg,fn,*args,tag,folder):
 client=Client(do_version_check=False);meta=client.get_endpoint_metadata(cfg['endpoint'])
 if 'aurora' not in meta.get('hostname','') or 'type: LocalProvider' not in meta.get('user_config_template',''):raise RuntimeError('Wrong endpoint')
 for index,variant in enumerate([None,{'worker_init':'export TMPDIR=/tmp\nexport OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'}]):
  ex=Executor(endpoint_id=cfg['endpoint'],client=client,user_endpoint_config=variant);ex.serializer=ComputeSerializer(strategy_code=AllCodeStrategies());receipt=dict(started=time.time(),variant=index);future=None
  try:
   future=ex.submit(fn,*args);receipt['task_id']=str(future.task_id);(folder/(tag+'.pending.json')).write_text(json.dumps(receipt,indent=2)+'\n')
   result=future.result(timeout=45);receipt.update(status='completed',result=result);return receipt
  except Exception as error:
   receipt.update(status='unresolved',error=repr(error))
   if 'ENDPOINT_NOT_ONLINE' not in str(error) or index:raise
  finally:
   if future is not None:receipt['task_id']=str(future.task_id)
   (folder/(tag+f'_config{index}.json')).write_text(json.dumps(receipt,indent=2)+'\n');ex.shutdown(wait=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('action',choices=['preflight','runtime','support','stage','submit','status']);p.add_argument('--job');o=p.parse_args();cfg=json.loads(o.config.read_text());folder=o.config.resolve().parent
 if o.action=='preflight':r=rpc(cfg,preflight,cfg,tag='preflight',folder=folder)
 elif o.action=='runtime':r=rpc(cfg,runtime_inventory,cfg,tag='runtime_inventory',folder=folder)
 elif o.action=='support':r=rpc(cfg,support_files,cfg,tag='support_files',folder=folder)
 elif o.action=='stage':
  m=json.loads((folder/'manifest.json').read_text());payload={name:dict(sha256=h,data=base64.b64encode(gzip.compress((folder/name).read_bytes())).decode()) for name,h in m['payload_sha256'].items()};r=rpc(cfg,stage,cfg,payload,m,tag='stage',folder=folder)
 elif o.action=='submit':r=rpc(cfg,submit,cfg,tag='submission',folder=folder)
 else:r=rpc(cfg,status,cfg,o.job,tag='status_latest',folder=folder)
 if o.action=='status':
  d=r['result'];j=next(iter(d['pbs'].get('Jobs',{}).values()),{});print(json.dumps(dict(pbs_state=j.get('job_state'),exit_status=j.get('Exit_status'),workers=len(d['workers']),complete=len(d['complete']),failed=len(d['failed']),log_tails=d['logs'])))
 else:print(json.dumps(r,indent=2))

if __name__=='__main__':main()
