"""Bounded continuation: local MSA batch -> AF3 pilot -> remaining AME refolds.

Submit the remaining designs only after every pilot prediction completes and
its sequence/atom correspondence is verified. No failed scientific attempt is
retried automatically. Download complete structures and metrics as they appear.
"""
from pathlib import Path
import argparse,base64,csv,gzip,hashlib,json,math,os,shutil,time,traceback,subprocess
from aurora_refold_control import rpc,status,stage,submit

def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fetch_models(cfg,records):
 import base64,gzip,hashlib
 from pathlib import Path
 root=Path(cfg['remote_root']).resolve();out=[]
 if len(records)>3:raise ValueError('At most three structures per request')
 for row in records:
  p=Path(row['model']).resolve()
  if not p.is_relative_to(root/'results') or not p.name.endswith('_model.cif'):raise ValueError('Outside refolding result directory')
  raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
  if digest!=row['model_sha256']:raise ValueError('Remote model checksum changed')
  out.append(dict(id=row['id'],sha256=digest,data=base64.b64encode(gzip.compress(raw,compresslevel=1)).decode()))
 return out

def invoke(folder,fn,*args,tag):
 cfg=json.loads((folder/'control.json').read_text())
 receipt=rpc(cfg,fn,*args,tag=tag,folder=folder)
 with (folder/'control_receipts.jsonl').open('a') as stream:stream.write(json.dumps(dict(operation=tag,task_id=receipt.get('task_id'),time=time.time()))+'\n')
 return receipt['result']

def pbs_state(snapshot):return next(iter(snapshot['pbs'].get('Jobs',{}).values()),{})

def build_full(root,pilot,pilot_snapshot,pilot_job):
 full=root/'af3_remaining';full.mkdir(exist_ok=True);cfg=json.loads((pilot/'control.json').read_text());cfg['remote_root']=str(Path(cfg['remote_root']).parent/'af3_remaining');save(full/'control.json',cfg)
 m=json.loads((pilot/'manifest.json').read_text());done={x['id'] for x in pilot_snapshot['complete'].values()};eligible={x['id'] for x in json.loads((root/'preparation_status.json').read_text())['rows'] if x['generation_clash_fraction']==0};jobs=[x for x in json.loads((root/'design_index.json').read_text()) if x['id'] in eligible-done]
 if not jobs:raise RuntimeError('No remaining eligible designs')
 seconds=max(x['process_wall_seconds'] for x in pilot_snapshot['complete'].values());hours=max(1,math.ceil((seconds*math.ceil(len(jobs)/6)*1.3+600)/3600))
 if hours>24:raise RuntimeError('Measured full-refolding walltime exceeds bounded one-node 24-hour allocation')
 for sub in ('inputs','references'):(full/sub).mkdir(exist_ok=True)
 for job in jobs:
  ident=job['id'];shutil.copy2(root/'inputs_af3_pending_msa'/f'{ident}.json',full/'inputs'/f'{ident}.json');shutil.copy2(root/'references'/f'{ident}.json',full/'references'/f'{ident}.json')
 for name in ('score_independent_refold.py','geometry.py','ame_filter.py'):shutil.copy2(pilot/name,full/name)
 shutil.copy2(Path(__file__).with_name('aurora_af3_refold_worker.py'),full/'worker.py')
 body=(pilot/'campaign.pbs').read_text().replace('as9_ame_pilot','as9_ame_all').replace('#PBS -q debug','#PBS -q capacity').replace('#PBS -l walltime=01:00:00',f'#PBS -l walltime={hours:02d}:00:00').replace(m['dependency'],pilot_job).replace(json.loads((pilot/'control.json').read_text())['remote_root'],cfg['remote_root']);(full/'campaign.pbs').write_text(body)
 m.update(expected_designs=len(jobs),jobs=jobs,purpose='All remaining designs passing the pre-refolding clash gate; pilot outputs reused',dependency=pilot_job,empirical_pilot_max_seconds=seconds,walltime_hours=hours,producer_revision='Fix output discovery to use JSON filename stem and report collection failures through exit status; model and sampling protocol unchanged')
 paths=[p for sub in ('inputs','references') for p in (full/sub).glob('*')]+[full/x for x in ('worker.py','score_independent_refold.py','geometry.py','ame_filter.py','campaign.pbs')];m['payload_sha256']={str(p.relative_to(full)):sha(p) for p in paths};save(full/'manifest.json',m)
 payload={name:dict(sha256=h,data=base64.b64encode(gzip.compress((full/name).read_bytes())).decode()) for name,h in m['payload_sha256'].items()};chunks=[];chunk={};count=0
 for name,row in payload.items():
  if chunk and count+len(row['data'])>500000:chunks.append(chunk);chunk={};count=0
  chunk[name]=row;count+=len(row['data'])
 if chunk:chunks.append(chunk)
 for i,chunk in enumerate(chunks):invoke(full,stage,cfg,chunk,m,tag=f'stage_{i:03d}')
 save(full/'staging_complete.json',dict(chunks=len(chunks),payload_files=len(payload),completed=time.time()))
 return full

def download(root,folder,snapshot):
 cfg=json.loads((folder/'control.json').read_text());rows=list(snapshot['complete'].values());dest=root/'independent_refolds';dest.mkdir(exist_ok=True);missing=[]
 for row in rows:
  p=dest/(row['id']+'.cif')
  if p.exists() and sha(p)!=row['model_sha256']:raise RuntimeError('Preserve differing local independent model '+row['id'])
  if not p.exists():missing.append(row)
  save(dest/(row['id']+'.remote.json'),row['metrics'])
 for i in range(0,len(missing),3):
  for item in invoke(folder,fetch_models,cfg,missing[i:i+3],tag='download_models'):
   raw=gzip.decompress(base64.b64decode(item['data']))
   if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Downloaded refold checksum mismatch')
   p=dest/(item['id']+'.cif');tmp=p.with_name(p.name+'.download.tmp');tmp.write_bytes(raw);tmp.replace(p)
 for row in rows:
  model=dest/(row['id']+'.cif');metrics=dest/(row['id']+'.metrics.json')
  old=json.loads(metrics.read_text()) if metrics.exists() else {}
  if old.get('scoring_version')=='motif-symmetry-v1' and old.get('independent_model_sha256')==row['model_sha256']:continue
  command=[root_config_python(root),str(Path(__file__).with_name('score_independent_refold.py')),str(root/'references'/(row['id']+'.json')),str(model),'--output',str(metrics)]
  result=subprocess.run(command,capture_output=True,text=True,timeout=120)
  if result.returncode:raise RuntimeError('Local symmetry-aware refold scoring failed: '+result.stderr[-2000:])

def root_config_python(root):
 config=json.loads((root/'local_analysis_config.json').read_text());return config['python']

def summary(root):
 rows=[];gen={x['id']:x for x in json.loads((root/'preparation_status.json').read_text())['rows']}
 for p in (root/'independent_refolds').glob('*.metrics.json'):rows.append(json.loads(p.read_text()))
 complete={r['id'] for r in rows}
 for ident,item in gen.items():
  if item['generation_clash_fraction']>0 and ident not in complete:rows.append(dict(id=ident,ame_gate='fail',ame_gate_reason='Pre-refolding protein-ligand clash gate failed',**{'clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A':item['generation_clash_fraction']}))
 fields=['id','ame_gate','ame_gate_reason','refolded_backbone_rmsd.rmsd','refolded_motif_allatom_rmsd.rmsd','clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A','mean_protein_plddt','distance_A','angle_deg','independent_model']
 with (root/'independent_refolding_metrics.csv').open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 result=dict(total_designs=len(gen),independent_models_downloaded=len(complete),AME_pass=sum(r['ame_gate']=='pass' for r in rows),AME_fail=sum(r['ame_gate']=='fail' for r in rows),pending=len(gen)-len(rows),scope='AF3/OpenFold3 exact-mode single-refold AME thresholds; subsequent 50-pose independent geometry evaluation not yet run',updated=time.time());save(root/'refolding_summary.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('--max-seconds',type=int,default=21600);p.add_argument('--interval',type=int,default=120);a=p.parse_args();root=a.evaluation.resolve();settings=json.loads((root/'continuation_config.json').read_text()) if (root/'continuation_config.json').exists() else {};msa=root/settings.get('msa_folder','msa_search');pilot=root/settings.get('pilot_folder','af3_pilot');statefile=root/'continuation_status.json';registry=root/'continuation_jobs.json';jobs=json.loads(registry.read_text()) if registry.exists() else {};start=time.monotonic();failures=0
 for name,folder in (('msa',msa),('pilot',pilot),('remaining',root/'af3_remaining')):
  if name in jobs:continue
  for file in folder.glob('submission_config*.json'):
   d=json.loads(file.read_text())
   if d.get('status')=='completed' and d.get('result',{}).get('job_id'):jobs[name]=d['result']['job_id']
 save(registry,jobs);summary(root)
 while time.monotonic()-start<a.max_seconds:
  try:
   mcfg=json.loads((msa/'control.json').read_text());ms=invoke(msa,status,mcfg,jobs['msa'],tag='status_latest');mj=pbs_state(ms)
   if mj.get('job_state')=='F' and (str(mj.get('Exit_status'))!='0' or not ms['complete']):raise RuntimeError('MSA search failed; preserve logs and dependent job without scientific retries')
   if 'pilot' not in jobs and mj.get('job_state') in ('R','F'):
    pcfg=json.loads((pilot/'control.json').read_text());jobs['pilot']=invoke(pilot,submit,pcfg,tag='submission')['job_id'];save(registry,jobs)
   phase='waiting_for_MSA_allocation' if 'pilot' not in jobs else 'waiting_for_MSA_or_AF3_pilot'
   if 'pilot' in jobs:
    pcfg=json.loads((pilot/'control.json').read_text());ps=invoke(pilot,status,pcfg,jobs['pilot'],tag='status_latest');download(root,pilot,ps);pj=pbs_state(ps);phase='AF3_pilot_'+str(pj.get('job_state'))
    completed_ids={x['id'] for x in ps['complete'].values()};unresolved_failures={k:v for k,v in ps['failed'].items() if v['id'] not in completed_ids}
    if unresolved_failures or (pj.get('job_state')=='F' and len(ps['complete'])!=6):raise RuntimeError('AF3 pilot has incomplete/failed predictions; do not launch remaining designs')
    if len(ps['complete'])==6 and 'remaining' not in jobs:
     full=root/'af3_remaining'
     if not (full/'staging_complete.json').exists():full=build_full(root,pilot,ps,jobs['pilot'])
     fcfg=json.loads((full/'control.json').read_text());jobs['remaining']=invoke(full,submit,fcfg,tag='submission')['job_id'];save(registry,jobs)
   if 'remaining' in jobs:
    full=root/'af3_remaining';fcfg=json.loads((full/'control.json').read_text());fs=invoke(full,status,fcfg,jobs['remaining'],tag='status_latest');download(root,full,fs);fj=pbs_state(fs);phase='AF3_remaining_'+str(fj.get('job_state'))
    if fj.get('job_state')=='F':
     final=summary(root);save(statefile,dict(status='refolding_stage_complete' if not fs['failed'] and final['pending']==0 else 'finished_with_incomplete_refolds',jobs=jobs,summary=final,failed_ids=list(fs['failed']),updated=time.time()));return
   current=summary(root);save(statefile,dict(status=phase,jobs=jobs,msa_state=mj.get('job_state'),summary=current,updated=time.time()));failures=0
  except Exception as error:
   text=str(error)
   if 'would exceed queue' in text:save(statefile,dict(status='waiting_for_PBS_queue_slot',jobs=jobs,error=text,updated=time.time()))
   else:
    failures+=1;save(statefile,dict(status='control_error_cooldown',jobs=jobs,error=traceback.format_exc(),consecutive_errors=failures,updated=time.time()))
    if failures>=3 or isinstance(error,TimeoutError):raise
  time.sleep(a.interval)
 save(statefile,dict(status='bounded_continuation_time_limit',jobs=jobs,summary=summary(root),updated=time.time()))

if __name__=='__main__':main()
