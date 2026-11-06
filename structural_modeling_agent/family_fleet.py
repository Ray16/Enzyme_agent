"""Launch the shared family queue across verified free Lambda resources."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from enzyme_discovery_agent.state_store import atomic_write_json


NFS=Path('/nfs/lambda_stor_01/homes/rzhu')
GATE=NFS/'bin/gpu_reserve'
STOCK_PYTHON='/homes/rzhu/miniforge3/envs/boltz-2/bin/python'
STOCK_LIB='/homes/rzhu/miniforge3/envs/boltz-2/lib'
THREADS=('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')


def inspect_node(host,project):
    python=str(NFS/'boltz2-venv/bin/python') if host=='lambda13' else STOCK_PYTHON
    profile='exact' if host=='lambda13' else 'stock_fp32_serial'
    code='''import os,json,psutil,subprocess,importlib
occupied=set();allowed=set(os.sched_getaffinity(0))
for process in psutil.process_iter(['cmdline']):
 try:
  affinity=set(process.cpu_affinity())
  if process.info['cmdline'] and len(affinity)<len(allowed) and len(affinity)<=32:occupied|=affinity
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
for name in ['torch','boltz.main','rdkit','gemmi','structural_modeling_agent.family_cofold']:importlib.import_module(name)
import torch
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,compute_cap','--format=csv,noheader,nounits'],text=True)
print(json.dumps({'host':os.uname().nodename,'online':os.cpu_count(),'load':os.getloadavg(),'available_ram':psutil.virtual_memory().available,
 'free_cpus':sorted(allowed-occupied),'gpus':[line.split(', ') for line in rows.splitlines()],'torch':torch.__version__,'compiled_archs':torch._C._cuda_getArchFlags()}))'''
    env=['env','PYTHONPATH='+str(project),*['%s=1'%key for key in THREADS]]
    if profile!='exact':env+=['LD_LIBRARY_PATH='+STOCK_LIB]
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
        shlex.join(env+[python,'-c',code])],capture_output=True,text=True,timeout=180,check=True)
    report=json.loads(result.stdout.strip().splitlines()[-1]);report.update(python=python,profile=profile)
    return report


def launch(run,specification,out,launch_workers=False):
    project=Path(__file__).resolve().parent.parent;out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    task=json.loads(Path(specification).read_text())
    minimum_vram=task.get('minimum_gpu_memory_mib',24000)
    free=subprocess.check_output([str(GATE),'list'],text=True,timeout=90)
    by_host={}
    for line in free.splitlines():
        host,index=line.split();by_host.setdefault(host,[]).append(int(index))
    provider=out/'fleet_native_provider.py'
    if not provider.exists():provider.write_text('import sys\nsys.path.insert(0,'+repr(str(project))+')\n'+(project/'structural_modeling_agent/lambda_boltz_backend.py').read_text())
    original=json.loads((Path(run)/'rolling_modeling/backend.json').read_text())
    reports={};workers=[];errors=[]
    for host,indices in by_host.items():
        try:node=inspect_node(host,project)
        except (ValueError,OSError,subprocess.SubprocessError) as exc:
            errors.append({'host':host,'reason':getattr(exc,'stderr','')[-1000:] or str(exc)});continue
        reports[host]=node
        eligible=[];small=False
        for gpu in node['gpus']:
            index=int(gpu[0]);memory=int(gpu[2]);capability=float(gpu[3])
            if index not in indices or capability<7 or memory<minimum_vram:continue
            small=memory<24000
            eligible.append((index,small))
        capacity=min(len(eligible),len(node['free_cpus'])//5,
                     max(0,int((node['online']-max(node['load']))//5)),node['available_ram']//(16*1024**3))
        cpus=node['free_cpus']
        for slot,(gpu,small) in enumerate(eligible[:capacity]):
            worker_id=host+'-slot-'+str(slot);directory=out/'workers'/worker_id;directory.mkdir(parents=True,exist_ok=True)
            owner=directory/'owner.json'
            if owner.exists():
                errors.append({'host':host,'worker_id':worker_id,'reason':'Existing owner record requires reconciliation; no duplicate launched.'});continue
            front=cpus[slot*5];native=cpus[slot*5+1:slot*5+5]
            backend={**original,'argv':[node['python'],str(provider),'--state-dir',str(directory/'provider'),'--pool',original['argv'][-1]],
                'resource_policy':{**original['resource_policy'],'execution_host':host,'worker_python':node['python'],
                    'numerical_profile':node['profile'],'serial_producer':str(project/'structural_modeling_agent/cofold/boltz_serial_predict.py'),
                    'library_path':STOCK_LIB if node['profile']!='exact' else '',
                    'gpu_index':gpu,'gpu_cpu_affinity':native,'allow_any_free_gpu':True,
                    'minimum_gpu_memory_mib':minimum_vram,'allowed_gpu_indices':[index for index,_ in eligible],
                    'gpu_dispatch_root':str(out/'gpu_dispatch'),'max_candidate_ensembles':None}}
            backend_path=directory/'backend.json';atomic_write_json(backend_path,backend)
            argv=['taskset','-c',str(front),node['python'],'-u','-m','structural_modeling_agent','family-cofold',
                '--run',str(Path(run).resolve()),'--specification',str(Path(specification).resolve()),'--backend',str(backend_path),
                '--out',str(out),'--worker-id',worker_id,'--expand-after-pilot','--watch-seconds','30']
            task=json.loads(Path(specification).read_text())
            if task.get('pocket_guided_from'):
                argv+=['--pocket-guided-from',str(task['pocket_guided_from'])]
            pilot=task.get('fleet_pilots',{}).get('small' if small else 'standard')
            if not pilot:
                queue=json.loads((out/'queue.json').read_text())
                eligible_jobs=sorted(queue['items'],key=lambda j:(j['rank'],j['step']))
                if not eligible_jobs:continue
                pilot=eligible_jobs[0]['accession']
            argv+=['--pilot-accession',pilot]
            if small:
                argv+=['--pilot-steps','1']
                if task.get('small_card_max_length'):argv+=['--max-protein-length',str(task['small_card_max_length'])]
            env=['env','PYTHONPATH='+str(project),'MPLBACKEND=Agg',*['%s=1'%key for key in THREADS]]
            if node['profile']!='exact':env+=['LD_LIBRARY_PATH='+STOCK_LIB]
            command=env+argv
            record={'host':host,'worker_id':worker_id,'frontend_cpu':[front],'native_cpus':native,'preferred_gpu':gpu,
                'profile':node['profile'],'small_models_only':small,'argv':command,'status':'prepared'}
            if launch_workers:
                remote='nohup '+shlex.join(command)+' > '+shlex.quote(str(directory/'frontend.log'))+' 2>&1 < /dev/null &'
                submitted=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,remote],capture_output=True,text=True,timeout=25)
                record['status']='submitted' if submitted.returncode==0 else 'submission_unknown'
            workers.append(record)
    report={'workers':workers,'node_preflight':reports,'unavailable':errors,'setup_llm_calls':0,
            'policy':'Use verified free resources; five disjoint CPU slots and 16 GiB host RAM reserved per worker. Small-memory cards receive shorter proteins. Exact-match results are reused.'}
    atomic_write_json(out/'fleet_launch_manifest.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--specification',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();report=launch(args.run,args.specification,args.out,args.launch)
    print(json.dumps({'workers':len(report['workers']),'hosts':list(report['node_preflight']),'unavailable':report['unavailable']}))


if __name__=='__main__':main()
