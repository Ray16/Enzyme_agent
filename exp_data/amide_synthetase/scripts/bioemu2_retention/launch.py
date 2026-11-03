"""Preflight and launch a bounded Lambda fleet using the shared GPU gate."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from core import digest, write_json

NFS=Path('/nfs/lambda_stor_01/homes/rzhu')
PYTHON=str(NFS/'Tool/envs/bioemu2/bin/python')
GATE=str(NFS/'bin/gpu_reserve')
THREADS=['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
         'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']

CPU_PROBE='''
import os,pathlib,json
allowed=set(os.sched_getaffinity(0)); occupied=set()
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  argv=(p/'cmdline').read_bytes().decode().split('\\0')
  affinity=set(os.sched_getaffinity(int(p.name)))
  if argv and 'python' in pathlib.Path(argv[0]).name and len(affinity)<len(allowed):
   occupied.update(affinity)
 except (OSError,UnicodeError):pass
print(json.dumps(dict(online=os.cpu_count(),load=max(os.getloadavg()),
 allowed=sorted(allowed),occupied=sorted(occupied),free=sorted(allowed-occupied))))
'''


def remote(host,argv):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
        shlex.join(argv)],text=True,timeout=55).strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--host',default='lambda5')
    p.add_argument('--count',type=int,default=12)
    p.add_argument('--only')
    args=p.parse_args()
    if not 1<=args.count<=12: raise ValueError('Pilot fleet is capped at 12 GPUs')
    root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    helper=NFS/'.codex-personal/skills/lambda-fleet/scripts/preflight_node.py'
    spec=importlib.util.spec_from_file_location('fleet_preflight',helper)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg=dict(modules=['bioemu2','mdtraj','rdkit','scipy','yaml','psutil'],
        binaries=[GATE],paths=[str(root/'manifest.json'),manifest['checkpoint']],
        temp_dir='/tmp',cpu_set='',min_available_ram_mib=12000*args.count)
    preflight=module.run_remote(cfg,PYTHON,args.host,None,55)
    if not preflight['ok']: raise RuntimeError(json.dumps(preflight))
    probe=json.loads(remote(args.host,['python3','-c',CPU_PROBE]))
    free_gpu=subprocess.check_output([GATE,'list','--hosts',args.host],text=True,timeout=55)
    indices=[int(row.split()[1]) for row in free_gpu.splitlines() if row.split()[0]==args.host]
    count=min(args.count,len(indices),len(probe['free'])//2)
    if not count or probe['load']+count*2>probe['online']:
        raise RuntimeError('Insufficient uncontended GPU/CPU capacity')
    source=Path(__file__).resolve().parent
    sources=sorted(source.glob('*.py'))
    code={p.name:digest(p) for p in sources}
    code_id=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()[:12]
    snapshot=root/'source'/code_id
    snapshot.mkdir(parents=True,exist_ok=True)
    for f in sources:
        target=snapshot/f.name
        if target.exists() and digest(target)!=digest(f): raise RuntimeError('Immutable source snapshot differs')
        if not target.exists(): shutil.copyfile(f,target)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch=root/'launches'/stamp
    launch.mkdir(parents=True)
    record=dict(host=args.host,preflight=preflight,cpu_probe=probe,source_id=code_id,source_hashes=code,
        manifest_sha256=digest(root/'manifest.json'),slots=[],only=args.only,
        requested_count=args.count,actual_count=count,
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip())
    for i,gpu in enumerate(indices[:count]):
        current=json.loads(remote(args.host,['python3','-c',CPU_PROBE]))
        # Recheck capacity and choose disjoint sets immediately before each launch.
        free=[c for c in current['free'] if c not in {n for s in record['slots'] for n in s['cpus']}]
        if len(free)<2 or current['load']+2>current['online']:
            record['capacity_stop']=current
            break
        cpus=free[:2]
        log=launch/f'gpu{gpu}.log'
        argv=['env',*[name+'=1' for name in THREADS],
            'HF_HUB_OFFLINE=1','HF_HOME='+str(NFS/'.cache/huggingface'),
            'taskset','-c',','.join(map(str,cpus)),GATE,'run',str(gpu),
            '--label','bioemu2-retention-panel','--',PYTHON,str(snapshot/'worker.py'),'--root',str(root)]
        if args.only: argv+=['--only',args.only]
        slot=dict(gpu=gpu,cpus=cpus,argv=argv,log=str(log))
        record['slots'].append(slot)
        write_json(launch/'manifest.json',record)
        command='nohup '+shlex.join(argv)+' >'+shlex.quote(str(log))+' 2>&1 </dev/null & echo $!'
        slot['gate_pid']=int(remote(args.host,['bash','-lc',command]).splitlines()[-1])
        write_json(launch/'manifest.json',record)
        print('LAUNCHED',args.host,gpu,cpus,slot['gate_pid'],log,flush=True)
        # Starting the next worker entails SSH/probing overhead, naturally staggering imports.
    print('launch_manifest',launch/'manifest.json',flush=True)


if __name__=='__main__':
    main()
