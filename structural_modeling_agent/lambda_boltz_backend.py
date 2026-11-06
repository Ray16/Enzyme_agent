"""Idempotent Lambda Boltz provider with explicit state storage and owned GPU execution."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import time

ROOT = None
POOL = None
NFS = Path('/nfs/lambda_stor_01/homes/rzhu')
GATE = NFS/'bin/gpu_reserve'
KIT = NFS/'Tool/uplifting-biomolecular-modeling/boltz2'
PYTHON = NFS/'boltz2-venv/bin/python'
THREADS = ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')

def cap_native_threads():
    for variable in THREADS:os.environ[variable]='1'

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(tmp,path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def alive(pid, ticks):
    try: return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19] == ticks
    except OSError: return False

def allocation(slots=4,requested=None):
    import psutil
    available = set(os.sched_getaffinity(0)); occupied = set()
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid == os.getpid(): continue
        try:
            affinity = set(process.cpu_affinity())
            if len(affinity) <= 8 and process.info['cmdline']: occupied |= affinity
        except psutil.NoSuchProcess: continue
        except psutil.AccessDenied:
            # Affinity remains available through the kernel even when cmdline is private.
            try:
                affinity=set(os.sched_getaffinity(process.pid))
                if len(affinity)<=8: occupied |= affinity
            except ProcessLookupError: pass
    cpus = list(requested) if requested is not None else sorted(available-occupied)[-slots:]
    if len(cpus)!=slots or len(set(cpus))!=slots or not set(cpus)<=available-occupied or max(os.getloadavg())+slots > os.cpu_count():
        raise RuntimeError('Insufficient verifiable CPU capacity')
    return cpus

def unexpected_residents(predictor,residents,descendants,live_pids,sessions):
    """A completed producer or stale NVML observation cannot own a live conflict."""
    if predictor.poll() is not None:return set()
    owned={predictor.pid}|set(descendants)
    owned|={pid for pid,session in sessions.items() if session==predictor.pid}
    return (set(residents)&set(live_pids))-owned

def chemical_component_sources(document,state_dir):
    """Cache authoritative CCD definitions once per component, independently of targets."""
    import requests
    import gemmi
    result={};cache=Path(state_dir)/'chemical_components';cache.mkdir(exist_ok=True)
    for ccd in sorted({entity['ligand']['ccd'] for entity in document['sequences'] if entity.get('ligand',{}).get('ccd')}):
        if not re.fullmatch('[A-Za-z0-9]{1,12}',ccd):raise ValueError('Invalid chemical component identifier')
        path=cache/(ccd+'.cif');url='https://files.rcsb.org/ligands/download/'+ccd+'.cif'
        if not path.exists():
            response=requests.get(url,timeout=(30,60));response.raise_for_status()
            temporary=path.with_suffix('.tmp');temporary.write_bytes(response.content);os.replace(temporary,path)
        block=gemmi.cif.read_file(str(path)).sole_block()
        if gemmi.cif.as_string(block.find_value('_chem_comp.id'))!=ccd:raise ValueError('CCD definition mismatches requested component')
        result[ccd]={'path':str(path),'sha256':sha(path),'source_url':url,'source_id':'CCD:'+ccd}
    return result

def worker(job):
    cap_native_threads()
    import psutil
    import yaml
    request=read(job/'request.json'); key=request['idempotency_key']
    receipt={'idempotency_key':key,'status':'queued','host':socket.gethostname(),
             'pid':os.getpid(),'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}
    write(job/'receipt.json',receipt)
    predictor=None;dispatch_claim=None
    def stop(*_):
        if predictor and predictor.poll() is None:
            os.killpg(predictor.pid,signal.SIGTERM)
            try: predictor.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(predictor.pid,signal.SIGKILL); predictor.wait()
        raise SystemExit('terminated; partial outputs retained')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        policy=request.get('resource_policy',{});profile=policy.get('numerical_profile','exact')
        host=policy.get('execution_host','lambda13')
        requested_cpus=policy.get('gpu_cpu_affinity')
        os.sched_setaffinity(0,set(allocation(requested=requested_cpus)))
        if socket.gethostname().split('.')[0]!=host: raise RuntimeError('Wrong accelerator host')
        if sha(request['input_yaml'])!=request['input_sha256']:
            raise RuntimeError('Rendered input changed')
        for source in request.get('msa_evidence',[]):
            if sha(source['path'])!=source['sha256']:raise RuntimeError('Prepared MSA changed')
        document=yaml.safe_load(Path(request['input_yaml']).read_text())
        component_sources=chemical_component_sources(document,policy.get('ccd_cache_root',ROOT))
        for entity in document['sequences']:
            if 'protein' in entity:
                msa=entity['protein'].get('msa')
                if not msa or msa=='empty' or not Path(msa).is_file():
                    raise RuntimeError('Every protein requires a prepared sequence-specific MSA')
        # Preflight before allocating any accelerator. No downloads or environment changes.
        import boltz.main, torch, rdkit
        from rdkit import Chem
        ligand_atom_maps={}
        for entity in document['sequences']:
            ligand=entity.get('ligand') or {}
            if not ligand.get('smiles'):continue
            molecule=Chem.AddHs(Chem.MolFromSmiles(ligand['smiles']))
            ranks=list(Chem.CanonicalRankAtoms(molecule))
            ligand_atom_maps[ligand['id']]={'smiles':ligand['smiles'],'rdkit_version':rdkit.__version__,
                'atoms':{str(atom.GetIdx()):{'name':atom.GetSymbol().upper()+str(ranks[atom.GetIdx()]+1),'element':atom.GetSymbol().upper()}
                         for atom in molecule.GetAtoms() if atom.GetAtomicNum()!=1}}
        for name in ('boltz2_conf.ckpt','ccd.pkl','mols'):
            if not (NFS/'.boltz'/name).exists(): raise RuntimeError('Missing checkpoint/cache: '+name)
        total_length=sum(len(e['protein']['sequence']) for e in document['sequences'] if 'protein' in e)
        required_bytes=2*1024**3+request['expected_predictions']*(total_length+200)**2*16
        if shutil.disk_usage(job).free<required_bytes:
            raise RuntimeError('Insufficient shared storage for this ensemble and safety reserve; no cofold submitted')
        deadline=time.monotonic()+21600
        receipt.update(phase='waiting_for_free_gpu',requested_gpu=request.get('resource_policy',{}).get('gpu_index'))
        write(job/'receipt.json',receipt)
        while True:
            free=subprocess.run([str(GATE),'list','--hosts',host],capture_output=True,text=True,check=True,timeout=30).stdout
            choices=[int(line.split()[1]) for line in free.splitlines()
                     if len(line.split())>=2 and line.split()[0]==host and line.split()[1].isdigit()]
            policy=request.get('resource_policy',{});fixed_gpu=policy.get('gpu_index')
            if fixed_gpu is not None and not policy.get('allow_any_free_gpu'):
                choices=[index for index in choices if index==fixed_gpu]
            elif fixed_gpu in choices:choices=[fixed_gpu]+[index for index in choices if index!=fixed_gpu]
            dispatch_root=Path(policy['gpu_dispatch_root']) if policy.get('gpu_dispatch_root') else None
            if dispatch_root:
                dispatch_root.mkdir(parents=True,exist_ok=True);selected=None
                for candidate_gpu in choices:
                    candidate_claim=dispatch_root/(host+'_gpu_'+str(candidate_gpu))
                    try:candidate_claim.mkdir()
                    except FileExistsError:
                        owner_path=candidate_claim/'owner.json'
                        if owner_path.exists():
                            owner=read(owner_path)
                            if owner.get('host')==socket.gethostname() and not alive(owner['pid'],owner['process_start_ticks']):
                                archive=dispatch_root/'stale';archive.mkdir(exist_ok=True)
                                os.rename(candidate_claim,archive/(candidate_claim.name+'_'+str(time.time_ns())))
                        continue
                    dispatch_claim=candidate_claim
                    write(candidate_claim/'owner.json',{'pid':os.getpid(),'host':socket.gethostname(),
                        'process_start_ticks':receipt['process_start_ticks'],'job':key})
                    selected=candidate_gpu;break
                choices=[selected] if selected is not None else []
            if choices:break
            if time.monotonic()>deadline: raise RuntimeError('GPU wait allowance exhausted; no cofold submitted')
            time.sleep(30)
        cpus=allocation(requested=requested_cpus); os.sched_setaffinity(0,set(cpus)); gpu=choices[0]
        env=os.environ.copy(); env.update({name:'1' for name in THREADS})
        env.pop('PYTHONPATH',None)
        command=['taskset','-c',','.join(map(str,cpus)),str(GATE),'run',str(gpu),'--label','discovery-'+key[:12],'--',
                 'bash',str(KIT/'_kit_run.sh'),'pred','--config','h100','--mode','exact',
                 '--input',request['input_yaml'],'--out_dir',str(job/'outputs'),
                 '--diffusion_samples',str(request['expected_predictions']),'--seed','42',
                 '--num_workers','0','--preprocessing-threads','1','--output_format','pdb']
        producer=KIT/'_kit_run.sh';batch_size=request['expected_predictions']
        if profile=='stock_fp32_serial':
            producer=Path(policy['serial_producer']);batch_size=1
            options=['predict',request['input_yaml'],'--out_dir',str(job/'outputs'),'--diffusion_samples',str(request['expected_predictions']),
                     '--seed','42','--num_workers','0','--preprocessing-threads','1','--output_format','pdb',
                     '--max_parallel_samples','1','--cache',str(NFS/'.boltz'),'--model','boltz2','--no_kernels']
            boltz.main.cli.commands['predict'].make_context('predict',options[1:])
            prefix=command[:command.index('--')+1]
            command=prefix+[policy['worker_python'],'-u',str(producer)]+options
        elif profile=='exact':
            # Verify the installed predictor's real interface before reserving.
            from boltz2_opt.cli import build_parser
            kit_args=command[command.index('pred'):]
            index=kit_args.index('--config');del kit_args[index:index+2]
            build_parser().parse_args(kit_args)
        else:raise ValueError('Unsupported numerical profile')
        receipt.update(status='running',phase='predicting',gpu=gpu,cpu_affinity=cpus,argv=command,
                       load_before=os.getloadavg(),thread_limits={name:env[name] for name in THREADS},
                       started_at=time.time(),producer_sha256=sha(producer),
                       lengths=[len(e['protein']['sequence']) for e in document['sequences'] if 'protein' in e],
                       requested_samples=request['expected_predictions'],batch_size=batch_size,
                       numerical_profile=profile,peak_allocated_memory=None,peak_reserved_memory=None)
        write(job/'receipt.json',receipt)
        with (job/'predictor.log').open('ab') as log:
            predictor=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            receipt['predictor_pid']=predictor.pid;write(job/'receipt.json',receipt)
            started=time.monotonic()
            while predictor.poll() is None:
                if time.monotonic()-started>21600: stop()
                # Any resident process outside our worker tree invalidates exclusivity.
                query=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
                if predictor.poll() is not None:break
                residents={int(v.strip()) for v in query.stdout.splitlines() if v.strip().isdigit()}
                try:descendants={p.pid for p in psutil.Process(predictor.pid).children(recursive=True)}
                except psutil.NoSuchProcess:descendants=set()
                live={pid for pid in residents if psutil.pid_exists(pid)}
                sessions={}
                for pid in live:
                    try:sessions[pid]=os.getsid(pid)
                    except ProcessLookupError:pass
                foreign=unexpected_residents(predictor,residents,descendants,live,sessions)
                if foreign:
                    receipt.update(error='Foreign resident compute process appeared',foreign_resident_pids=sorted(foreign),
                                   observed_residents=sorted(residents),observed_descendants=sorted(descendants))
                    stop()
                time.sleep(5)
        files=sorted((job/'outputs').rglob('*_model_*.pdb'))
        confidence=sorted((job/'outputs').rglob('confidence_*_model_*.json'))
        for line in (job/'predictor.log').read_text(errors='replace').splitlines():
            if line.startswith('{'):
                try:measurement=json.loads(line)
                except ValueError:continue
                if measurement.get('producer')=='boltz_serial_fp32_v1_finish':
                    receipt['peak_allocated_memory']=measurement.get('peak_allocated_bytes')
                    receipt['peak_reserved_memory']=measurement.get('peak_reserved_bytes')
        if predictor.returncode or len(files)!=request['expected_predictions'] or len(confidence)!=len(files):
            raise RuntimeError(f'Incomplete prediction: rc={predictor.returncode}, coordinates={len(files)}, confidence={len(confidence)}')
        destination=job/'ensemble';destination.mkdir(exist_ok=True)
        for file in files+confidence:
            target=destination/file.name
            if target.exists() or target.is_symlink(): raise RuntimeError('Output slot already exists; preserve for reconciliation')
            target.symlink_to(file)
        producer={'input_sha256':request['input_sha256'],'execution_sha256':request['execution_sha256'],
                  'predictions':[{'path':str(p),'sha256':sha(p)} for p in destination.glob('*.pdb')],
                  'producer':receipt,'mode':profile,'seed':42,'chemical_component_sources':component_sources,
                  'ligand_atom_maps':ligand_atom_maps,
                  'format_policy':{'generic_ligand_residue_names':['LIG']}}
        write(job/'prediction_manifest.json',producer)
        receipt.update(status='complete',prediction_dir=str(destination),prediction_manifest=str(job/'prediction_manifest.json'),
                       completed_samples=len(files),process_wall_seconds=time.monotonic()-started)
    except BaseException as exc:
        if predictor and predictor.poll() is None:
            os.killpg(predictor.pid,signal.SIGTERM); predictor.wait(timeout=60)
        receipt.update(status='failed',error=receipt.get('error') or f'{type(exc).__name__}: {exc}')
    finally:
        receipt['finished_at']=time.time(); write(job/'receipt.json',receipt)
        if dispatch_claim:
            (dispatch_claim/'owner.json').unlink();dispatch_claim.rmdir()

def check_prediction_pool(document,pool):
    """Only the same molecular system can block a new target-substrate prediction."""
    from campaign_catalog.lookup import lookup_predictions
    from collections import Counter
    from rdkit import Chem
    import yaml
    def system(doc):
        entities=[]
        for item in doc.get('sequences',[]):
            value=item.get('protein')
            if value:
                copies=len(value['id']) if isinstance(value.get('id'),list) else 1
                entities.extend([('protein',value['sequence'])]*copies)
            value=item.get('ligand')
            if value:
                copies=len(value['id']) if isinstance(value.get('id'),list) else 1
                identity=('ccd',value['ccd']) if value.get('ccd') else ('smiles',Chem.MolToSmiles(Chem.MolFromSmiles(value['smiles'])))
                entities.extend([identity]*copies)
        return Counter(entities),doc.get('constraints') or []
    wanted=system(document)
    checks=[]
    for entity in document.get('sequences',[]):
        if 'protein' not in entity:continue
        protein=entity['protein']
        check=lookup_predictions(pool,sequence=protein['sequence'])
        compatible=[]
        for match in check['matches']:
            source=(match.get('setup') or {}).get('input_source') or {}
            path=Path(source.get('path',''))
            if (path.is_file() and sha(path)==source.get('sha256')
                    and system(yaml.safe_load(path.read_text()))==wanted):compatible.append(match)
        check={**check,'same_system_matches':compatible}
        checks.append(check)
    return {'pool':str(Path(pool).resolve()),'checks':checks,
            'existing_predictions':any(check['same_system_matches'] for check in checks)}


def validate_sample_count(count,policy):
    minimum=policy.get('min_predictions_per_job',1);limit=policy.get('max_predictions_per_job')
    if (type(count) is not int or type(minimum) is not int or type(limit) is not int
            or not 1<=minimum<=count<=limit):
        raise ValueError('Sample count must fit the explicit quality and resource allowance')


def adapter():
    request=json.load(sys.stdin); key=request['idempotency_key']
    if not re.fullmatch('[0-9a-f]{64}',key): raise ValueError('Unsafe job identity')
    if request['operation'] not in {'submit','status'}:raise ValueError('Unsupported provider operation')
    if request['operation']=='submit':
        validate_sample_count(request.get('expected_predictions'),request['resource_policy'])
    job=ROOT/'backend_jobs'/key; job.mkdir(parents=True,exist_ok=True)
    # Serialize claims across different job identities as well as repeated requests.
    with (ROOT/'submission.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (job/'receipt.json').exists():
            receipt=read(job/'receipt.json')
        elif request['operation']=='status': receipt={'idempotency_key':key,'status':'not_found'}
        else:
            import yaml
            if POOL is None:raise RuntimeError('Supply --pool for a current canonical-pool lookup before submission')
            document=yaml.safe_load(Path(request['input_yaml']).read_text())
            pool_check=check_prediction_pool(document,POOL)
            write(job/'reuse_check.json',pool_check)
            if pool_check['existing_predictions']:
                receipt={'idempotency_key':key,'status':'failed',
                         'error':'Existing canonical-pool cofold: inspect and validate it; automatic recofolding is disabled.'}
                write(job/'receipt.json',receipt);print(json.dumps(receipt));return
            attempts=list((ROOT/'backend_jobs').glob('*/request.json'))
            ensemble_limit=request['resource_policy'].get('max_candidate_ensembles',8)
            if ensemble_limit is not None and len(attempts)>=ensemble_limit:
                receipt={'idempotency_key':key,'status':'failed','error':'Campaign ensemble allowance reached'}
                write(job/'receipt.json',receipt);print(json.dumps(receipt));return
            # All jobs pass through one campaign owner, one GPU at a time.
            active=[read(p) for p in (ROOT/'backend_jobs').glob('*/receipt.json')]
            if any(r['status'] in {'queued','running','unknown'} for r in active):
                raise RuntimeError('A cofold is already outstanding; reconcile it first')
            write(job/'request.json',request)
            write(job/'receipt.json',{'idempotency_key':key,'status':'queued'})
            policy=request['resource_policy'];host=policy.get('execution_host','lambda13')
            argv=[policy.get('worker_python',str(PYTHON)),'-u',str(Path(__file__).resolve()),'--state-dir',str(ROOT),'--worker',str(job)]
            if host==socket.gethostname().split('.')[0]:
                if policy.get('gpu_cpu_affinity'):
                    argv=['taskset','-c',','.join(map(str,policy['gpu_cpu_affinity']))]+argv
                environment=os.environ.copy()
                if policy.get('library_path'):environment['LD_LIBRARY_PATH']=policy['library_path']
                with (job/'worker.log').open('ab') as log:
                    process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        env=environment,start_new_session=True)
            else:
                if policy.get('library_path'):argv=['env','LD_LIBRARY_PATH='+policy['library_path']]+argv
                remote='nohup '+shlex.join(argv)+' > '+shlex.quote(str(job/'worker.log'))+' 2>&1 < /dev/null &'
                result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,remote],capture_output=True,text=True,timeout=25)
                if result.returncode:
                    write(job/'receipt.json',{'idempotency_key':key,'status':'unknown','error':'SSH submission uncertain; reconcile instead of retrying',
                        'submission_stderr':result.stderr[-1000:]})
            receipt=read(job/'receipt.json')
        if request['operation']=='status' and receipt.get('status') in {'queued','running'}:
            # A producer may still be alive after its coordinator vanished: never reset its claim here.
            receipt={**receipt,'reconcile_note':'Durable worker receipt; status calls never allocate compute.'}
    print(json.dumps(receipt))

if __name__=='__main__':
    cap_native_threads()
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',required=True,help='Explicit NFS run/provider state directory; never writes into the installed package')
    parser.add_argument('--pool',type=Path,help='Canonical prediction pool; required for new submissions, never used as discovery context')
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    ROOT=Path(args.state_dir).resolve();ROOT.mkdir(parents=True,exist_ok=True)
    POOL=args.pool
    if args.worker:worker(args.worker)
    else:adapter()
