"""Automatically refold every generated AS9 design after generation completes.

Fresh batched MSAs, unrestrained Boltz-2 ensembles, no generation-QC exclusion.
Uses the existing reservation-aware provider, with explicit state per design.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback

THREADS=('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(root,phase,**fields):
    save(root/'status.json',dict(phase=phase,expected_designs=100,updated=time.time(),**fields))


def wait_generation(root,cfg):
    gen=Path(cfg['generation_dir']);began=time.monotonic()
    m=json.loads((gen/'manifest.json').read_text())
    while time.monotonic()-began<cfg['generation_wait_seconds']:
        complete={p.stem for p in (gen/'complete').glob('*.json')}
        failed={p.stem for p in (gen/'failed').glob('*.json')}
        files=list((gen/'runs').glob('*/inference_samples/**/*_generated.cif'))
        if len(complete|failed)==len(m['jobs']):
            return m,files
        status(root,'waiting_for_generation',generated_files=len(files),expected_draws=100,
               completed_generation_chunks=len(complete),failed_generation_chunks=sorted(failed))
        time.sleep(30)
    raise RuntimeError('Generation wait limit reached; no unverified completion inferred')


def collect(root,cfg,manifest,files):
    from biotite.structure.io import pdbx
    from biotite.sequence import ProteinSequence
    gen=Path(cfg['generation_dir']);jobs={j['id']:j for j in manifest['jobs']}
    qc={row['path']:row for f in (gen/'complete').glob('*.json') for row in json.loads(f.read_text())['rows']}
    records=[];errors=[]
    for path in sorted(files):
        try:
            atoms=pdbx.get_structure(pdbx.CIFFile.read(path),model=1)
            protein=atoms[atoms.chain_id=='A'];ca=protein[protein.atom_name=='CA']
            seq=''.join(ProteinSequence.convert_letter_3to1(str(a)) for a in ca.res_name)
            if not seq or len(set(map(int,ca.res_id)))!=len(seq):raise ValueError('Invalid protein residue inventory')
            relative=path.relative_to(gen/'runs');job=jobs[relative.parts[0]]
            ident='AS9D_'+hashlib.sha256(str(relative).encode()).hexdigest()[:12]
            input_id=job['input_id']
            records.append(dict(id=ident,sequence=seq,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),
                design_cif=str(path),design_sha256=sha(path),input_id=input_id,
                motif_input=str(gen/'inputs'/(input_id+'.cif')),
                motif_metadata=str(gen/'inputs'/(input_id+'.metadata.json')),
                motif_input_sha256=sha(gen/'inputs'/(input_id+'.cif')),
                motif_metadata_sha256=sha(gen/'inputs'/(input_id+'.metadata.json')),
                generation_QC=qc.get(str(path),{'scope':'generation QC unavailable; included anyway'})))
        except Exception:
            errors.append(dict(path=str(path),error=traceback.format_exc()))
    save(root/'design_inventory.json',dict(designs=records,unreadable_outputs=errors,
        policy='Every readable generated sequence included; no geometry, Pro311, AME or generation-QC selection'))
    if not records:raise RuntimeError('No readable generated sequences')
    (root/'sequences.fasta').write_text(''.join('>'+r['id']+'\n'+r['sequence']+'\n' for r in records))
    return records


def prepare_msas_and_inputs(root,cfg,records):
    env=dict(os.environ,LD_LIBRARY_PATH=cfg['library_path'])
    for k in THREADS:env[k]='1'
    status(root,'searching_fresh_batched_MSAs',designs=len(records))
    node=inspect_host(os.uname().nodename.split('.')[0])
    if not node['free_cpus'] or node['load']+1>node['online']:
        raise RuntimeError('No verified CPU capacity for batched MSA search')
    msa_cpu=cfg['msa_cpu'] if cfg['msa_cpu'] in node['free_cpus'] else node['free_cpus'][-1]
    # Resilient batched MSA: small chunks + retries so a congested/flaky ColabFold
    # server degrades to a resume rather than a total loss. Completed chunks cache by
    # sequence hash; each retry (and batch_msa's own per-chunk retries) only fetches
    # what is still uncached. Correctness gate is cache COMPLETENESS, not index.tsv
    # existence (batch_msa writes index.tsv even on a partial run).
    command=['taskset','-c',str(msa_cpu),cfg['boltz_python'],'-u',str(root/'producer/batch_msa.py'),
             '--fasta',str(root/'sequences.fasta'),'--out',str(root/'msa'),
             '--chunk',str(cfg.get('msa_chunk',25)),'--also-a3m']
    def _unique_hashes():
        import hashlib as _h
        seqs=set();cur=None;buf=''
        for ln in (root/'sequences.fasta').read_text().splitlines():
            if ln.startswith('>'):
                if cur is not None:seqs.add(_h.sha1(buf.encode()).hexdigest()[:16])
                cur=ln;buf=''
            elif cur is not None:buf+=ln.strip()
        if cur is not None:seqs.add(_h.sha1(buf.encode()).hexdigest()[:16])
        return seqs
    want=_unique_hashes()
    cached=lambda:sum(1 for h in want if (root/'msa'/(h+'.csv')).exists())
    attempts=int(cfg.get('msa_attempts',8))
    # Always invoke batch_msa at least once even when every MSA is already cached:
    # it is instant with nothing to fetch but (re)writes a COMPLETE index.tsv for all
    # ids. Skipping it here would leave a stale/partial index.tsv and KeyError later.
    for attempt in range(1,attempts+1):
        with (root/f'msa_{time.time_ns()}.log').open('x') as log:
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'msa_owner.json',dict(pid=process.pid,argv=command,started=time.time(),attempt=attempt))
            try:rc=process.wait(timeout=cfg['msa_timeout_seconds'])
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        status(root,'searching_fresh_batched_MSAs',designs=len(records),msa_attempt=attempt,
               msa_cached=cached(),msa_expected=len(want),last_rc=rc)
        if rc==0 and cached()>=len(want):break
        if attempt<attempts:time.sleep(min(30*attempt,180))
    if cached()<len(want):
        raise RuntimeError(f'Batched MSA search incomplete after {attempts} attempts: {cached()}/{len(want)} unique sequences cached')
    index={r['id']:r for r in csv.DictReader((root/'msa/index.tsv').open(),delimiter='\t')}
    out=[]
    for record in records:
        row=index[record['id']];msa=Path(row['csv'])
        sequences=[r['sequence'] for r in csv.DictReader(msa.open())]
        ungap=lambda s:''.join(c for c in s if c.isalpha()).upper()
        if not sequences or ungap(sequences[0])!=record['sequence']:
            raise RuntimeError('Fresh MSA query does not match design '+record['id'])
        unique=len(set(sequences))
        meta=json.loads(Path(record['motif_metadata']).read_text())
        document={'version':1,'sequences':[
            {'protein':{'id':'A','sequence':record['sequence'],'msa':str(msa)}},
            {'ligand':{'id':'B','smiles':meta['ligand_smiles']['B']}},
            {'ligand':{'id':'C','ccd':'MG'}},
            {'ligand':{'id':'D','smiles':meta['ligand_smiles']['D']}}]}
        # JSON is a YAML subset; no templates, coordinates or restraints supplied.
        yaml=root/'inputs'/(record['id']+'.yaml');save(yaml,document)
        record.update(input_yaml=str(yaml),input_sha256=sha(yaml),msa=str(msa),msa_sha256=sha(msa),
            msa_unique_aligned_sequences=unique,msa_scope='fresh searched MSA' if unique>1 else 'fresh search; query-only alignment')
        out.append(record)
    save(root/'refold_inputs.json',out)
    return out


def inspect_host(host):
    code="""import os,json
from pathlib import Path
occupied=set()
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid<1000 or not(p/'cmdline').read_bytes():continue
  slots=os.sched_getaffinity(int(p.name))
  if len(slots)<os.cpu_count():occupied|=slots
 except OSError:pass
print(json.dumps(dict(free_cpus=sorted(set(os.sched_getaffinity(0))-occupied),online=os.cpu_count(),load=max(os.getloadavg()))))
"""
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
        shlex.join(['/usr/bin/python3','-c',code])],capture_output=True,text=True,timeout=20,check=True)
    return json.loads(result.stdout)


def launch(root,cfg,record,host,gpu,cpus):
    for field in ('design','motif_input','motif_metadata'):
        path=record['design_cif'] if field=='design' else record[field]
        if sha(path)!=record[field+'_sha256']:raise RuntimeError('Frozen design/reference changed: '+field)
    job=root/'jobs'/record['id'];job.mkdir(exist_ok=True)
    key=hashlib.sha256((record['id']+record['input_sha256']).encode()).hexdigest()
    request=dict(idempotency_key=key,operation='submit',input_yaml=record['input_yaml'],
        input_sha256=record['input_sha256'],execution_sha256=sha(root/'config.json'),
        expected_predictions=50,msa_evidence=[dict(path=record['msa'],sha256=record['msa_sha256'])],
        resource_policy=dict(execution_host=host,worker_python=cfg['boltz_python'],
            numerical_profile='stock_fp32_serial',serial_producer=str(root/'producer/boltz_serial_predict.py'),
            library_path=cfg['library_path'],gpu_index=gpu,allow_any_free_gpu=False,
            gpu_cpu_affinity=cpus,minimum_gpu_memory_mib=30000,
            min_predictions_per_job=50,max_predictions_per_job=50,ccd_cache_root=str(root)))
    save(job/'request.json',request)
    env=['env','LD_LIBRARY_PATH='+cfg['library_path'],*['%s=1'%k for k in THREADS]]
    argv=env+['taskset','-c',','.join(map(str,cpus)),cfg['boltz_python'],'-u',
        str(root/'producer/boltz_backend.py'),'--state-dir',str(root),'--worker',str(job)]
    code="import subprocess,json,time\nfrom pathlib import Path\np=Path("+repr(str(job))+")\nwith (p/'worker.log').open('x') as log:q=subprocess.Popen("+repr(argv)+",stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nr=dict(pid=q.pid,host="+repr(host)+",gpu="+str(gpu)+",cpu_affinity="+repr(cpus)+",started=time.time());(p/'launch.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))"
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
        shlex.join(['/usr/bin/python3','-c',code])],capture_output=True,text=True,timeout=25)
    if result.returncode:
        save(job/'submission_uncertain.json',dict(stderr=result.stderr,stdout=result.stdout))
        raise RuntimeError('Uncertain SSH submission; inspect before retry')
    return json.loads(result.stdout)


def refold(root,cfg,records):
    if list((root/'jobs').glob('*/submission_uncertain.json')):
        raise RuntimeError('Uncertain earlier submission requires reconciliation')
    submitted={p.parent.name for p in (root/'jobs').glob('*/launch.json')}
    submitted|={p.parent.name for p in (root/'jobs').glob('*/receipt.json')}
    scored={p.stem for p in (root/'metrics').glob('*.json')}
    scored|={p.stem for p in (root/'scoring_errors').glob('*.json')}
    began=time.monotonic()
    while time.monotonic()-began<cfg['refold_timeout_seconds']:
        receipts={}
        for record in records:
            job=root/'jobs'/record['id'];p=job/'receipt.json'
            if p.exists():receipts[record['id']]=json.loads(p.read_text())
        for record in records:
            ident=record['id'];receipt=receipts.get(ident,{})
            if receipt.get('status')!='complete' or ident in scored:continue
            try:
                command=[cfg['analysis_python'],str(root/'producer/score_boltz_design_refolds.py'),
                    '--design',record['design_cif'],'--motif',record['motif_input'],
                    '--metadata',record['motif_metadata'],'--boltz-dir',str(root/'jobs'/ident/'outputs'),
                    '--output',str(root/'metrics'/(ident+'.json'))]
                result=subprocess.run(command,text=True,capture_output=True,timeout=300)
                if result.returncode:raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])
                scored_result=json.loads((root/'metrics'/(ident+'.json')).read_text())
                if scored_result.get('status')!='complete':
                    raise RuntimeError('Scoring incomplete; retained per-pose errors in metrics JSON')
                scored.add(ident)
            except Exception:
                save(root/'scoring_errors'/(ident+'.json'),dict(error=traceback.format_exc()))
                scored.add(ident)  # Explicit error is retained; no repeated busy loop.
        completed=[k for k,v in receipts.items() if v.get('status')=='complete']
        failed=[k for k,v in receipts.items() if v.get('status')=='failed']
        active=[k for k in submitted if k not in completed+failed]
        status(root,'refolding',designs=len(records),submitted=len(submitted),active=active,
            complete=len(completed),failed=failed,scored=len(list((root/'metrics').glob('*.json'))),
            scoring_errors=[p.stem for p in (root/'scoring_errors').glob('*.json')],
            samples_per_design=50,profile='stock_fp32_serial',attack_geometry_source='independent Boltz-2 refolds')
        if len(completed)+len(failed)==len(records):
            errors=[p.stem for p in (root/'scoring_errors').glob('*.json')]
            status(root,'finished_with_failures' if failed or errors or len(records)!=cfg['expected_designs'] else 'finished',
                designs=len(records),complete_refold_ensembles=len(completed),failed=failed,
                metrics_dir=str(root/'metrics'),scoring_errors=[p.stem for p in (root/'scoring_errors').glob('*.json')])
            return
        if len(failed)>=3:
            status(root,'stopped_three_prediction_failures',complete=len(completed),failed=failed,active=active)
            return
        pending=[r for r in records if r['id'] not in submitted]
        # Validate the first full 50-pose prediction before increasing concurrency.
        capacity=cfg['max_workers'] if completed else 1
        if pending and len(active)<capacity:
            result=subprocess.run([cfg['gpu_reserve'],'list','--hosts',','.join(cfg['hosts'])],
                text=True,capture_output=True,timeout=45,check=True)
            free={}
            for line in result.stdout.splitlines():
                host,gpu=line.split();free.setdefault(host,[]).append(int(gpu))
            active_slots=set()
            for ident in active:
                launched=root/'jobs'/ident/'launch.json'
                if launched.exists():
                    assignment=json.loads(launched.read_text())
                    active_slots.add((assignment['host'],assignment['gpu']))
            for host,gpus in free.items():
                node=inspect_host(host);cpus=node['free_cpus'];assigned=0
                for gpu in gpus:
                    if not pending or len(active)>=capacity:break
                    if (host,gpu) in active_slots or len(cpus)<4 or node['load']+assigned+4>node['online']:continue
                    record=pending.pop(0);slots=cpus[:4];cpus=cpus[4:];assigned+=4
                    launch(root,cfg,record,host,gpu,slots);submitted.add(record['id']);active.append(record['id'])
                    time.sleep(3)
        time.sleep(20)
    raise RuntimeError('Refolding coordinator time limit; existing workers retain ownership')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();cfg=json.loads((root/'config.json').read_text())
    for name,digest in json.loads((root/'producer_manifest.json').read_text()).items():
        if sha(root/'producer'/name)!=digest:raise RuntimeError('Frozen refolding producer changed: '+name)
    for name in ('inputs','msa','jobs','metrics','scoring_errors'):(root/name).mkdir(exist_ok=True)
    save(root/'owner.json',dict(pid=os.getpid(),host=os.uname().nodename,
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        cpu_affinity=sorted(os.sched_getaffinity(0)),started=time.time()))
    try:
        manifest,files=wait_generation(root,cfg)
        records=collect(root,cfg,manifest,files)
        records=prepare_msas_and_inputs(root,cfg,records)
        refold(root,cfg,records)
    except BaseException:
        status(root,'blocked_or_failed_preserved',error=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
