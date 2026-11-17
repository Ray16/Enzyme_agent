"""One bounded, claimed AS9 pose job: matched preparation then reserved GPU MD."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'structural_modeling_agent/runs/as9_step1_multipose_md_20261117'
AMBER=Path('/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2')
MDPY='/nfs/lambda_stor_01/homes/rzhu/Tool/envs/processivity-md-20261106/bin/python'
GATE='/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve'
SPEC=ROOT/'structural_modeling_agent/barrier/specs/adenylation_adipate_cofold.json'
job=json.loads((RUN/'manifest.json').read_text())['jobs'][int(sys.argv[1])-1]
work=RUN/job['label'];work.mkdir(exist_ok=True)
claim=work/'claim';claim.mkdir()  # Never reclaim an existing owner automatically.
os.chdir(ROOT)
env=os.environ.copy()
for name in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OPENMM_CPU_THREADS']:env[name]='1'
env.update(AMBERHOME=str(AMBER),PATH=str(AMBER/'bin')+':'+env['PATH'],BARRIER_COFACTOR_ASSET=str(ROOT/'structural_modeling_agent/runs/cofactor_assets_atp_1264'))
owner=dict(host=os.uname().nodename,pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],cpus=sorted(os.sched_getaffinity(0)),job=job,thread_caps={k:v for k,v in env.items() if k.endswith('THREADS') or k=='OMP_THREAD_LIMIT'})
(claim/'owner.json').write_text(json.dumps(owner,indent=2))
(work/'owner.json').write_text(json.dumps(owner,indent=2))
proc=None

def status(state,**extra):
    tmp=work/'status.tmp';tmp.write_text(json.dumps(dict(state=state,time=time.time(),**extra),indent=2));tmp.replace(work/'status.json')
    print(state,extra,flush=True)

def stop_child():
    if proc and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def interrupted(signum,frame):
    stop_child();raise SystemExit(128+signum)
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)

def run(argv,log,monitor=False):
    global proc
    (work/(log+'.argv.json')).write_text(json.dumps(argv,indent=2))
    with (work/log).open('a') as out:
        proc=subprocess.Popen(argv,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        uuid=None
        if monitor:
            uuid=subprocess.check_output(['nvidia-smi','-i',str(job['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        while proc.poll() is None:
            time.sleep(5)
            if monitor:
                text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
                for line in text.splitlines():
                    gpu,pid=[x.strip() for x in line.split(',')]
                    if gpu!=uuid:continue
                    try:own=os.getpgid(int(pid))==proc.pid
                    except ProcessLookupError:continue
                    if not own:
                        stop_child();raise RuntimeError('GPU contention: stopped our job; preserved partial outputs')
        if proc.returncode:raise RuntimeError(f'{log} exited {proc.returncode}; no automatic destructive retry')

try:
    assert owner['host']==job['host'] and owner['cpus']==job['cpus']
    assert hashlib.sha256(Path(job['source']).read_bytes()).hexdigest()==job['source_sha256']
    if max(os.getloadavg())+2>os.cpu_count():raise RuntimeError('Insufficient CPU capacity')
    prep=work/'prep'; md=work/'md_hmr_seed17'
    if not (prep/'preparation.json').exists():
        if prep.exists():raise RuntimeError('Partial prep exists; preserve for review')
        status('preparing')
        run([str(AMBER/'bin/python'),'-m','enzyme_barrier.prepare','--cofold',job['source'],'--out',str(prep),'--spec',str(SPEC),'--polarizabilities',str(AMBER/'dat/leap/parm/lj_1264_pol.dat'),'--overrides-json',str(ROOT/'runs/barrier_maturation_20261110/polarizability_overrides_native.json'),'--ph','8.0'],'prep.log')
    review=json.loads((prep/'preparation.json').read_text())
    if not review['validation']['physically_valid']:raise RuntimeError('Physical validation failed; MD not launched')
    if md.exists():raise RuntimeError('Existing MD outputs preserved; explicit resume review required')
    if max(os.getloadavg())+2>os.cpu_count():raise RuntimeError('Insufficient CPU capacity before MD')
    status('submitting_md',preparation_status=review['status'],gpu=job['gpu'])
    command=[GATE,'run',str(job['gpu']),'--label','AS9_matched_'+job['label'],'--',MDPY,'-u','-m','enzyme_barrier.md','--prmtop',str(prep/'complex_1264.prmtop'),'--coords',str(prep/'complex.inpcrd'),'--spec',str(SPEC),'--out',str(md),'--seed','17','--ns','0.5','--equil-ns','0.5']
    run(command,'md.log',monitor=True)
    frames=[json.loads(x) for x in (md/'geometry.jsonl').read_text().splitlines()]
    assert len(frames)==50 and abs(frames[-1]['time_ps']-500)<.01
    assert all((md/f).exists() for f in ['result.json','production.dcd','final.chk'])
    status('complete_requires_scientific_review',frames=len(frames))
except BaseException as exc:
    stop_child();status('stopped',error=str(exc));raise
finally:
    (claim/'owner.json').unlink();claim.rmdir()
