"""Attach the independent structural agent to a growing ranked discovery run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from .production import live, owner, read, require_capacity
from .state_store import atomic_write_json, state_lock


THREADS=('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
         'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
TARGET_POLICY={
    'description':'Investigate the requested target substrate combination, including non-native partners when supplied, from the current discovery target contract.',
    'decision_policy':'Gather and weigh primary literature, curated and inferred reaction records, sequence/domain evidence and experimental or predicted structures. Distinguish exact-protein evidence, homolog transfers, contradictions and unknowns. Missing native reaction annotation alone neither justifies compute nor excludes an informative conditional hypothesis. Record why the proposed cofold resolves an important uncertainty and why further source retrieval would not answer it more cheaply.',
    'molecular_setup':'The modeling agent selects the relevant reaction step, intact target partners, protonation, construct, assembly and required cofactors from evidence. Account for every target partner; if a partial-step model omits one, explain the limited conclusion and remaining partner-specific test. Never substitute a native-only model for the requested non-native test. Avoid arbitrary combinations of states or cofactors. A native control is optional when it distinguishes a model/setup failure from a target-specific difference.',
    'quality':'Boltz-2 exact mode, sequence-specific batched MSAs, 50 independent predictions per justified setup, no imposed reaction geometry. Reuse only after sequence and complete setup review. Preserve failed poses and unsupported setups.',
    'analysis':'Use the existing frozen inverse-distance-weighted attack-angle definition where chemically applicable, reporting harmonic distance separately. Compare only supported steps and mechanisms. Do not refit the metric or interpret a favorable pose as enzyme activity.'}


def _available_cpus():
    import psutil
    occupied=set()
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid==os.getpid():continue
        try:
            affinity=set(process.cpu_affinity())
            if process.info['cmdline'] and len(affinity)<=8:occupied|=affinity
        except psutil.NoSuchProcess:continue
        except psutil.AccessDenied:
            affinity=set(os.sched_getaffinity(process.pid))
            if len(affinity)<=8:occupied|=affinity
    return sorted(set(os.sched_getaffinity(0))-occupied)


def _snapshot(folder):
    project=Path(__file__).resolve().parent.parent
    hashes={}
    for package in ('enzyme_discovery_agent','structural_modeling_agent','campaign_catalog','enzyme_design_agent'):
        source=project/package
        for directory,dirs,files in os.walk(source):
            dirs[:]=[d for d in dirs if d not in {'runs','tests','eval','archive','analysis','campaigns','cofold','integrated_campaigns','figures','__pycache__','knowledge'}]
            for name in files:
                path=Path(directory)/name
                if path.suffix not in {'.py','.json','.yaml','.yml','.txt'}:continue
                relative=path.relative_to(project);target=folder/'source'/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
                hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    # Declared static modeling contracts; archives and historical outputs stay out.
    for name in ('attack_angle_metric.json','mlmm_engine_options.json','family_mechanism_fallback.json'):
        asset=project/'structural_modeling_agent/knowledge'/name
        target=folder/'source'/asset.relative_to(project)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(asset,target)
        hashes[str(asset.relative_to(project))]=hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def attach(run_dir, *, discovery_owner=None, model=None, runtime_seconds=43200,
           per_candidate_tokens=100000, modeling_token_budget=None, msa_cache=None):
    """Start once; repeated attachment preserves live ownership and the existing queue."""
    if type(per_candidate_tokens) is not int or per_candidate_tokens<1 or runtime_seconds<=0:
        raise ValueError('Require positive per-candidate research and elapsed-time allowances')
    if modeling_token_budget is not None and (type(modeling_token_budget) is not int or modeling_token_budget<1):
        raise ValueError('Require a positive total allowance or per-candidate budgeting')
    run=Path(run_dir).resolve();folder=run/'rolling_modeling';folder.mkdir(exist_ok=True)
    with state_lock(folder/'launch'):
        previous=read(folder/'launch_manifest.json')
        if previous and live(previous) is True:return previous
        if previous:
            raise RuntimeError('A previous coordinator stopped; reconcile its saved queue before restarting it')
        cpus=_available_cpus()
        if len(cpus)<3:raise RuntimeError('Three disjoint CPU slots are required for rolling preparation and setup')
        coordinator_cpus=[cpus[-3]];modeling_cpus=cpus[-2:]
        require_capacity(coordinator_cpus+modeling_cpus)
        project=Path(__file__).resolve().parent.parent
        pool=folder/'prediction_pool';pool.mkdir(exist_ok=True)
        atomic_write_json(pool/'sources.json',{'campaign_id':run.name,'project_root':str(project),
            'discovery_sources':[],'modeling_sources':[],'structural_agent_sources':[]})
        from campaign_catalog.catalog import build
        build(pool)
        cache=Path(msa_cache) if msa_cache else run.parent/'.sequence_msa_cache'
        settings={'project_root':str(project),'model':model or 'claude-code:claude-sonnet-5-5',
            'cofold_pool':str(pool),'cofold_pool_target_id':run.name,'msa_cache':str(cache),
            'prediction_cache_pools':[str(path.parent.parent) for path in
                run.parent.glob('*/rolling_modeling/prediction_pool/catalog/current.json')
                if path.parent.parent.resolve()!=pool.resolve()],
            'max_modeling_candidates':None,'protected_unannotated_slots':0,'modeling_selection':'ranked',
            'modeling_token_budget':modeling_token_budget,'modeling_item_token_budget':per_candidate_tokens,
            'modeling_max_turns':20,'setup_turn_limit':8,'modeling_max_attempts':2,
            'runtime_budget_seconds':runtime_seconds,'modeling_runtime_seconds':21600,
            'modeling_cpu_affinity':modeling_cpus,'coordinator_cpu_affinity':coordinator_cpus,
            'modeling_target':TARGET_POLICY,'expected_predictions':50,
            'feedback_instruction':'Read returned structural evidence, distinguish setup failure from target incompatibility, and revise source-grounded priorities. Structural confidence is not activity. The structural agent owns cofold submissions; do not duplicate them.'}
        config_path=folder/'campaign_input.json';atomic_write_json(config_path,settings)
        atomic_write_json(folder/'campaign_state.json',{'run_dir':str(run)})
        atomic_write_json(folder/'discovery_owner.json',discovery_owner or owner())
        hashes=_snapshot(folder)
        worker=folder/'worker.py'
        worker.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent/'source'))\nfrom enzyme_discovery_agent.production import main\nif __name__=='__main__':main()\n")
        provider=folder/'provider.py'
        source=(folder/'source/structural_modeling_agent/lambda_boltz_backend.py').read_text()
        provider.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent/'source'))\n"+source)
        atomic_write_json(folder/'backend.json',{'provider':'lambda13-boltz2-exact','engine':'boltz2',
            'argv':[sys.executable,str(provider),'--state-dir',str(folder/'provider'),'--pool',str(pool)],
            'timeout_seconds':60,'resource_policy':{'reservation_managed':True,'max_jobs':1,
                'min_predictions_per_job':50,'max_predictions_per_job':50,'max_candidate_ensembles':None}})
        env=os.environ.copy();env.update({key:'1' for key in THREADS});env['PYTHONUNBUFFERED']='1'
        argv=['taskset','-c',','.join(map(str,coordinator_cpus)),sys.executable,'-u',str(worker),
            '--config',str(config_path),'--role','coordinator']
        manifest={'host':socket.gethostname(),'run_dir':str(run),'source_hashes':hashes,
            'argv':argv,'cpu_affinity':coordinator_cpus,'modeling_cpu_affinity':modeling_cpus,
            'gpu_policy':'One reserved exact-mode ensemble at a time; the backend checks actual free GPUs and dependencies before claiming.',
            'scope':'Every new published rank enters agent-led scientific setup review; GPU work requires justified inputs. Per-candidate research is bounded.'}
        with (folder/'coordinator.log').open('ab') as log:
            process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=env,start_new_session=True,cwd=project)
        manifest.update(pid=process.pid,process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        atomic_write_json(folder/'launch_manifest.json',manifest)
        return manifest
