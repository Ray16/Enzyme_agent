"""Agent tools for reaction-conditioned GENzyme generation in its own conda env.

This module is stdlib-only. Model code/weights stay in the external checkout;
the worker is packaged here and executes through the shared GPU gate.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from enzyme_discovery_agent.tool_registry import tool

_ROOT = Path(__file__).resolve().parents[2]
_NFS = Path('/nfs/lambda_stor_01/homes/rzhu')
_THREAD_KEYS = ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _paths():
    default_repo = _ROOT/'external/GENzyme'
    if not default_repo.is_dir(): default_repo = _NFS/'Enzyme_agent/external/GENzyme'
    repo = Path(os.environ.get('ENZYME_AGENT_GENZYME_DIR', str(default_repo))).expanduser().resolve()
    return dict(repo=str(repo),
        python=os.environ.get('ENZYME_AGENT_GENZYME_PYTHON', str(_NFS/'miniforge3/envs/genzyme/bin/python')),
        checkpoint=os.environ.get('ENZYME_AGENT_GENZYME_CHECKPOINT', str(repo/'genzyme_ckpt/genzyme.ckpt')),
        gpu_reserve=os.environ.get('ENZYME_AGENT_GENZYME_GPU_RESERVE', str(_NFS/'bin/gpu_reserve')),
        hf_home=os.environ.get('ENZYME_AGENT_GENZYME_HF_HOME', str(_NFS/'.cache/huggingface')),
        worker=str(Path(__file__).with_name('_genzyme_worker.py')))


def _readiness():
    paths = _paths(); blocking = []
    for key in ('python', 'checkpoint', 'gpu_reserve', 'worker'):
        if not Path(paths[key]).is_file(): blocking.append(f'{key} missing: {paths[key]}')
    if not (Path(paths['repo'])/'generate.py').is_file(): blocking.append('GENzyme generate.py is missing')
    for key in ('python', 'gpu_reserve'):
        if Path(paths[key]).exists() and not os.access(paths[key], os.X_OK): blocking.append(f'{key} is not executable')
    hub = Path(paths['hf_home'])/'hub/models--EvolutionaryScale--esm3-sm-open-v1'
    required = ('esm3_sm_open_v1.pth', 'esm3_structure_encoder_v0.pth',
                'esm3_structure_decoder_v0.pth', 'esm3_function_decoder_v0.pth')
    snapshots = [p for p in (hub/'snapshots').glob('*')
                 if all((p/'data/weights'/name).is_file() for name in required)]
    if not snapshots: blocking.append('Complete ESM3 weights are not cached under hf_home')
    return dict(ready=not blocking, blocking=blocking, paths=paths, minimum_gpu_memory_GiB=24,
                esm3_snapshots=[str(p) for p in snapshots])


def _env(paths, threads):
    env = dict(os.environ)
    for k in _THREAD_KEYS: env[k] = str(threads)
    env['HF_HOME'] = paths['hf_home']; env['HF_HUB_OFFLINE'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'; env['PYTHONUNBUFFERED'] = '1'
    env['LD_LIBRARY_PATH'] = ':'.join(p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                                      if p and 'cuda' not in p.lower())
    # GPU visibility is set solely by gpu_reserve run, not by this wrapper.
    return env


def _cpu_slots(n):
    allowed = set(os.sched_getaffinity(0)); occupied = set()
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if not (p/'cmdline').read_bytes(): continue
            affinity = set(os.sched_getaffinity(int(p.name)))
            if len(affinity) < len(allowed): occupied.update(affinity)
        except OSError: continue
    available = sorted(allowed-occupied)
    if os.cpu_count() is None or max(os.getloadavg())+n > os.cpu_count() or len(available) < n:
        raise RuntimeError('Insufficient verified CPU capacity')
    return available[-n:]


def _free_gpus(paths, minimum_gib=24):
    host = socket.gethostname().split('.')[0]
    result = subprocess.run([paths['gpu_reserve'], 'list', '--hosts', host],
                            text=True, capture_output=True, timeout=30, check=True)
    free = [p[1] for line in result.stdout.splitlines()
            if len(p := line.split()) == 2 and p[0] == host and p[1].isdigit()]
    memory = subprocess.run(['nvidia-smi','--query-gpu=index,memory.total','--format=csv,noheader,nounits'],
                            text=True,capture_output=True,timeout=15,check=True)
    capacity = {parts[0].strip():float(parts[1]) for line in memory.stdout.splitlines()
                if len(parts := line.split(',')) == 2}
    return [gpu for gpu in free if capacity.get(gpu,0) >= minimum_gib*1024]


def _verify_protein(path, sequence=None):
    """Check actual backbone coordinates and residue identities without torch."""
    mapping = dict(zip(('ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE',
                        'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'),
                       'ARNDCEQGHILKMFPSTWYV'))
    residues = {}
    for line in path.read_text().splitlines():
        if not line.startswith('ATOM  '): continue
        if not all(math.isfinite(float(line[a:b])) for a,b in ((30,38),(38,46),(46,54))):
            raise ValueError('Non-finite GENzyme coordinates')
        key = (line[21],line[22:27]); name = line[12:16].strip()
        aa, atoms = residues.setdefault(key,(mapping.get(line[17:20].strip(),'?'),set()))
        atoms.add(name)
    observed = ''.join(v[0] for v in residues.values())
    if len({k[0] for k in residues}) != 1 or '?' in observed or (sequence is not None and observed != sequence):
        raise ValueError('GENzyme PDB sequence differs from FASTA')
    if not all({'N','CA','C'} <= atoms for _,atoms in residues.values()):
        raise ValueError('GENzyme PDB has an incomplete backbone')
    return observed


def _validate_options(substrate, product, length, pocket, count, steps, inpaint):
    if not isinstance(substrate, str) or not substrate.strip() or not isinstance(product, str) or not product.strip():
        raise ValueError('Both substrate_smiles and product_smiles are required')
    for name, value, low, high in [('n_protein_res', length, 64, 2000),
        ('n_pocket_res', pocket, 16, 128), ('num_designs', count, 1, 64),
        ('pocket_steps', steps, 2, 1000), ('inpaint_steps', inpaint, 2, 1000)]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{name} must be an integer between {low} and {high}')
    if pocket >= length: raise ValueError('n_pocket_res must be smaller than n_protein_res')


@tool(description='Check the GENzyme checkout, dedicated Python environment, model checkpoint, cached ESM3 weights and locally available reserved-GPU candidates. This check does not launch generation.',
      category='structure', parallel_safe=True, params={})
def check_genzyme_design_readiness() -> dict:
    info = _readiness()
    try: info['free_gpus'] = _free_gpus(info['paths']) if info['ready'] else []
    except (OSError, subprocess.SubprocessError) as exc:
        info['free_gpus'] = []; info['availability_error'] = str(exc)
    return dict(status='success' if info['ready'] else 'error',
        summary='GENzyme installation ready' if info['ready'] else 'GENzyme is not ready: '+'; '.join(info['blocking']), data=info)


@tool(description='Generate de novo enzyme candidates with GENzyme conditioned on substrate and product SMILES. Produces pockets, full-protein backbone models and sequences. Set chain length from the intended architecture. Starts one asynchronous job in the dedicated conda environment through gpu_reserve; use get_genzyme_design_status to inspect completion. Confidence is a diagnostic, not evidence of catalysis. This tool does not promise native-scaffold preservation or run cofold evaluation.',
      category='structure', parallel_safe=False, injected_params=['run_dir'], params={
          'substrate_smiles': {'type':'string','description':'Reactant molecular graph(s); disconnected participants use dot-separated SMILES.'},
          'product_smiles': {'type':'string','description':'Product molecular graph(s) describing the intended transformation.'},
          'n_protein_res': {'type':'integer','description':'Target full-protein length, chosen for the task.','default':200},
          'n_pocket_res': {'type':'integer','description':'Number of generated pocket residues.','default':64},
          'num_designs': {'type':'integer','description':'Requested candidate count; each candidate is generated independently.','default':1},
          'seed': {'type':'integer','description':'Reproducible initial random seed.','default':0},
          'pocket_steps': {'type':'integer','description':'Pocket flow sampling steps.','default':50},
          'inpaint_steps': {'type':'integer','description':'Structure inpainting steps.','default':100},
          'run_dir': {'type':'string','description':'Agent-injected durable run directory.','default':''}})
def design_enzyme_with_genzyme(substrate_smiles: str, product_smiles: str,
        n_protein_res: int = 200, n_pocket_res: int = 64, num_designs: int = 1,
        seed: int = 0, pocket_steps: int = 50, inpaint_steps: int = 100,
        run_dir: str = '') -> dict:
    try:
        _validate_options(substrate_smiles, product_smiles, n_protein_res, n_pocket_res,
                          num_designs, pocket_steps, inpaint_steps)
        if type(seed) is not int or not 0 <= seed < 2**32-num_designs*3:
            raise ValueError('seed must fit the unsigned 32-bit RNG range')
        info = _readiness()
        if not info['ready']: return dict(status='error', summary='GENzyme is not ready', data=info)
        paths = info['paths']; cpus = _cpu_slots(2); env = _env(paths, 2)
        out = (Path(run_dir) if run_dir else _ROOT/'enzyme_design_agent/campaigns')/'genzyme'/str(time.time_ns())
        out = out.resolve(); out.mkdir(parents=True)
        request = dict(repo=paths['repo'], checkpoint=paths['checkpoint'], output_dir=str(out),
            substrate_smiles=substrate_smiles, product_smiles=product_smiles,
            n_protein_res=n_protein_res, n_pocket_res=n_pocket_res, num_designs=num_designs,
            seed=seed, pocket_steps=pocket_steps, inpaint_steps=inpaint_steps,
            max_attempts=3*num_designs, inverse_fold_microbatch=1,
            ptm_filter=0.65, plddt_filter=0.65)
        req = out/'request.json'; req.write_text(json.dumps(request, indent=2)+'\n')
        affinity = ','.join(map(str, cpus))
        worker = [paths['python'], '-u', paths['worker'], '--request', str(req)]
        # Real imports and SMILES featurization happen before any GPU is claimed.
        with (out/'preflight.log').open('w') as log:
            subprocess.run(['taskset','-c',affinity,*worker,'--validate-only'],
                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
        minimum_gib = 24 if n_protein_res <= 512 else (32 if n_protein_res <= 1024 else 80)
        free = _free_gpus(paths, minimum_gib)
        if not free:
            return dict(status='error', summary=f'No free GPU with at least {minimum_gib} GiB on this host; request is prepared',
                        data={'job_dir':str(out),'request':str(req)})
        # Recheck CPU capacity immediately before submission.
        if set(cpus) != set(_cpu_slots(2)):
            raise RuntimeError('CPU availability changed during preflight; retry submission')
        argv = [paths['gpu_reserve'],'run',free[0],'--label','genzyme-design','--',
                'taskset','-c',affinity,*worker]
        with (out/'run.log').open('w') as log:
            child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(pid=child.pid, start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                      host=socket.gethostname().split('.')[0], gpu=free[0], cpu_affinity=cpus,
                      argv=argv, python=paths['python'], requested_designs=num_designs)
        (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        return dict(status='running', summary=f'GENzyme launched for {num_designs} candidate(s)',
                    data={'job_dir':str(out),'launch':launch,'status_tool':'get_genzyme_design_status'})
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return dict(status='error', summary=str(exc), data={'job_dir':str(out) if 'out' in locals() else ''})


@tool(description='Read a GENzyme job directory returned by design_enzyme_with_genzyme. Checks actual FASTA and structure outputs before reporting completion; incomplete or failed jobs are reported explicitly.',
      category='structure', parallel_safe=True,
      params={'job_dir':{'type':'string','description':'GENzyme job directory containing request.json and launch.json.'}})
def get_genzyme_design_status(job_dir: str) -> dict:
    try:
        out = Path(job_dir).expanduser().resolve()
        request = json.loads((out/'request.json').read_text())
        if (out/'failure.json').exists():
            return dict(status='error', summary='GENzyme generation failed',
                        data=json.loads((out/'failure.json').read_text()))
        if (out/'result.json').exists():
            result = json.loads((out/'result.json').read_text())
            for d in result['designs']:
                seq_path, pdb = out/d['fasta'], out/d['protein_pdb']
                seq = ''.join(s.strip() for s in seq_path.read_text().splitlines() if not s.startswith('>'))
                if seq != d['sequence'] or len(seq) != request['n_protein_res'] or not pdb.is_file() or pdb.stat().st_size < 100:
                    raise ValueError('GENzyme output verification failed')
                _verify_protein(pdb, seq)
                pocket = out/d['pocket_pdb']
                if len(_verify_protein(pocket)) != request['n_pocket_res']:
                    raise ValueError('GENzyme pocket residue count differs from the request')
            if len(result['designs']) != request['num_designs']:
                return dict(status='error', summary='GENzyme produced fewer candidates than requested', data=result)
            return dict(status='success', summary=f"GENzyme generated {len(result['designs'])} candidate(s)", data=result)
        launch = json.loads((out/'launch.json').read_text())
        if launch['host'] != socket.gethostname().split('.')[0]:
            return dict(status='unknown', summary='GENzyme outputs pending; process liveness must be checked on the worker host',data=launch)
        try:
            fields = Path(f"/proc/{launch['pid']}/stat").read_text().split()
            alive = fields[21] == launch['start_ticks'] and fields[2] != 'Z'
        except OSError: alive = False
        if not alive:
            tail = (out/'run.log').read_text(errors='replace')[-2000:]
            return dict(status='error', summary='GENzyme process exited without verified outputs',data={'log_tail':tail,'job_dir':str(out)})
        return dict(status='running', summary='GENzyme generation in progress',data={'job_dir':str(out),'launch':launch})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(status='error',summary=f'Cannot inspect GENzyme job: {exc}')
