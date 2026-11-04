"""Lightweight agent interface to the isolated RFD4-Proteina runtime on NFS."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

from enzyme_discovery_agent.tool_registry import tool

_NFS = Path('/nfs/lambda_stor_01/homes/rzhu')
_THREAD_KEYS = ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _paths():
    return dict(
        repo=os.environ.get('ENZYME_AGENT_RFD4_DIR', str(_NFS/'Tool/RFD4-Proteina')),
        python=os.environ.get('ENZYME_AGENT_RFD4_PYTHON', str(_NFS/'Tool/envs/rfd4-proteina/bin/python')),
        host=os.environ.get('ENZYME_AGENT_RFD4_HOST', 'lambda13'),
        hf_home=os.environ.get('ENZYME_AGENT_RFD4_HF_HOME', str(_NFS/'Tool/hf_cache')),
        gpu_reserve=os.environ.get('ENZYME_AGENT_RFD4_GPU_RESERVE', str(_NFS/'bin/gpu_reserve')),
        worker=str(Path(__file__).with_name('_rfd4_worker.py')))


def _command(paths, *args):
    host = paths['host']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host):
        raise ValueError('RFD4 host must be a hostname')
    env_args = [f'{key}=2' for key in _THREAD_KEYS]
    env_args += [f"HF_HOME={paths['hf_home']}", 'PYTHONUNBUFFERED=1',
                 'TOKENIZERS_PARALLELISM=false', 'WANDB_MODE=disabled']
    argv = ['env', *env_args, paths['python'], '-u', paths['worker'], *args]
    if host in (socket.gethostname(), socket.gethostname().split('.')[0], 'localhost'):
        return argv
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(argv)]


def _preflight(paths):
    result = subprocess.run(_command(paths, '--preflight', json.dumps(paths)),
                            capture_output=True, text=True, timeout=90, check=True)
    return json.loads(result.stdout)


@tool(description='Check the RFD4-Proteina isolated environment, pinned checkpoint cache, compatible GPU hardware and resource availability on the configured host. Does not reserve a GPU or generate designs.',
      category='structure', parallel_safe=True, params={})
def check_rfd4_design_readiness() -> dict:
    try:
        data = _preflight(_paths())
        return dict(status='success' if data['ready'] else 'error',
                    summary='RFD4-Proteina ready' if data['ready'] else '; '.join(data['blocking']), data=data)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(status='error', summary=f'RFD4 preflight failed: {exc}')


@tool(description='Generate sequence and structure designs with RFD4-Proteina from one annotated CIF or one YAML/JSON condition-spec file. Supports motif scaffolding and ligand conditioning encoded in the input. Preserves supplied conditions. Launches one asynchronous job on the configured H100 host through gpu_reserve, with pinned checkpoint, CPU limits, batch size one, eager inference and scoring disabled. Select the recycling schedule for the task. Use get_rfd4_design_status to verify outputs; independent refolding and catalytic assessment follow generation.',
      category='structure', parallel_safe=False, injected_params=['run_dir'], params={
          'input_path': {'type':'string','description':'Absolute shared-filesystem path to an annotated .cif/.cif.zst/.cif.gz or YAML/JSON condition spec. Relative structure references in specs resolve from the spec directory.'},
          'num_replicates': {'type':'integer','description':'Designs per input example, 1–64.','default':1},
          'seed': {'type':'integer','description':'Reproducible RNG seed.','default':0},
          'recycle_schedule': {'type':'string','description':'Task schedule: midpeak for enzymes/small molecules, front_loaded for protein/DNA binders. All use the upstream medium budget.','enum':['midpeak','front_loaded','flat'],'default':'midpeak'},
          'run_dir': {'type':'string','description':'Agent-injected durable run directory.','default':''}})
def design_enzyme_with_rfd4(input_path: str, num_replicates: int = 1, seed: int = 0,
                          recycle_schedule: str = 'midpeak', run_dir: str = '') -> dict:
    try:
        source = Path(input_path).expanduser().resolve()
        if not source.is_file() or not any(str(source).endswith(s) for s in ('.cif','.cif.zst','.cif.gz','.yaml','.yml','.json')):
            raise ValueError('Provide an existing annotated CIF or YAML/JSON spec file')
        if type(num_replicates) is not int or not 1 <= num_replicates <= 64:
            raise ValueError('num_replicates must be an integer from 1 to 64')
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError('seed must be an integer from 0 to 2**31-1')
        if recycle_schedule not in ('midpeak','front_loaded','flat'):
            raise ValueError('Invalid recycle_schedule')
        paths = _paths()
        info = _preflight(paths)
        if not info['ready']:
            return dict(status='error', summary='RFD4 is not ready', data=info)
        if not info['free_gpus']:
            return dict(status='error', summary='No compatible GPU available on the configured host', data=info)
        base = Path(run_dir).expanduser().resolve() if run_dir else Path(__file__).resolve().parents[1]/'runs'
        out = base/'rfd4'/str(time.time_ns())
        out.mkdir(parents=True)
        request = dict(paths=paths, input_path=str(source), num_replicates=num_replicates,
                       seed=seed, recycle_schedule=recycle_schedule, output_dir=str(out))
        req = out/'request.json'
        req.write_text(json.dumps(request, indent=2)+'\n')
        argv = _command(paths, '--run', str(req))
        with (out/'run.log').open('w') as log:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(pid=proc.pid, start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21],
                      launcher_host=socket.gethostname().split('.')[0], worker_host=paths['host'], argv=argv)
        (out/'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
        return dict(status='running', summary='RFD4 job submitted; resource claim and generation pending',
                    data=dict(job_dir=str(out), status_tool='get_rfd4_design_status'))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(status='error', summary=str(exc))


@tool(description='Inspect an RFD4-Proteina job. Completion requires the expected number of parseable structures with finite coordinates and a protein backbone; reports generation separately from skipped scoring/refolding.',
      category='structure', parallel_safe=True,
      params={'job_dir':{'type':'string','description':'RFD4 job directory returned at submission.'}})
def get_rfd4_design_status(job_dir: str) -> dict:
    try:
        out = Path(job_dir).expanduser().resolve()
        request = json.loads((out/'request.json').read_text())
        result = subprocess.run(_command(request['paths'], '--status', str(out/'request.json')),
                                capture_output=True, text=True, timeout=60, check=True)
        data = json.loads(result.stdout)
        if data['status'] == 'pending':
            launch = json.loads((out/'launch.json').read_text())
            if launch['launcher_host'] == socket.gethostname().split('.')[0]:
                try:
                    fields = Path(f"/proc/{launch['pid']}/stat").read_text().split()
                    alive = fields[21] == launch['start_ticks'] and fields[2] != 'Z'
                except OSError:
                    alive = False
                if not alive:
                    data = dict(status='error', summary='Launcher exited before worker startup',
                                log_tail=(out/'run.log').read_text(errors='replace')[-3000:])
        return dict(status=data['status'], summary=data.get('summary', 'RFD4 '+data['status']), data=data)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        return dict(status='error', summary=f'Cannot inspect RFD4 job: {exc}')
