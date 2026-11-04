"""Standalone worker; model imports run only inside the shared GPU reservation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

REPO_ID = 'nvidia/RFD4-Proteina'
REVISION = 'a06e2509c7033b010fc43614e9fd840303588502'


def _write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def _cpus():
    allowed = set(os.sched_getaffinity(0))
    occupied = set()
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if not (entry/'cmdline').read_bytes():
                continue
            affinity = set(os.sched_getaffinity(int(entry.name)))
            if len(affinity) < len(allowed):
                occupied.update(affinity)
        except OSError:
            continue
    free = sorted(allowed-occupied)
    if len(free) < 2 or max(os.getloadavg())+2 > (os.cpu_count() or 0):
        raise RuntimeError('Insufficient verified CPU capacity for two slots')
    import psutil
    if psutil.virtual_memory().available < 16*1024**3:
        raise RuntimeError('Less than 16 GiB host RAM available')
    return free[-2:]


def preflight(paths):
    blocking = []
    if sys.version_info[:2] != (3, 12):
        blocking.append('RFD4-Proteina requires Python 3.12')
    versions = {}
    for name in ('rfd4-proteina','torch','atomworks','tmol','cuequivariance-torch',
                 'cuequivariance-ops-cu12','cuequivariance-ops-torch-cu12','psutil'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            blocking.append(f'Missing dependency: {name}')
    for path in (Path(paths['repo'])/'configs/inference.yaml', Path(paths['gpu_reserve']),
                 Path(sys.executable).parent/'rfd4-generate'):
        if not path.is_file():
            blocking.append(f'Missing file: {path}')
    snapshot = Path(paths['hf_home'])/'hub/models--nvidia--RFD4-Proteina/snapshots'/REVISION
    for name in ('config.json','model.safetensors'):
        if not (snapshot/name).is_file():
            blocking.append(f'Missing cached model: {snapshot/name}')
    try:
        cpu_slots = _cpus()
    except (OSError, RuntimeError, ImportError) as exc:
        blocking.append(str(exc)); cpu_slots = []
    free = []; hardware = []
    try:
        query = subprocess.run(['nvidia-smi','--query-gpu=index,name,compute_cap,memory.total',
                                '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
        for line in query.stdout.splitlines():
            idx, name, cap, memory = [p.strip() for p in line.split(',')]
            hardware.append(dict(index=idx, name=name, compute_capability=float(cap), memory_MiB=float(memory)))
        compatible = {g['index'] for g in hardware if g['compute_capability'] >= 8 and g['memory_MiB'] >= 24000}
        if not compatible:
            blocking.append('No GPU with compute capability >=8.0 and >=24 GiB (V100 unsupported)')
        gate = subprocess.run([paths['gpu_reserve'],'list','--hosts',socket.gethostname().split('.')[0]],
                              check=True, capture_output=True, text=True, timeout=60)
        free = [parts[1] for line in gate.stdout.splitlines()
                if len(parts := line.split()) == 2 and parts[0] == socket.gethostname().split('.')[0] and parts[1] in compatible]
        with tempfile.TemporaryFile() as handle:
            handle.write(b'rfd4 preflight')
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        blocking.append(str(exc))
    return dict(ready=not blocking, blocking=blocking, versions=versions, free_gpus=free,
                hardware=hardware, cpu_slots=cpu_slots, host=socket.gethostname(), paths=paths,
                model_repo=REPO_ID, model_revision=REVISION)


def _input(request):
    """Snapshot the spec and absolutize references without changing conditions."""
    source = Path(request['input_path'])
    out = Path(request['output_dir'])
    if source.suffix not in ('.yaml','.yml','.json'):
        import shutil
        target = out/source.name
        shutil.copyfile(source, target)
        return target, 1
    if source.suffix == '.json':
        data = json.loads(source.read_text())
    else:
        import yaml
        data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError('Condition spec must be a nonempty example-name mapping')
    for spec in data.values():
        if not isinstance(spec, dict):
            raise ValueError('Every condition-spec example must be a mapping')
        if 'input' in spec:
            values = spec['input'] if isinstance(spec['input'], list) else [spec['input']]
            resolved = []
            for value in values:
                path = Path(value).expanduser()
                path = (source.parent/path).resolve() if not path.is_absolute() else path
                if not path.is_file():
                    raise ValueError(f'Missing structure input: {path}')
                resolved.append(str(path))
            spec['input'] = resolved if isinstance(spec['input'], list) else resolved[0]
    target = out/'condition_spec.json'
    target.write_text(json.dumps(data, indent=2)+'\n')
    return target, len(data)


def _verify(out, expected):
    files = sorted((out/'inference_samples').rglob('*_generated.cif'))
    if len(files) != expected:
        raise ValueError(f'Expected {expected} CIF designs, found {len(files)}')
    import numpy as np
    from biotite.structure.io.pdbx import CIFFile, get_structure
    from biotite.structure import filter_amino_acids, get_residue_count, residue_iter
    designs = []
    for path in files:
        atoms = get_structure(CIFFile.read(path), model=1)
        protein = atoms[filter_amino_acids(atoms)]
        if (not len(protein) or not np.isfinite(atoms.coord).all()
                or not all({'N','CA','C'} <= set(residue.atom_name) for residue in residue_iter(protein))):
            raise ValueError(f'Invalid protein coordinates: {path}')
        designs.append(dict(cif=str(path), atoms=len(atoms), protein_residues=get_residue_count(protein),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return designs


def generate(req_path):
    request = json.loads(req_path.read_text()); out = Path(request['output_dir'])
    paths = request['paths']
    os.chdir(paths['repo'])
    os.environ['HF_HOME'] = paths['hf_home']
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['DATA_PATH'] = str(Path(paths['repo'])/'data_path')
    # Exercise real CUDA imports/checkpoint availability while the gate owns the GPU.
    subprocess.run([sys.executable, 'skills/rfd4-proteina-setup/scripts/preflight.py'], check=True)
    target, examples = _input(request)
    argv = [str(Path(sys.executable).parent/'rfd4-generate'),
            'inference_experiment=quickstart', f'inputs={json.dumps(str(target))}',
            f'out_dir={json.dumps(str(out))}', f"num_replicates={request['num_replicates']}",
            f"seed={request['seed']}", f"recycle_strategy=medium/{request['recycle_schedule']}",
            'batch_size=1', 'num_workers=0', 'prevalidate_inputs=true',
            f'hf_ckpt_path={REPO_ID}', f'hf_ckpt_revision={REVISION}',
            'compress_outputs=false', f'hydra.run.dir={json.dumps(str(out/"hydra"))}']
    manifest = dict(request=request, argv=argv, cpu_affinity=sorted(os.sched_getaffinity(0)),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    source_sha256=hashlib.sha256(Path(request['input_path']).read_bytes()).hexdigest(),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    model_revision=REVISION, expected_designs=examples*request['num_replicates'])
    _write(out/'manifest.json', manifest)
    subprocess.run(argv, check=True)
    designs = _verify(out, manifest['expected_designs'])
    _write(out/'result.json', dict(status='success', summary=f'Generated and checked {len(designs)} RFD4 designs',
                                 designs=designs, scoring='skipped', refolding='not run', manifest=str(out/'manifest.json')))


def run(req_path):
    request = json.loads(req_path.read_text()); out = Path(request['output_dir'])
    state = dict(status='running', summary='RFD4 worker preflight', host=socket.gethostname().split('.')[0],
                 pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21])
    _write(out/'state.json', state)
    try:
        info = preflight(request['paths'])
        _write(out/'preflight.json', info)
        if not info['ready'] or not info['free_gpus']:
            raise RuntimeError('; '.join(info['blocking']) or 'No compatible free GPU')
        cpus = _cpus()
        gpu = info['free_gpus'][0]
        state.update(summary='RFD4 reserved generation starting', gpu=gpu, cpu_affinity=cpus)
        _write(out/'state.json', state)
        argv = [request['paths']['gpu_reserve'], 'run', gpu, '--label', 'rfd4-design', '--',
                'taskset', '-c', ','.join(map(str,cpus)), sys.executable, '-u', __file__, '--generate', str(req_path)]
        subprocess.run(argv, check=True)
        if not (out/'result.json').is_file():
            raise RuntimeError('Generation exited without verified results')
    except Exception as exc:
        _write(out/'failure.json', dict(status='error', summary=str(exc), job_dir=str(out)))
        raise


def status(req_path):
    request = json.loads(req_path.read_text()); out = Path(request['output_dir'])
    if (out/'failure.json').is_file():
        return json.loads((out/'failure.json').read_text())
    if (out/'result.json').is_file():
        result = json.loads((out/'result.json').read_text())
        manifest = json.loads((out/'manifest.json').read_text())
        verified = _verify(out, manifest['expected_designs'])
        if verified != result['designs']:
            raise ValueError('Output files changed after generation')
        return result
    if not (out/'state.json').is_file():
        return dict(status='pending', summary='Remote worker startup pending')
    state = json.loads((out/'state.json').read_text())
    try:
        fields = Path(f"/proc/{state['pid']}/stat").read_text().split()
        alive = fields[21] == state['start_ticks'] and fields[2] != 'Z'
    except OSError:
        alive = False
    if not alive:
        return dict(status='error', summary='RFD4 worker exited without verified results', state=state,
                    log_tail=(out/'run.log').read_text(errors='replace')[-3000:])
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('preflight','run','generate','status'):
        group.add_argument('--'+name)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(json.loads(args.preflight))))
    elif args.run:
        run(Path(args.run))
    elif args.generate:
        generate(Path(args.generate))
    else:
        try:
            result = status(Path(args.status))
        except Exception as exc:
            result = dict(status='error', summary=f'Output verification failed: {exc}')
        print(json.dumps(result))
