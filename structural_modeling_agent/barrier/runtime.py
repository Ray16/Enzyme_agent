"""Typed calculation entry point executed only by a reservation-managed worker."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_plan(path,expected_sha256=None):
    path=Path(path);plan=json.loads(path.read_text())
    if expected_sha256 is not None and digest(path)!=expected_sha256:
        raise ValueError('Calculation plan changed after submission')
    for name,h in plan['input_sha256'].items():
        if digest(name)!=h:raise ValueError(f'Calculation input changed: {name}')
    for name,h in plan['producer_sha256'].items():
        if digest(Path(plan['producer_directory'])/name)!=h:raise ValueError('Frozen numerical producer changed')
    return plan


def preflight(path,minimum_gpu_memory_GiB,expected_sha256=None,minimum_host_memory_GiB=0):
    """Dependency/device checks before a task claim; no model inference."""
    plan=checked_plan(path,expected_sha256);started=time.monotonic();errors=[];versions={};device=None;checkpoint=None
    memory={line.split(':')[0]:line.split(':')[1].strip().split()[0] for line in Path('/proc/meminfo').read_text().splitlines()}
    available=int(memory['MemAvailable'])*1024
    if available<minimum_host_memory_GiB*1024**3:
        report=dict(passed=False,errors=['Insufficient available host memory before a task claim'],available_host_bytes=available)
        Path(plan['work_dir'],'preflight.json').write_text(json.dumps(report,indent=2))
        raise RuntimeError(report['errors'][0])
    modules=['numpy','scipy','ase','parmed','openmm','torch']
    modules+=['pyscf','gpu4pyscf'] if plan['method']=='dft' else ['fairchem.core']
    for module in modules:
        try:
            imported=importlib.import_module(module)
            versions[module]=getattr(imported,'__version__',None)
        except Exception as exc:errors.append(f'{module}: {type(exc).__name__}: {exc}')
    try:
        import torch
        if not torch.cuda.is_available():raise ValueError('Reserved CUDA device unavailable')
        properties=torch.cuda.get_device_properties(0)
        device=dict(name=properties.name,total_memory_bytes=properties.total_memory)
        if properties.total_memory/1024**3<minimum_gpu_memory_GiB:raise ValueError('Reserved GPU does not meet declared memory requirement')
        if plan['method']=='dft':
            import openmm
            openmm.Platform.getPlatformByName('CUDA')
        else:
            from fairchem.core import pretrained_mlip
            resolved=pretrained_mlip.resolve_checkpoint(plan['settings']['model'])
            checkpoint=dict(path=str(Path(resolved.path).resolve()),sha256=digest(resolved.path),
                reference_tables_sha256=hashlib.sha256(json.dumps(dict(atom_refs=resolved.atom_refs,
                    form_elem_refs=resolved.form_elem_refs),sort_keys=True,allow_nan=False).encode()).hexdigest())
        if plan['method']=='uma-xtb':
            import os
            command=os.environ.get('BARRIER_XTB_COMMAND','/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb')
            if not Path(command).is_file():raise ValueError('Explicit xTB executable unavailable')
    except Exception as exc:errors.append(str(exc))
    report=dict(passed=not errors,errors=errors,versions=versions,device=device,checkpoint=checkpoint,
                available_host_bytes=available,minimum_host_memory_GiB=minimum_host_memory_GiB,wall_s=time.monotonic()-started,
                scope='Imports and reserved-device capability; no enzyme accuracy validation')
    Path(plan['work_dir'],'preflight.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if errors:raise RuntimeError('Barrier runtime preflight failed: '+'; '.join(errors))
    return report


def execution_metrics(plan,numerical,wall_s):
    """Report measured scope explicitly; unavailable timings/memory stay null."""
    completed=len(numerical.get('images',numerical.get('scan_curve',[])))
    preflight=Path(plan['work_dir'])/'preflight.json'
    profile=json.loads(preflight.read_text()) if preflight.exists() else {}
    peak_allocated=peak_reserved=None
    torch=sys.modules.get('torch')
    if plan['method'] in {'uma','uma-xtb'} and torch is not None and torch.cuda.is_initialized():
        peak_allocated=int(torch.cuda.max_memory_allocated());peak_reserved=int(torch.cuda.max_memory_reserved())
    keys=('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OPENMM_CPU_THREADS')
    return dict(protein_lengths=[plan.get('protein_length')],requested_samples=plan['settings']['images'],
        completed_samples=completed,sample_unit='constrained path image, not an independent ensemble sample',batch_size=1,
        accelerator=profile.get('device'),cpu_affinity=sorted(os.sched_getaffinity(0)),
        native_thread_limits={key:os.environ.get(key) for key in keys},
        numerical_profile=('embedded DFT/MM; stated basis/functional and backend grid/MM profile' if plan['method']=='dft' else
                           'explicit_ieee_fp32_general_no_compile'+('+GFN2 embedding response' if plan['method']=='uma-xtb' else '')),
        peak_allocated_device_bytes=peak_allocated,peak_reserved_device_bytes=peak_reserved,
        dependency_preflight_wall_s=profile.get('wall_s'),startup_import_overhead_s=None,
        warmup_to_first_evaluation_s=None,scheduler_wait_s=None,calculation_entrypoint_wall_s=wall_s,
        total_process_wall_s=None,average_wall_s_per_completed_image=wall_s/completed if completed else None,
        timing_definitions='Entrypoint wall includes imports, setup and optimizations; image average is throughput, not per-energy latency. Interpreter startup and scheduler wait are unavailable, not zero.')


def calculate(path,expected_sha256=None):
    started=time.monotonic();plan=checked_plan(path,expected_sha256);root=Path(plan['work_dir']);settings=plan['settings']
    if Path(plan['result_path']).exists():raise FileExistsError('Preserve an existing result; inspect it or declare a new protocol version')
    preflight=root/'preflight.json'
    preflight_data=json.loads(preflight.read_text()) if preflight.exists() else {}
    checkpoint=preflight_data.get('checkpoint')
    if checkpoint and digest(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Model checkpoint changed after preflight')
    mode=plan['mode'];prepared=Path(plan['prepared_directory']);files=[]
    if mode=='cluster':
        from .scan import scan_barrier
        work=root/'cluster';work.mkdir(exist_ok=False)
        for name in plan['input_sha256']:
            source=Path(name)
            if source.parent==prepared and source.name.startswith('cluster_'):
                shutil.copy2(source,work/source.name)
        meta=list(work.glob('cluster_*.meta.json'))
        if len(meta)!=1:raise ValueError('Expected one immutable cluster input')
        # Explicit generic settings share the optimizer and wall controls with
        # the embedded path; no fixed hidden scan size/budget is substituted.
        numerical=scan_barrier(meta[0],model=settings['model'],concerted=settings['route']=='concerted',
            images=settings['images'],max_steps=settings['max_steps'],wall_seconds=settings['wall_seconds'])
        converged=bool(numerical.get('all_optimizations_converged') and numerical.get('mechanism_gate',{}).get('ok')
                       and numerical['i_reactant']<numerical['i_ts']<numerical['i_product']
                       and len(numerical.get('scan_curve',[]))==settings['images'])
        files=list(work.glob('*'))
        gap=numerical.get('dE_act_kcal') if converged else None
    else:
        from .path import explore
        from .stationary import summarize
        numerical=explore(prepared,root/'path',engine=plan['method'],route=settings['route'],
            images=settings['images'],max_steps=settings['max_steps'],wall_seconds=settings['wall_seconds'],
            basis=settings['basis'],xc=settings['xc'],ecp=settings.get('ecp'),model=settings['model'])
        summary=summarize(root/'path/result.json',root/'chemical_summary.json')
        converged=bool(numerical['all_optimizations_converged'] and summary.get('has_interior_maximum')
                       and len(numerical['images'])==settings['images'])
        gap=summary.get('constrained_path_gap_kcal') if converged else None
        files=[root/'path/result.json',root/'chemical_summary.json']
        files += [Path(row['positions']) for row in numerical['images']]
    checked_plan(path,expected_sha256)
    if checkpoint and digest(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Model checkpoint changed during calculation')
    if preflight.exists():files.append(preflight)
    wall_s=time.monotonic()-started
    result=dict(status='calculation_complete',calculation_status='converged_profile' if converged else 'unresolved_profile',
        plan_sha256=digest(path),mode=mode,method=plan['method'],quantity=plan['quantity'],
        dE_act_kcal=gap,output_sha256={str(p.resolve()):digest(p) for p in files if p.is_file()},
        wall_s=wall_s,execution_metrics=execution_metrics(plan,numerical,wall_s),settings=settings,transition_state_certified=False,
        validated_activation_free_energy=False,
        limitations=['Constrained path maximum; saddle index/connectivity are not certified',
                     'No ensemble activation free energy or kinetic prediction',
                     'Embedding, QM-region, basis and reference-state sensitivity require separate checks'])
    Path(plan['result_path']).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('plan')
    parser.add_argument('--minimum-gpu-memory-GiB',type=float,default=0)
    parser.add_argument('--minimum-host-memory-GiB',type=float,default=0)
    parser.add_argument('--expected-plan-sha256',required=True)
    args=parser.parse_args(argv)
    if args.preflight and not all(0<value<float('inf') for value in (args.minimum_host_memory_GiB,args.minimum_gpu_memory_GiB)):
        raise ValueError('Explicit positive GPU and host-memory requirements are required for worker preflight')
    result=(preflight(args.plan,args.minimum_gpu_memory_GiB,args.expected_plan_sha256,args.minimum_host_memory_GiB) if args.preflight else
            calculate(args.plan,args.expected_plan_sha256))
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
