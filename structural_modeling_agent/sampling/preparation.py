"""Agent-owned preparation studies. No chemical defaults or inferred successes.

The ledger binds each alternative to one setup and exact artifacts. Numerical
operators supply reports; the decision agent evaluates their scientific meaning.
"""
from pathlib import Path
import hashlib
import json
import math

from ..validation import file_digest, fingerprint
from .setup import assess

STAGES = ('chemical_state', 'parameters', 'solvation', 'minimization', 'equilibration', 'pilot_review')


def contract():
    return dict(schema_version='preparation-study-1.0',
        required=['setup_id', 'question', 'variants'],
        variant_required=['id', 'rationale', 'setup_sha256', 'stages'],
        stages=list(STAGES),
        stage_required=['status', 'interpretation', 'artifacts', 'next_action'],
        statuses=['pending', 'passed', 'failed', 'not_applicable'],
        artifact_required=['path', 'sha256', 'role'],
        optional_stage_fields=['criteria', 'measurements'],
        rules=[
            'Create a separate sampling setup for each consequential chemical microstate; never pool proton inventories by raw energy.',
            'Record all alternatives, including failed preparations. Every stage requires a concrete next_action until passed.',
            'Passed stages require explicit criteria, measured results and hashed output evidence. A process exit code alone is insufficient.',
            'Stages depend on all preceding stages; changed evidence invalidates downstream readiness.',
            'No automatic chemistry, metal count, tautomer, force field, solvent or restraint choices.',
            'Use explicit preparation operators in their validated environment. Unavailable operators remain actionable gaps.',
            'Pilot review covers independent replicas, coordination/hydration, substrate motion, timestep stability and uncertainty.',
            'Readiness is conditional on agent interpretation, not chemical accuracy or authorization to launch production.'])


def inspect_pdb(path):
    """Bounded geometric observations only; no bonding or protonation inference."""
    path = Path(path)
    if path.stat().st_size > 16*1024*1024:
        raise ValueError('Structure exceeds inspection limit')
    lines = path.read_text().splitlines()
    if sum(line.startswith('MODEL ') for line in lines) > 1:
        raise ValueError('Select one PDB model explicitly')
    atoms = []
    keys = set()
    for line in lines:
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        if line[16:17].strip():
            raise ValueError('Resolve alternate locations explicitly before inspection')
        key = (line[21:22], line[22:27].strip(), line[12:16].strip())
        if key in keys:
            raise ValueError('Duplicate atom identity')
        keys.add(key)
        element = line[76:78].strip().upper()
        if not element:
            raise ValueError('PDB element fields are required; do not infer metal identity')
        xyz = tuple(float(line[i:i+8]) for i in (30,38,46))
        if not all(math.isfinite(x) for x in xyz):
            raise ValueError('Nonfinite coordinates')
        atoms.append(dict(id=':'.join(key), residue=line[17:20].strip(), element=element, xyz=xyz))
    if not atoms:
        raise ValueError('No PDB atoms found')
    metals = []
    for metal in atoms:
        if metal['element'] not in {'MG','CA','MN','FE','CO','NI','CU','ZN'}:
            continue
        neighbors = [dict(atom=a['id'], residue=a['residue'], distance_A=math.dist(a['xyz'],metal['xyz']))
                     for a in atoms if a['element'] in {'O','N','S'} and math.dist(a['xyz'],metal['xyz']) <= 3.0]
        metals.append(dict(atom=metal['id'], element=metal['element'], neighbors_within_3A=neighbors))
    return dict(path=str(path.resolve()), sha256=file_digest(path), atom_count=len(atoms),
                hydrogen_count=sum(a['element'] in {'H','D'} for a in atoms),
                water_oxygen_count=sum(a['residue'] in {'HOH','WAT'} and a['element']=='O' for a in atoms),
                metals=metals, scope='Coordinate observations only; 3 A is an inspection radius, not a coordination rule')


def evaluate(study, setups, sequence, target, authorize):
    if study.get('schema_version') != 'preparation-study-1.0':
        raise ValueError('Require preparation-study-1.0')
    if not isinstance(study.get('question'), str) or not study['question'].strip():
        raise ValueError('Explicit preparation question required')
    variants = study.get('variants')
    if not isinstance(variants, list) or not variants:
        raise ValueError('At least one explicit preparation variant is required')
    seen, results = set(), []
    for variant in variants:
        identifier = variant.get('id')
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError('Variant IDs must be nonempty and unique')
        seen.add(identifier)
        if not variant.get('rationale'):
            raise ValueError('Each alternative needs a scientific rationale')
        setup_id = variant.get('setup_id', study.get('setup_id'))
        if setup_id not in setups:
            raise ValueError('Unknown sampling setup')
        setup = setups[setup_id]['setup']
        gaps = []
        if variant.get('setup_sha256') != fingerprint(setup):
            gaps.append(dict(stage='setup', reason='Variant setup identity changed'))
        for source in setup['sources']:
            authorize(source['path'])
        assessment = assess(setup, sequence, target)
        gaps.extend(dict(stage='setup', **g) for g in assessment['gaps'])
        stage_results = []
        for name in STAGES:
            stage = variant.get('stages', {}).get(name, {})
            status = stage.get('status', 'pending')
            if status not in {'pending','failed','passed','not_applicable'}:
                raise ValueError('Unknown preparation stage status')
            errors = []
            if not stage.get('interpretation'):
                errors.append('Scientific interpretation missing')
            if status in {'pending','failed'}:
                errors.append(stage.get('next_action') or 'Specify the next investigation or preparation operation')
            artifacts = stage.get('artifacts', [])
            if status == 'passed' and (not artifacts or not stage.get('criteria') or not stage.get('measurements')):
                errors.append('Passed stage lacks criteria, measurements or output evidence')
            if status == 'not_applicable' and name in {'chemical_state','parameters','minimization','equilibration','pilot_review'}:
                errors.append('This stage cannot be omitted for a prepared dynamics study')
            for artifact in artifacts:
                try:
                    path = authorize(artifact['path'])
                    if not artifact.get('role') or file_digest(path) != artifact.get('sha256'):
                        errors.append('Output artifact changed or has no declared role')
                except (OSError, ValueError, KeyError) as exc:
                    errors.append(str(exc))
            if not errors and gaps:
                errors.append('Upstream preparation or setup remains unresolved')
            gaps.extend(dict(stage=name, reason=reason) for reason in errors)
            stage_results.append(dict(stage=name, ready=not errors, declared_status=status, errors=errors))
        results.append(dict(id=identifier, setup_id=setup_id, ready=not gaps, gaps=gaps, stages=stage_results))
    return dict(status='ready_for_sampling_review' if all(r['ready'] for r in results) else 'needs_preparation',
                variants=results, study_sha256=fingerprint(study), execution_authorized=False,
                limitations='Evidence integrity and agent-reviewed criteria; chemical accuracy and production convergence are not established.')


def run_operator(folder, python, code, inputs, outputs, seconds, authorize):
    """Run an explicit CPU preparation/diagnostic script, retaining failed attempts.

    This is an agent code-execution tool, not a security sandbox. Expensive GPU
    operators belong in the reserved scheduler workflow, not this short operator.
    """
    import os
    import signal
    import subprocess
    import time
    if type(seconds) is not int or not 1 <= seconds <= 60:
        raise ValueError('Local operator wall allowance must be 1..60 seconds')
    if not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError('Choose an explicit existing Python interpreter')
    if not isinstance(code, str) or not code.strip() or len(code) > 100000:
        raise ValueError('Provide a bounded explicit Python script')
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError('Declare distinct expected output files')
    for name in outputs:
        if Path(name).is_absolute() or '..' in Path(name).parts or name in {'preparation_step.py','launch_preparation.py','environment.json','request.json','stdout.log','stderr.log','result.json'}:
            raise ValueError('Output must be a safe relative path')
    bound = []
    for item in inputs:
        path = authorize(item['path'])
        if file_digest(path) != item['sha256']:
            raise ValueError('Operator input changed')
        bound.append(dict(path=str(path), sha256=item['sha256']))
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2 or os.getloadavg()[0] + 2 > os.cpu_count():
        raise RuntimeError('Insufficient verified CPU capacity for controller and operator')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'preparation_step.py').write_text(code)
    (folder/'launch_preparation.py').write_text(
        "import importlib.metadata as m, json, platform, runpy, sys\n"
        "from pathlib import Path\n"
        "Path('environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),"
        "'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions() if d.metadata['Name'])},indent=2))\n"
        "runpy.run_path('preparation_step.py',run_name='__main__')\n")
    request = dict(python=python, python_sha256=file_digest(python), inputs=bound, outputs=outputs,
                   seconds=seconds, worker_cpu=cpus[-1], script_sha256=file_digest(folder/'preparation_step.py'))
    (folder/'request.json').write_text(json.dumps(request, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    for name in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','OPENMM_CPU_THREADS'):
        env[name] = '1'
    started = time.monotonic()
    timed_out = False
    process = None
    try:
        os.sched_setaffinity(0, cpus[:-1])
        with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
            process = subprocess.Popen(['taskset','-c',str(cpus[-1]),python,'launch_preparation.py'], cwd=folder,
                                       env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Reap helpers even when the main script exits before them.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    finally:
        os.sched_setaffinity(0, cpus)
    errors = []
    if timed_out or process.returncode != 0:
        errors.append('Operator timed out' if timed_out else f'Operator exited {process.returncode}')
    for item in bound:
        if not Path(item['path']).is_file() or file_digest(item['path']) != item['sha256']:
            errors.append('Input changed during operator execution')
    artifacts = []
    for name in outputs:
        path = folder/name
        if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            errors.append(f'Missing or external output: {name}')
        else:
            artifacts.append(dict(path=str(path.resolve()), sha256=file_digest(path), role=name))
    if (folder/'environment.json').is_file():
        artifacts.append(dict(path=str((folder/'environment.json').resolve()), sha256=file_digest(folder/'environment.json'), role='runtime_environment'))
    result = dict(status='operator_complete_requires_review' if not errors else 'operator_failed',
                  errors=errors, artifacts=artifacts, request=request, elapsed_seconds=time.monotonic()-started,
                  scope='Execution and file integrity only; inspect measurements before declaring a stage passed')
    (folder/'result.json').write_text(json.dumps(result, indent=2))
    return dict(artifact=str(folder/'result.json'), **result)
