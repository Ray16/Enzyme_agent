"""Independent discovery and rolling modeling controllers configured by task data."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from .state_store import atomic_write_json, state_lock


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owner():
    return {'host': socket.gethostname(), 'pid': os.getpid(),
            'process_start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            'cpu_affinity': sorted(os.sched_getaffinity(0))}


def live(record):
    if not record:
        return False
    if record['host'] != socket.gethostname():
        return None
    try:
        return Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19] == record['process_start_ticks']
    except FileNotFoundError:
        return False


def require_capacity(cpus):
    import psutil
    cpus = set(cpus)
    if not cpus or any(type(cpu) is not int or cpu < 0 for cpu in cpus):
        raise RuntimeError('Requested CPU allocation is unavailable')
    # The coordinator has its own narrow binding. A child can use another
    # permitted set, which must be verified against the kernel's actual policy.
    probe = subprocess.run(['taskset', '-c', ','.join(map(str, sorted(cpus))), sys.executable,
                            '-c', 'import json,os;print(json.dumps(sorted(os.sched_getaffinity(0))))'],
                           capture_output=True, text=True, timeout=10, check=True)
    if set(json.loads(probe.stdout)) != cpus:
        raise RuntimeError('The child CPU allocation was not accepted exactly')
    occupied = set()
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid == os.getpid():
            continue
        try:
            affinity = set(process.cpu_affinity())
            if len(affinity) <= 8 and process.info['cmdline']:
                occupied |= affinity
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            try:
                affinity = set(os.sched_getaffinity(process.pid))
                if len(affinity) <= 8:
                    occupied |= affinity
            except ProcessLookupError:
                pass
    if cpus & occupied or max(os.getloadavg()) + len(cpus) > os.cpu_count():
        raise RuntimeError('CPU capacity cannot be assigned without contention')


def configure_discovery(settings):
    from . import config, tool_registry
    from .bootstrap import register_builtin_tools
    register_builtin_tools()
    # Structural work has an independent process, allowance and turn ledger.
    for name, info in list(tool_registry._TOOLS.items()):
        if info['category'] in {'structure', 'design'} or name in {'map_candidate_neighbors', 'expand_candidates_esm'}:
            del tool_registry._TOOLS[name]
    config.ADAPTIVE_REASONING = True
    config.ENABLE_CANDIDATE_MAP = False
    config.ENABLE_MECHANISM_FAMILY_CACHE = False
    config.MODEL_CONTEXT_MAX_CHARS = settings['model_context_max_chars']
    for name in ('TOOL_THREAD_POOL_WORKERS', 'EVAL_POOL_WORKERS', 'FULLTEXT_HARVEST_WORKERS', 'KEGG_THREAD_POOL_WORKERS'):
        setattr(config, name, min(getattr(config, name), len(os.sched_getaffinity(0))))
    config.FAMILY_SURVEY_MEMBER_WORKERS = 1
    from .gates import family_survey
    family_survey._WORKERS = min(family_survey._WORKERS, len(os.sched_getaffinity(0)))


def prepare_input_queries(settings, root, agent):
    """Retrieve explicit task-derived source probes once, without supplying candidate answers."""
    from .tool_registry import execute_tool
    ledger=read(root/'input_source_queries.json',{})
    allowed={'search_pubmed','search_openalex','search_crossref','search_europepmc'}
    for query in settings.get('input_literature_queries',[])[:8]:
        name=query.get('tool');arguments=query.get('arguments') or {}
        if name not in allowed or not isinstance(arguments.get('query'),str):
            raise ValueError('Input source probes require a permitted literature query')
        key=hashlib.sha256(json.dumps(query,sort_keys=True).encode()).hexdigest()
        if key in ledger:continue
        result=execute_tool(name,arguments,injected_args={'run_dir':str(agent.run_dir)})
        agent._save_artifact('literature',name,result,arguments,0)
        ledger[key]={'tool':name,'arguments':arguments,'status':result.get('status'),
                     'query_origin':'Explicit source query derived from the current user task; no enzyme identities are supplied by this controller.',
                     'retrieved_at':datetime.now(timezone.utc).isoformat()}
        atomic_write_json(root/'input_source_queries.json',ledger)


def discovery(settings, root):
    configure_discovery(settings)
    from . import config
    from .agent import EnzymeAgent
    from .models.target_profile import load_target_profile
    saved = read(root / 'campaign_state.json', {})
    resume = Path(saved['run_dir']) if saved.get('run_dir') else None
    spent = read(resume / 'agent_state/token_usage.json', {}).get('total_tokens', 0) if resume else 0
    remaining = settings['discovery_token_budget'] - spent
    if remaining <= 0:
        raise RuntimeError('Discovery allowance is exhausted; saved work remains resumable under an explicitly revised allowance')
    agent = EnzymeAgent(model=settings['model'], output_dir=settings['discovery_output_root'],
                        query=settings['query'], resume_run_dir=resume, token_budget=remaining,
                        max_iterations=settings.get('discovery_max_turns',config.MAX_ITERATIONS),
                        runtime_budget_seconds=settings['runtime_budget_seconds'])
    agent.capabilities['pocket_analysis'] = False
    agent.capabilities['boltz2'] = False
    atomic_write_json(root / 'discovery_owner.json', owner())
    if not resume:
        atomic_write_json(root / 'campaign_state.json', {'run_dir': str(agent.run_dir),
                          'created_at': datetime.now(timezone.utc).isoformat(), 'discovery_independent': True})
    try:
        prepare_input_queries(settings,root,agent)
        if resume:
            agent.run('')
        else:
            agent.run(settings['query'], substrate_smiles=settings['substrate_smiles'],
                      target_profile=load_target_profile(root / settings['target_profile']))
    finally:
        atomic_write_json(root / 'discovery_status.json', {**owner(),
            'status': read(agent.run_dir / 'manifest.json', {}).get('status', 'failed'),
            'run_dir': str(agent.run_dir), 'finished_at': time.time()})


def candidate_sources(value, run):
    """Only explicitly referenced sources from the current discovery are forwarded."""
    sources = set()
    if isinstance(value, dict):
        for child in value.values():
            sources |= candidate_sources(child, run)
    elif isinstance(value, list):
        for child in value:
            sources |= candidate_sources(child, run)
    elif isinstance(value, str) and len(value) < 4096 and '\n' not in value:
        try:
            path = Path(value)
            path = (path if path.is_absolute() else run / path).resolve()
            if path.is_relative_to((run / 'artifacts').resolve()) and path.is_file() and path.stat().st_size <= 16 * 1024 * 1024:
                sources.add(str(path))
        except OSError:
            pass
    return sources


def prepare_tasks(settings, run, rows, directory):
    """Resolve sequence sources, check the shared pool, batch MSAs and emit task data."""
    import requests
    from campaign_catalog.lookup import lookup_predictions
    from structural_modeling_agent.msa import prepare
    directory.mkdir(parents=True, exist_ok=True)
    resolved = []
    errors = []
    for row in rows:
        accession = row['accession']
        slot = directory / hashlib.sha256(accession.encode()).hexdigest()[:20]
        slot.mkdir(exist_ok=True)
        try:
            if re.fullmatch(r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}', accession):
                response = requests.get('https://rest.uniprot.org/uniprotkb/' + accession + '.json', timeout=(30, 60))
                response.raise_for_status()
                identity = response.json()
                if identity.get('primaryAccession') != accession or identity.get('inactiveReason'):
                    raise ValueError('Active database identity does not match the candidate')
                sequence = identity.get('sequence', {}).get('value', '')
                identity_path = slot / 'identity.json'
                atomic_write_json(identity_path, identity)
            else:
                data = row['data']
                identity_path = Path(data.get('identity_source_path', ''))
                sequence = data.get('sequence', '')
                if not identity_path.is_file() or checksum(identity_path) != data.get('identity_source_sha256'):
                    raise ValueError('Source-scoped sequence needs explicit fingerprinted identity evidence')
            if not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+', sequence):
                raise ValueError('Require the exact complete protein sequence')
            pools = list(dict.fromkeys([settings['cofold_pool']] + settings.get('prediction_cache_pools', [])))
            lookups = [lookup_predictions(pool, accession=accession, sequence=sequence,
                target_id=settings.get('cofold_pool_target_id'),
                required_ligand_smiles=settings.get('modeling_required_ligand_smiles')) for pool in pools]
            def unique_matches(key):
                found = {}
                for result in lookups:
                    for match in result.get(key, []):
                        found[match['execution']['execution_id']] = match
                return list(found.values())
            lookup = {'source_pools': [{'pool': pool, **result} for pool, result in zip(pools, lookups)],
                      'matches': unique_matches('matches'),
                      'other_target_matches': unique_matches('other_target_matches')}
            atomic_write_json(slot / 'pool_lookup.json', lookup)
            facts = {key: value for key, value in row['data'].items() if key != 'modeling_input'}
            facts_path = slot / 'discovery_evidence.json'
            atomic_write_json(facts_path, facts)
            evidence_paths = candidate_sources(facts, run) | {str(identity_path.resolve()), str(facts_path.resolve())}
            for match in lookup['matches']:
                for source in (match['execution'].get('artifacts') or []) + [match['setup'].get('input_source') or {}]:
                    path = Path(source.get('path', ''))
                    if path.is_file() and checksum(path) == source.get('sha256'):
                        evidence_paths.add(str(path.resolve()))
                producer_source=match['execution'].get('source') or {}
                producer_path=Path(producer_source.get('path',''))
                if producer_path.suffix=='.json' and producer_path.is_file() and checksum(producer_path)==producer_source.get('sha256'):
                    producer_report=read(producer_path,{})
                    for old in producer_report.get('candidates',[]):
                        for ref in old.get('msa_evidence') or []:
                            dependency=Path(ref['path'])
                            if dependency.is_file() and checksum(dependency)==ref.get('sha256'):evidence_paths.add(str(dependency.resolve()))
            candidate = {'accession': accession, 'sequence': sequence, 'evidence': {
                'identity_source_path': str(identity_path.resolve()), 'identity_source_sha256': checksum(identity_path),
                'discovery_evidence_path': str(facts_path), 'discovery_evidence_sha256': checksum(facts_path),
                'priority_hypothesis': row['data']['priority_assessment'],
                'native_substrates_resolved': row['native_substrates_resolved'],
                'existing_cofolds': lookup['matches'], 'do_not_recofold': bool(lookup['matches']),
                'other_target_cofolds': lookup.get('other_target_matches', []),
                'evidence_paths': sorted(evidence_paths)}}
            resolved.append((row, slot, candidate))
        except Exception as exc:
            errors.append({'accession': accession, 'error': f'{type(exc).__name__}: {exc}'})
    atomic_write_json(directory / 'preparation_errors.json', errors)
    if not resolved:
        return []
    fasta = directory / 'sequences.fasta'
    new_sequences=[(i,c) for i,(_,_,c) in enumerate(resolved) if not c['evidence']['existing_cofolds']]
    fasta.write_text(''.join(f'>{i}\n{c["sequence"]}\n' for i,c in new_sequences))
    alignments = prepare(fasta, directory / 'msa', settings['msa_cache']) if new_sequences else []
    by_id = {row['id']: row['csv'] for row in alignments}
    tasks = []
    for i, (row, slot, candidate) in enumerate(resolved):
        if not candidate['evidence']['existing_cofolds']:candidate['evidence']['msa_path'] = by_id[str(i)]
        task = {'schema_version': '1.0', 'objective': 'model_and_validate',
                'workflow_mode': 'production_cofold', 'setup_turn_limit': settings['setup_turn_limit'],
                'target': settings['modeling_target'], 'candidates': [candidate]}
        path = slot / 'task.json'; atomic_write_json(path, task)
        tasks.append({'accession': row['accession'], 'task': str(path),
                      'existing_predictions': bool(candidate['evidence']['existing_cofolds'])})
    return tasks


def modeling(settings, root, item_path):
    from structural_modeling_agent.production import run_candidate
    item = read(item_path)
    atomic_write_json(item_path.parent / 'owner.json', owner())
    report = run_candidate(task_path=item['task'], run_dir=item['modeling_run'],
        backend_path=root / 'backend.json', execution_intent='submit_after_checks_pass',
        model=settings['model'], token_budget=item['token_allowance'],
        max_turns=settings['modeling_max_turns'], runtime_budget_seconds=settings['modeling_runtime_seconds'])
    atomic_write_json(item_path.parent / 'worker_status.json', {**owner(),
                      'status': report['status'], 'finished_at': time.time()})


def register_completed_models(settings, run, model_report=None):
    """Refresh the target's discovery pool and add a modeling report when supplied."""
    from campaign_catalog.catalog import build
    pool = Path(settings['cofold_pool'])
    with state_lock(pool / 'catalog/source_registration'):
        config = read(pool / 'sources.json')
        ranking = run / 'ranked_candidates.csv'
        if not ranking.exists():
            ranking = run / 'artifacts/production_modeling/catalog_ranking.csv'
            ranking.parent.mkdir(parents=True, exist_ok=True)
            if not ranking.exists():
                ranking.write_text('accession,rank\n')
        source = {'path': str(run), 'label': run.name, 'target_id': settings['cofold_pool_target_id'],
                  'ranking': str(ranking), 'select_ranked_only': False}
        config['discovery_sources'] = [entry for entry in config['discovery_sources'] if Path(entry['path']).resolve() != run.resolve()] + [source]
        reports = config.setdefault('structural_agent_sources', [])
        if model_report is not None and not any(Path(entry['report']).resolve() == model_report.resolve() for entry in reports):
            reports.append({'report': str(model_report), 'run_id': run.name + ':' + model_report.parent.name,
                            'discovery_run_id': run.name, 'target_id': settings['cofold_pool_target_id']})
        atomic_write_json(pool / 'sources.json', config)
        return build(pool)


def evidence_basis(row):
    data = row['data']
    evidence = {key: data.get(key) for key in ('sequence', 'identity_resolution',
                'mechanism_evidence', 'catalytic_activities', 'structural_data', 'literature_evidence')}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()


# A queue item holds its slot both while its worker runs and while the coordinator
# waits for that worker's backend jobs to settle after it exits.
OCCUPIED_STATUSES = {'running', 'awaiting_backend_jobs'}


def coordinator(settings, root, config_path):
    from .models.modeling_portfolio import select_modeling_portfolio
    from .import_structural_evidence import import_structural_evidence
    from .steering import submit
    from .provenance import snapshot_source_code
    atomic_write_json(root / 'coordinator_owner.json', owner())
    deadline = time.monotonic() + settings['runtime_budget_seconds']
    while not (root / 'campaign_state.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Discovery did not initialize a run')
        time.sleep(5)
    run = Path(read(root / 'campaign_state.json')['run_dir'])
    queue = run / 'agent_state/production_modeling'; queue.mkdir(parents=True, exist_ok=True)
    snapshot_source_code(run)
    consecutive_failures = 0
    last_pool_refresh = 0
    last_pool_inputs = None

    def usage():
        return sum((read(path, {}).get('usage') or {}).get('total_tokens', 0)
                   for path in (run / 'structural_modeling').glob('*/agent_state.json'))

    with state_lock(queue / 'coordinator'):
        active = None
        current = None
        while time.monotonic() < deadline:
            from .run_layout import run_file
            pool_inputs = [run_file(run, 'candidate_registry.json'), run / 'ranked_candidates.csv', run / 'manifest.json']
            signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                              for path in pool_inputs if path.exists())
            if (pool_inputs[0].exists() and signature != last_pool_inputs
                    and time.monotonic() - last_pool_refresh >= 120):
                try:
                    result = register_completed_models(settings, run)
                    atomic_write_json(root / 'candidate_pool_status.json', result)
                    last_pool_inputs = signature
                except Exception as exc:
                    atomic_write_json(root / 'candidate_pool_status.json', {'error': f'{type(exc).__name__}: {exc}'})
                last_pool_refresh = time.monotonic()
            entries = [read(path) for path in queue.glob('*/item.json')]
            if active is None:
                running = [entry for entry in entries if entry['status'] in OCCUPIED_STATUSES]
                if running:
                    raise RuntimeError('A recorded model worker needs explicit ownership reconciliation')
            if active is not None and active.poll() is not None:
                if current['status'] != 'awaiting_backend_jobs':
                    # The worker process is gone. Record that now: the loop below can wait
                    # many minutes for its backend jobs to settle, and until it does the
                    # queue would otherwise keep reporting an exited worker as running.
                    current.update(status='awaiting_backend_jobs', exit_code=active.returncode,
                                   worker_finished_at=time.time())
                    atomic_write_json(Path(current['item_dir']) / 'item.json', current)
                outstanding = [read(path) for path in (root / 'provider/backend_jobs').glob('*/receipt.json')
                               if read(path, {}).get('status') in {'queued', 'running', 'unknown'}]
                if outstanding:
                    published = read(root / 'campaign_status.json', {})
                    published.update(updated_at=time.time(), modeling_live=False,
                                     awaiting_backend_jobs=len(outstanding),
                                     queue=[read(path) for path in queue.glob('*/item.json')])
                    atomic_write_json(root / 'campaign_status.json', published)
                    time.sleep(10)
                    continue
                report = read(Path(current['modeling_run']) / 'report.json', {})
                current.update(status=report.get('status', 'worker_failed'), exit_code=active.returncode,
                               finished_at=time.time())
                evidence = Path(current['modeling_run']) / 'structural_modeling_evidence.json'
                if evidence.exists():
                    import_structural_evidence(run, evidence)
                    current['evidence_path'] = str(evidence)
                    exported = read(evidence)
                    if any(c.get('execution_status') == 'complete' for c in exported.get('candidates', [])):
                        try:
                            current['pool_registration'] = register_completed_models(settings, run, Path(current['modeling_run']) / 'report.json')
                        except Exception as exc:
                            current['pool_registration_error'] = f'{type(exc).__name__}: {exc}'
                atomic_write_json(Path(current['item_dir']) / 'item.json', current)
                claim = Path(current['item_dir']) / 'claim'
                if claim.exists():
                    claim.rmdir()
                consecutive_failures = consecutive_failures + 1 if active.returncode else 0
                submit(run, 'instruction', message=settings['feedback_instruction'])
                active = None
                current = None
                entries = [read(path) for path in queue.glob('*/item.json')]
            rows = select_modeling_portfolio(run, max_candidates=settings['max_modeling_candidates'],
                    protected_unannotated_slots=settings['protected_unannotated_slots'])
            new = [row for row in rows if
                   not any(entry['accession'] == row['accession'] and (entry['status'] in OCCUPIED_STATUSES or
                       entry.get('basis_sha256') == evidence_basis(row)) for entry in entries)
                   and sum(entry['accession'] == row['accession'] for entry in entries) < settings['modeling_max_attempts']]
            remaining = settings['modeling_token_budget'] - usage()
            known_ids = {entry['accession'] for entry in entries}
            selected = []
            for row in new:
                if row['accession'] in known_ids or len(known_ids) < settings['max_modeling_candidates']:
                    selected.append(row)
                    known_ids.add(row['accession'])
            if selected and remaining > 0 and consecutive_failures < 3:
                batch = hashlib.sha256(json.dumps([(r['accession'], evidence_basis(r)) for r in selected]).encode()).hexdigest()[:20]
                directory = run / 'artifacts/production_modeling' / batch
                try:
                    prepared = prepare_tasks(settings, run, selected, directory)
                except Exception as exc:
                    atomic_write_json(directory / 'batch_error.json', {'error': f'{type(exc).__name__}: {exc}'})
                    prepared = []
                    consecutive_failures += 1
                prepared_by_id = {item['accession']: item for item in prepared}
                for row in selected:
                    key = hashlib.sha256((row['accession'] + evidence_basis(row)).encode()).hexdigest()[:20]
                    item_dir = queue / key; item_dir.mkdir(exist_ok=True)
                    item = {**prepared_by_id.get(row['accession'], {}), 'accession': row['accession'],
                            'item_dir': str(item_dir), 'modeling_run': str(run / 'structural_modeling' / key),
                            'basis_sha256': evidence_basis(row),
                            'status': 'ready' if row['accession'] in prepared_by_id else 'preparation_unresolved'}
                    atomic_write_json(item_dir / 'item.json', item)
                entries = [read(path) for path in queue.glob('*/item.json')]
            if active is None and remaining > 0 and consecutive_failures < 3:
                current = next((item for item in entries if item['status'] == 'ready'), None)
                if current:
                    cpus = settings['modeling_cpu_affinity']; require_capacity(cpus)
                    item_path = Path(current['item_dir']) / 'item.json'
                    current['token_allowance'] = min(remaining, settings['modeling_item_token_budget'])
                    atomic_write_json(item_path, current)
                    claim = Path(current['item_dir']) / 'claim'; claim.mkdir()
                    argv = ['taskset', '-c', ','.join(map(str, cpus)), sys.executable,
                            str(root / 'worker.py'), '--config', str(config_path), '--role', 'modeling', '--item', str(item_path)]
                    try:
                        with (Path(current['item_dir']) / 'worker.log').open('ab') as log:
                            active = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        claim.rmdir(); raise
                    current.update(status='running', pid=active.pid, host=socket.gethostname(),
                                   process_start_ticks=Path(f'/proc/{active.pid}/stat').read_text().rsplit(')', 1)[1].split()[19],
                                   cpu_affinity=cpus, argv=argv, started_at=time.time())
                    atomic_write_json(item_path, current)
                    entries = [read(path) for path in queue.glob('*/item.json')]
            discovery_owner = read(root / 'discovery_owner.json')
            discovery_live = live(discovery_owner)
            complete_ids = set();new_predictions=set();reused_reviews=set()
            for item in entries:
                report = read(Path(item['modeling_run']) / 'structural_modeling_evidence.json', {})
                for candidate in report.get('candidates',[]):
                    if candidate.get('execution_status')!='complete':continue
                    reused=candidate.get('existing_prediction_review')
                    identity=reused['source_execution_id'] if reused else item['modeling_run']
                    complete_ids.add(identity)
                    if reused:reused_reviews.add((candidate['accession'],identity))
                    else:new_predictions.add(identity)
            status = {'updated_at': time.time(), 'run_dir': str(run), 'owner': owner(),
                      'discovery_live': discovery_live, 'modeling_live': active is not None and active.poll() is None,
                      'discovery_status': read(run / 'manifest.json', {}).get('status'),
                      'discovery_tokens': read(run / 'agent_state/token_usage.json', {}).get('total_tokens', 0),
                      'modeling_tokens': usage(), 'modeling_token_allowance': settings['modeling_token_budget'],
                      'modeling_turns_charged_to_discovery': 0, 'complete_ensembles': len(complete_ids),
                      'new_prediction_ensembles':len(new_predictions),'reused_prediction_reviews':len(reused_reviews),
                      'queue': entries, 'consecutive_failures': consecutive_failures}
            atomic_write_json(root / 'campaign_status.json', status)
            atomic_write_json(run / 'artifacts/production_modeling_feedback.json', status)
            atomic_write_json(run / 'structural_modeling_current.json', {'mode': 'rolling', 'running': active is not None,
                'root': str(run / 'structural_modeling'), 'candidates': entries})
            if discovery_live is False and active is None and (remaining <= 0 or consecutive_failures >= 3 or
                    not any(item['status'] == 'ready' for item in entries)):
                break
            time.sleep(10)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--role', required=True, choices=['discovery', 'modeling', 'coordinator'])
    parser.add_argument('--item', type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve(); root = config_path.parent; settings = read(config_path)
    if args.role == 'discovery':
        discovery(settings, root)
    elif args.role == 'modeling':
        modeling(settings, root, args.item)
    else:
        coordinator(settings, root, config_path)


if __name__ == '__main__':
    main()
