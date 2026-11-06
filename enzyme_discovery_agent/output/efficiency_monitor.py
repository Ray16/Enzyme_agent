"""Passive discovery/token/explicit-reasoning monitoring; never sends model context."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.models.candidate_registry import discovery_status, load_candidate_registry
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def summarize(root):
    root = Path(root)
    usage = read(run_file(root, 'token_usage.json'), {})
    events = read_events(root)
    candidates = load_candidate_registry(root, strict=True).get('candidates', {})
    stages = defaultdict(lambda: {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
    for call in usage.get('calls', []):
        label = call.get('iteration')
        stage = 'controller' if str(label).isdigit() else str(label)
        row = stages[stage]
        row['calls'] += 1
        for key in ('prompt_tokens', 'completion_tokens'):
            row[key] += call.get(key, 0)
    profiles = {e['request_profile_id']: e for e in events
                if e.get('type') == 'llm_request_profile' and e.get('request_profile_id')}
    # Count successful, metered calls only; retries/abandoned requests must not
    # inflate the amount of input supposedly saved.
    matched = [profiles[c['request_profile_id']] for c in usage.get('calls', [])
               if c.get('request_profile_id') in profiles]
    eligible = [p for p in matched if 'messages_characters_before_json_compaction' in p]
    before = sum(p['messages_characters_before_json_compaction'] for p in eligible)
    after = sum(p['messages_characters'] for p in eligible)
    assessments = []
    sequence_verified = 0
    annotated_families = set()
    from enzyme_discovery_agent.models.target_review import identity_verified
    for accession, entry in candidates.items():
        metadata = entry.get('metadata') or {}
        if metadata.get('sequence') and identity_verified(root, accession, metadata):
            sequence_verified += 1
        family = metadata.get('interpro_family_id') or metadata.get('screening_family_id')
        if family:
            annotated_families.add(json.dumps(family, sort_keys=True))
        assessment = metadata.get('target_assessment') or {}
        if assessment:
            assessments.append({'accession': accession, 'protein_name': metadata.get('protein_name') or '(name not recorded)',
                                'decision': assessment.get('decision'), 'evidence_level': assessment.get('evidence_level'),
                                'source_refs': assessment.get('source_refs', []),
                                'reason': assessment.get('reason'), 'unknowns': assessment.get('unknowns', [])})
    tools = [e for e in events if e.get('type') == 'tool_call']
    queries = Counter((e.get('tool'), json.dumps(e.get('args') or {}, sort_keys=True)) for e in tools
                      if str(e.get('tool', '')).startswith(('search_', 'deep_literature_search')))
    return {'run': root.name, 'status': read(root/'manifest.json', {}).get('status'),
            'tokens': usage.get('total_tokens', 0), 'accounting_complete': usage.get('accounting_complete'),
            'cache_read_fraction': usage.get('cache_read_fraction_of_prompt'), 'stages': dict(stages),
            'json_compaction': {'matched_calls': len(eligible), 'before_characters': before,
                                'after_characters': after, 'saved_characters': before-after,
                                'scope': 'Measured message characters, not provider token savings or cost.'},
            'registered_candidates': len(candidates),
            'verified_sequence_records': sequence_verified,
            'status_counts': dict(Counter(discovery_status(v) or 'unknown' for v in candidates.values())),
            'annotated_family_groups': len(annotated_families),
            'assessment_counts': dict(Counter(a['decision'] for a in assessments)),
            'assessment_evidence_levels': dict(Counter(a['evidence_level'] for a in assessments)),
            'assessments': assessments,
            'tool_calls': len(tools), 'tool_errors': sum(e.get('status') == 'error' for e in tools),
            'exact_repeat_searches': sum(n-1 for n in queries.values() if n > 1),
            'proposal_repairs': sum(e.get('type') == 'research_proposal_rejected' for e in events),
            'last_event_at': events[-1].get('ts') if events else None,
            'recent_activity': [{k: e.get(k) for k in ('ts', 'type', 'tool', 'status', 'summary', 'message') if k in e}
                                for e in events[-8:]]}


def refresh(root, baseline=None):
    from enzyme_discovery_agent.export_reasoning_trace import export_trace
    root = Path(root)
    current = summarize(root)
    try:
        # Export recorded explicit claims only. Never invent or reconstruct a
        # prospective explanation from later results while the campaign is live.
        trace = export_trace(root, reconstruct_missing=False)
        examples = trace['examples']
        reasoning = {'events': len(trace['events']), 'hash_chain_valid': trace['hash_chain_valid'],
                     'source_files_intact': trace['source_files_intact'],
                     'event_types': dict(Counter(e['type'] for e in trace['events'])),
                     'registered_tests': len(examples),
                     'tests_with_observation': sum(bool(e.get('observation')) for e in examples),
                     'prospective_tests_verified': sum(bool(e.get('prospective_capture_verified')) for e in examples),
                     'tests_with_revision': sum(bool(e.get('followup_decisions')) for e in examples),
                     'planned_actions': (trace.get('audit_summary') or {}).get('planned_actions'),
                     'linked_request_tokens': (trace.get('audit_summary') or {}).get('linked_request_tokens'),
                     'trace_page': '../reasoning/trace.html'}
    except (ValueError, OSError, KeyError) as exc:
        reasoning = {'status': 'verification_failed', 'error': str(exc)}
    alerts = []
    if reasoning.get('status') == 'verification_failed' or reasoning.get('source_files_intact') is False:
        alerts.append('Reasoning trace/source integrity requires inspection.')
    if current['tool_errors'] >= 3:
        alerts.append('At least three tool errors recorded; inspect source availability and recovery.')
    if current['exact_repeat_searches'] >= 3:
        alerts.append('Repeated identical searches recorded; inspect whether they added evidence.')
    if current['tokens'] >= 250000 and not current['registered_candidates']:
        alerts.append('Substantial spend before candidate registration; inspect literature-to-sequence progress.')
    stamp = current.get('last_event_at')
    if current['status'] == 'running' and stamp:
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(stamp)).total_seconds()
        if age >= 600:
            alerts.append('No recorded event for ten minutes; verify the live process and current service call.')
    result = {'generated_at': datetime.now(timezone.utc).isoformat(), 'current': current,
              'baseline': baseline, 'reasoning': reasoning, 'alerts': alerts,
              'quality_comparison': 'Not yet assessable' if not current['assessments'] else 'Requires source-level review of the recorded assessments',
              'limits': ['Baseline and current run have different progress and retrieved evidence; totals are not a matched efficiency experiment.',
                         'Candidate counts, model decisions and trace integrity do not establish target activity or better candidates.',
                         'This observer writes derived monitoring/trace exports only; it does not steer discovery or call models.']}
    folder = root/'monitoring'
    atomic_write_json(folder/'current.json', result)
    with (folder/'history.jsonl').open('a') as handle:
        small = {k: result[k] for k in ('generated_at', 'reasoning', 'alerts')}
        small['current'] = {k: v for k, v in current.items() if k not in {'assessments', 'recent_activity'}}
        handle.write(json.dumps(small)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--max-seconds', type=float, default=18000)
    args = parser.parse_args()
    if args.interval < 10 or args.max_seconds <= 0:
        parser.error('Require interval >=10 seconds and positive duration')
    baseline = summarize(args.baseline) if args.baseline else None
    deadline = time.monotonic()+args.max_seconds
    with state_lock(args.run/'monitoring/observer'):
        while True:
            try:
                result = refresh(args.run, baseline)
                print(json.dumps({'tokens': result['current']['tokens'], 'candidates': result['current']['registered_candidates'],
                                  'alerts': result['alerts']}), flush=True)
                if result['current']['status'] not in {'running', 'created', None}:
                    return
            except (OSError, ValueError, KeyError) as exc:
                atomic_write_json(args.run/'monitoring/observer_error.json', {'error': str(exc)})
            if time.monotonic() >= deadline:
                return
            time.sleep(min(args.interval, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
