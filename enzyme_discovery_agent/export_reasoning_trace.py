"""Export source-linked scientific decisions without generating retrospective reasoning.

Usage: python -m enzyme_discovery_agent.export_reasoning_trace RUN_DIR
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from enzyme_discovery_agent.models.reasoning_trace import capture_scientific_record, read_scientific_trace
from enzyme_discovery_agent.state_store import atomic_write_json


def synchronize_saved_records(run_dir):
    """Recover existing artifacts, explicitly labelled as reconstructed capture."""
    root=Path(run_dir)
    base=root/'agent_state/research_decisions'
    known={event['id'] for event in read_scientific_trace(root)}
    sources=[]
    for folder in base.iterdir() if base.exists() else []:
        if folder.name in {'evidence','reviews','interpretations'} or not folder.is_dir():
            continue
        for filename,kind,key in [('plan.json','prediction_registered','created_at'),
                                  ('outcome.json','observation_recorded','completed_at')]:
            path=folder/filename
            if path.exists() and f'{kind}:{folder.name}' not in known:
                value=json.loads(path.read_text())
                sources.append((value[key],kind,path))
    for path in (base/'reviews').glob('*.json'):
        if f'review_concluded:{path.stem}' not in known:
            value=json.loads(path.read_text())
            sources.append((value['created_at'],'review_concluded',path))
    for path in (base/'interpretations').glob('*.json'):
        if f'outcome_interpreted:{path.stem}' not in known:
            value=json.loads(path.read_text())
            sources.append((value['created_at'],'outcome_interpreted',path))
    for path in (root/'artifacts/mechanism_hypotheses').glob('*.json'):
        if f'mechanism_hypotheses_recorded:{path.stem}' not in known:
            value=json.loads(path.read_text())
            sources.append((value['created_at'],'mechanism_hypotheses_recorded',path))
    for path in (root/'artifacts/target_reviews').glob('*.json'):
        kind='target_review_batch_recorded' if path.stem.startswith('batch_') else 'target_assessment_recorded'
        if path.stem.endswith('_priority') or f'{kind}:{path.stem}' in known:
            continue
        value=json.loads(path.read_text())
        if value.get('id') and value.get('created_at'):
            sources.append((value['created_at'],kind,path))
    for folder,kind in [('reaction_roles','participant_roles_recorded'),('reaction_steps','reaction_step_recorded')]:
        for path in (root/'artifacts'/folder).glob('*/*.json'):
            if f'{kind}:{path.stem}' not in known:
                value=json.loads(path.read_text())
                if value.get('recorded_at'):
                    sources.append((value['recorded_at'],kind,path))
    for _,kind,path in sorted(sources,key=lambda row:(row[0],row[1],str(row[2]))):
        capture_scientific_record(root,kind,path,capture_mode='reconstructed_from_saved_artifact')


def _verify_sources(root,events):
    checks=[]
    def visit(value,event_id):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                try:
                    path=(root/value['path']).resolve()
                    path.relative_to(root)
                    intact=path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==value['sha256']
                except (ValueError,TypeError,OSError):
                    intact=False
                checks.append({'event':event_id,'path':value['path'],'intact':intact})
            for child in value.values():
                visit(child,event_id)
        elif isinstance(value,list):
            for child in value:
                visit(child,event_id)
    for event in events:
        visit(event['record'],event['id'])
    return checks


def export_trace(run_dir, *, reconstruct_missing=True):
    root=Path(run_dir).resolve()
    if reconstruct_missing:
        synchronize_saved_records(root)
    events=read_scientific_trace(root)
    predictions={event['record']['id']:event for event in events if event['type']=='prediction_registered'}
    observations={event['record']['decision_id']:event for event in events if event['type']=='observation_recorded'}
    conclusions=[event for event in events if event['type']=='review_concluded']
    interpretations=[event['record'] for event in events if event['type']=='outcome_interpreted']
    checks=_verify_sources(root,events)
    cases=[]
    for identifier,event in predictions.items():
        plan=event['record']
        observation=observations.get(identifier)
        revisions=[other['record']['id'] for other in predictions.values() if other['record'].get('revises')==identifier]
        closed=[item['record']['id'] for item in conclusions if item['record'].get('revises')==identifier]
        cases.append({'decision_id':identifier,'registered_at':event['occurred_at'],
            'capture_mode':event['capture_mode'],'hypothesis':plan['hypothesis'],'alternative':plan['alternative'],
            'scope':plan['scope'],'prediction':plan['prediction'],'contrasting_prediction':plan['contrasting_prediction'],
            'why_this_test':plan['why_this_test'],'revision_trigger':plan['revision_trigger'],
            'action':{'tool':plan['tool_name'],'arguments':plan['arguments']},
            'evidence_refs':plan['evidence_refs'],'model_provenance':plan.get('model_provenance'),
            'observation':observation['record'] if observation else None,
            'revises':plan.get('revises'),'belief_update':plan.get('belief_update'),
            'followup_decisions':revisions,'concluded_reviews':closed,
            'outcome_interpretations':[item for item in interpretations if item.get('decision_id')==identifier],
            'prospective_capture_verified':event['capture_mode']=='live' and bool(observation)
                and event['sequence']<observation['sequence'],
            'source':plan['source']})
    cases.sort(key=lambda row:(row['registered_at'],row['decision_id']))
    result={'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),'run_dir':str(root),
        'journal':'reasoning/trace.jsonl','hash_chain_valid':True,
        'source_files_intact':all(item['intact'] for item in checks),'source_checks':checks,
        'scope':'Explicit scientific hypotheses and decisions. Provider-private thoughts are not requested or stored.',
        'limits':['Recorded reasoning and source integrity do not prove scientific entailment or activity.',
                  'Reconstructed historical captures do not establish prospective prediction registration.',
                  'An action change is not necessarily an improved scientific decision.'],
        'events':events,'examples':cases,'conclusions':[item['record'] for item in conclusions]}
    destination=root/'reasoning'
    atomic_write_json(destination/'trace.json',result)
    lines=['# Scientific reasoning trace','',result['scope'],'',
        f"Journal events: {len(events)}. Source files intact: {result['source_files_intact']}.",
        '','This export copies saved records; no model wrote a retrospective explanation.','']
    for index,case in enumerate(cases,1):
        outcome=case['observation'] or {}
        lines.extend([f"## {index}. {case['action']['tool']} — {case['decision_id'][:8]}",'',
            f"Recorded: {case['registered_at']} · Capture: {case['capture_mode']}",'',
            f"**Hypothesis:** {case['hypothesis']}",'',f"**Alternative:** {case['alternative']}",'',
            f"**Prediction:** {case['prediction']}",'',f"**Contrasting prediction:** {case['contrasting_prediction']}",'',
            f"**Why this action:** {case['why_this_test']}",'',
            '**Action arguments:**','', '```json',json.dumps(case['action']['arguments'],indent=2),'```','',
            f"**Observed result:** {outcome.get('status','not yet recorded')} — {outcome.get('summary','')}",'',
            f"**Revision condition:** {case['revision_trigger']}",'',
            f"**Belief update:** {case.get('belief_update') or 'not stated'}; revises {case.get('revises') or 'no earlier decision'}.",'',
            '**Evidence snapshots:**',''])
        for ref in case['evidence_refs']:
            digest=ref.removeprefix('evidence:')
            lines.append(f'- [{ref}](../agent_state/research_decisions/evidence/{digest}.json)')
        lines.extend(['',f"[Saved prediction](../{case['source']['path']})",''])
        if outcome:
            lines.extend([f"[Complete observed result](../{outcome['result_source']['path']})",''])
        for conclusion in result['conclusions']:
            if conclusion.get('revises')==case['decision_id']:
                lines.extend([f"**Provisional conclusion:** {conclusion['conclusion']}",''])
    hypotheses=[event for event in events if event['type']=='mechanism_hypotheses_recorded']
    if hypotheses:
        lines.extend(['## Mechanism search hypotheses','',
            'Provisional source-informed search maps; neither exhaustive coverage nor established target activity.',''])
        for event in hypotheses:
            record=event['record']
            lines.extend([f"Recorded: {event['occurred_at']} · Capture: {event['capture_mode']}",''])
            for family in record.get('families',[]):
                lines.append('- '+family['name']+': '+str(family.get('required_catalytic_capability') or 'Unresolved catalytic capability.'))
            lines.extend(['',f"[Saved hypotheses and source inputs](../{record['source']['path']})",''])
    assessments=[event for event in events if event['type']=='target_assessment_recorded']
    if assessments:
        lines.extend(['## Target-specific candidate assessments',''])
        for event in assessments:
            record=event['record']
            lines.extend([f"### {record['accession']} · {record['decision']}",'',
                record['reason'],'',record['mechanism_and_step'],'',
                '**Participant roles:**','',json.dumps(record['target_roles'],ensure_ascii=False),'',
                '**Reopen when:** '+record['revisit_when'],'',
                f"[Complete assessment and source evidence](../{record['source']['path']})",''])
    batches=[event for event in events if event['type']=='target_review_batch_recorded']
    if batches:
        lines.extend(['## Assessment attempts','',
            'Incomplete or invalid responses are unresolved work, not negative scientific results.',''])
        for event in batches:
            record=event['record'];batch_result=record['result']
            lines.extend([f"Recorded: {event['occurred_at']} · Capture: {event['capture_mode']}",'',
                '**Assessed:** '+', '.join(batch_result['assessed']),
                '**Unassessed:** '+', '.join(batch_result['unassessed']),
                '**Errors:** '+json.dumps(batch_result['errors']+batch_result.get('provider_faults',[])), '',
                f"[Complete attempt, feedback and source inputs](../{record['source']['path']})",''])
    interpretations=[event for event in events if event['type'] in {'participant_roles_recorded','reaction_step_recorded'}]
    if interpretations:
        lines.extend(['## Participant and reaction-step interpretations','',
            'These are source-linked hypotheses. Structural and source-integrity checks do not establish enzyme activity.',''])
        for event in interpretations:
            record=event['record'];value=record['interpretation']
            lines.extend([f"### {record.get('accession')} · {event['type']}",'',
                f"Recorded: {event['occurred_at']} · Capture: {event['capture_mode']}",'',
                value.get('interpretation') or json.dumps(value.get('participants'),ensure_ascii=False),'',
                '**Activation assessment:** '+(value.get('activation_assessment') or 'Not recorded.'), '',
                '**Continuation assessment:** '+(value.get('continuation_assessment') or 'Not recorded.'), '',
                f"[Complete saved interpretation](../{record['source']['path']})",''])
    lines.extend(['## Verification limits','']+['- '+limit for limit in result['limits']]+[''])
    (destination/'trace.md').write_text('\n'.join(lines))
    from enzyme_discovery_agent.output.reasoning_viewer import write_reasoning_viewer
    write_reasoning_viewer(destination,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--live',action='store_true',help='Export only already journaled records; do not reconstruct artifacts during active execution')
    args=parser.parse_args()
    result=export_trace(args.run_dir,reconstruct_missing=not args.live)
    print(json.dumps({'run_dir':result['run_dir'],'events':len(result['events']),
                      'examples':len(result['examples']),'source_files_intact':result['source_files_intact']}))
    return 0 if result['source_files_intact'] else 1


if __name__=='__main__':
    raise SystemExit(main())
