"""Reasoning trace: structured JSON event log for analytics."""

from __future__ import annotations

import json
import logging
import threading
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")

_trace_lock = threading.Lock()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read_scientific_trace(run_dir):
    """Verify the append-only journal; never silently discard damaged evidence."""
    path=Path(run_dir)/'reasoning/trace.jsonl'
    with state_lock(path):
        return _read_scientific_trace(path)


def _read_scientific_trace(path):
    """Read under the same lock as append, including live exports on NFS."""
    if not path.exists():
        return []
    events=[]
    previous=None
    with path.open() as handle:
        for number,line in enumerate(handle,1):
            try:
                event=json.loads(line)
                digest=event['event_sha256']
                payload={key:value for key,value in event.items() if key!='event_sha256'}
                if event['sequence']!=number or event['previous_sha256']!=previous or _digest(payload)!=digest:
                    raise ValueError('sequence or checksum mismatch')
            except (ValueError,KeyError,TypeError) as exc:
                raise ValueError(f'Invalid scientific trace at line {number}; original journal preserved') from exc
            events.append(event)
            previous=digest
    return events


def append_scientific_trace(run_dir,event_type,record,*,record_id,occurred_at,
                            capture_mode='live'):
    """Durably record explicit scientific reasoning, not provider-private thoughts.

    New predictions are recorded before execution. Recovered records are marked
    reconstructed, so exporting history cannot masquerade as a prospective test.
    """
    path=Path(run_dir)/'reasoning/trace.jsonl'
    with state_lock(path):
        events=read_scientific_trace(run_dir)
        identity=f'{event_type}:{record_id}'
        existing=next((event for event in events if event['id']==identity),None)
        if existing:
            if existing['record']!=record:
                raise ValueError('Scientific trace record changed after capture')
            return existing
        payload={'schema_version':1,'sequence':len(events)+1,'id':identity,'type':event_type,
                 'occurred_at':occurred_at,'captured_at':datetime.now(timezone.utc).isoformat(),
                 'capture_mode':capture_mode,'previous_sha256':events[-1]['event_sha256'] if events else None,
                 'record':record}
        event={**payload,'event_sha256':_digest(payload)}
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        return event


def source_reference(run_dir,path):
    root=Path(run_dir).resolve()
    source=Path(path).resolve()
    relative=source.relative_to(root)
    return {'path':str(relative),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}


def capture_scientific_record(run_dir,kind,path,*,capture_mode='live'):
    """Capture an immutable decision, outcome or closure using its saved source."""
    root=Path(run_dir)
    path=Path(path)
    value=json.loads(path.read_text())
    source=source_reference(root,path)
    if kind=='prediction_registered':
        fields=('id','hypothesis','alternative','scope','prediction','contrasting_prediction',
                'why_this_test','revision_trigger','tool_name','arguments','evidence_refs','revises',
                'belief_update','historical_evidence_refs','requested_tool_name','requested_evidence_refs','model_provenance')
        record={key:value.get(key) for key in fields}
        record['source']=source
        record['evidence_snapshots']=[]
        for ref in value.get('evidence_refs') or []:
            candidate=root/'agent_state/research_decisions/evidence'/(ref.removeprefix('evidence:')+'.json')
            if candidate.is_file():
                snapshot={'id':ref,**source_reference(root,candidate)}
                source=json.loads(candidate.read_text()).get('source') or {}
                if source.get('path') and source.get('sha256'):
                    snapshot['underlying_source']={'path':source['path'],'sha256':source['sha256']}
                record['evidence_snapshots'].append(snapshot)
        identifier,occurred=value['id'],value['created_at']
    elif kind=='observation_recorded':
        identifier=path.parent.name
        result_path=path.with_name('result.json')
        result=json.loads(result_path.read_text())
        record={'decision_id':identifier,'outcome':value,'result_source':source_reference(root,result_path),
                'outcome_source':source,'status':result.get('status'),'summary':result.get('summary'),
                'count':result.get('count'),'deduplicated':result.get('deduplicated',False),
                'interpretation':'Tool completion is not proof of biological activity.'}
        data=result.get('data')
        if isinstance(data,dict):
            record['assessment']={key:data.get(key) for key in ('attempted','supported','evidence_pending',
                'rejected','failed','actual_sample','scope_limit','existing_assessments','new_evidence') if key in data}
        occurred=value['completed_at']
    elif kind in {'review_concluded','outcome_interpreted','mechanism_hypotheses_recorded',
                  'target_assessment_recorded','target_review_batch_recorded'}:
        record={**value,'source':source}
        identifier,occurred=value['id'],value['created_at']
    elif kind in {'participant_roles_recorded','reaction_step_recorded'}:
        record={'accession':value.get('accession'),'interpretation':value,'source':source,
                'scope':'A source-linked agent interpretation, not experimentally established target activity.'}
        identifier,occurred=path.stem,value['recorded_at']
    else:
        raise ValueError('Unknown scientific trace record kind')
    return append_scientific_trace(root,kind,record,record_id=identifier,occurred_at=occurred,capture_mode=capture_mode)


def record_trace_event(
    run_dir: Path,
    stage: str,
    event_type: str,
    data: dict,
) -> None:
    """Append a structured event to run_dir/reasoning_trace.json."""
    trace_path = run_file(run_dir, 'reasoning_trace.json')

    with _trace_lock, state_lock(trace_path):
        if trace_path.exists():
            try:
                with open(trace_path) as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, OSError):
                trace = {"events": []}
        else:
            trace = {"events": []}

        trace["events"].append({
            "timestamp": datetime.now().astimezone().isoformat(),
            "stage": stage,
            "event_type": event_type,
            "data": data,
        })

        try:
            atomic_write_json(trace_path, trace)
        except Exception as e:
            logger.warning("Failed to write reasoning trace: %s", e)
