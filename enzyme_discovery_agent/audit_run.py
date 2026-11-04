"""Audit a saved campaign without model calls or changes to its evidence."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .models.candidate_registry import load_candidate_registry
from .run_layout import run_file


def audit(run_dir):
    root=Path(run_dir)
    manifest=json.loads((root/'manifest.json').read_text())
    usage=json.loads(run_file(root,'token_usage.json').read_text())
    registry=load_candidate_registry(root,strict=True)['candidates']
    groups=defaultdict(Counter)
    for call in usage.get('calls',[]):
        label=call.get('iteration'); label='controller' if isinstance(label,int) else str(label)
        groups[label]['calls']+=1
        for key in ('total_tokens','prompt_tokens','completion_tokens','cache_creation_tokens','cache_read_tokens'):
            groups[label][key]+=call.get(key,0)
    grouped={label:dict(values) for label,values in groups.items()}
    for label,values in grouped.items():
        values['fraction_of_total_tokens']=values['total_tokens']/usage['total_tokens'] if usage.get('total_tokens') else None
        values['cache_read_fraction_of_prompt']=values['cache_read_tokens']/values['prompt_tokens'] if values['prompt_tokens'] else None
    assessments=Counter();leads=[]
    for accession,entry in registry.items():
        meta=entry.get('metadata') or {};record=meta.get('target_assessment') or {}
        if record:assessments[record.get('decision','unknown')]+=1
        if meta.get('is_literature_lead'):
            leads.append({'lead_id':accession,'enzyme_name':meta.get('literature_name') or meta.get('protein_name'),
                          'status':meta.get('identity_resolution_status')})
    events_path=run_file(root,'events.jsonl')
    events=[json.loads(line) for line in events_path.read_text().splitlines() if line] if events_path.exists() else []
    csv_path=root/'ranked_candidates.csv'
    rows=list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
    projection=[]
    for row in rows:
        assessment=((registry.get(row.get('accession')) or {}).get('metadata') or {}).get('target_assessment') or {}
        if assessment and not row.get('target_assessment_decision'):
            projection.append({'enzyme':row.get('protein_name','')+' ('+row.get('accession','')+')',
                               'saved_decision':assessment.get('decision')})
    fits=Counter(row['substrate_fit'] for row in rows if row.get('substrate_fit'))
    return {'run_id':root.name,'status':manifest.get('status'),
        'wall_clock_seconds':(manifest.get('timing') or {}).get('wall_clock_s'),
        'run_configuration':{key:(manifest.get('config') or {}).get(key) for key in
            ('ADAPTIVE_REASONING','AUTO_TARGET_REVIEW','TARGET_REVIEW_BATCH_SIZE','MODEL_CONTEXT_MAX_CHARS')},
        'tokens':{key:usage.get(key) for key in ('total_tokens','cache_read_fraction_of_prompt')},
        'tokens_by_workload':grouped,'registered_records':len(registry),
        'saved_target_assessments':dict(assessments),'literature_leads':leads,
        'identity_resolution_attempt_events':sum(e.get('type')=='literature_lead_resolution_pending' for e in events),
        'ranked_rows':len(rows),'csv_assessment_projection_gaps':projection,
        'most_common_reported_substrate_fits':fits.most_common(5),
        'limits':['Current saved records are reported, not reconstructed earlier states.',
            'An assessment or a database match does not establish target activity.',
            'Repeated substrate resemblance values are not per-protein affinity measurements.',
            'Cache accounting is provider-reported; this audit makes no timing or speedup estimate.']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.run_dir),indent=2))
