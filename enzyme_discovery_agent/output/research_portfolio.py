"""Expose source-cited hypotheses and unresolved identities without inventing activity ranks."""
import csv
import io
import json
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry,discovery_status
from enzyme_discovery_agent.state_store import atomic_write_text
from enzyme_discovery_agent.run_layout import run_file


def write_research_portfolio(run_dir):
    from enzyme_discovery_agent.tools.discovery_proposals import write_primary_portfolio
    primary=write_primary_portfolio(run_dir)
    from enzyme_discovery_agent.output.discovery_portfolio import write_discovery_portfolio
    from enzyme_discovery_agent import config
    discovery=write_discovery_portfolio(run_dir,limit=config.DISCOVERY_PORTFOLIO_SIZE)
    root=Path(run_dir);registry=load_candidate_registry(root,strict=True)['candidates']
    hypotheses=[];unresolved=[];mentions=[]
    for accession,entry in registry.items():
        meta=entry.get('metadata') or {};data={**(meta.get('evaluation_data') or {}),**meta}
        priority=data.get('priority_assessment') or {}
        if data.get('is_literature_lead'):
            if discovery_status(entry)=='rejected':continue
            if data.get('identity_resolution_status')!='resolved':
                paper=(data.get('literature_lead') or {}).get('paper') or {}
                row={'lead_id':accession,'name':data.get('literature_name') or data.get('protein_name'),
                    'paper':paper.get('doi') or paper.get('pmid') or paper.get('title'),
                    'identity_status':data.get('identity_resolution_status','pending'),
                    'relevance_status':data.get('target_relevance_status','not_yet_assessed'),
                    'scope':'Paper-scoped lead; not a verified sequence or target activity.'}
                # An unresolved identity is the actionable gap and must be reported as one.
                # Routing on relevance alone used to file every lead as a bare mention and
                # leave the unresolved-leads deliverable empty while leads went untested.
                if data.get('target_relevance_status')=='needs_relevance_review':mentions.append(row)
                unresolved.append(row)
            continue
        if discovery_status(entry)=='rejected' or priority.get('decision') not in {'prioritize','investigate'}:continue
        if not priority.get('strengths') or not priority.get('evidence_refs'):continue
        hypotheses.append({'accession':accession,'name':data.get('literature_name') or data.get('protein_name'),
            'work_priority':priority['decision'],'discovery_status':discovery_status(entry),
            'native_annotation_available':bool(data.get('catalytic_activities') or data.get('native_substrates')),
            'hypothesis_rationale':priority.get('reason'),'strengths':'; '.join(priority.get('strengths') or []),
            'unknowns':'; '.join(priority.get('unknowns') or []),'revisit_when':priority.get('revisit_when'),
            'source_refs':'; '.join(priority.get('evidence_refs') or []),
            'modeling_review_status':data.get('modeling_review_status','not_returned'),
            'scope':'Revisable investigation priority; target activity remains unestablished.'})
    def save(name,rows,fields):
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        atomic_write_text(run_file(root,name),stream.getvalue())
    fields=['accession','name','work_priority','discovery_status','native_annotation_available','hypothesis_rationale',
        'strengths','unknowns','revisit_when','source_refs','modeling_review_status','scope']
    hypotheses.sort(key=lambda row:(row['work_priority']!='prioritize',row['accession']))
    save('research_portfolio.csv',hypotheses,fields)
    lead_fields=['lead_id','name','paper','identity_status','relevance_status','scope']
    save('unresolved_literature_leads.csv',unresolved,lead_fields)
    save('literature_mentions.csv',mentions,lead_fields)
    return {'source_cited_hypotheses':len(hypotheses),'unresolved_paper_leads':len(unresolved),
            'discovery_proposals':discovery['selected'],'primary_candidates':primary['selected']}
