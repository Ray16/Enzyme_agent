"""Durable source-to-identity follow-up, independent of native annotation richness."""
import csv
import hashlib
import io
import json
import re
import time
from pathlib import Path

from .candidate_registry import load_candidate_registry,update_candidate_registry,discovery_status
from .target_reasoning import target_claims
from .target_profile import is_opposite_direction
from enzyme_discovery_agent.state_store import atomic_write_json,atomic_write_text


def direction_evidence(root,accession,entry,registry):
    meta=entry.get('metadata') or {};origin=meta.get('literature_lead') or {}
    if origin.get('text_path'):
        path=Path(origin['text_path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=origin.get('text_sha256'):return []
    records=[meta.get('literature_evidence') or {},meta.get('source_function_evidence') or {}]
    paper=(origin.get('paper') or {}).get('doi');name=(meta.get('literature_name') or '').casefold()
    for other in registry.values():
        data=other.get('metadata') or {};evidence=data.get('literature_evidence') or {}
        if paper and (evidence.get('paper') or {}).get('doi')==paper and str(data.get('literature_name','')).casefold()==name:
            records.append(evidence)
    found=[]
    for record in records:
        if record.get('text_path') and record.get('text_sha256'):
            path=Path(record['text_path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['text_sha256']:continue
        value=str((record.get('mechanism') or {}).get('direction') or record.get('reaction_direction') or '').lower()
        match=re.match(r'\s*(formation|cleavage|hydrolysis|degradation|synthesis|ligation|condensation)\b',value)
        if match:
            direction='cleavage' if match[1] in {'cleavage','hydrolysis','degradation'} else 'formation'
            found.append({'direction':direction,'basis':'recorded functional evidence','evidence':record})
    path=origin.get('text_path')
    if path and name:
        source=Path(path);content=source.read_bytes()
        if hashlib.sha256(content).hexdigest()==origin.get('text_sha256'):
            # Attribute the role to the named subject, not every enzyme in a cascade.
            pattern=r'\b(?:hydrolysis|degradation)\s+of\s+(?:(?!(?:with|by|and)\b)[^.!?\n]){1,100}?\s+(?:with|by)\s+'+re.escape(meta['literature_name'])+r'\b'
            for match in re.finditer(pattern,content.decode(errors='replace'),re.I):
                found.append({'direction':'cleavage','basis':'explicit named source clause',
                              'quote':match[0],'path':str(source),'sha256':origin['text_sha256']})
    return found


def reconcile_directions(run_dir,accessions=None):
    root=Path(run_dir);registry=load_candidate_registry(root,strict=True)['candidates']
    target,_,_=target_claims(root);excluded=[]
    for accession,entry in registry.items():
        if accessions is not None and accession not in accessions:continue
        meta=entry.get('metadata') or {}
        if not (meta.get('is_literature_lead') or meta.get('is_literature_deposit')):continue
        evidence=direction_evidence(root,accession,entry,registry)
        directions={row['direction'] for row in evidence}
        if len(directions)==1 and is_opposite_direction(next(iter(directions)),target):
            update_candidate_registry(root,accession,'rejected',{'target_relevance_status':'excluded_opposite_direction',
                'rejection_reason':'Retrieved functional evidence is opposite to the requested reaction direction.',
                'direction_filter_evidence':evidence})
            excluded.append(accession)
        elif meta.get('is_literature_lead') and not meta.get('target_relevance_status'):
            update_candidate_registry(root,accession,None,{'target_relevance_status':
                'source_supported_direction' if target in directions else 'needs_relevance_review'})
    return excluded


LEAD_ATTEMPTS=3
# 'identity_pending' is not a verdict about the lead: its own reason asks for a taxon or a
# deposited identifier. Retry it under the same bounded budget as a transient outage, or a
# lead that misses once is dropped from the campaign for good.
RETRYABLE_STAGES={'unavailable','identity_pending'}
MAX_LEADS_PER_CALL=8


def lead_key(meta):
    """Ledger identity that survives the registry re-keying a lead to its canonical name.

    The registry keys a lead by `literature:<hash>` until its name is known and then by that
    name. Keying follow-up history by the registry key therefore orphans every earlier attempt
    and makes a drained backlog look untouched. Key by the source instead: the name, the paper
    it came from, and the fingerprint of the text the lead was extracted from.
    """
    origin=meta.get('literature_lead') or {};paper=origin.get('paper') or {}
    seed=json.dumps([(meta.get('literature_name') or '').casefold(),
                     paper.get('doi') or paper.get('pmid') or paper.get('title') or '',
                     origin.get('text_sha256') or ''],sort_keys=True)
    return 'lead:'+hashlib.sha256(seed.encode()).hexdigest()[:20]


def sibling_organism(registry,accession,meta):
    """A taxon for this lead already recorded on another entry for the same enzyme.

    Resolution stalls on 'ambiguous identity' or 'retrieve a source linking this name to
    a taxon', yet the run often already holds the organism on a separate registry entry
    carrying the same name. Both facts sat in the same registry, unjoined, so every retry
    re-asked the same under-specified question. Only an unambiguous match is used: two
    entries disagreeing about the taxon is exactly the conflict that must stay unresolved.
    """
    name=(meta.get('literature_name') or '').casefold()
    if not name:return None
    found={}
    for other,entry in registry.items():
        if other==accession:continue
        data=entry.get('metadata') or {}
        organism=data.get('organism') or ((data.get('evaluation_data') or {}).get('organism'))
        if not organism:continue
        for label in (data.get('literature_name'),data.get('protein_name'),other):
            if label and str(label).casefold()==name:
                found[str(organism).strip().casefold()]=str(organism).strip()
                break
    return next(iter(found.values())) if len(found)==1 else None


def service_leads(run_dir,limit=None,lead_ids=None):
    """Give newly discovered and older leads bounded, source-grounded follow-through.

    ``limit=None`` scales the batch to the outstanding backlog. Servicing a lead makes no LLM
    call, so a slow fixed cadence buys no tokens -- it only strands leads that the run already
    paid to discover.
    """
    from enzyme_discovery_agent.tools.identity_resolution import resolve_literature_lead
    from enzyme_discovery_agent.tools.literature import search_pubmed
    from enzyme_discovery_agent.tools.research_decisions import assess_next_batch
    root=Path(run_dir);excluded=reconcile_directions(root)
    registry=load_candidate_registry(root,strict=True)['candidates']
    ledger_path=root/'agent_state/lead_followup.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    pending=[]
    for accession,entry in registry.items():
        if lead_ids is not None and accession not in lead_ids:continue
        m=entry.get('metadata') or {}
        if not m.get('is_literature_lead') or discovery_status(entry)=='rejected' or m.get('identity_resolution_status')=='resolved':continue
        extra=[]
        for folder in (root/'artifacts/literature',root/'artifacts/identity_followup'):
            for path in folder.rglob('*.json'):
                if path.parent.name==accession.replace(':','_'):continue
                content=path.read_bytes()
                if re.search(r'(?<!\w)'+re.escape(m['literature_name'])+r'(?!\w)',content.decode(errors='replace'),re.I):
                    extra.append((str(path),hashlib.sha256(content).hexdigest()))
        # The taxon and any deposited identifier are what resolution actually turns on, and
        # they often arrive after the first attempt. Including them means a lead reopens with
        # a full budget the moment the thing it was waiting for exists, instead of staying
        # 'pending: retrieve a source linking this name to a taxon' while that taxon sits in
        # its own metadata.
        hints=[m.get('organism') or sibling_organism(registry,accession,m),
               (m.get('literature_lead') or {}).get('deposited_identifier')]
        basis=hashlib.sha256(json.dumps({'source':m.get('literature_lead'),
            'retrieved_sources':sorted(extra),'identity_hints':hints,
            'implementation':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()
        previous=ledger.get(lead_key(m),{})
        if previous.get('basis')==basis and (previous.get('stage') not in RETRYABLE_STAGES or
                previous.get('attempts',0)>=LEAD_ATTEMPTS or time.time()<previous.get('retry_after',0)):continue
        pending.append((accession,entry,basis,extra))
    pending.sort(key=lambda row:row[1].get('first_seen',''))
    ordered=[]
    while pending:
        ordered.append(pending.pop())
        if pending:ordered.append(pending.pop(0))
    resolved=[];attempted=[]
    if limit is None:
        limit=max(2,min(MAX_LEADS_PER_CALL,-(-len(ordered)//4)))
    for accession,entry,basis,extra in ordered[:limit]:
        meta=entry['metadata'];folder=root/'artifacts/identity_followup'/accession.replace(':','_');folder.mkdir(parents=True,exist_ok=True)
        taxon=meta.get('organism') or sibling_organism(registry,accession,meta)
        if taxon and not meta.get('organism'):
            # Give the resolver the taxon it keeps asking for, recorded as a derived link
            # rather than as an observation: it narrows the lookup, it is not evidence.
            update_candidate_registry(root,accession,None,{'organism':taxon,
                'organism_basis':'Taxon carried from another registry entry for the same enzyme name; narrows identity lookup only.'})
            meta={**meta,'organism':taxon}
        attempts=[' '.join(filter(None,[meta['literature_name'],taxon,'enzyme'])) if taxon else None,
                  meta['literature_name']+' enzyme characterization',
                  meta['literature_name']]
        result=None
        for query in [q for q in attempts if q]:
            result=search_pubmed(query,max_results=8)
            if result.get('status') not in {'success','partial'} or result.get('data'):
                break
        path=folder/(basis+'.json');atomic_write_json(path,{'tool_name':'search_pubmed','args':{'query':query},'result':result})
        # JSON retains the exact retrieved title/abstract and its source identifiers.
        try:
            outcome=resolve_literature_lead(accession,source_paths=list(dict.fromkeys([str(path)]+[p for p,_ in extra])),run_dir=root)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            outcome={'status':'partial','summary':'Identity source is unavailable; retain the lead for retry.',
                     'data':{'status':'unavailable','reason':str(exc)}}
        identity=outcome.get('data') or {};canonical=identity.get('resolved_accession')
        key=lead_key(meta);previous=ledger.get(key,{})
        ledger[key]={'basis':basis,'source':str(path),'identity_outcome':outcome,'registry_key':accession,
            'literature_name':meta.get('literature_name'),
            'stage':'unavailable' if identity.get('status')=='unavailable' else 'identity_pending',
            'needs':identity.get('reason') or outcome.get('summary'),
            'attempts':previous.get('attempts',0)+1 if previous.get('basis')==basis else 1,'retry_after':time.time()+300}
        if canonical:
            assessment=assess_next_batch(root,llm_client=None,limit=1,accessions=[canonical])
            ledger[key].update(stage='assessed',assessment=assessment);resolved.append(canonical)
            update_candidate_registry(root,accession,None,{'target_relevance_status':'linked_sequence_assessed'})
        atomic_write_json(ledger_path,ledger);attempted.append(accession)
    write_coverage(root)
    return {'attempted':attempted,'resolved':resolved,'excluded_opposite_direction':excluded}


def write_coverage(root):
    registry=load_candidate_registry(Path(root),strict=True)['candidates'];rows=[]
    for accession,entry in registry.items():
        meta=entry.get('metadata') or {};data={**(meta.get('evaluation_data') or {}),**meta}
        rows.append({'candidate_id':accession,'name':entry.get('enzyme_name') or data.get('protein_name'),
            'status':discovery_status(entry),'relevance_status':meta.get('target_relevance_status','not_yet_assessed'),
            'resolved_accession':meta.get('resolved_accession',''),'native_substrates_available':bool(data.get('native_substrates')),
            'priority':(meta.get('priority_assessment') or {}).get('decision',''),
            'rejection_reason':meta.get('rejection_reason','')})
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]) if rows else ['candidate_id']);writer.writeheader();writer.writerows(rows)
    atomic_write_text(Path(root)/'candidate_coverage.csv',out.getvalue())
