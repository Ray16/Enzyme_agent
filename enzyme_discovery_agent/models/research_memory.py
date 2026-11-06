"""Retrieve campaign-local research history without model-written summaries."""

from enzyme_discovery_agent.models.candidate_evidence import candidate_record
import json
import re
from pathlib import Path

from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.run_layout import run_file


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def memory_records(run_dir, topic):
    """Newest records first; source locators refer only to this campaign."""
    root = Path(run_dir)
    if topic == "searches":
        records, seen = [], set()
        for event in reversed(read_events(root)):
            name = str(event.get("tool") or "")
            if event.get("type") != "tool_call" or not (
                    name.startswith(("search_", "fetch_"))
                    or name in {"follow_paper_citations", "deep_literature_search", "extract_reaction_mechanism"}):
                continue
            args = event.get("args") or {}
            key = json.dumps([name, args], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            records.append({"tool": name, "arguments": args, "status": event.get("status"),
                            "count": event.get("count"), "iteration": event.get("iteration"),
                            "source": str(run_file(root, "events.jsonl").relative_to(root)),
                            "event_sequence": event.get("seq")})
        return records
    if topic == "leads":
        papers = _read(root / "artifacts/fulltext/papers_enzymes.json", [])
        records, seen, resolved = [], set(), set()
        for paper in papers:
            for lead in paper.get("enzymes", []) + paper.get("homologs", []):
                if lead.get("accessions"):
                    resolved.add((lead.get("name"), lead.get("organism"), paper.get("doi")))
        for paper in reversed(papers):
            for lead in paper.get("enzymes", []) + paper.get("homologs", []):
                key = (lead.get("name"), lead.get("organism"), paper.get("doi"))
                if not lead.get("name") or key in seen or key in resolved:
                    continue
                seen.add(key)
                records.append({**lead, "doi": paper.get("doi"),
                                "source": "artifacts/fulltext/papers_enzymes.json",
                                "evidence_status": "unresolved identity; activity unverified"})
        return records
    if topic == "notes":
        notes = _read(run_file(root, "notebook.json"), {})
        records = []
        for section, content in (notes.get("sections") or {}).items():
            text = str(content)
            for start in reversed(range(0, len(text), 1600)):
                records.append({"section": section, "start_character": start,
                                "text": text[start:start + 1600],
                                "source": str(run_file(root, "notebook.json").relative_to(root)),
                                "evidence_status": "agent notes; not primary evidence"})
        return records
    if topic == "decisions":
        decisions = _read(run_file(root, "family_decisions.json"), {}).get("decisions", {})
        return [{"family": family, **value,
                 "source": str(run_file(root, "family_decisions.json").relative_to(root)),
                 "evidence_status": "revisable agent decision; not primary evidence"}
                for family, value in sorted(decisions.items(), key=lambda item: item[1].get("ts", ""), reverse=True)]
    if topic == 'reasoning':
        from enzyme_discovery_agent.models.reasoning_trace import read_scientific_trace
        from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
        from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
        records=[]
        registry=load_candidate_registry(root,strict=True)['candidates']
        journal_sources=set()
        for event in reversed(read_scientific_trace(root)):
            value=event['record']
            interpretation=value.get('interpretation')
            artifact=(value.get('source') or {}).get('path')
            if isinstance(interpretation,dict) and artifact:
                journal_sources.add(artifact)
            metadata=(registry.get(value.get('accession')) or {}).get('metadata') or {}
            current=artifact in {metadata.get('reaction_role_artifact'),metadata.get('reaction_step_artifact')} if artifact else False
            current_data=candidate_record(metadata)
            if event['type'] == 'target_assessment_recorded':
                from enzyme_discovery_agent.models.target_review import assessment_receipt
                assessment = {key: item for key, item in value.items() if key != 'source'}
                receipt = assessment_receipt(root, value.get('accession'), metadata,
                                             assessment, artifact)
                records.append({'kind': event['type'], 'id': event['id'],
                    'accession': value.get('accession'), 'protein_name': receipt.get('protein_name'),
                    'assessment_receipt': receipt, 'source': artifact,
                    'event_sequence': event['sequence'], 'event_sha256': event['event_sha256'],
                    'evidence_status': 'Saved target interpretation; inspect freshness before reuse.'})
                continue
            records.append({'kind':event['type'],'id':event['id'],
                **{key:value.get(key) for key in ('id','decision_id','scope','hypothesis','alternative','prediction',
                    'contrasting_prediction','revision_trigger','tool_name','evidence_refs','revises',
                    'belief_update','outcome','conclusion','result_source','evidence_snapshots') if value.get(key) is not None},
                **({'accession':value.get('accession'),'interpretation':interpretation,
                    'artifact_source':value.get('source'),'current_interpretation':current,
                    **({'participant_roles':role_transfer_summary(current_data)} if current else {})}
                    if isinstance(interpretation,dict) else {}),
                'source':'reasoning/trace.jsonl','event_sequence':event['sequence'],
                'event_sha256':event['event_sha256'], 'capture_mode':event['capture_mode'],
                'evidence_status':'Verified journal integrity; interpretations remain revisable and are not primary evidence.'})
        # Current interpretations remain available even when originally recorded
        # outside a controller-mediated prediction transaction.
        for accession,entry in registry.items():
            metadata=entry.get('metadata') or {}
            data=candidate_record(metadata)
            if data.get('reaction_roles') or data.get('reaction_step'):
                artifact=metadata.get('reaction_step_artifact') or metadata.get('reaction_role_artifact')
                if artifact in journal_sources:
                    continue
                records.append({'kind':'participant_interpretation','accession':accession,
                    'participant_roles':role_transfer_summary(data),
                    'source':artifact,
                    'evidence_status':'Current candidate interpretation; inspect source and freshness before reuse.'})
        return records
    raise ValueError("Unknown research memory topic")


def recall_memory(run_dir, topic, query="", offset=0, limit=10):
    """Return a bounded, paginated projection while retaining full disk records."""
    if offset < 0 or not 1 <= limit <= 20:
        raise ValueError("offset must be nonnegative and limit must be between 1 and 20")
    records = memory_records(run_dir, topic)
    if query.strip():
        needle = query.casefold().strip()
        texts=[json.dumps(row,ensure_ascii=False).casefold() for row in records]
        exact=[row for row,text in zip(records,texts) if needle in text]
        if exact:
            records=exact
        else:
            terms=set(re.findall(r'[\w.-]+',needle)) - {'the','a','an','of','to','for','and','or','in','is'}
            scored=[(sum(term in text for term in terms),index,row)
                    for index,(row,text) in enumerate(zip(records,texts))]
            records=[row for score,_,row in sorted(scored,key=lambda value:(-value[0],value[1])) if score]
    page, used = [], 0
    for row in records[offset:offset + limit]:
        # The complete record remains in its source file; previews must also
        # fit the context budget when arguments contain unusually long text.
        preview = {key: value[:1800] if isinstance(value, str) else value for key, value in row.items()}
        if topic=='reasoning' and isinstance(row.get('interpretation'),dict):
            # Preserve the decision-bearing fields as structured values. Cutting
            # a serialized mapped reaction at its first N characters can lose
            # both the comparison result and the continuation uncertainty.
            value=row['interpretation']
            preview['interpretation']={key:(item[:600]+' [preview]' if isinstance(item,str) and len(item)>600 else item)
                for key,item in value.items() if key in {
                    'interpretation','activation_assessment','continuation_assessment',
                    'local_bond_changes_match','validation_scope','basis_sha256','target_sha256',
                    'native_bond_changes','target_bond_changes'}}
            if value.get('participants'):
                preview['interpretation']['participants']=[{key:(item[:300] if isinstance(item,str) else item)
                    for key,item in participant.items() if key in {
                        'native_smiles','participant_name','role','transferred_unit','site_relation','interpretation'}}
                    for participant in value['participants'][:3]]
                preview['interpretation']['participant_count']=len(value['participants'])
            preview['step_limits']='Local bond changes alone do not establish target activity, activation feasibility or repeated extension.'
            roles=row.get('participant_roles') or {}
            step=roles.get('reaction_step') or {}
            preview['participant_roles']={key:roles.get(key) for key in ('status','review_required','record_status','artifacts','next_question')}
            preview['participant_roles']['reaction_step']={key:step.get(key) for key in (
                'status','local_bond_changes_match','activation_assessment','continuation_assessment')}
            for key in ('activation_assessment','continuation_assessment'):
                if isinstance(preview['participant_roles']['reaction_step'].get(key),str):
                    preview['participant_roles']['reaction_step'][key]=step[key][:400]
            # The exact source quote and full atom mapping stay in this immutable
            # artifact; retrieval returns its locator and verified checksum.
            preview.pop('record_preview',None)
        if "arguments" in preview:
            preview["arguments"] = {key: value[:600] if isinstance(value, str) else value
                                    for key, value in preview["arguments"].items()}
        size = len(json.dumps(preview, ensure_ascii=False))
        if size>5000 and topic=='reasoning':
            def bounded(value,key=''):
                if isinstance(value,dict):
                    return {k:bounded(v,k) for k,v in value.items()}
                if isinstance(value,list):
                    return [bounded(v,key) for v in value[:2]]
                if isinstance(value,str) and len(value)>200 and key not in {'path','source','locator','native_smiles','target_smiles'}:
                    return value[:200]+' [preview; inspect source]'
                return value
            preview=bounded(preview)
            preview['truncated']=True
            size=len(json.dumps(preview,ensure_ascii=False))
        if size > 5000:
            serialized = json.dumps(preview, ensure_ascii=False)
            preview = {"source": row.get("source"), "record_preview": serialized[:4000], "truncated": True}
            while len(json.dumps(preview, ensure_ascii=False)) > 5000:
                preview["record_preview"] = preview["record_preview"][:len(preview["record_preview"]) // 2]
            size = len(json.dumps(preview, ensure_ascii=False))
        if page and used + size > 6000:
            break
        page.append(preview)
        used += size
    end = offset + len(page)
    return {"topic": topic, "records": page, "total": len(records),
            "next_offset": end if end < len(records) else None,
            "authority": "Saved campaign records. A failed lookup or search is not evidence of biological absence."}
