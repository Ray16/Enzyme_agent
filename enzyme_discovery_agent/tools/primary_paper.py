"""Bounded primary-text inspection for campaign-local scientific questions."""
from pathlib import Path
import hashlib

from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.state_store import atomic_write_json


@tool(description='Read open-access primary text to verify protein identity or functional evidence; returns source-hashed passages.',
      category='literature', parallel_safe=True, injected_params=['run_dir'], params={
          'run_dir':{'type':'string'},
          'identifier':{'type':'string','description':'DOI, PMID, PMCID, or exact paper title from a retrieved record.'},
          'query':{'type':'string','default':'','description':'Optional phrase to locate in the primary text, such as a protein identifier or assay term.'},
          'offset_characters':{'type':'integer','default':0,'description':'Absolute text offset for continued inspection. A nonzero offset takes precedence over query location.'},
          'max_characters':{'type':'integer','default':4000,'description':'Bounded excerpt size, from 500 to 8000 characters.'}})
def read_primary_paper(run_dir, identifier, query='', offset_characters=0, max_characters=4000):
    from enzyme_discovery_agent.tools.measurement_extraction import resolve_paper, fetch_jats_or_text, _jats_to_text, _pdf_to_text
    from enzyme_discovery_agent.models.candidate_registry import UNIPROT_ACC_RE
    if offset_characters < 0 or not 500 <= max_characters <= 8000:
        return {'status':'error','summary':'Use a nonnegative offset and 500–8000 excerpt characters.'}
    paper=resolve_paper(identifier)
    content,kind=fetch_jats_or_text(paper)
    text=(_jats_to_text(content) if kind=='xml' else _pdf_to_text(content) if kind=='pdf' else '') if content else ''
    if not text.strip():
        return {'status':'unavailable','summary':'Open-access primary text was not retrieved. This is an access limitation, not evidence of biological absence.',
                'data':{'paper':paper,'text_available':False,'retrieval_kind':kind}}
    digest=hashlib.sha256(text.encode()).hexdigest()
    folder=Path(run_dir)/'artifacts/primary_papers'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(digest+'.txt')
    if not path.exists():
        path.write_text(text)
    atomic_write_json(folder/(digest+'.json'),{'paper':paper,'text_sha256':digest,'text_kind':kind})
    found=text.casefold().find(query.casefold()) if query.strip() else -1
    start=min(offset_characters,len(text)) if offset_characters else max(0,found-500) if found>=0 else 0
    excerpt=text[start:start+max_characters]
    mentions=sorted(set(UNIPROT_ACC_RE.findall(text)))[:30]
    data={'paper':paper,'text_available':True,'text_kind':kind,'text_sha256':digest,
          'source_path':str(path.relative_to(Path(run_dir))),'start_character':start,
          'next_offset':start+len(excerpt) if start+len(excerpt)<len(text) else None,
          'total_characters':len(text),'passage':excerpt,'mentioned_uniprot_accessions':mentions,
          'scope':'A mention is a lead, not a verified paper-to-sequence association or target-activity claim.'}
    return {'status':'success','summary':f'Read {len(excerpt)} characters of primary {kind} text; {len(mentions)} protein-identifier mentions require source-scoped verification.',
            'data':data}
