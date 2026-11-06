"""Batch exact-sequence preparation for shared family protocols, without model calls."""
import hashlib
import json
from pathlib import Path
import re

from enzyme_discovery_agent.state_store import atomic_write_json,state_lock
from .msa import prepare,valid_csv


def prepare_members(handoff,legacy,directory,cache,fetch=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    index_path=directory/'index.json'
    with state_lock(directory/'prepare'):
        candidates={**legacy,**(json.loads(index_path.read_text()) if index_path.exists() else {})}
        needed=[member for group in handoff['families'] for member in group['members']
                if member['accession'] not in candidates]
        if not needed:return candidates
        if fetch is None:
            import requests
            def fetch(accession):
                response=requests.get('https://rest.uniprot.org/uniprotkb/'+accession+'.json',timeout=(15,45))
                response.raise_for_status();return response.json()
        fresh={};errors=[]
        for member in needed:
            accession=member['accession']
            try:
                if not re.fullmatch(r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}',accession):
                    raise ValueError('A source-scoped identity must be supplied explicitly')
                path=directory/(accession+'.json')
                identity=json.loads(path.read_text()) if path.exists() else fetch(accession)
                sequence=identity.get('sequence',{}).get('value','')
                if identity.get('primaryAccession')!=accession or identity.get('inactiveReason') or not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',sequence):
                    raise ValueError('Exact active protein identity is unresolved')
                atomic_write_json(path,identity)
                fresh[accession]={'accession':accession,'sequence':sequence,'evidence':{
                    'identity_source_path':str(path.resolve()),'identity_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'enzyme_name':member['enzyme_name']}}
            except (ValueError,OSError,RuntimeError) as exc:errors.append({'accession':accession,'enzyme_name':member['enzyme_name'],'error':str(exc)})
        if fresh:
            fasta=directory/'batch.fasta';fasta.write_text(''.join('>'+acc+'\n'+candidate['sequence']+'\n' for acc,candidate in fresh.items()))
            alignments=prepare(fasta,directory/'msa',cache)
            for alignment in alignments:fresh[alignment['id']]['evidence']['msa_path']=alignment['csv']
            candidates.update(fresh);atomic_write_json(index_path,candidates)
        atomic_write_json(directory/'pending.json',errors)
        return candidates
