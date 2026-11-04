"""Reproducible, CPU-only E. coli expression triage; no expression probability claim.

Run ``python -m enzyme_design_agent.expression_screen --help`` for inputs.
Native CDSs are recovered through UniProt's EMBL protein cross-references and
NCBI coded_by coordinates, then validated against the exact requested protein.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import html
import io
import json
import os
import re
import subprocess
from pathlib import Path
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import Bio
from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition, Location
from Bio.SeqUtils import CodonAdaptationIndex
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis

BL21_ACCESSION = 'CP001509.3'
BL21_URL = f'https://www.ebi.ac.uk/ena/browser/api/embl/{BL21_ACCESSION}'
CANONICAL_AA = set('ACDEFGHIKLMNPQRSTVWY')
LIMITATIONS = [
    'This is a sequence-feature screen, not a calibrated probability of expression or soluble yield.',
    'CAI and low-adaptiveness codons describe the supplied/native CDS, not a synonymous optimized construct.',
    'CAI reference is BL21(DE3) ribosomal-protein CDSs; no universal CAI pass/fail threshold is applied.',
    'Amino-acid hydropathy windows are a heuristic; they are not a transmembrane, signal-peptide or aggregation predictor.',
    'Promoter, vector, RBS, tags, 5-prime UTR, induction and chaperones are unspecified; initiation accessibility is not assessed.',
    'Missing or translation-mismatched DNA is unassessed, never counted as a favorable result.',
    'Native AS9 DNA is a sequence comparator; the DNA construct used in the activity assay is not established here.',
]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class CachedHTTP:
    """URL-addressed raw snapshots; globally limit NCBI calls to two per second."""
    def __init__(self, cache, offline=False):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.offline=offline;self.lock=threading.Lock();self.last_ncbi=0.

    def get(self, url):
        path=self.cache/(sha(url)+'.txt')
        if path.exists():return path.read_text()
        if self.offline:raise RuntimeError('Not cached in offline mode: '+url)
        last=None
        for attempt in range(3):
            try:
                if 'eutils.ncbi.nlm.nih.gov' in url:
                    with self.lock:
                        time.sleep(max(0,.51-(time.monotonic()-self.last_ncbi)))
                        self.last_ncbi=time.monotonic()
                req=Request(url,headers={'User-Agent':'EnzymeAgent-expression-screen/1.0'})
                with urlopen(req,timeout=35) as response:text=response.read().decode()
                if not text.strip() or text.lstrip().startswith('<!DOCTYPE html'):raise ValueError('Empty or HTML response')
                tmp=path.with_suffix('.'+str(threading.get_ident())+'.tmp');tmp.write_text(text);tmp.replace(path)
                path.with_suffix('.json').write_text(json.dumps(dict(url=url,sha256=sha(text),retrieved_utc=datetime.now(timezone.utc).isoformat()),indent=2))
                return text
            except Exception as exc:
                last=exc
                if attempt<2:time.sleep(attempt+1)
        raise RuntimeError(f'{url}: {last}')


def verify_cds(dna, protein, table=11):
    """Accept bacterial initiating Met and optional terminal stop; reject edits/frame errors."""
    dna=''.join(str(dna).upper().split()).replace('U','T')
    protein=''.join(protein.upper().split())
    if not dna or set(dna)-set('ACGT'):raise ValueError('DNA is empty or contains ambiguous/noncanonical bases')
    if len(dna)%3:raise ValueError('CDS length is not divisible by three')
    translated=str(Seq(dna).translate(table=table))
    terminal_stop=translated.endswith('*')
    translated=translated[:-1] if terminal_stop else translated
    if '*' in translated:raise ValueError('Internal stop codon')
    start=dna[:3];alternate_start=False
    if translated!=protein and protein.startswith('M') and start in CodonTable.unambiguous_dna_by_id[table].start_codons:
        candidate='M'+translated[1:]
        if candidate==protein:translated=candidate;alternate_start=start!='ATG'
    if translated!=protein:raise ValueError(f'CDS translation mismatch: {len(translated)} translated vs {len(protein)} expected amino acids')
    return dict(dna=dna,terminal_stop=terminal_stop,start_codon=start,alternate_initiator=alternate_start,
                translation_table=table,translation_matches=True,sequence_sha256=sha(protein),cds_sha256=sha(dna))


def uniprot_entries(accessions, http):
    output={}
    for start in range(0,len(accessions),80):
        query=' OR '.join('accession:'+x for x in accessions[start:start+80])
        url='https://rest.uniprot.org/uniprotkb/stream?'+urlencode(dict(query=query,format='json'))
        for x in json.loads(http.get(url))['results']:output[x['primaryAccession']]=x
    return output


def protein_accessions(entry):
    ids=[]
    for ref in entry.get('uniProtKBCrossReferences',[]):
        if ref['database']!='EMBL':continue
        props={p['key']:p['value'] for p in ref.get('properties',[])}
        ident=props.get('ProteinId')
        if ident and ident!='-' and ident not in ids:ids.append(ident)
    return ids


class CDSResolver:
    def __init__(self, http, entries):
        self.http=http;self.entries=entries;self.proteins={}

    def prefetch(self):
        ids=sorted(set(p for e in self.entries.values() for p in protein_accessions(e)[:1]))
        if not ids:return
        url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+urlencode(dict(db='protein',id=','.join(ids),rettype='gp',retmode='text',tool='EnzymeAgentExpressionScreen'))
        self.proteins.update({r.id:r for r in SeqIO.parse(io.StringIO(self.http.get(url)),'genbank')})

    def recover(self, ident, protein):
        entry=self.entries.get(ident)
        if entry is None:raise ValueError('No UniProt entry linked to input identifier')
        if entry['sequence']['value']!=protein:raise ValueError('Current UniProt protein differs from requested sequence')
        errors=[]
        for pid in protein_accessions(entry):
            try:
                record=self.proteins.get(pid)
                if record is None:
                    url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+urlencode(dict(db='protein',id=pid,rettype='gp',retmode='text',tool='EnzymeAgentExpressionScreen'))
                    record=SeqIO.read(io.StringIO(self.http.get(url)),'genbank')
                if str(record.seq)!=protein:raise ValueError('Linked protein translation differs from input')
                features=[f for f in record.features if f.type=='CDS' and 'coded_by' in f.qualifiers]
                if len(features)!=1:raise ValueError('No unique coded_by feature')
                q=features[0].qualifiers;coded_by=q['coded_by'][0]
                location=Location.fromstring(coded_by)
                parts=[];urls=[]
                for part in location.parts:
                    if not part.ref or not isinstance(part.start,ExactPosition) or not isinstance(part.end,ExactPosition):
                        raise ValueError('Partial or unresolved CDS coordinates')
                    url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+urlencode(dict(
                        db='nuccore',id=part.ref,rettype='fasta',retmode='text',seq_start=int(part.start)+1,
                        seq_stop=int(part.end),strand=1,tool='EnzymeAgentExpressionScreen'))
                    dna=SeqIO.read(io.StringIO(self.http.get(url)),'fasta').seq
                    if len(dna)!=len(part):raise ValueError('Retrieved nucleotide range has wrong length')
                    parts.append(str(dna.reverse_complement() if part.strand==-1 else dna));urls.append(url)
                dna=''.join(parts)[int(q.get('codon_start',['1'])[0])-1:]
                verified=verify_cds(dna,protein,int(q.get('transl_table',['11'])[0]))
                verified.update(source='UniProt EMBL cross-reference -> NCBI coded_by -> native nucleotide ranges',
                    linked_protein=pid,coded_by=coded_by,source_urls=urls)
                return verified
            except Exception as exc:errors.append(f'{pid}: {exc}')
        raise ValueError('; '.join(errors) if errors else 'No EMBL coding-sequence links')


def host_reference(http, reference_path=None, reference_format='embl'):
    text=Path(reference_path).read_text() if reference_path else http.get(BL21_URL)
    record=SeqIO.read(io.StringIO(text),reference_format)
    cds=[];names=[]
    for f in record.features:
        if f.type!='CDS' or 'pseudo' in f.qualifiers or 'pseudogene' in f.qualifiers:continue
        product=' '.join(f.qualifiers.get('product',[])).lower()
        # Restrict to actual 30S/50S structural ribosomal proteins, not modifiers.
        if not re.fullmatch(r'(?:30s|50s) ribosomal (?:subunit )?protein [sl]\d+(?:/[sl]\d+)?(?: type [ab])?(?:;.*)?',product):continue
        dna=str(f.extract(record.seq)).upper()
        if len(dna)%3 or set(dna)-set('ACGT'):continue
        cds.append(dna);names.append(f.qualifiers.get('locus_tag',['unknown'])[0])
    if len(cds)<20:raise ValueError('Insufficient ribosomal-protein CDSs in host reference')
    cai=CodonAdaptationIndex(cds,table=CodonTable.unambiguous_dna_by_id[11])
    sense=CodonTable.unambiguous_dna_by_id[11].forward_table
    low=sorted(c for c in sense if cai[c]<.1)
    return cai,dict(host=record.description,accession=record.id,reference_sha256=sha(text),
        cai_reference='30S/50S structural ribosomal-protein CDSs from the host genome',
        reference_genes=names,reference_CDS_count=len(cds),low_adaptiveness_definition='CAI codon relative adaptiveness < 0.1 in this reference',
        low_adaptiveness_codons=low,codon_relative_adaptiveness=dict(cai))


def protein_features(protein):
    if not protein or set(protein)-CANONICAL_AA:raise ValueError('Protein contains noncanonical or ambiguous amino acids')
    analysis=ProteinAnalysis(protein)
    kd=analysis.protein_scale(ProtParamData.kd,19) if len(protein)>=19 else []
    maximum=max(kd) if kd else None
    return dict(protein_length=len(protein),molecular_weight_kDa=analysis.molecular_weight()/1000,
        theoretical_pI=analysis.isoelectric_point(),GRAVY=analysis.gravy(),
        maximum_19aa_Kyte_Doolittle=maximum,
        hydrophobic_window_flag=maximum is not None and maximum>1.6,
        hydrophobic_window_definition='Any 19-aa mean Kyte-Doolittle > 1.6; heuristic, not a membrane/solubility prediction',
        cysteine_count=protein.count('C'),sequence_sha256=sha(protein))


def dna_features(verified, cai, reference):
    dna=verified['dna'];coding=dna[:-3] if verified['terminal_stop'] else dna
    codons=[coding[i:i+3] for i in range(0,len(coding),3)]
    low=set(reference['low_adaptiveness_codons'])
    hits=[i+1 for i,c in enumerate(codons) if c in low]
    window_counts=[sum(c in low for c in codons[i:i+10]) for i in range(max(1,len(codons)-9))]
    return dict(CAI=float(cai.calculate(coding)),CDS_GC_pct=100*sum(c in 'GC' for c in coding)/len(coding),
        GC3_pct=100*sum(c[2] in 'GC' for c in codons)/len(codons),
        low_adaptiveness_codon_count=len(hits),low_adaptiveness_codon_pct=100*len(hits)/len(codons),
        low_adaptiveness_in_first_30_codons=sum(p<=30 for p in hits),
        maximum_low_adaptiveness_in_10_codons=max(window_counts),
        low_adaptiveness_positions=hits,low_adaptiveness_codon_counts=dict(Counter(c for c in codons if c in low)),
        start_codon=verified['start_codon'],terminal_stop=verified['terminal_stop'],
        translation_matches=True,cds_sha256=verified['cds_sha256'])


def screen_one(record, cai, reference, resolver):
    ident=record['id'];protein=record['sequence'].upper()
    row=dict(id=ident,arms='+'.join(record.get('arms',[])),organism=record.get('organism',''),
        **protein_features(protein),dna_status='unavailable',dna_error='',CAI=None,CDS_GC_pct=None,GC3_pct=None,
        low_adaptiveness_codon_count=None,low_adaptiveness_codon_pct=None,
        low_adaptiveness_in_first_30_codons=None,maximum_low_adaptiveness_in_10_codons=None)
    verified=None
    try:
        if record.get('cds'):
            verified=verify_cds(record['cds'],protein,record.get('translation_table',11))
            verified.update(source='User-supplied CDS; exact protein translation checked')
        else:verified=resolver.recover(ident,protein)
        row.update(dna_features(verified,cai,reference));row['dna_status']='verified'
    except Exception as exc:row['dna_error']=str(exc)
    return row,verified


def add_netsolp(records, rows, out, directory, python, threads=2):
    """One batched invocation of the published protein predictor, explicitly CPU-only."""
    directory=Path(directory).expanduser().resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    script=directory/'predict.py';models=directory/'models'
    required=[models/f'{name}_ESM1b_distilled_quantized.onnx' for name in ['Solubility','Usability']]
    provenance=dict(method='NetSolP-1.0 Distilled',citation='https://doi.org/10.1093/bioinformatics/btab801',
        backend='CPU-only ONNX; authors predict.py',threads=threads,status='unavailable')
    for row in rows:row.update(NetSolP_solubility=None,NetSolP_usability=None,NetSolP_note='Not assessed')
    if not script.exists() or not all(p.exists() for p in required):
        provenance['reason']='Installed predictor script/models unavailable';return provenance
    if any(len(r['sequence'])>1022 for r in records):
        provenance['reason']='NetSolP truncates sequences above 1022 aa; this wrapper declines truncated predictions';return provenance
    fasta=out/'netsolp_input.fasta';fasta.write_text(''.join(f">{r['id']}\n{r['sequence']}\n" for r in records))
    output=out/'netsolp_predictions.csv'
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]=str(threads)
    command=[str(python),str(Path(__file__).with_name('netsolp_cpu_worker.py')),str(script),
        '--FASTA_PATH',str(fasta),'--OUTPUT_PATH',str(output),'--MODELS_PATH',str(models),
        '--NUM_THREADS',str(threads),'--MODEL_TYPE','Distilled','--PREDICTION_TYPE','SU']
    try:
        with (out/'netsolp.log').open('w') as log:
            subprocess.run(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True)
        predictions=list(csv.DictReader(output.open()))
        by_id={p['sid']:p for p in predictions}
        if len(predictions)!=len(records) or set(by_id)!={r['id'] for r in records}:raise ValueError('NetSolP returned missing/duplicate/unexpected IDs')
        sequences={r['id']:r['sequence'] for r in records}
        values={}
        for ident,prediction in by_id.items():
            if prediction.get('fasta')!=sequences[ident]:raise ValueError('NetSolP output protein mismatch')
            scores=[float(prediction[key]) for key in ['predicted_solubility','predicted_usability']]
            if not all(0<=v<=1 for v in scores):raise ValueError('Invalid predictor score')
            values[ident]=scores
        for row in rows:
            row.update(NetSolP_solubility=values[row['id']][0],NetSolP_usability=values[row['id']][1],
                NetSolP_note='Published predictor score; not a measured expression yield')
        provenance.update(status='complete',input_sha256=sha(fasta.read_text()),
            predictor_code_sha256=sha(script.read_text()),
            model_sha256={p.name:file_sha(p) for p in required})
    except Exception as exc:provenance.update(status='failed',reason=str(exc))
    (out/'netsolp_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def write_outputs(out,rows,verified,reference,predictor=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'candidate_expression_features.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    (out/'native_CDS.fasta').write_text(''.join(f'>{ident}\n{x["dna"]}\n' for ident,x in sorted(verified.items())))
    (out/'CDS_provenance.json').write_text(json.dumps(verified,indent=2)+'\n')
    arms={}
    for arm in sorted(set(a for r in rows for a in r['arms'].split('+') if a)):
        group=[r for r in rows if arm in r['arms'].split('+')];dna=[r for r in group if r['dna_status']=='verified']
        arms[arm]=dict(candidates=len(group),verified_DNA=len(dna),hydrophobic_window_flags=sum(r['hydrophobic_window_flag'] for r in group))
        for key in ['CAI','CDS_GC_pct','GC3_pct','low_adaptiveness_codon_pct','NetSolP_solubility','NetSolP_usability']:
            values=sorted(r[key] for r in group if r.get(key) is not None)
            arms[arm][key]=dict(mean=sum(values)/len(values),minimum=min(values),maximum=max(values)) if values else None
    summary=dict(completed_utc=datetime.now(timezone.utc).isoformat(),candidates=len(rows),
        verified_DNA=len(verified),host_reference=reference,arms=arms,limitations=LIMITATIONS,
        protein_predictor=predictor,code_sha256=sha(Path(__file__).read_text()),biopython_version=Bio.__version__)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    cols=['id','arms','organism','dna_status','CAI','CDS_GC_pct','low_adaptiveness_codon_pct','maximum_low_adaptiveness_in_10_codons','NetSolP_solubility','NetSolP_usability','GRAVY']
    def fmt(v):return 'not assessed' if v is None else f'{v:.3f}' if isinstance(v,float) else html.escape(str(v))
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><title>E. coli expression sequence screen</title><style>body{font:15px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:15px}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}</style>'
    doc+='<h1>E. coli expression: sequence-feature screen</h1><p>Native DNA verified for '+str(len(verified))+'/'+str(len(rows))+' proteins. Host reference: '+html.escape(reference['host'])+'.</p>'
    doc+='<p>CAI, GC and low-adaptiveness codons describe native DNA; they do not establish expression success. Low-adaptiveness codons are defined from ribosomal-gene preferences, not measured tRNA supply. Protein predictor scores, when available, are separate from codon metrics.</p>'
    doc+='<p><a href="codon_comparison.png">Codon compatibility comparison</a> · <a href="codon_preferences.png">Synonymous codon preferences</a></p>'
    doc+='<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'
    doc+=''.join('<tr>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in cols)+'</tr>' for r in rows)+'</table>'
    doc+='<h2>Interpretation limits</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in LIMITATIONS)+'</ul><p><a href="candidate_expression_features.csv">Features CSV</a> · <a href="native_CDS.fasta">Verified native CDS</a> · <a href="CDS_provenance.json">DNA provenance</a> · <a href="summary.json">Reference and settings</a></p></html>'
    (out/'report.html').write_text(doc)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-json',type=Path,help='List of records with id, sequence; optional arms, organism, cds, translation_table')
    inputs.add_argument('--uniprot',nargs='+',help='UniProt accessions; retrieves protein and native CDS links')
    inputs.add_argument('--protein-fasta',type=Path,help='Protein FASTA; IDs matching UniProt accessions can retrieve native DNA')
    p.add_argument('--cds-fasta',type=Path,help='Optional supplied CDS FASTA, matched by input ID')
    p.add_argument('--uniprot-json',type=Path,help='Optional cached UniProt stream JSON, containing results')
    p.add_argument('--protein-genbank',type=Path,help='Optional cached GenPept records with coded_by features')
    p.add_argument('--reference-embl',type=Path,help='Host reference EMBL genome; defaults to BL21(DE3) CP001509.3')
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--offline',action='store_true')
    p.add_argument('--workers',type=int,default=3,choices=range(1,5))
    p.add_argument('--netsolp-dir',type=Path,help='Optional installed NetSolP PredictionServer directory; adds CPU-only batch protein predictions')
    p.add_argument('--netsolp-python',type=Path,default=Path(os.sys.executable),help='Python environment with torch and onnxruntime')
    a=p.parse_args(argv);http=CachedHTTP(a.cache,a.offline)
    if a.input_json:records=json.loads(a.input_json.read_text())
    elif a.protein_fasta:
        records=[dict(id=(r.id.split('|')[1] if r.id.startswith(('sp|','tr|')) else r.id),sequence=str(r.seq)) for r in SeqIO.parse(a.protein_fasta,'fasta')]
    else:records=[]
    entries={x['primaryAccession']:x for x in json.loads(a.uniprot_json.read_text())['results']} if a.uniprot_json else {}
    ids=a.uniprot or [r['id'] for r in records if not r.get('cds')]
    missing=[ident for ident in ids if ident not in entries]
    if missing:
        # Arbitrary FASTA IDs are legal: retrieval failure must not prevent protein assessment.
        try:entries.update(uniprot_entries(missing,http))
        except Exception as exc:print('UniProt retrieval incomplete:',exc,flush=True)
    if a.uniprot:
        absent=[i for i in a.uniprot if i not in entries]
        if absent:raise ValueError('UniProt accessions could not be retrieved: '+','.join(absent))
        records=[dict(id=i,sequence=entries[i]['sequence']['value'],organism=entries[i]['organism']['scientificName']) for i in a.uniprot]
    if not records or len({r['id'] for r in records})!=len(records):raise ValueError('Inputs must be nonempty with unique IDs')
    if a.cds_fasta:
        supplied_records=list(SeqIO.parse(a.cds_fasta,'fasta'))
        supplied={r.id:str(r.seq) for r in supplied_records}
        if len(supplied)!=len(supplied_records):raise ValueError('Supplied CDS FASTA has duplicate IDs')
        if set(supplied)-{r['id'] for r in records}:raise ValueError('Supplied CDS IDs do not match input protein IDs')
        for r in records:
            if r['id'] in supplied:r['cds']=supplied[r['id']]
    cai,reference=host_reference(http,a.reference_embl)
    resolver=CDSResolver(http,entries)
    if a.protein_genbank:resolver.proteins.update({r.id:r for r in SeqIO.parse(a.protein_genbank,'genbank')})
    else:
        try:resolver.prefetch()
        except Exception as exc:print('Protein prefetch incomplete; trying individual links:',exc,flush=True)
    rows={};verified={}
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        tasks={executor.submit(screen_one,r,cai,reference,resolver):r['id'] for r in records}
        for future in as_completed(tasks):
            row,dna=future.result();ident=row['id'];rows[ident]=row
            if dna:verified[ident]=dna
            print(f"{len(rows)}/{len(records)} {ident}: DNA {row['dna_status']}",flush=True)
    ordered=[rows[r['id']] for r in records]
    predictor=add_netsolp(records,ordered,a.out,a.netsolp_dir,a.netsolp_python) if a.netsolp_dir else None
    summary=write_outputs(a.out,ordered,verified,reference,predictor)
    print(json.dumps(dict(candidates=summary['candidates'],verified_DNA=summary['verified_DNA'],arms=summary['arms'],report=str(a.out/'report.html')),indent=2))


if __name__=='__main__':main()
