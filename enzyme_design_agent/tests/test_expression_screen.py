"""Biological edge cases and offline reproducibility of expression triage."""
import json
from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import CodonAdaptationIndex

from enzyme_design_agent.expression_screen import (
    CDSResolver, add_netsolp, dna_features, main, protein_features, verify_cds,
)


def test_alternate_bacterial_initiator_and_optional_stop():
    x=verify_cds('GTGGCCTAA','MA')
    assert x['translation_matches'] and x['alternate_initiator']
    assert x['terminal_stop'] and x['start_codon']=='GTG'
    assert not verify_cds('ATGGCC','MA')['terminal_stop']


@pytest.mark.parametrize('dna,protein',[
    ('ATGGC','MA'),('ATGNNNTAA','MX'),('ATGTAGGCCTAA','MA'),('ATGGACTAA','MA'),
])
def test_invalid_or_mismatched_cds_rejected(dna,protein):
    with pytest.raises(ValueError):verify_cds(dna,protein)


class FakeHTTP:
    def __init__(self, sequences):self.sequences=sequences
    def get(self,url):
        from urllib.parse import parse_qs,urlsplit
        args=parse_qs(urlsplit(url).query)
        start=int(args['seq_start'][0]);stop=int(args['seq_stop'][0])
        return '>segment\n'+self.sequences[(start,stop)]+'\n'


def resolver_for(location,sequences):
    entry=dict(sequence=dict(value='MA'),uniProtKBCrossReferences=[dict(database='EMBL',id='TEST',properties=[dict(key='ProteinId',value='PID.1')])])
    r=CDSResolver(FakeHTTP(sequences),{'U1':entry})
    protein=SeqRecord(Seq('MA'),id='PID.1')
    protein.features=[SeqFeature(SimpleLocation(0,2),type='CDS',qualifiers=dict(coded_by=[location],transl_table=['11']))]
    r.proteins['PID.1']=protein
    return r


def test_reverse_strand_cds_uses_native_reverse_complement():
    r=resolver_for('complement(TEST.1:5..13)',{(5,13):'TTAGGCCAT'})
    assert r.recover('U1','MA')['dna']=='ATGGCCTAA'


def test_joined_cds_preserves_part_order():
    r=resolver_for('join(TEST.1:1..3,TEST.1:10..15)',{(1,3):'ATG',(10,15):'GCCTAA'})
    assert r.recover('U1','MA')['dna']=='ATGGCCTAA'


def test_fuzzy_cds_is_not_silently_treated_as_complete():
    r=resolver_for('TEST.1:<1..9',{})
    with pytest.raises(ValueError,match='Partial'):r.recover('U1','MA')


def test_first_30_and_sliding_codon_clusters_have_correct_boundaries():
    codons=['ATG']+['GCC']*34
    for i in [1,2,3,30,31]:codons[i]='GCG'
    verified=verify_cds(''.join(codons)+'TAA','M'+'A'*34)
    cai=CodonAdaptationIndex(['ATG'+'GCC'*100+'GCG'+'TAA'])
    x=dna_features(verified,cai,{'low_adaptiveness_codons':['GCG']})
    assert x['low_adaptiveness_codon_count']==5
    assert x['low_adaptiveness_in_first_30_codons']==3
    assert x['maximum_low_adaptiveness_in_10_codons']==3


def test_protein_features_reject_ambiguous_amino_acids():
    with pytest.raises(ValueError):protein_features('MAX')


def test_missing_protein_predictor_does_not_fabricate_score(tmp_path):
    rows=[{'id':'one'}]
    status=add_netsolp([dict(id='one',sequence='MA')],rows,tmp_path/'out',tmp_path/'missing','python')
    assert status['status']=='unavailable' and rows[0]['NetSolP_solubility'] is None


@pytest.mark.parametrize('returned_sequence,expected_status',[('MA','complete'),('MD','failed')])
def test_protein_predictor_requires_exact_output_sequence(tmp_path,monkeypatch,returned_sequence,expected_status):
    install=tmp_path/'netsolp';models=install/'models';models.mkdir(parents=True)
    (install/'predict.py').write_text('# synthetic test predictor')
    for name in ['Solubility','Usability']:(models/f'{name}_ESM1b_distilled_quantized.onnx').write_bytes(b'test')
    def predict(command,**kwargs):
        target=Path(command[command.index('--OUTPUT_PATH')+1])
        target.write_text('sid,fasta,predicted_solubility,predicted_usability\none,'+returned_sequence+',0.7,0.6\n')
    monkeypatch.setattr('enzyme_design_agent.expression_screen.subprocess.run',predict)
    rows=[dict(id='one')]
    result=add_netsolp([dict(id='one',sequence='MA')],rows,tmp_path/'out',install,'python')
    assert result['status']==expected_status
    assert rows[0]['NetSolP_solubility']==(0.7 if expected_status=='complete' else None)


def test_offline_cli_reports_bad_dna_as_unassessed(tmp_path):
    # A compact synthetic host annotation permits a fully offline end-to-end run.
    from Bio import SeqIO
    host=SeqRecord(Seq('ATGGCCTAA'*20),id='HOST.1',description='Synthetic test reference')
    host.annotations['molecule_type']='DNA'
    host.features=[SeqFeature(SimpleLocation(9*i,9*i+9),type='CDS',qualifiers={'product':[f'30S ribosomal subunit protein S{i+1}'],'locus_tag':[f'rps{i}']}) for i in range(20)]
    ref=tmp_path/'host.embl';SeqIO.write(host,ref,'embl')
    inp=tmp_path/'inputs.json';inp.write_text(json.dumps([
        dict(id='good',sequence='MA',cds='ATGGCCTAA'),
        dict(id='bad',sequence='MA',cds='ATGGACTAA'),
    ]))
    main(['--input-json',str(inp),'--reference-embl',str(ref),'--cache',str(tmp_path/'cache'),'--out',str(tmp_path/'out'),'--offline'])
    result=json.loads((tmp_path/'out/records.json').read_text())
    assert result[0]['dna_status']=='verified' and result[0]['CAI'] is not None
    assert result[1]['dna_status']=='unavailable' and result[1]['CAI'] is None
    assert 'mismatch' in result[1]['dna_error']
