"""Shared protocol compilation, new family members and chemical representation."""
import copy
import csv
import json
from pathlib import Path

import pytest

from structural_modeling_agent import family_cofold as family
from structural_modeling_agent.modeling_reasoning.chemistry import characterize_smiles


@pytest.fixture
def specification():
    acid={'source_smiles':'CC(=O)O','states':{}}
    partner={'source_smiles':'CCN','states':{}}
    intermediates={'acyl_adenylate':'CC(=O)OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O',
                   'acyl_phosphate':'CC(=O)OP(=O)([O-])[O-]'}
    templates={}
    for mode,intermediate in intermediates.items():
        acid['states'][mode]={'1':{'role':'carboxylate_substrate','include_in_model':True,'modeling_smiles':'CC(=O)[O-]'},
                              '2':{'role':mode+'_intermediate','include_in_model':True,'modeling_smiles':intermediate}}
        partner['states'][mode]={'1':{'role':'amine_substrate','include_in_model':False,'modeling_smiles':'CCN'},
                                 '2':{'role':'amine_substrate','include_in_model':True,'modeling_smiles':'CCN'}}
        templates[mode]={'nucleotide_ccd':'ATP','cofactors':[{'ccd':'ATP','count':1,'role':'nucleotide'}],
                         'metal':{'metal':'MG','per_site_metal_count':1 if mode=='acyl_adenylate' else 2,'rationale':'Explicit shared test protocol'}}
    return {'components':[acid,partner],'mechanism_templates':templates,'protein_copies':1,
            'assembly_basis':'Shared test unit','step_rationale':{'1':'First declared step','2':'Second declared step'},
            'assignment_rules':[{'mode':'acyl_adenylate','required':['AMP','diphosphate'],'excluded':['ADP']},
                                {'mode':'acyl_phosphate','required':['ADP'],'excluded':['AMP']}]}


def test_both_steps_share_mechanism_metal_but_not_consumed_nucleotide(specification):
    for mode,metal in [('acyl_adenylate',1),('acyl_phosphate',2)]:
        protocol=family.compile_protocol(mode,specification)
        first,second=(protocol['steps'][str(i)] for i in [1,2])
        assert {c['ccd']:c['count'] for c in first['cofactors']}=={'ATP':1,'MG':metal}
        assert {c['ccd']:c['count'] for c in second['cofactors']}=={'MG':metal}
        assert len(family.modeled_substrates(first))==1
        assert len(family.modeled_substrates(second))==2
        assert protocol['setup_llm_calls']==0


def test_new_members_share_one_protocol_and_conflicting_mechanisms_split(tmp_path,monkeypatch,specification):
    records={identifier:{'metadata':{'interpro_family_id':'SYNTHETIC_FAMILY','catalytic_activities':[reaction]}}
             for identifier,reaction in [('SYNTH_A','ATP = AMP + diphosphate'),('SYNTH_B','ATP = AMP + diphosphate')]}
    monkeypatch.setattr(family,'load_candidate_registry',lambda *a,**k:{'candidates':records})
    def write():
        with (tmp_path/'ranked_candidates.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=['rank','accession','protein_name']);writer.writeheader()
            for index,identifier in enumerate(records,1):writer.writerow({'rank':index,'accession':identifier,'protein_name':'Synthetic catalyst'})
    write();handoff=family.build_handoff(tmp_path,specification)
    assert len(handoff['protocols'])==len(handoff['families'])==1
    assert len(handoff['families'][0]['members'])==2
    records['SYNTH_C']={'metadata':{'interpro_family_id':'SYNTHETIC_FAMILY','catalytic_activities':['ATP = ADP']}}
    write();handoff=family.build_handoff(tmp_path,specification)
    assert len(handoff['protocols'])==len(handoff['families'])==2
    assert sum(len(g['members']) for g in handoff['families'])==3


def test_unknown_family_is_pending_without_a_guessed_protocol(tmp_path,monkeypatch,specification):
    monkeypatch.setattr(family,'load_candidate_registry',lambda *a,**k:{'candidates':{'SYNTH_X':{'metadata':{}}}})
    (tmp_path/'ranked_candidates.csv').write_text('rank,accession,protein_name\n1,SYNTH_X,Unknown synthetic protein\n')
    handoff=family.build_handoff(tmp_path,specification)
    assert not handoff['protocols'] and len(handoff['pending_families'])==1


def test_phosphate_resonance_rule_is_not_an_enzyme_or_substrate_rule():
    ordinary=characterize_smiles('COP(=O)([O-])OCC')
    isotope=characterize_smiles('COP(=[18O])([O-])OCC')
    sulfur=characterize_smiles('COP(=S)([O-])OCC')
    carbon=characterize_smiles('CC(O)CC')
    assert ordinary['stereochemistry']['status']=='not_applicable'
    assert isotope['stereochemistry']['status']=='unspecified'
    assert sulfur['stereochemistry']['status']=='unspecified'
    assert carbon['stereochemistry']['status']=='unspecified'


def test_physical_reuse_key_ignores_labels_but_preserves_required_input_differences():
    from campaign_catalog.lookup import physical_system_key
    original={'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},
        {'ligand':{'id':'B','smiles':'CC(=O)[O-]'}},{'ligand':{'id':'C','ccd':'MG'}}]}
    renamed=copy.deepcopy(original);renamed['sequences'][0]['protein']['id']='X'
    renamed['sequences']=renamed['sequences'][::-1]
    assert physical_system_key(renamed)==physical_system_key(original)
    changed=copy.deepcopy(original);changed['sequences'][1]['ligand']['smiles']='CC(=O)O'
    assert physical_system_key(changed)!=physical_system_key(original)
    changed=copy.deepcopy(original);changed['sequences'][0]['protein']['id']=['A','D']
    assert physical_system_key(changed)!=physical_system_key(original)
    changed=copy.deepcopy(original);changed['sequences'].append({'ligand':{'id':'D','ccd':'MG'}})
    assert physical_system_key(changed)!=physical_system_key(original)


def test_sequence_preparation_batches_only_new_members(tmp_path,monkeypatch):
    from structural_modeling_agent import family_sequences
    handoff={'families':[{'members':[{'accession':'P12345','enzyme_name':'Synthetic enzyme A'},
                                   {'accession':'Q12345','enzyme_name':'Synthetic enzyme B'}]}]}
    fetches=[];batches=[]
    def fetch(accession):fetches.append(accession);return {'primaryAccession':accession,'sequence':{'value':'AGGA'}}
    def prepare(fasta,out,cache):
        batches.append(fasta.read_text())
        return [{'id':accession,'csv':'synthetic-msa'} for accession in fetches]
    monkeypatch.setattr(family_sequences,'prepare',prepare)
    first=family_sequences.prepare_members(handoff,{},tmp_path/'sequence',tmp_path/'cache',fetch)
    assert len(first)==2 and len(batches)==1 and len(fetches)==2
    family_sequences.prepare_members(handoff,{},tmp_path/'sequence',tmp_path/'cache',fetch)
    assert len(batches)==1 and len(fetches)==2


def test_same_job_id_in_an_older_run_is_reused_not_skipped(tmp_path,monkeypatch):
    import hashlib
    import yaml
    old=tmp_path/'old_run';new=tmp_path/'new_run';new.mkdir()
    document={'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},{'ligand':{'id':'B','smiles':'CCO'}}]}
    old.mkdir();old_input=old/'input.yaml';old_input.write_text(yaml.safe_dump(document))
    new_input=new/'input.yaml';new_input.write_text(yaml.safe_dump(document))
    old_job={'id':'same-id','input_yaml':str(old_input),'input_sha256':hashlib.sha256(old_input.read_bytes()).hexdigest(),
             'modeling_input':{'geometry':{}}}
    saved=old/'family_cofold/jobs/same-id/complete.json';saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({'job':old_job,'receipt':{'status':'complete'}}))
    job={**old_job,'input_yaml':str(new_input)}
    monkeypatch.setattr(family,'analyze_outputs',lambda *a:{'identity_pass_poses':50})
    result=family.find_reusable(job,new,{'prediction_pools':[]})
    assert result['status']=='reused' and result['job']['discovery_run_id']=='old_run'


def test_registered_protocol_does_not_copy_old_cohort_pilot_selectors(tmp_path,specification):
    specification['fleet_pilots']={'standard':'OLD_COHORT_SELECTOR'}
    path=tmp_path/'protocol.json';path.write_text(json.dumps(specification))
    registry=tmp_path/'registry/index.json'
    family.register_protocol(path,registry)
    record=next(iter(json.loads(registry.read_text()).values()))
    saved=json.loads(Path(record['path']).read_text())
    assert 'fleet_pilots' not in saved
    assert saved['components']==specification['components']


def test_readable_input_index_has_the_consuming_run_identity(tmp_path):
    path=tmp_path/'model.yaml';path.write_text('version: 1\n')
    job={'accession':'SYNTH_A','enzyme_name':'Synthetic enzyme','family_id':'SYNTH_FAMILY','step':1,
         'discovery_run_id':'synthetic_run','input_yaml':str(path),'modeling_input':{
             'cofactors':[],'target_substrates':[{'role':'substrate','modeling_smiles':'CCO','include_in_model':True}]}}
    family.write_input_index([job],tmp_path)
    rows=list(csv.DictReader((tmp_path/'input_index.csv').open()))
    assert rows[0]['discovery_run_id']=='synthetic_run'
    assert Path(rows[0]['input_yaml']).resolve()==path
