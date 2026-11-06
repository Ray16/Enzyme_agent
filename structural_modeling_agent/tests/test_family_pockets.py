"""Pocket guidance changes localization inputs, never the reactive geometry."""
import copy
import hashlib
import json

import numpy as np
import pytest
import yaml

from structural_modeling_agent import family_pockets as pockets


def test_activation_pocket_uses_the_mapped_reactive_carboxyl_end(tmp_path, monkeypatch):
    path=tmp_path/'pose.pdb';path.write_text('synthetic pose')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'input_sha256':'input','ligand_atom_maps':{'L':{'smiles':'CC(=O)[O-]',
        'atoms':{'0':{'name':'C0'},'1':{'name':'C1'},'2':{'name':'O2'},'3':{'name':'O3'}}}}}))
    atoms={('L','1','C1'):{'xyz':np.zeros(3),'protein':False},
        **{('P',str(i),'CA'):{'xyz':np.array([i,0,0]),'protein':True} for i in [1,2,3]}}
    monkeypatch.setattr(pockets,'_read_structure',lambda path:(atoms,{'P':'AAA'}))
    saved={'job':{'input_sha256':'input'},'receipt':{'prediction_manifest':str(manifest)},
        'analysis':{'poses':[{'distance_A':3.,'identity_issues':[],'path':str(path),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nucleophile':['L','O3']}]}}
    result=pockets.activation_pocket(saved)
    assert result['contacts']==[['P',1],['P',2],['P',3]]
    assert result['reactive_carboxyl_carbon']==['L','C1']
    broken=copy.deepcopy(saved);broken['analysis']['poses'][0]['distance_A']=20.
    with pytest.raises(ValueError,match='does not localize'):
        pockets.activation_pocket(broken)


def test_prepare_adds_only_broad_protein_pocket_and_retains_original_input(tmp_path, monkeypatch):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'guided'
    document={'version':1,'sequences':[{'protein':{'id':'P','sequence':'AAA','msa':'prepared.csv'}},
        {'ligand':{'id':'M','smiles':'CCN'}}]}
    input_path=tmp_path/'input.yaml';input_path.write_text(yaml.safe_dump(document));msa=tmp_path/'prepared.csv';msa.write_text('original msa')
    first={'accession':'SYNTH','enzyme_name':'Synthetic enzyme','family_id':'FAMILY','step':1,'id':'first'}
    second={**first,'step':2,'id':'second','input_yaml':str(input_path),'input_sha256':pockets.file_hash(input_path),
        'msa_path':str(msa),'msa_sha256':pockets.file_hash(msa),'expected_predictions':50,
        'entities':{'amine':'M'},'modeling_input':{'geometry':{'nucleophile':{'role':'amine'}},'cofactors':[],
        'target_substrates':[]}}
    (source/'queue.json').write_text(json.dumps({'items':[first,second]}))
    pose=tmp_path/'pose.pdb';pose.write_text('verified pose')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    for job in [first,second]:
        p=source/'jobs'/job['id']/'complete.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'job':job,'receipt':{'prediction_manifest':str(manifest)},'analysis':{
            'geometry':{'harmonic_distance_A':20.},'poses':[{'path':str(pose),'sha256':pockets.file_hash(pose)}]}}))
    pocket={'contacts':[['P',1],['P',2],['P',3]],'reference_pdb':str(pose)}
    monkeypatch.setattr(pockets,'activation_pocket',lambda saved:pocket)
    monkeypatch.setattr(pockets,'carbonyl_atom_references',lambda *args:(('L','C'),('L','O'),[('M','N')]))
    atoms={('L','1','C'):{'xyz':np.zeros(3),'protein':False},
        **{('P',str(i),'CA'):{'xyz':np.array([i,0,0]),'protein':True} for i in [1,2,3]}}
    monkeypatch.setattr(pockets,'_read_structure',lambda path:(atoms,{'P':'AAA'}))
    monkeypatch.setattr(pockets,'write_input_index',lambda *args:None)
    jobs=pockets.prepare(tmp_path,source,out,10.)
    guided=jobs[1];saved=yaml.safe_load(open(guided['input_yaml']))
    assert saved['sequences']==document['sequences']
    assert saved['constraints']==[{'pocket':{'binder':'M','contacts':pocket['contacts'],'max_distance':10.,'force':True}}]
    assert guided['expected_predictions']==50 and guided['msa_sha256']==second['msa_sha256']
    assert guided['id']!=second['id']
    assert yaml.safe_load(input_path.read_text())==document
    assert not guided['pocket_protocol']['reactive_atom_distance_restraint']
    assert not guided['pocket_protocol']['attack_angle_restraint']
    assert pockets.prepare(tmp_path,source,out,10.)==jobs
