"""Legacy coordinates yield conditional frozen descriptors without invented provenance."""
import json

import gemmi
import pytest

from test_validation import write_pose
from structural_modeling_agent.agent import ModelingAgent
from structural_modeling_agent.validation import file_digest
from structural_modeling_agent.evidence import modeling_evidence


def pose(path,displacement):
    write_pose(path,displacement=displacement)
    structure=gemmi.read_structure(str(path));model=structure[0]
    oxygen=gemmi.Atom();oxygen.name='O2';oxygen.element=gemmi.Element('O')
    oxygen.pos=gemmi.Position(3+displacement,-1.2,0);model['B'][0].add_atom(oxygen)
    chain=gemmi.Chain('C');residue=gemmi.Residue();residue.name='ATP';residue.seqid=gemmi.SeqId(1,' ');residue.het_flag='H'
    for name,element,xyz in [('PA','P',(10,0,0)),('O3A','O',(11,0,0))]:
        atom=gemmi.Atom();atom.name=name;atom.element=gemmi.Element(element);atom.pos=gemmi.Position(*xyz);residue.add_atom(atom)
    chain.add_residue(residue);model.add_chain(chain);structure.write_pdb(str(path));return path


def test_conditional_legacy_geometry_is_remeasured_and_never_marked_producer_verified(tmp_path):
    source=tmp_path/'collection.csv';source.write_text('accession,source\nSYNTHETIC1,legacy\n')
    paths=[]
    for i in range(2):
        paths.append(pose(tmp_path/f'model_{i}.pdb',float(i)))
        confidence=tmp_path/f'confidence_model_{i}.json';confidence.write_text('{}');paths.append(confidence)
    task=tmp_path/'task.json';task.write_text(json.dumps({'schema_version':'1.0','workflow_mode':'production_cofold',
        'objective':'model_and_validate','target':{'reaction':'synthetic transfer'},'candidates':[{
        'accession':'SYNTHETIC1','sequence':'AG','evidence':{'do_not_recofold':True,'existing_cofolds':[{
            'setup':{'setup_id':'setup:legacy','input_source':None,'declared_legacy_recipe':{'phosphate':'PA'}},
            'execution':{'execution_id':'execution:legacy','expected_poses':2,
                'source':{'path':str(source),'sha256':file_digest(source)},
                'artifacts':[{'path':str(path),'sha256':file_digest(path)} for path in paths]}}]}}]}))
    agent=ModelingAgent(task_path=task,run_dir=tmp_path/'review')
    result=agent.dispatch('review_existing_predictions',{'accession':'SYNTHETIC1','setup_id':'setup:legacy',
        'applicability':'limited_transfer','interpretation':'The recorded selector permits conditional geometry; molecular provenance remains unknown.',
        'unresolved':['Original input and producer binding are unavailable.']})
    assert result['source_execution_status']=='unverified_legacy'
    assert next(iter(result['geometry_metrics'].values()))['n_poses']==2
    assert next(iter(result['geometry_metrics'].values()))['weighted_attack_angle_deg']==pytest.approx(180)
    agent.state['status']='inconclusive';agent._save();agent._write_report()
    evidence=modeling_evidence(agent.run_dir/'report.json')['candidates'][0]
    assert evidence['execution_status']=='unverified_legacy'
    assert evidence['pose_qc_status']=='not_assessed'
    assert evidence['experimental_activity']=='not_established_by_modeling'
    assert not list(agent.run_dir.glob('models/*/input.yaml'))
