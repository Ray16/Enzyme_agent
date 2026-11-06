"""Preparation tools must work from the lightweight decision environment."""
from pathlib import Path
import json

from test_modeling_agent import make_agent
from test_sampling_setup import declaration


def test_readonly_preparation_tools_from_controller(tmp_path):
    agent=make_agent(tmp_path, objective='prepare_sampling')
    setup=declaration(tmp_path)
    for name,args in [
        ('inspect_candidate',{'accession':'SYNTHETIC1'}),
        ('read_candidate_section',{'accession':'SYNTHETIC1','section':'evidence'}),
        ('read_evidence',{'path':setup['sources'][0]['path']}),
        ('inspect_result_locations',{}),
        ('inspect_mlmm_engine_options',{}),
        ('inspect_preparation_contract',{}),
        ('inspect_sampling_setup_contract',{}),
        ('inspect_processivity_toolchain',{}),
        ('inspect_molecule',{'smiles':'CC(=O)[O-]','smarts':'C(=O)[O-]'})]:
        result=agent.dispatch(name,args)
        assert result
        json.dumps(result,allow_nan=False)
    inventory=agent.dispatch('inspect_processivity_toolchain',{})
    assert inventory['recorded_evidence_current']
    assert all(e.get('interpreter_available',e.get('executable_available')) for e in inventory['environments'])
    saved=agent.dispatch('declare_sampling_setup',{'accession':'SYNTHETIC1','setup':setup})
    assert agent.dispatch('inspect_sampling_setup',{'accession':'SYNTHETIC1','setup_id':saved['id']})['status']=='ready_for_preparation'
    assert agent.dispatch('finish_task',{'outcome':'complete','rationale':'Synthetic test'})['status']=='blocked'


def test_assembly_requires_namespace_and_exact_sequence(tmp_path,monkeypatch):
    from structural_modeling_agent.modeling_reasoning import modeling_unit
    agent=make_agent(tmp_path)
    def forbidden(*a,**k): raise AssertionError('Invalid namespace must not reach UniProt')
    monkeypatch.setattr(modeling_unit,'gather_evidence',forbidden)
    assert agent.dispatch('assess_assembly',{'accession':'SYNTHETIC1','ligand_heavy_atoms':10})['status']=='needs_identity_mapping'
    monkeypatch.setattr(modeling_unit,'gather_evidence',lambda *a,**k:(None,{'sequence':'WRONG'}))
    assert agent.dispatch('assess_assembly',{'accession':'SYNTHETIC1','uniprot_accession':'P12345','ligand_heavy_atoms':10})['status']=='sequence_mismatch'


def test_pdb_inspection_is_observation_not_assignment(tmp_path):
    agent=make_agent(tmp_path)
    p=agent.run_dir/'artifacts'/'input.pdb';p.parent.mkdir(exist_ok=True)
    p.write_text('HETATM    1 MG    MG C   1       0.000   0.000   0.000  1.00  0.00          MG\n'
                 'HETATM    2  O   HOH W   1       2.000   0.000   0.000  1.00  0.00           O\n')
    result=agent.dispatch('inspect_preparation_structure',{'path':str(p)})
    assert result['water_oxygen_count']==1
    assert result['hydrogen_count']==0
    assert len(result['metals'][0]['neighbors_within_3A'])==1
    assert Path(result['artifact']).is_file()


def test_working_memory_survives_context_and_revokes_changed_evidence(tmp_path):
    agent=make_agent(tmp_path,objective='prepare_sampling'); setup=declaration(tmp_path)
    source=setup['sources'][0]
    args=dict(accession='SYNTHETIC1',summary='Synthetic unresolved state; do not repeat the same diagnostic.',
              sources=[{'path':source['path'],'sha256':source['sha256']}],next_action='Compare a justified alternative')
    agent.dispatch('record_preparation_findings',args)
    identity=agent._preparation_progress()
    assert agent.dispatch('record_preparation_findings',args)['status']=='unchanged'
    assert agent._preparation_progress()==identity
    assert args['summary'] in json.dumps(agent._context())
    Path(source['path']).write_text('changed')
    assert agent._preparation_findings(agent.state['candidates']['SYNTHETIC1'])['source_errors']


def test_two_read_only_turns_require_a_decision_update(tmp_path):
    from test_modeling_agent import Client
    class InspectClient(Client):
        def call(self,**kwargs):
            names={t['function']['name'] for t in kwargs['tools']}
            if len(self.contexts)==2:
                assert 'read_evidence' not in names
                assert 'record_preparation_findings' in names
            return super().call(**kwargs)
    client=InspectClient([[('inspect_preparation_contract',{})], [('inspect_sampling_setup_contract',{})],
                          [('finish_task',{'outcome':'inconclusive','rationale':'Synthetic missing evidence'})]])
    agent=make_agent(tmp_path,client,objective='prepare_sampling')
    assert agent.run(max_turns=3)['status']=='inconclusive'
    assert all(len(json.dumps(c))<=24000 for c in client.contexts)


def test_paid_response_can_save_findings_at_token_boundary(tmp_path):
    from test_modeling_agent import Client
    client=Client([[('record_preparation_findings',dict(accession='SYNTHETIC1',summary='Unresolved synthetic choice',sources=[],next_action='Gather evidence'))]])
    agent=make_agent(tmp_path,client,objective='prepare_sampling')
    result=agent.run(max_turns=1,token_budget=1)
    assert result['status']=='budget_exhausted'
    assert agent.state['candidates']['SYNTHETIC1']['preparation_findings']['summary']=='Unresolved synthetic choice'


def test_setup_binds_bookkeeping_hashes_without_model_arithmetic(tmp_path):
    agent=make_agent(tmp_path,objective='prepare_sampling');setup=declaration(tmp_path)
    expected=setup.pop('sequence_sha256');setup.pop('target_sha256')
    result=agent.dispatch('declare_sampling_setup',{'accession':'SYNTHETIC1','setup':setup})
    saved=agent.state['candidates']['SYNTHETIC1']['sampling_setups'][result['id']]['setup']
    assert saved['sequence_sha256']==expected


def test_empty_bookkeeping_is_not_scientific_progress(tmp_path):
    agent=make_agent(tmp_path,objective='prepare_sampling')
    before=agent._preparation_progress()
    agent.state['candidates']['SYNTHETIC1']['sampling_setups']={}
    agent.state['candidates']['SYNTHETIC1']['preparation_studies']={}
    assert agent._preparation_progress()==before
