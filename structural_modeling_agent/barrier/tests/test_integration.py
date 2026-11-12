import importlib
import json
import subprocess
import sys

import pytest


def test_compatibility_imports_share_module_and_class_identity():
    for name in ('spec','region','jobs','validation'):
        legacy=importlib.import_module('enzyme_barrier.'+name)
        canonical=importlib.import_module('structural_modeling_agent.barrier.'+name)
        assert legacy is canonical
    from enzyme_barrier import ReactionSpec as old
    from structural_modeling_agent.barrier import ReactionSpec as new
    assert old is new


@pytest.mark.parametrize('module',['enzyme_barrier','structural_modeling_agent.barrier'])
def test_cli_help_remains_available_without_loading_a_potential(module):
    proc=subprocess.run([sys.executable,'-m',module,'--help'],capture_output=True,text=True,timeout=15)
    assert proc.returncode==0,proc.stderr
    assert 'embedded-path' in proc.stdout and 'carve' in proc.stdout


def test_stored_validated_claim_does_not_bypass_evidence_gate(tmp_path):
    from structural_modeling_agent.barrier import review_result
    path=tmp_path/'result.json'
    path.write_text(json.dumps(dict(dE_act_kcal=12,quantity='constrained_electronic_energy_gap',
                                    validated_activation_free_energy=True)))
    result=review_result(path)
    assert result['stored_validation_claim']
    assert not result['validated_activation_free_energy']
    assert result['validation']['failures']


@pytest.fixture
def generic_inputs(tmp_path):
    import numpy as np
    pmd=pytest.importorskip('parmed');pytest.importorskip('ase')
    from structural_modeling_agent.barrier.fixtures import water_pair
    from structural_modeling_agent.barrier import ReactionSpec,AtomRef
    p=water_pair();top=tmp_path/'system.prmtop'
    pmd.amber.AmberParm.from_structure(p).save(str(top))
    coords=tmp_path/'frame.npz';np.savez(coords,positions_A=p.coordinates,
                                      box_lengths_angles=[20,20,20,90,90,90])
    spec=ReactionSpec('synthetic_exchange',(('WAT',0),),(),
                      AtomRef('WAT','O',1),AtomRef('WAT','O',0),AtomRef('WAT','H1',0),
                      forming_distance_A=1.5,breaking_distance_A=2.5)
    path=tmp_path/'spec.json';spec.save(path)
    return top,coords,path


@pytest.mark.parametrize('mode',['cluster','embedded'])
def test_prepare_generic_nonphosphate_inputs_with_real_topology(tmp_path,mode,generic_inputs):
    from structural_modeling_agent.barrier import prepare_inputs
    top,coords,path=generic_inputs
    result=prepare_inputs(top,coords,path,tmp_path/'prepared',mode=mode)
    assert result['mode']==mode and not result['validated_activation_free_energy']
    if mode=='embedded':
        region=json.loads((tmp_path/'prepared/region.json').read_text())
        assert region['path_definition']['forming_distance_A']==1.5
        assert region['metal_indices']==[]
    else:
        meta=result['result']
        assert meta['cluster_bonds'] and meta['n_atoms']==6
    with pytest.raises(FileExistsError):
        prepare_inputs(top,coords,path,tmp_path/'prepared',mode=mode)


def test_generic_path_uses_declared_targets(tmp_path,generic_inputs,monkeypatch):
    import numpy as np
    from ase.calculators.calculator import Calculator,all_changes
    from structural_modeling_agent.barrier import prepare_inputs,path,embedded
    class Harmonic(Calculator):
        implemented_properties=['energy','forces']
        def __init__(self,*args,**kwargs):
            super().__init__();self.last_components={}
        def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            self.results=dict(energy=float(np.sum(atoms.positions**2)*.001),forces=-.002*atoms.positions)
    prepare_inputs(*generic_inputs,tmp_path/'snapshot')
    monkeypatch.setattr(embedded,'EmbeddedDFTMM',Harmonic)
    result=path.explore(tmp_path/'snapshot',tmp_path/'path',engine='dft',images=3,max_steps=20)
    assert result['completed_images']==3
    assert result['images'][-1]['forming_A']==pytest.approx(1.5,abs=1e-7)
    assert result['images'][-1]['breaking_A']==pytest.approx(2.5,abs=1e-7)
    assert not result['validated_activation_free_energy']


def test_agent_operator_prepares_hashed_inputs_in_selected_runtime(tmp_path,generic_inputs):
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from structural_modeling_agent.barrier.api import prepare_for_agent
    if len(os.sched_getaffinity(0))<2:
        pytest.skip('Bounded controller/operator test requires two disjoint CPUs')
    top,coords,spec=generic_inputs
    state={'candidates':{'candidate':{}}}
    agent=SimpleNamespace(run_dir=tmp_path/'agent',state=state,
        _candidate=lambda _:None,_authorized_evidence=lambda value:Path(value).resolve(strict=True))
    result=prepare_for_agent(agent,dict(accession='candidate',prmtop=str(top),coordinates=str(coords),
                                      spec_path=str(spec),python=sys.executable,mode='embedded',seconds=30))
    assert result['status']=='operator_complete_requires_review',result
    assert state['candidates']['candidate']['barrier_preparations']
    prepared=next((tmp_path/'agent/artifacts').glob('barrier_*/prepared/region.json'))
    assert json.loads(prepared.read_text())['path_definition']['forming_distance_A']==1.5
