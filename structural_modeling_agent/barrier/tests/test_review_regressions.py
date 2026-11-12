import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('ase')
pytest.importorskip('parmed')
from enzyme_barrier.fixtures import water_pair
from enzyme_barrier.spec import AtomRef, ReactionSpec


def test_atom_resolution_rejects_ambiguity_and_honors_auto_residue():
    from enzyme_barrier.carve import _resolve
    p=water_pair()
    with pytest.raises(ValueError,match='Ambiguous'):
        _resolve(p,AtomRef('WAT','O'))
    assert _resolve(p,AtomRef('WAT','O',1))==3
    assert _resolve(p,AtomRef('WAT','AUTO',1),p.coordinates[0],p.coordinates)==3


def test_cap_redistribution_cannot_recharge_link_parent():
    from enzyme_barrier.region import redistribute_cap_charges
    x=np.array([[0,0,0],[1.54,0,0],[1.09,.1,0],[5,0,0]],float)
    r=dict(indices=[0],links=[dict(qm=0,mm=1,ratio=1.09/1.54)])
    original=np.array([0,0,.3,-.3])
    charges,deleted=redistribute_cap_charges(original,x,r,.2)
    assert deleted==[2]
    assert charges[1]==0 and charges[2]==0
    assert charges.sum()==pytest.approx(original.sum(),abs=1e-14)
    with pytest.raises(ValueError,match='No MM recipients'):
        redistribute_cap_charges(original[:3],x[:3],r,.2)


def test_embedding_selects_whole_residues_and_reports_finite_charge():
    from enzyme_barrier.ee_barrier import full_cell_embedding
    p=water_pair()
    r=dict(indices=[0,1,2],links=[],charge=0,
           mm_charge_updates={'0':0.,'1':0.,'2':0.})
    # Only the MM oxygen lies within 3.21 A; its hydrogens must come with it.
    result=full_cell_embedding(p,p.coordinates,r,embedding_radius=3.21)
    assert result['embedding_indices']==[3,4,5]
    assert result['embedding_selection']=='whole_residue_shell'
    assert result['cap_charge_delete_A'] is None
    assert result['embedded_total_charge']==pytest.approx(0)


def _path(tmp_path,monkeypatch):
    import parmed as pmd
    from enzyme_barrier.jobs import sha256
    p=water_pair();top=tmp_path/'system.prmtop';top.write_text('synthetic topology')
    monkeypatch.setattr(pmd,'load_file',lambda _:p)
    region=dict(indices=[0,1,2],links=[],charge=0,multiplicity=1,
                embedding_indices=[3,4,5],embedding_charges=[-.834,.417,.417],
                mm_charge_updates={},mobile_indices=list(range(6)))
    snaps={}
    for label in ('R','P'):
        root=tmp_path/label;root.mkdir()
        (root/'region.json').write_text(json.dumps(region));np.save(root/'positions.npy',p.coordinates)
        manifest=dict(prmtop=str(top),box_lengths_angles=[20,20,20,90,90,90],
                      region_sha256=sha256(root/'region.json'),positions_sha256=sha256(root/'positions.npy'))
        (root/'input.json').write_text(json.dumps(manifest));snaps[label]=str(root)
    (tmp_path/'path.json').write_text(json.dumps(dict(snapshots=snaps)))
    return p,region


def test_path_rejects_changed_partition_even_with_valid_hash(tmp_path,monkeypatch):
    from enzyme_barrier.jobs import consistent_path_positions,sha256
    p,region=_path(tmp_path,monkeypatch)
    assert len(consistent_path_positions(tmp_path)[3])==2
    region['charge']=2
    (tmp_path/'P/region.json').write_text(json.dumps(region))
    manifest=json.loads((tmp_path/'P/input.json').read_text())
    manifest['region_sha256']=sha256(tmp_path/'P/region.json')
    (tmp_path/'P/input.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='partition differs'):
        consistent_path_positions(tmp_path)


def test_path_uses_authoritative_cell_and_rejects_changed_cell(tmp_path,monkeypatch):
    from enzyme_barrier.jobs import consistent_path_positions,_recorded_box,_min_image
    _path(tmp_path,monkeypatch)
    manifest=json.loads((tmp_path/'R/input.json').read_text())
    manifest['box']=[40,40,40,90,90,90]
    assert _recorded_box(tmp_path/'R',manifest)[:3]==[20]*3
    assert np.allclose(_min_image(np.array([[60.3,-40.2,20.1]]),np.eye(3)*20),[[.3,-.2,.1]])
    manifest=json.loads((tmp_path/'P/input.json').read_text());manifest['box_lengths_angles'][0]=21
    (tmp_path/'P/input.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='cell differs'):
        consistent_path_positions(tmp_path)


def test_unconverged_cluster_image_is_rejected_without_loading_a_model(monkeypatch):
    from ase import Atoms
    from ase.calculators.calculator import Calculator,all_changes
    from structural_modeling_agent.barrier import scan
    class Harmonic(Calculator):
        implemented_properties=['energy','forces']
        def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            self.results=dict(energy=float(np.sum(atoms.positions**2)),forces=-2*atoms.positions)
    a=Atoms('H4',positions=[[0,1,0],[1,1,0],[0,0,1],[1,0,2]])
    with pytest.raises(RuntimeError,match='did not converge'):
        scan._relax_held(a,0,1,Harmonic(),(0,1),1,[],[],.01,0)


def test_scan_initialization_preserves_fragments_and_frozen_environment():
    from ase import Atoms
    from structural_modeling_agent.barrier.scan_geometry import reaction_groups,set_group_distance
    atoms=Atoms('NCOCH',positions=[[0,0,3],[0,0,0],[1.4,0,0],[0,0,4.4],[5,0,0]])
    original=atoms.positions.copy()
    forming,leaving=reaction_groups(len(atoms),[(0,3),(1,2)],[0,1,2])
    assert forming==[0,3] and leaving==[2]
    set_group_distance(atoms,0,1,1.4,forming,frozen=[4])
    set_group_distance(atoms,1,2,3.4,leaving,frozen=[4])
    assert atoms.get_distance(0,3)==pytest.approx(1.4)
    assert atoms.get_distance(0,1)==pytest.approx(1.4)
    assert atoms.get_distance(1,2)==pytest.approx(3.4)
    assert np.array_equal(atoms.positions[[1,4]],original[[1,4]])
    with pytest.raises(ValueError,match='frozen atoms'):
        set_group_distance(atoms,0,1,1.3,forming,frozen=[3])


def test_nonphosphate_states_use_declared_targets_and_preserve_other_bonds():
    from structural_modeling_agent.barrier.chemistry import reaction_state
    numbers=[6,8,8,7,6]
    atoms=[SimpleNamespace(idx=i,atomic_number=z,bond_partners=[]) for i,z in enumerate(numbers)]
    for i,j in [(0,1),(0,2),(3,4)]:
        atoms[i].bond_partners.append(atoms[j]);atoms[j].bond_partners.append(atoms[i])
    p=SimpleNamespace(atoms=atoms)
    x=np.array([[0,0,0],[1.4,0,0],[-1.2,0,0],[0,0,3],[0,0,4.4]],float)
    r=dict(indices=list(range(5)),reaction_atoms=[3,0,1],path_definition=dict(
        reactant_forming_A=3.,reactant_breaking_A=1.4,forming_distance_A=1.4,breaking_distance_A=3.4))
    assert reaction_state(p,x,r)['state']=='reactant'
    x[1]=[3.4,0,0];x[3]=[0,0,1.4];x[4]=[0,0,2.8]
    assert reaction_state(p,x,r)['state']=='product'
    x[4]=[0,0,7]
    assert reaction_state(p,x,r)['state']=='off_path_or_mapping_requires_review'


def test_nonphosphate_targets_and_nonmagnesium_charges_are_explicit(tmp_path):
    spec=ReactionSpec('generic_exchange',(('SUB',0),),('ZN',),
                      AtomRef('SUB','N'),AtomRef('SUB','C'),AtomRef('SUB','O'),
                      metal_charges=(('ZN',2),),forming_distance_A=1.4,breaking_distance_A=3.4)
    assert spec.qm_residue_charges()=={'SUB':0,'ZN':2}
    target=spec.path_definition([7,6,8],[[0,0,3],[0,0,0],[0,0,-1.4]],[0,1,2])
    assert target['forming_distance_A']==1.4 and target['breaking_distance_A']==3.4
    assert not target['targets_inferred']
    path=tmp_path/'spec.json';spec.save(path)
    assert ReactionSpec.load(path)==spec
    from dataclasses import replace
    with pytest.raises(ValueError,match='Explicit metal charge'):
        replace(spec,metal_charges=()).qm_residue_charges()


def test_nonmagnesium_shell_and_open_shell_partition():
    from parmed.topologyobjects import Atom
    from enzyme_barrier.region import build_region
    p=water_pair();x=np.vstack([p.coordinates,[[0,0,2.5]]])
    p.add_atom(Atom(name='ZN',atomic_number=30,mass=65.38,charge=2),'ZN',3)
    r=build_region(p,x,[0,1,2],{'ZN':2},radius=.1)
    assert r['metal_indices']==[6]
    assert set(range(7))<=set(r['indices'])  # both complete coordination waters
    with pytest.raises(ValueError,match='multiplicity'):
        build_region(p,x,[0,1,2],{'ZN':2},radius=.1,multiplicity=2)
