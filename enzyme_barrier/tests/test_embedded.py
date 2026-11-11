import numpy as np
import pytest
pytest.importorskip('ase')
pytest.importorskip('openmm')
pytest.importorskip('parmed')
pytest.importorskip('pyscf')
from enzyme_barrier.region import capped_coordinates,project_link_forces


def test_link_force_projection_matches_coordinate_derivative():
    region={'indices':[0], 'links':[dict(qm=0,mm=1,local=0,ratio=0.7)]}
    p=np.array([[0.1,0.2,0.3],[1.5,0.5,0.4]])
    def energy(x):return np.sum(capped_coordinates(x,region)**2)
    analytic=project_link_forces(-2*capped_coordinates(p,region),region,2)
    for i in range(2):
        for k in range(3):
            a=p.copy();b=p.copy();a[i,k]+=1e-5;b[i,k]-=1e-5
            assert analytic[i,k]==pytest.approx(-(energy(a)-energy(b))/2e-5,abs=1e-8)


def test_mm_eliminates_qm_coulomb_exceptions_and_internal_bonds():
    import openmm as mm
    from openmm import unit as u
    from enzyme_barrier.embedded import remove_qm_terms
    s=mm.System()
    for _ in range(3):s.addParticle(12)
    nb=mm.NonbondedForce()
    for _ in range(3):nb.addParticle(0.3,0.3,0.1)
    nb.addException(0,2,0.045,0.3,0.01)
    s.addForce(nb);b=mm.HarmonicBondForce();b.addBond(0,1,0.15,100);b.addBond(1,2,0.15,100);s.addForce(b)
    remove_qm_terms(s,[0,1])
    assert b.getBondParameters(0)[-1].value_in_unit(u.kilojoule_per_mole/u.nanometer**2)==0
    assert b.getBondParameters(1)[-1].value_in_unit(u.kilojoule_per_mole/u.nanometer**2)>0
    for i in (0,1):assert nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge)==0
    for k in range(nb.getNumExceptions()):
        i,j,q,_,e=nb.getExceptionParameters(k)
        assert q.value_in_unit(u.elementary_charge**2)==0


def water_pair():
    import parmed as pmd
    from parmed.topologyobjects import Atom,AtomType,Bond,BondType,Angle,AngleType
    p=pmd.Structure();ot=AtomType('OW',1,16,8);ot.set_lj_params(0.152,1.7682)
    ht=AtomType('HW',2,1,1);ht.set_lj_params(0,0)
    for r in range(2):
        for nm,z,q,m,t in [('O',8,-0.834,16,ot),('H1',1,0.417,1,ht),('H2',1,0.417,1,ht)]:
            a=Atom(name=nm,type=t.name,atomic_number=z,charge=q,mass=m);a.atom_type=t;p.add_atom(a,'WAT',r+1)
        base=3*r;bt=BondType(553,0.9572);p.bond_types.append(bt)
        for j in (1,2):p.bonds.append(Bond(p.atoms[base],p.atoms[base+j],type=bt))
        at=AngleType(100,104.52);p.angle_types.append(at)
        p.angles.append(Angle(p.atoms[base+1],p.atoms[base],p.atoms[base+2],type=at))
    p.coordinates=np.array([[0,0,0],[0.9572,0,0],[-0.24,0.927,0],[0,0,3.2],[0.9572,0,3.2],[-0.24,0.927,3.2]])
    return p


def test_cpu_embedded_qm_and_mm_forces_are_energy_derivatives():
    from ase import Atoms
    from enzyme_barrier.embedded import EmbeddedDFTMM
    p=water_pair();r={'indices':[0,1,2],'links':[],'charge':0,'multiplicity':1,
                      'embedding_indices':[3,4,5],'embedding_charges':[-.834,.417,.417]}
    a=Atoms(numbers=[x.atomic_number for x in p.atoms],positions=p.coordinates)
    a.calc=EmbeddedDFTMM(p,r,engine='cpu',platform='CPU',xc='pbe',basis='sto-3g',grid_level=1,conv_tol=1e-11,require_magnesium_c4=False)
    force=a.get_forces()
    for atom,axis in [(0,2),(3,2),(4,0)]:
        pos=a.positions.copy();a.positions[atom,axis]+=1e-4;ep=a.get_potential_energy()
        a.positions[:]=pos;a.positions[atom,axis]-=1e-4;em=a.get_potential_energy();a.positions[:]=pos
        assert force[atom,axis]==pytest.approx(-(ep-em)/2e-4,abs=3e-4)


def test_embedding_field_changes_QM_density_not_only_classical_energy():
    from enzyme_barrier.embedded import EmbeddedDFTMM
    p=water_pair();r=dict(indices=[0,1,2],links=[],charge=0,multiplicity=1,
        embedding_indices=[3,4,5],embedding_charges=[-.834,.417,.417])
    calc=EmbeddedDFTMM(p,r,engine='cpu',platform='CPU',xc='pbe',basis='sto-3g',
        grid_level=1,conv_tol=1e-11,require_magnesium_c4=False)
    q=p.coordinates[:3];positions=p.coordinates[3:]
    vacuum=calc.high_energy_forces(['O','H','H'],q,positions,np.zeros(3),False)[0]
    dm_vacuum=np.asarray(calc.dm).copy()
    embedded=calc.high_energy_forces(['O','H','H'],q,positions,np.array(r['embedding_charges']),False)[0]
    assert abs(embedded-vacuum)>1e-4
    assert np.linalg.norm(np.asarray(calc.dm)-dm_vacuum)>1e-4


def test_complete_embedded_system_preserves_translation_and_total_force():
    from ase import Atoms
    from enzyme_barrier.embedded import EmbeddedDFTMM
    p=water_pair();r=dict(indices=[0,1,2],links=[],charge=0,multiplicity=1,
        embedding_indices=[3,4,5],embedding_charges=[-.834,.417,.417])
    a=Atoms(numbers=[at.atomic_number for at in p.atoms],positions=p.coordinates)
    a.calc=EmbeddedDFTMM(p,r,engine='cpu',platform='CPU',xc='pbe',basis='sto-3g',grid_level=2,
        conv_tol=1e-11,require_magnesium_c4=False,grid_response=True)
    forces=a.get_forces();energy=a.get_potential_energy()
    assert np.linalg.norm(forces.sum(axis=0))<1e-6
    a.positions+=np.array([3.,-4.,2.])
    assert a.get_potential_energy()==pytest.approx(energy,abs=1e-6)
    assert np.allclose(a.get_forces(),forces,atol=1e-6)


@pytest.mark.parametrize('grid_response',[False,True])
def test_actual_SCF_link_and_embedding_forces(grid_response):
    import parmed as pmd
    from parmed.topologyobjects import Atom,AtomType,Bond,BondType,Angle,AngleType
    from ase import Atoms
    from enzyme_barrier.embedded import EmbeddedDFTMM
    from enzyme_barrier.embedding_forces import force_checks
    p=pmd.Structure();ct=AtomType('CT',1,12,6);ct.set_lj_params(.1094,1.908)
    ht=AtomType('HC',2,1,1);ht.set_lj_params(.0157,1.487)
    for name,z,typ in [('CB',6,ct),('CA',6,ct)]+[(f'H{i}',1,ht) for i in range(6)]:
        atom=Atom(name=name,type=typ.name,atomic_number=z,mass=typ.mass,charge=0)
        atom.atom_type=typ;p.add_atom(atom,'ETH',1)
    p.coordinates=np.array([[0,0,0],[1.54,0,0],[-.36,1.027,0],[-.36,-.5135,.8894],[-.36,-.5135,-.8894],
        [1.90,-1.027,0],[1.90,.5135,.8894],[1.90,.5135,-.8894]])
    neighbors={0:[1,2,3,4],1:[0,5,6,7]}
    for i,j in [(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(1,7)]:
        typ=BondType(310 if j==1 else 340,1.54 if j==1 else 1.09);p.bond_types.append(typ)
        p.bonds.append(Bond(p.atoms[i],p.atoms[j],type=typ))
    import itertools
    for center,ids in neighbors.items():
        for i,j in itertools.combinations(ids,2):
            typ=AngleType(40,109.5);p.angle_types.append(typ);p.angles.append(Angle(p.atoms[i],p.atoms[center],p.atoms[j],type=typ))
    water=water_pair()[':1'];water.coordinates+=np.array([0,0,3.2]);p=p+water
    r=dict(indices=[0,2,3,4],links=[dict(qm=0,mm=1,local=0,ratio=1.09/1.54)],charge=0,multiplicity=1,
        embedding_indices=[8,9,10],embedding_charges=[-.834,.417,.417])
    a=Atoms(numbers=[at.atomic_number for at in p.atoms],positions=p.coordinates)
    a.calc=EmbeddedDFTMM(p,r,engine='cpu',platform='CPU',xc='pbe',basis='sto-3g',grid_level=2,
        conv_tol=1e-11,require_magnesium_c4=False,grid_response=grid_response)
    result=force_checks(a,[dict(atom_zero_based=i,axis=j) for i,j in [(0,0),(1,1),(8,2)]],tolerance_eV_A=.002)
    assert result['passed'],result
