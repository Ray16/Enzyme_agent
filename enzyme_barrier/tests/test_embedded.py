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
    a.calc=EmbeddedDFTMM(p,r,engine='cpu',platform='Reference',xc='pbe',basis='sto-3g',grid_level=1,conv_tol=1e-11,require_magnesium_c4=False)
    force=a.get_forces()
    for atom,axis in [(0,2),(3,2),(4,0)]:
        pos=a.positions.copy();a.positions[atom,axis]+=1e-4;ep=a.get_potential_energy()
        a.positions[:]=pos;a.positions[atom,axis]-=1e-4;em=a.get_potential_energy();a.positions[:]=pos
        assert force[atom,axis]==pytest.approx(-(ep-em)/2e-4,abs=3e-4)
