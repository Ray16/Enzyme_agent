import numpy as np
import pytest

pytest.importorskip('ase')
pytest.importorskip('parmed')
pytest.importorskip('openmm')
from enzyme_barrier.fixtures import water_pair


def test_audit_checks_actual_remaining_MM_charges_and_exports_atoms(tmp_path,monkeypatch):
    from ase import Atoms
    from enzyme_barrier import embedding_audit
    p=water_pair();a=Atoms(numbers=[v.atomic_number for v in p.atoms],positions=p.coordinates)
    r=dict(indices=[0,1,2],links=[],charge=0,multiplicity=1,embedding_indices=[3,4,5],
        embedding_charges=[-.834,.417,.417],mobile_indices=list(range(6)),entries=[dict(resid=0,resname='WAT',charge=0)],
        embedding_radius_A=20,mobile_radius_A=8)
    root=tmp_path/'snapshot';root.mkdir();(root/'input.json').write_text('{}')
    monkeypatch.setattr(embedding_audit,'load_snapshot',lambda _: (p,r,a,dict(input_sha256={})))
    good=embedding_audit.audit_snapshot(root,tmp_path/'good')
    assert good['passed'] and all(good['checks'].values())
    assert len((tmp_path/'good/embedding_charges.csv').read_text().splitlines())==4
    r['embedding_charges'][0]=-.2
    bad=embedding_audit.audit_snapshot(root,tmp_path/'bad')
    assert not bad['passed'] and not bad['checks']['embedding_matches_remaining_MM_charges']


def test_force_check_restores_coordinates_and_rejects_bad_derivatives():
    from ase import Atoms
    from ase.calculators.calculator import Calculator,all_changes
    from enzyme_barrier.embedding_forces import force_checks
    class Harmonic(Calculator):
        implemented_properties=['energy','forces']
        def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            self.results=dict(energy=float(np.sum(atoms.positions**2)/2),forces=-atoms.positions)
    a=Atoms('H',positions=[[1.,2,3]]);a.calc=Harmonic();x=a.positions.copy()
    assert force_checks(a,[dict(atom_zero_based=0,axis=1)],tolerance_eV_A=1e-8)['passed']
    assert np.array_equal(a.positions,x)
    with pytest.raises(ValueError):force_checks(a,[dict(atom_zero_based=4,axis=0)])
    assert np.array_equal(a.positions,x)


def test_changing_zero_charge_keeps_12_and_13_Coulomb_excluded():
    import openmm as mm
    from openmm import unit
    from enzyme_barrier.embedded import build_mm_system
    p=water_pair();p.atoms[3].charge=0
    r=dict(indices=[0,1,2],mm_charge_updates={'3':-.834})
    system=build_mm_system(p,r)
    nb=next(f for f in system.getForces() if isinstance(f,mm.NonbondedForce))
    for k in range(nb.getNumExceptions()):
        i,j,q,_,_=nb.getExceptionParameters(k)
        if {i,j}<={3,4,5}:assert q.value_in_unit(unit.elementary_charge**2)==0


def test_LJ_C4_energy_and_force_vanish_smoothly_at_cutoff():
    import openmm as mm
    from openmm import unit as u
    from enzyme_barrier.embedded import configure_mm_switching
    s=mm.System();s.addParticle(12);s.addParticle(12)
    nb=mm.NonbondedForce();nb.setNonbondedMethod(nb.CutoffNonPeriodic);nb.setCutoffDistance(1.2)
    for _ in range(2):nb.addParticle(0,.3,.5)
    s.addForce(nb)
    c4=mm.CustomNonbondedForce('-.001/r^4');c4.setNonbondedMethod(c4.CutoffNonPeriodic);c4.setCutoffDistance(1.2)
    c4.addParticle([]);c4.addParticle([]);s.addForce(c4);configure_mm_switching(s)
    integrator=mm.VerletIntegrator(.001);ctx=mm.Context(s,integrator,mm.Platform.getPlatformByName('CPU'),{'Threads':'1'})
    def evaluate(r):
        ctx.setPositions([[0,0,0],[r,0,0]]*u.nanometer)
        st=ctx.getState(getEnergy=True,getForces=True)
        return st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole/u.nanometer)
    for radius in [1.199999,1.2,1.200001]:
        energy,force=evaluate(radius)
        # The CPU kernels evaluate some nonbonded arithmetic in float32.
        # These limits are still >10,000-fold below the unswitched energy jump.
        assert abs(energy)<1e-8 and np.max(np.abs(force))<1e-6
    e,f=evaluate(1.1999);numeric=-(evaluate(1.2)[0]-evaluate(1.1998)[0])/.0002
    assert abs(numeric-f[1,0])<2e-5
