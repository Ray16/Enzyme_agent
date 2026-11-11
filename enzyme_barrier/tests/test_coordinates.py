import numpy as np
import pytest
pytest.importorskip('ase')
pytest.importorskip('parmed')
from enzyme_barrier.fixtures import water_pair
from enzyme_barrier.coordinates import reimage_molecules
from enzyme_barrier.sander_backend import scc_converged,electronic_converged


def test_reimaging_preserves_molecules_and_recovers_nearby_water():
    p=water_pair();p.box=np.array([20,20,20,90,90,90]);x=p.coordinates.copy();x[3:]+=np.array([40,-20,20])
    before=np.linalg.norm(x[3]-x[4]);out=reimage_molecules(p,x,0)
    assert np.linalg.norm(out[3]-out[4])==pytest.approx(before)
    assert np.allclose(out,p.coordinates,atol=1e-10)


def test_scc_reset_is_not_convergence():
    assert not scc_converged('SCC-DFTB FOR STEP 1 DID NOT CONVERGE AFTER 70 cycles. Resetting Broyden mixing.')
    assert scc_converged('SCC-DFTB FOR STEP 1 DID NOT CONVERGE AFTER 70 cycles.\nSCC-DFTB for step 1 converged in 90 cycles.')
    assert not scc_converged('The calculation will continue, but energies and forces will not be accurate.')


def test_nddo_energy_line_cannot_hide_scf_failure():
    assert electronic_converged('QMMM: SCF Converged to 0.1000E-08 in: 65 Cycles','AM1/D')
    assert not electronic_converged('QMMM: SCF Energy = -2877.0','AM1/D')
    assert not electronic_converged('QMMM: Job will continue with unconverged SCF.\nQMMM: SCF Energy = -2877.0','AM1/D')


def test_pdb_writer_uses_current_barostat_cell(tmp_path):
    mm=pytest.importorskip('openmm')
    import openmm.app as app
    from openmm import unit
    from types import SimpleNamespace
    from enzyme_barrier.md import write_state_pdb
    p=water_pair();top=p.topology;old=np.eye(3)*20*unit.angstrom;new=np.eye(3)*30*unit.angstrom
    top.setPeriodicBoxVectors(old)
    state=SimpleNamespace(getPeriodicBoxVectors=lambda:new,getPositions=lambda:p.positions)
    path=tmp_path/'frame.pdb';write_state_pdb(top,state,path)
    row=next(l for l in path.read_text().splitlines() if l.startswith('CRYST1'))
    assert float(row[6:15])==30.0


def test_npz_coordinates_preserve_cell_and_precision(tmp_path):
    import parmed as pmd
    from enzyme_barrier.coordinates import load_system
    p=water_pair();parm=pmd.amber.AmberParm.from_structure(p)
    top=tmp_path/'water.prmtop';parm.save(str(top))
    x=p.coordinates+.000012345;box=np.array([31.123456789]*3+[90]*3)
    path=tmp_path/'frame.npz';np.savez(path,positions_A=x,box_lengths_angles=box)
    loaded=load_system(top,path)
    assert np.array_equal(loaded.coordinates,x)
    assert np.array_equal(loaded.box,box)


def test_magnesium_cannot_silently_use_a_topology_without_c4():
    from types import SimpleNamespace
    from enzyme_barrier.parameters import require_c4
    parm=SimpleNamespace(atoms=[SimpleNamespace(atomic_number=12)],parm_data={})
    with pytest.raises(ValueError,match='C4'):require_c4(parm)
