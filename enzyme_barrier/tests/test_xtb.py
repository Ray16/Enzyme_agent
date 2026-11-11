import os
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('ase')
from enzyme_barrier.xtb import run_xtb,read_engrad


def test_gradient_parser_checks_atom_count(tmp_path):
    f=tmp_path/'molecule.engrad';f.write_text('# atom count\n1\n# energy\n-1\n# gradient\n0\n0\n0\n')
    energy,force=read_engrad(f,1)
    assert force.shape==(1,3)
    with pytest.raises(ValueError,match='atom count'):read_engrad(f,2)


def test_external_xtb_embedding_force_derivatives(tmp_path):
    executable=Path(os.environ.get('BARRIER_XTB_COMMAND','/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb'))
    if not executable.is_file():pytest.skip('External xTB is not installed in this test environment')
    q=np.array([[0.,0,0],[.9572,0,0],[-.24,.927,0]]);mm=q+np.array([0,0,3.2]);charges=[-.834,.417,.417]
    options=dict(symbols=['O','H','H'],charge=0,multiplicity=1,command=str(executable),workdir=tmp_path,threads=1,accuracy=.05)
    e,fq,fm=run_xtb(positions=q,mm_positions=mm,mm_charges=charges,**options)
    for subsystem in ['qm','mm']:
        qp=q.copy();qm=q.copy();mp=mm.copy();mn=mm.copy();eps=1e-4
        if subsystem=='qm':qp[0,2]+=eps;qm[0,2]-=eps;analytic=fq[0,2]
        else:mp[0,2]+=eps;mn[0,2]-=eps;analytic=fm[0,2]
        ep=run_xtb(positions=qp,mm_positions=mp,mm_charges=charges,**options)[0]
        em=run_xtb(positions=qm,mm_positions=mn,mm_charges=charges,**options)[0]
        assert analytic==pytest.approx(-(ep-em)/(2*eps),abs=.002)
