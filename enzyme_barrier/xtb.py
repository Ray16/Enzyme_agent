"""Standalone external-xTB interface using its documented file formats.

No toolkit internals are imported. Embedding input follows the public xTB PCEM
specification: https://xtb-docs.readthedocs.io/en/latest/pcem.html . Coordinates
are explicitly Angstrom via interface=orca; gradients are atomic units.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import numpy as np
from ase.units import Hartree,Bohr


def _number(text):return float(text.replace('D','E').replace('d','e'))


def read_engrad(path,natoms):
    fields=[line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if int(fields[0])!=natoms:raise ValueError('xTB gradient atom count mismatch')
    energy=_number(fields[1])*Hartree
    gradient=np.array([_number(v) for v in fields[2:2+3*natoms]])
    if gradient.size!=3*natoms or not np.isfinite(gradient).all() or not np.isfinite(energy):raise ValueError('Invalid xTB gradient')
    return energy,-gradient.reshape(natoms,3)*Hartree/Bohr


def run_xtb(symbols,positions,charge,multiplicity,*,mm_positions=None,mm_charges=None,
            forces=True,command,workdir,threads=1,accuracy=.2,timeout=120):
    coordinates=np.asarray(positions,float)
    if coordinates.shape!=(len(symbols),3):raise ValueError('xTB coordinate shape mismatch')
    mm=np.asarray(mm_positions if mm_positions is not None else [],float).reshape(-1,3)
    charges=np.asarray(mm_charges if mm_charges is not None else [],float).reshape(-1)
    if len(mm)!=len(charges):raise ValueError('xTB MM charge count mismatch')
    directory=Path(tempfile.mkdtemp(prefix='evaluation_',dir=workdir));success=False
    try:
        geometry=directory/'molecule.xyz'
        geometry.write_text(f'{len(symbols)}\nreactive subsystem\n'+''.join(f'{s} {x:.12f} {y:.12f} {z:.12f}\n' for s,(x,y,z) in zip(symbols,coordinates)))
        argv=[str(command),geometry.name,'--gfn','2','--chrg',str(int(charge)),'--uhf',str(int(multiplicity)-1),'--acc',str(accuracy),'--norestart']
        if forces:argv.append('--grad')
        if len(mm):
            (directory/'charges.pc').write_text(f'{len(mm)}\n'+''.join(f'{q:.12f} {x:.12f} {y:.12f} {z:.12f} 99\n' for q,(x,y,z) in zip(charges,mm)))
            (directory/'embedding.inp').write_text('$embedding\n input=charges.pc\n interface=orca\n$end\n')
            argv += ['--input','embedding.inp']
        env=os.environ.copy();env['OMP_NUM_THREADS']=str(threads)
        proc=subprocess.run(argv,cwd=directory,env=env,text=True,capture_output=True,timeout=timeout)
        (directory/'stdout.txt').write_text(proc.stdout);(directory/'stderr.txt').write_text(proc.stderr)
        converged=bool(re.search(r'convergence criteria satisfied|SCC converged',proc.stdout,re.IGNORECASE))
        if proc.returncode or (directory/'NOT_CONVERGED').exists() or not converged:
            raise RuntimeError(f'xTB failed/convergence not established; retained {directory}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}')
        if forces:
            energy,qforce=read_engrad(directory/'molecule.engrad',len(symbols))
            if len(mm):
                values=np.loadtxt(directory/'pcgrad',ndmin=2)
                if values.shape!=(len(mm),3) or not np.isfinite(values).all():raise ValueError('Invalid point-charge gradient')
                mforce=-values*Hartree/Bohr
            else:mforce=np.empty((0,3))
        else:
            values=re.findall(r'TOTAL ENERGY\s+([-+0-9.EeDd]+)',proc.stdout)
            if not values:raise ValueError('xTB total energy not found')
            energy=_number(values[-1])*Hartree;qforce=mforce=None
        success=True;return float(energy),qforce,mforce
    finally:
        if success:shutil.rmtree(directory)
