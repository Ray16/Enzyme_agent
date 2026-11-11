"""ASE adapter to the installed native SANDER/DFTB3 QM/MM implementation."""
import os
from pathlib import Path
import subprocess
import re
import numpy as np
from ase.calculators.calculator import Calculator,all_changes
from .sander_worker import send,receive


def scc_converged(output):
    return bool(re.search(r'SCC-DFTB for step\s+\d+\s+converged in\s+\d+\s+cycles',output,re.IGNORECASE))


def electronic_converged(output,method):
    if method=='DFTB3':return scc_converged(output)
    if method=='AM1/D':
        # Native NDDO can continue after failure, so an energy line alone is
        # insufficient. Require its explicit final convergence message.
        return bool(re.search(r'QMMM:\s*SCF Converged to\s+[-+0-9.EeDd]+\s+in:\s*\d+\s+Cycles',output,re.IGNORECASE)) and not bool(re.search(r'unconverged SCF|No convergence in SCF|Unable to achieve self consistency',output,re.IGNORECASE))
    return False


class SanderQMMM(Calculator):
    implemented_properties=['energy','forces']
    def __init__(self,prmtop,positions,box,region,*,workdir,method='DFTB3',native_python=None,electronic_temperature=0.0,max_scc_iterations=200):
        super().__init__()
        if method not in {'DFTB3','AM1/D'}:raise ValueError('Unsupported native QM method/status parser')
        self.method=method
        self.logfile=Path(workdir)/'sander_native.log';self.logfile.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.logfile.open('a');env=os.environ.copy()
        prefix=Path(os.environ.get('BARRIER_AMBERHOME','/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mlmm-toolkit'))
        env['AMBERHOME']=str(prefix)
        env['GFORTRAN_UNBUFFERED_ALL']='1'
        py=native_python or str(prefix/'bin/python')
        # Absolute worker file avoids depending on native-env package installation.
        self.proc=subprocess.Popen([py,'-u',str(Path(__file__).with_name('sander_worker.py'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env=env,cwd=workdir)
        send(self.proc.stdin,dict(prmtop=str(Path(prmtop).resolve()),positions=positions,box=list(box) if box is not None else None,
                periodic=box is not None,qm_indices=region['indices'],charge=region['charge'],multiplicity=region.get('multiplicity',1),method=method,dftb_telec=electronic_temperature,dftb_maxiter=max_scc_iterations))
        result=receive(self.proc.stdout)
        if result is None or not result.get('ready'):raise RuntimeError(f'SANDER setup failed; inspect {self.logfile}')
        self.setup=result;self.last_components={}
    def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        offset=self.logfile.stat().st_size
        send(self.proc.stdin,dict(command='evaluate',positions=atoms.positions))
        result=receive(self.proc.stdout)
        if result is None:raise RuntimeError(f'SANDER force call failed; inspect {self.logfile}')
        with self.logfile.open('rb') as log:log.seek(offset);scf_output=log.read().decode(errors='replace')
        if not electronic_converged(scf_output,self.method):raise RuntimeError(f'No positive electronic convergence evidence; reject energy/forces: {self.logfile}')
        if not np.isfinite(result['energy_kcal']) or not np.isfinite(result['forces_kcal_A']).all():
            raise RuntimeError(f'Nonfinite native energy/forces: {self.logfile}')
        self.results=dict(energy=result['energy_kcal']/23.06054783,forces=result['forces_kcal_A']/23.06054783)
        self.last_components=result['components_kcal']|{'wall_s':result['wall_s']}
    def close(self):
        if self.proc.poll() is None:
            send(self.proc.stdin,dict(command='stop'));self.proc.wait(timeout=30)
        self.log.close()


def benchmark(snapshot,out,force_check=False,electronic_temperature=0.0,method='DFTB3'):
    import json,time
    from .jobs import load_snapshot,atomic_json
    p,r,a,d=load_snapshot(snapshot)
    # Use the equilibrated frame's cell, rather than the original preparation cell.
    from .coordinates import load_system
    actual=load_system(d['prmtop'],d['coordinates'])
    box=actual.box.copy()
    if p.pointers.get('IFBOX')==2:
        # PDB rounds 109.4712206 to 109.47; native Amber validates IFBOX=2
        # against the exact truncated-octahedron angle.
        box[3:]=np.degrees(np.arccos(-1/3))
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    calc=SanderQMMM(d['prmtop'],a.positions,box,r,workdir=str(out.resolve()),electronic_temperature=electronic_temperature,method=method);a.calc=calc
    rows=[]
    try:
        for k in range(4):
            t0=time.monotonic()
            if k:a.positions[r['indices'][0],0]+=0.001
            f=a.get_forces();e=a.get_potential_energy();rows.append(dict(energy_eV=e,max_force_eV_A=float(np.linalg.norm(f,axis=1).max()),wall_s=time.monotonic()-t0,components=calc.last_components))
        result=dict(method=method+'/MM',setup=calc.setup,evaluations=rows,validated_activation_free_energy=False)
        if force_check:
            reference=a.positions.copy();analytic=a.get_forces().copy();checks=[]
            ids=[r['reaction_atoms'][0]]
            if r['links']:ids.append(r['links'][0]['mm'])
            ids += [at.idx for at in p.atoms if at.atomic_number==12][:1]
            for atom in ids:
                axis=0;epsilon=1e-4
                a.positions[:]=reference;a.positions[atom,axis]+=epsilon;ep=a.get_potential_energy()
                a.positions[:]=reference;a.positions[atom,axis]-=epsilon;em=a.get_potential_energy()
                numeric=float(-(ep-em)/(2*epsilon));error=abs(float(analytic[atom,axis])-numeric)
                checks.append(dict(atom=atom,axis=axis,analytic_eV_A=float(analytic[atom,axis]),numeric_eV_A=numeric,error_eV_A=error))
            result['force_checks']=checks;result['passed']=max(c['error_eV_A'] for c in checks)<.002
        atomic_json(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
    finally:calc.close()


def initialization_probe(snapshot,out,electronic_temperature=0.0):
    """Diagnostic repeats at IDENTICAL coordinates; no failed-SCC force is accepted.

    Used to separate initialization/coupled-field effects from geometry-induced
    state changes. Raw invalid iterations are retained, never used for dynamics.
    """
    import time,json
    from .jobs import load_snapshot,atomic_json
    p,r,a,d=load_snapshot(snapshot);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    box=d['box_lengths_angles'];rows=[];previous=None
    calc=SanderQMMM(d['prmtop'],a.positions,box,r,workdir=str(out.resolve()),electronic_temperature=electronic_temperature,max_scc_iterations=70)
    try:
        for iteration in range(4):
            offset=calc.logfile.stat().st_size;t0=time.monotonic()
            send(calc.proc.stdin,dict(command='evaluate',positions=a.positions))
            result=receive(calc.proc.stdout)
            if result is None:raise RuntimeError('Native initialization probe terminated')
            with calc.logfile.open('rb') as f:f.seek(offset);text=f.read().decode(errors='replace')
            force=result['forces_kcal_A']/23.06054783;energy=result['energy_kcal']/23.06054783
            row=dict(iteration=iteration,energy_eV=energy,scc_converged=scc_converged(text),wall_s=time.monotonic()-t0,
                     max_qm_force_eV_A=float(np.linalg.norm(force[r['indices']],axis=1).max()))
            if previous is not None:
                row['energy_change_eV']=energy-previous[0];row['max_force_change_eV_A']=float(np.max(np.abs(force-previous[1])))
            rows.append(row);previous=(energy,force);atomic_json(out/'progress.json',dict(iterations=rows));print(json.dumps(row),flush=True)
        atomic_json(out/'result.json',dict(scope='initialization_diagnostic_only',electronic_temperature_K=electronic_temperature,iterations=rows,
             last_pair_stable=rows[-1]['scc_converged'] and rows[-2]['scc_converged'] and abs(rows[-1]['energy_change_eV'])<1e-5 and rows[-1]['max_force_change_eV_A']<1e-4,
             validated_activation_free_energy=False))
    finally:calc.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);p.add_argument('--force-check',action='store_true')
    p.add_argument('--electronic-temperature',type=float,default=0.0)
    p.add_argument('--initialization-probe',action='store_true')
    p.add_argument('--method',default='DFTB3',choices=['DFTB3','AM1/D'])
    a=p.parse_args()
    if a.initialization_probe:initialization_probe(a.snapshot,a.out,a.electronic_temperature)
    else:benchmark(a.snapshot,a.out,force_check=a.force_check,electronic_temperature=a.electronic_temperature,method=a.method)
