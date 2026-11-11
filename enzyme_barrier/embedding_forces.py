"""Finite differences of the actual embedded DFT/MM enzyme Hamiltonian."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .jobs import atomic_json,sha256,load_snapshot


def force_checks(atoms,checks,*,step_A=1e-4,tolerance_eV_A=.002,on_result=None):
    if step_A<=0 or tolerance_eV_A<=0:raise ValueError('Invalid finite difference settings')
    x=atoms.positions.copy();analytic=atoms.get_forces().copy();rows=[]
    try:
        for check in checks:
            i=int(check['atom_zero_based']);axis=int(check['axis'])
            if not 0<=i<len(atoms) or axis not in (0,1,2):raise ValueError('Invalid derivative component')
            for sign in (1,-1):
                displaced=x.copy();displaced[i,axis]+=sign*step_A;atoms.set_positions(displaced)
                energy=float(atoms.get_potential_energy())
                if sign==1:ep=energy
                else:em=energy
            numeric=-(ep-em)/(2*step_A);error=abs(float(analytic[i,axis])-numeric)
            row=dict(check,analytic_eV_A=float(analytic[i,axis]),numeric_eV_A=numeric,error_eV_A=error,
                passed=bool(np.isfinite(error) and error<tolerance_eV_A));rows.append(row)
            if on_result:on_result(rows)
    finally:atoms.set_positions(x)
    return dict(passed=bool(rows) and all(r['passed'] for r in rows),checks=rows,
        step_A=step_A,tolerance_eV_A=tolerance_eV_A,scope='selected Cartesian derivatives, not a full Hessian or accuracy proof')


def run(config_path):
    from .embedded import EmbeddedDFTMM
    cfg=json.loads(Path(config_path).read_text());p,r,a,m=load_snapshot(cfg['snapshot'])
    if sha256(cfg['positions'])!=cfg['positions_sha256']:raise ValueError('Geometry changed')
    positions=np.load(cfg['positions'])
    if positions.shape!=a.positions.shape or not np.isfinite(positions).all():raise ValueError('Invalid geometry')
    a.positions[:]=positions;out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    a.calc=EmbeddedDFTMM(p,r,xc=cfg.get('xc','wb97m-v'),basis=cfg.get('basis','def2-svp'),
        engine=cfg.get('engine','gpu'),platform='CPU',conv_tol=1e-10,grid_response=cfg.get('grid_response',True))
    t0=time.monotonic()
    def save(rows):atomic_json(out/'progress.json',dict(checks=rows,wall_s=time.monotonic()-t0))
    result=force_checks(a,cfg['checks'],step_A=cfg.get('step_A',1e-4),
        tolerance_eV_A=cfg.get('tolerance_eV_A',.002),on_result=save)
    result.update(config_sha256=sha256(config_path),snapshot_sha256=sha256(Path(cfg['snapshot'])/'input.json'),
        positions_sha256=cfg['positions_sha256'],xc=cfg.get('xc','wb97m-v'),basis=cfg.get('basis','def2-svp'),
        grid_response=cfg.get('grid_response',True),wall_s=time.monotonic()-t0,validated_activation_free_energy=False)
    atomic_json(out/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config');args=parser.parse_args()
    result=run(args.config);print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)
