"""Bounded relaxed-path exploration with a conserved embedded Hamiltonian.

These are input-generating calculations, not validated activation barriers.
Two-dimensional seeds are allowed to represent different mechanistic hypotheses.
Every optimization records actual convergence and preserves partial geometries.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .jobs import load_snapshot,atomic_json,sha256


def explore(snapshot,out,*,route='concerted',images=9,max_steps=60,wall_seconds=3600,engine='uma-xtb'):
    from ase.constraints import FixAtoms,FixBondLengths
    from ase.optimize import LBFGS
    from .sampler import EmbeddedUMAXTB
    p,r,a,meta=load_snapshot(snapshot);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if engine=='uma-xtb':a.calc=EmbeddedUMAXTB(p,r,platform='CPU')
    elif engine=='dftb3':
        from .sander_backend import SanderQMMM
        a.calc=SanderQMMM(meta['prmtop'],a.positions,meta['box_lengths_angles'],r,workdir=str(out.resolve()))
    else:raise ValueError('Unknown reference engine')
    mobile=set(r['mobile_indices']);frozen=[i for i in range(len(a)) if i not in mobile]
    i,j,k=r['reaction_atoms'];t0=time.monotonic()
    # Each moved group follows topology, excluding the explicitly changing bond.
    graph={n:[] for n in range(len(a))}
    for bond in p.bonds:
        x,y=bond.atom1.idx,bond.atom2.idx
        if {x,y}=={j,k}:continue
        graph[x].append(y);graph[y].append(x)
    def component(start):
        seen={start};queue=[start]
        while queue:
            for nxt in graph[queue.pop()]:
                if nxt not in seen:seen.add(nxt);queue.append(nxt)
        return sorted(seen)
    forming_group=component(i);leaving_group=component(k)
    if j in forming_group or j in leaving_group:raise ValueError('Invalid reactive bond graph')
    if not set(forming_group+leaving_group)<=set(r['indices']):raise ValueError('Reactive fragments must be wholly QM')
    df0=float(a.get_distance(i,j));db0=float(a.get_distance(j,k))
    targets=[]
    for lam in np.linspace(0,1,images):
        if route=='concerted':lf=lb=lam
        elif route=='form-first':lf=min(2*lam,1);lb=max(2*lam-1,0)
        elif route=='break-first':lb=min(2*lam,1);lf=max(2*lam-1,0)
        else:raise ValueError('Unknown route')
        targets.append((df0+(1.65-df0)*lf,db0+(3.2-db0)*lb))
    records=[]
    for image,(df,db) in enumerate(targets):
        if time.monotonic()-t0>wall_seconds:break
        a.set_constraint()
        for moving,first,second,target in [(forming_group,i,j,df),(leaving_group,k,j,db)]:
            vector=a.positions[first]-a.positions[second];current=np.linalg.norm(vector)
            a.positions[moving]+=(target/current-1)*vector
        a.set_constraint([FixBondLengths([(i,j),(j,k)],bondlengths=[df,db],tolerance=1e-8),FixAtoms(indices=frozen)])
        opt=LBFGS(a,logfile=str(out/f'image{image:03d}.opt.log'),maxstep=0.08)
        def save():
            np.save(out/f'image{image:03d}.positions.npy',a.positions)
            if time.monotonic()-t0>wall_seconds:raise TimeoutError('Relaxed-path wall budget reached')
        opt.attach(save,interval=5)
        try:converged=bool(opt.run(fmax=0.1,steps=max_steps))
        except TimeoutError:
            save_path=out/f'image{image:03d}.positions.npy';np.save(save_path,a.positions);break
        np.save(out/f'image{image:03d}.positions.npy',a.positions)
        row=dict(image=image,forming_A=float(a.get_distance(i,j)),breaking_A=float(a.get_distance(j,k)),
                 energy_eV=float(a.get_potential_energy()),optimization_converged=converged,
                 steps=opt.nsteps,components=a.calc.last_components,high_diagnostics=getattr(a.calc,'high_diagnostics',{'potential':'DFTB3/3OB/MM'}),
                 positions=str((out/f'image{image:03d}.positions.npy').resolve()))
        records.append(row);atomic_json(out/'progress.json',dict(images=records,wall_s=time.monotonic()-t0))
        print(json.dumps(row),flush=True)
    complete=len(records)==images
    result=dict(snapshot=str(Path(snapshot).resolve()),snapshot_sha256=sha256(Path(snapshot)/'input.json'),
       route=route,reference_engine=engine,requested_images=images,completed_images=len(records),images=records,
       all_optimizations_converged=complete and all(x['optimization_converged'] for x in records),
       status='path_exploration_complete' if complete else 'partial_wall_budget',
       validated_activation_free_energy=False,quantity='reference_potential_constrained_path',
       limitations=['imposed seed path','uncalibrated reference potential','finite embedding sphere'],wall_s=time.monotonic()-t0)
    if records:result['diagnostic_max_gap_kcal']=(max(x['energy_eV'] for x in records)-records[0]['energy_eV'])*23.06054783
    atomic_json(out/'result.json',result)
    if hasattr(a.calc,'close'):a.calc.close()
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    p.add_argument('--route',default='concerted',choices=['concerted','form-first','break-first'])
    p.add_argument('--images',type=int,default=9);p.add_argument('--max-steps',type=int,default=60);p.add_argument('--wall-seconds',type=int,default=3600)
    p.add_argument('--engine',default='uma-xtb',choices=['uma-xtb','dftb3'])
    a=p.parse_args(argv);print(json.dumps(explore(a.snapshot,a.out,route=a.route,images=a.images,max_steps=a.max_steps,wall_seconds=a.wall_seconds,engine=a.engine),indent=2))


if __name__=='__main__':main()
