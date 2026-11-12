"""Classify path states, then relax genuine endpoint seeds without reaction restraints."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .jobs import load_snapshot,atomic_json,sha256
from .chemistry import reaction_state


def transfer_mode(positions,reaction_atoms,mobile_indices):
    """A deterministic bond-exchange direction, without environmental path noise."""
    x=np.asarray(positions,float);i,j,k=reaction_atoms;mode=np.zeros_like(x)
    for a,b,sign in [(i,j,-1),(j,k,1)]:
        v=x[a]-x[b];distance=np.linalg.norm(v)
        if distance<1e-8:raise ValueError('Coincident reaction atoms')
        mode[a]+=sign*v/distance;mode[b]-=sign*v/distance
    fixed=np.ones(len(x),bool);fixed[list(mobile_indices)]=False;mode[fixed]=0
    norm=np.linalg.norm(mode)
    if norm<1e-8:raise ValueError('Reaction mode has no mobile component')
    return mode/norm


def mode_diagnostics(atoms,mode,*,step=.002):
    """Central force difference along one mode; this does not count all saddles."""
    v=np.asarray(mode,float).copy();norm=np.linalg.norm(v)
    if norm<1e-8 or step<=0:raise ValueError('Invalid mode or difference step')
    v/=norm;x=atoms.positions.copy()
    try:
        atoms.set_positions(x+step*v);fp=atoms.get_forces()
        atoms.set_positions(x-step*v);fm=atoms.get_forces()
    finally:atoms.set_positions(x)
    hv=-(fp-fm)/(2*step);curvature=float(np.sum(v*hv))
    residual=float(np.linalg.norm(hv-curvature*v))
    return dict(curvature_eV_A2=curvature,eigenvector_residual_eV_A2=residual,
                finite_difference_step_A=step,negative_curvature=curvature<0,
                scope='one projected Cartesian mode; full Hessian index and IRC still required')


def refine_dimer(atoms,mode,out,*,max_steps=250,fmax=.03,wall_seconds=5400):
    """Refine a seed using only frozen outer atoms, with explicit real-force QC."""
    from ase.mep import DimerControl,MinModeAtoms,MinModeTranslate
    out=Path(out);out.mkdir(parents=True,exist_ok=False);t0=time.monotonic()
    seed_mode=np.asarray(mode,float);seed_mode=seed_mode/np.linalg.norm(seed_mode)
    with DimerControl(logfile=str(out/'rotation.log'),eigenmode_logfile=None,
            dimer_separation=.002,maximum_translation=.06,trial_trans_step=.001,
            max_num_rot=4,f_rot_min=.02,f_rot_max=.1,use_central_forces=False,
            extrapolate_forces=False) as control:
        dimer=MinModeAtoms(atoms,control,eigenmodes=[seed_mode.copy()])
        with MinModeTranslate(dimer,logfile=str(out/'optimization.log')) as opt:
            def save():
                np.save(out/'latest.positions.npy',atoms.positions)
                np.save(out/'latest.mode.npy',dimer.get_eigenmode())
                atomic_json(out/'progress.json',dict(steps=opt.nsteps,wall_s=time.monotonic()-t0,
                    dimer_curvature_eV_A2=float(dimer.get_curvature())))
                if time.monotonic()-t0>wall_seconds:raise TimeoutError('Saddle wall budget reached')
            opt.attach(save,interval=1)
            try:optimizer_converged=bool(opt.run(fmax=fmax,steps=max_steps));status='saddle_search_finished'
            except TimeoutError:optimizer_converged=False;status='partial_wall_budget'
            final_mode=dimer.get_eigenmode().copy()
            # Physical force, not the reflected force used by the dimer optimizer.
            physical_fmax=float(np.linalg.norm(atoms.get_forces(),axis=1).max())
            diagnostics=mode_diagnostics(atoms,final_mode)
            np.save(out/'final.positions.npy',atoms.positions);np.save(out/'final.mode.npy',final_mode)
            result=dict(status=status,optimizer_converged=optimizer_converged,
                stationary_candidate=optimizer_converged and physical_fmax<=fmax and diagnostics['negative_curvature'],
                max_free_force_eV_A=physical_fmax,energy_eV=float(atoms.get_potential_energy()),
                steps=opt.nsteps,wall_s=time.monotonic()-t0,mode_diagnostics=diagnostics,
                seed_mode_overlap=float(abs(np.sum(final_mode*seed_mode))),
                positions=str((out/'final.positions.npy').resolve()),mode=str((out/'final.mode.npy').resolve()),
                reaction_restraints=False,validated_activation_free_energy=False,
                transition_state_certified=False,limitations=['Full Hessian index and reaction connectivity not certified',
                    'Reference potential energy; no thermal or ensemble free energy'])
            atomic_json(out/'result.json',result);return result


def saddle(snapshot,start,out,*,max_steps=250,fmax=.03,wall_seconds=5400):
    from ase.constraints import FixAtoms
    from .sander_backend import SanderQMMM
    p,r,a,m=load_snapshot(snapshot);a.positions[:]=np.load(start)
    a.set_constraint(FixAtoms(indices=sorted(set(range(len(a)))-set(r['mobile_indices']))))
    # Backend log uses a separate sibling directory: refine_dimer creates out.
    native=Path(str(out)+'_native');native.mkdir(parents=True,exist_ok=False)
    a.calc=SanderQMMM(m['prmtop'],a.positions,m['box_lengths_angles'],r,workdir=str(native.resolve()),method='AM1/D')
    try:
        result=refine_dimer(a,transfer_mode(a.positions,r['reaction_atoms'],r['mobile_indices']),out,
            max_steps=max_steps,fmax=fmax,wall_seconds=wall_seconds)
        result.update(snapshot=str(Path(snapshot).resolve()),snapshot_sha256=sha256(Path(snapshot)/'input.json'),
            start_sha256=sha256(start),method='AM1/D',chemical_state=reaction_state(p,a.positions,r))
        atomic_json(Path(out)/'result.json',result);return result
    finally:a.calc.close()


def summarize(path,out):
    data=json.loads(Path(path).read_text());p,r,a,m=load_snapshot(data['snapshot']);rows=[]
    for image in data['images']:
        positions=np.load(image['positions']);state=reaction_state(p,positions,r)
        rows.append(dict(image=image['image'],energy_eV=image['energy_eV'],positions=image['positions'],optimization_converged=image['optimization_converged'],chemical_state=state))
    products=[i for i,row in enumerate(rows) if row['chemical_state']['state'] in {'product','activated_product'}]
    candidates=[i for i,row in enumerate(rows) if row['chemical_state']['state']=='reactant' and (not products or i<products[0])]
    result=dict(path=str(Path(path).resolve()),path_sha256=sha256(path),snapshot=data['snapshot'],images=rows,validated_activation_free_energy=False)
    if not rows or not all(row['optimization_converged'] and np.isfinite(row['energy_eV']) for row in rows):
        result.update(status='path_optimization_unresolved',failures=['all path images must have converged finite energies before seed selection'])
        atomic_json(out,result);return result
    if candidates and products:
        ir=min(candidates,key=lambda i:rows[i]['energy_eV']);ip=min([i for i in products if i>ir],key=lambda i:rows[i]['energy_eV'])
        it=max(range(ir,ip+1),key=lambda i:rows[i]['energy_eV'])
        result.update(reactant_seed=rows[ir],product_seed=rows[ip],saddle_seed=rows[it],has_interior_maximum=ir<it<ip,
            constrained_path_gap_kcal=(rows[it]['energy_eV']-rows[ir]['energy_eV'])*23.06054783,
            status='endpoint_seeds_identified_by_geometry',quantity='provisional_constrained_path_gap')
    else:result.update(status='chemical_endpoints_unresolved',failures=['both chemical basins were not identified'])
    atomic_json(out,result);return result


def optimize(snapshot,start,out,*,method='AM1/D',engine='native',basis='def2-svp',
             hold_reaction_distances=False,max_steps=350,fmax=.03,wall_seconds=3600):
    from ase.constraints import FixAtoms,FixBondLengths
    from ase.optimize import LBFGS
    p,r,a,m=load_snapshot(snapshot);reference=a.positions.copy();positions=np.load(start)
    if positions.shape!=a.positions.shape or not np.isfinite(positions).all():raise ValueError('Invalid starting geometry')
    frozen=sorted(set(range(len(a)))-set(r['mobile_indices']))
    if frozen and not np.allclose(positions[frozen],reference[frozen],atol=1e-7,rtol=0):
        raise ValueError('Frozen environment differs from the snapshot reference')
    if engine not in {'native','dft'}:raise ValueError('Unknown optimization engine')
    a.positions[:]=positions;out=Path(out);out.mkdir(parents=True,exist_ok=False)
    before=reaction_state(p,a.positions,r);constraints=[];held=[]
    if hold_reaction_distances:
        i,j,k=r['reaction_atoms'];pairs=[(i,j),(j,k)];held=[float(a.get_distance(*pair)) for pair in pairs]
        constraints.append(FixBondLengths(pairs,bondlengths=held,tolerance=1e-8))
    constraints.append(FixAtoms(indices=frozen));a.set_constraint(constraints)
    if engine=='dft':
        from .embedded import EmbeddedDFTMM
        a.calc=EmbeddedDFTMM(p,r,xc='wb97m-v',basis=basis,engine='gpu',platform='CUDA',
            conv_tol=1e-10,grid_response=True,mm_cutoff_nm=1.2,mm_switch_nm=1.0)
        method='wb97m-v/'+basis+'/electrostatically_embedded_DFT_MM'
    else:
        from .sander_backend import SanderQMMM
        a.calc=SanderQMMM(m['prmtop'],a.positions,m['box_lengths_angles'],r,workdir=str(out.resolve()),method=method)
    t0=time.monotonic();opt=LBFGS(a,logfile=str(out/'optimization.log'),maxstep=.06)
    def save():
        np.save(out/'latest.positions.npy',a.positions)
        atomic_json(out/'progress.json',dict(steps=opt.nsteps,wall_s=time.monotonic()-t0,
            energy_eV=float(a.get_potential_energy()),projected_fmax_eV_A=float(np.linalg.norm(a.get_forces(),axis=1).max()),
            chemical_state=reaction_state(p,a.positions,r),components=a.calc.last_components,
            reaction_restraints=hold_reaction_distances))
        if time.monotonic()-t0>wall_seconds:raise TimeoutError('Endpoint wall budget reached')
    opt.attach(save,interval=1)
    try:
        try:converged=bool(opt.run(fmax=fmax,steps=max_steps));status='endpoint_optimization_finished'
        except TimeoutError:converged=False;status='partial_wall_budget'
        np.save(out/'final.positions.npy',a.positions);forces=a.get_forces()
        result=dict(status=status,optimization_converged=converged,energy_eV=float(a.get_potential_energy()),
            max_free_force_eV_A=float(np.linalg.norm(forces,axis=1).max()),steps=opt.nsteps,before=before,
            chemical_state=reaction_state(p,a.positions,r),snapshot=str(Path(snapshot).resolve()),
            snapshot_sha256=sha256(Path(snapshot)/'input.json'),start_sha256=sha256(start),method=method,
            positions=str((out/'final.positions.npy').resolve()),wall_s=time.monotonic()-t0,
            reaction_restraints=hold_reaction_distances,held_reaction_distances_A=held,engine=engine,
            physical_mobile_fmax_eV_A=float(np.linalg.norm(a.get_forces(apply_constraint=False)[r['mobile_indices']],axis=1).max()),
            components=a.calc.last_components,transition_state_certified=False,validated_activation_free_energy=False)
        atomic_json(out/'result.json',result);return result
    finally:
        if hasattr(a.calc,'close'):a.calc.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('summarize');a.add_argument('--path',required=True);a.add_argument('--out',required=True)
    b=s.add_parser('optimize');b.add_argument('--snapshot',required=True);b.add_argument('--start',required=True);b.add_argument('--out',required=True)
    b.add_argument('--max-steps',type=int,default=350);b.add_argument('--fmax',type=float,default=.03);b.add_argument('--wall-seconds',type=int,default=3600)
    b.add_argument('--engine',choices=['native','dft'],default='native');b.add_argument('--basis',default='def2-svp')
    b.add_argument('--hold-reaction-distances',action='store_true')
    c=s.add_parser('saddle');c.add_argument('--snapshot',required=True);c.add_argument('--start',required=True);c.add_argument('--out',required=True)
    c.add_argument('--max-steps',type=int,default=250);c.add_argument('--fmax',type=float,default=.03);c.add_argument('--wall-seconds',type=int,default=5400)
    a=p.parse_args()
    if a.cmd=='summarize':value=summarize(a.path,a.out)
    elif a.cmd=='optimize':value=optimize(a.snapshot,a.start,a.out,max_steps=a.max_steps,fmax=a.fmax,
        wall_seconds=a.wall_seconds,engine=a.engine,basis=a.basis,hold_reaction_distances=a.hold_reaction_distances)
    else:value=saddle(a.snapshot,a.start,a.out,max_steps=a.max_steps,fmax=a.fmax,wall_seconds=a.wall_seconds)
    print(json.dumps(value,indent=2))
