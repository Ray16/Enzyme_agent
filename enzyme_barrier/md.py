"""Independent unrestrained classical pilots, with C4 checks and durable checkpoints."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .jobs import atomic_json,sha256


def write_state_pdb(topology,state,path):
    """Write the current state's cell, never the initial topology's stale cell."""
    import openmm.app as app
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with Path(path).open('w') as f:app.PDBFile.writeFile(topology,state.getPositions(),f)


def resolve_attack_geometry(parm,spec):
    """Resolve and validate angle atoms before creating an expensive MD context."""
    from .carve import _resolve
    j=_resolve(parm,spec.electrophile);k=_resolve(parm,spec.leaving)
    i=_resolve(parm,spec.nucleophile,near_xyz=parm.coordinates[j],xyz=parm.coordinates)
    if parm.atoms[j].atomic_number==6:
        if spec.carbonyl_oxygen is None:
            raise ValueError('Carbonyl attack requires the double-bonded oxygen, distinct from the leaving atom')
        angle_atom=_resolve(parm,spec.carbonyl_oxygen)
        oxygen=parm.atoms[angle_atom]
        heavy=[a.idx for a in oxygen.bond_partners if a.atomic_number>1]
        if angle_atom==k or oxygen.atomic_number!=8 or heavy!=[j]:
            raise ValueError('Carbonyl oxygen must be a distinct terminal oxygen bonded to the electrophile')
        angle_kind='nucleophile-carbonylC-carbonylO'
    elif parm.atoms[j].atomic_number==15:
        angle_atom=k;angle_kind='nucleophile-phosphorus-leavingO'
    else:raise ValueError('Unsupported electrophile for attack-angle reporting')
    return i,j,k,angle_atom,angle_kind


def equilibrate(prmtop,coordinates,spec_path,out,*,seed=17,production_ns=2.0,temperature=303.15):
    import parmed as pmd
    import openmm as mm
    import openmm.app as app
    from openmm import unit as u
    from .parameters import require_c4
    from .spec import ReactionSpec
    from .coordinates import load_system
    t0=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if production_ns<=0 or temperature<=0:raise ValueError('Production duration and temperature must be positive')
    parm=load_system(prmtop,coordinates);require_c4(parm)
    spec=ReactionSpec.load(spec_path)
    i,j,k,angle_atom,angle_kind=resolve_attack_geometry(parm,spec)
    positions_A=parm.coordinates.copy()
    print(f'Loaded {len(parm.atoms)} atoms in {time.monotonic()-t0:.1f}s',flush=True)
    system=parm.createSystem(nonbondedMethod=app.PME,nonbondedCutoff=1*u.nanometer,
                           constraints=app.HBonds,rigidWater=True)
    c4=[f for f in system.getForces() if isinstance(f,mm.CustomNonbondedForce) and 'ccoef' in f.getEnergyFunction()]
    if not c4:raise ValueError('C4 topology did not produce a C4 force')
    for f in c4:f.setForceGroup(31)
    restraint=mm.CustomExternalForce('0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2')
    restraint.addGlobalParameter('k',5*418.4)
    for name in ('x0','y0','z0'):restraint.addPerParticleParameter(name)
    for a in parm.atoms:
        if a.atomic_number>1 and a.residue.name not in {'WAT','HOH','Na+','Cl-'}:
            restraint.addParticle(a.idx,positions_A[a.idx]*0.1)
    system.addForce(restraint)
    barostat=mm.MonteCarloBarostat(1*u.bar,temperature*u.kelvin,25);barostat.setRandomNumberSeed(seed+10000);system.addForce(barostat)
    integ=mm.LangevinMiddleIntegrator(temperature*u.kelvin,1/u.picosecond,0.002*u.picosecond);integ.setRandomNumberSeed(seed)
    sim=app.Simulation(parm.topology,system,integ,mm.Platform.getPlatformByName('CUDA'),{'Precision':'mixed'})
    sim.context.setPositions(parm.positions)
    print(f'CUDA context ready after {time.monotonic()-t0:.1f}s; minimizing',flush=True)
    sim.minimizeEnergy(maxIterations=5000)
    minimized=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.angstrom)
    magnesium=[a.idx for a in parm.atoms if a.atomic_number==12]
    heavy=[a.idx for a in parm.atoms if a.atomic_number>1 and a.atomic_number!=12]
    for metal in magnesium:
        if np.linalg.norm(minimized[heavy]-minimized[metal],axis=1).min()<1.65:
            raise ValueError('Magnesium heavy-atom overlap remains after minimization; no dynamics launched')
    initial_c4=float(sim.context.getState(getEnergy=True,groups={31}).getPotentialEnergy().value_in_unit(u.kilocalorie_per_mole))
    if not np.isfinite(initial_c4) or initial_c4>=0:raise ValueError('C4 energy is not finite and attractive')
    print(f'Minimized after {time.monotonic()-t0:.1f}s; C4={initial_c4:.2f} kcal/mol',flush=True)
    sim.context.setVelocitiesToTemperature(temperature*u.kelvin,seed)
    # 200 ps staged release, then 200 ps completely unrestrained equilibration.
    for k in (5,2,0.5,0):sim.context.setParameter('k',k*418.4);sim.step(25000)
    sim.step(100000)
    sim.saveCheckpoint(str(out/'equilibrated.chk'))
    equil_state=sim.context.getState(getPositions=True)
    write_state_pdb(sim.topology,equil_state,out/'equilibrated.pdb')
    metals=[a.idx for a in parm.atoms if a.atomic_number==12]
    donors=[a.idx for a in parm.atoms if a.atomic_number in (7,8,16)]
    sim.reporters.append(app.DCDReporter(str(out/'production.dcd'),5000))
    sim.reporters.append(app.StateDataReporter(str(out/'thermodynamics.csv'),5000,step=True,potentialEnergy=True,temperature=True,density=True))
    total=int(round(production_ns*500000));done=0;observations=[]
    with (out/'geometry.jsonl').open('w') as log:
        while done<total:
            step=min(5000,total-done);sim.step(step);done+=step
            st=sim.context.getState(getPositions=True);x=st.getPositions(asNumpy=True).value_in_unit(u.angstrom)
            box=np.array(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.angstrom));inv=np.linalg.inv(box)
            def vec(a,b):
                v=x[a]-x[b];return (v@inv-np.round(v@inv))@box
            v1=vec(i,j);v2=vec(angle_atom,j);angle=np.degrees(np.arccos(np.clip(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2),-1,1)))
            coordination=[]
            for m in metals:
                v=x[donors]-x[m];v=(v@inv-np.round(v@inv))@box
                coordination.append(int(np.count_nonzero(np.linalg.norm(v,axis=1)<2.7)))
            row=dict(time_ps=done*0.002,forming_A=float(np.linalg.norm(v1)),attack_angle_deg=float(angle),metal_coordination=coordination)
            observations.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if done%50000==0:
                sim.saveCheckpoint(str(out/'latest.chk'))
                atomic_json(out/'progress.json',dict(completed_steps=done,requested_steps=total,wall_s=time.monotonic()-t0,last_geometry=row))
                print(json.dumps(row),flush=True)
    sim.saveCheckpoint(str(out/'final.chk'))
    final_state=sim.context.getState(getPositions=True,getVelocities=True)
    box_vectors=final_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.angstrom)
    lengths=np.linalg.norm(box_vectors,axis=1)
    angles=[np.degrees(np.arccos(np.clip(np.dot(box_vectors[a],box_vectors[b])/lengths[a]/lengths[b],-1,1))) for a,b in [(1,2),(0,2),(0,1)]]
    np.savez_compressed(out/'final.npz',positions_A=final_state.getPositions(asNumpy=True).value_in_unit(u.angstrom),
                        velocities=final_state.getVelocities(asNumpy=True).value_in_unit(u.angstrom/u.picosecond),
                        box_vectors_A=box_vectors,box_lengths_angles=np.r_[lengths,angles])
    write_state_pdb(sim.topology,final_state,out/'final.pdb')
    coords=np.array([r['metal_coordination'] for r in observations]);dist=np.array([r['forming_A'] for r in observations])
    result=dict(status='pilot_complete_requires_scientific_review',seed=seed,temperature_K=temperature,
        production_ns=production_ns,equilibration_ns=0.4,n_atoms=len(parm.atoms),magnesium_count=len(metals),
        coordination_definition='N/O/S donor atoms within 2.7 A (minimum image)',
        attack_angle_definition=angle_kind,
        c4_initial_kcal=initial_c4,metal_coordination_mean=coords.mean(axis=0).tolist(),metal_coordination_std=coords.std(axis=0).tolist(),
        forming_distance_mean_A=float(dist.mean()),forming_distance_std_A=float(dist.std()),
        unrestrained_production=True,wall_s=time.monotonic()-t0,
        input_sha256={str(Path(f).resolve()):sha256(f) for f in [prmtop,coordinates,spec_path]},
        final_coordinates=str((out/'final.pdb').resolve()),validated_activation_free_energy=False)
    atomic_json(out/'result.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['prmtop','coords','spec','out']:p.add_argument('--'+flag,required=True)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--ns',type=float,default=2)
    a=p.parse_args(argv);print(json.dumps(equilibrate(a.prmtop,a.coords,a.spec,a.out,seed=a.seed,production_ns=a.ns),indent=2))


if __name__=='__main__':main()
