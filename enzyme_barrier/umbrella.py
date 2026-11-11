"""Conservative distance-CV umbrellas with checkpointed, resumable chunks.

An exploratory reference PMF is never promoted without target-level ensemble
correction and validation. Use only through the reserved worker launcher.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ase.calculators.calculator import Calculator,all_changes
from .jobs import atomic_json,sha256,load_snapshot


def distance_cv(positions,terms):
    """Sum coefficient*distance, with its exact Cartesian gradient.

    Covers forming-minus-breaking and donor-H-minus-H-acceptor coordinates.
    """
    x=np.asarray(positions);gradient=np.zeros_like(x);value=0.
    for term in terms:
        i,j,coefficient=term;v=x[i]-x[j];distance=float(np.linalg.norm(v))
        if distance<1e-8:raise ValueError('Coincident CV atoms')
        value+=coefficient*distance;g=coefficient*v/distance;gradient[i]+=g;gradient[j]-=g
    return value,gradient


class UmbrellaCalculator(Calculator):
    implemented_properties=['energy','forces']
    def __init__(self,base,coordinates,centers,springs):
        super().__init__();self.base=base;self.coordinates=coordinates
        self.centers=np.asarray(centers,float);self.springs=np.asarray(springs,float)/23.06054783
        if len(self.coordinates)!=len(self.centers) or self.centers.shape!=self.springs.shape:raise ValueError('CV dimensions differ')
    def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        self.base.calculate(atoms,properties,system_changes)
        energy=float(self.base.results['energy']);force=self.base.results.get('forces')
        if force is not None:force=force.copy()
        bias=0.;values=[]
        for cv,center,spring in zip(self.coordinates,self.centers,self.springs):
            value,gradient=distance_cv(atoms.positions,cv);values.append(value);d=value-center
            bias+=.5*spring*d*d
            if force is not None:force-=spring*d*gradient
        self.unbiased_energy_eV=energy;self.bias_eV=bias;self.cv_values=values
        self.results={'energy':energy+bias}
        if force is not None:self.results['forces']=force


def sample(config_path):
    from ase import units
    from ase.constraints import FixAtoms
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from .sampler import EmbeddedUMAXTB
    cfg=json.loads(Path(config_path).read_text());out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    if cfg.get('sampling_role')!='reference_only':raise ValueError('Reference sampler requires explicit reference_only scope')
    p,r,a,d=load_snapshot(cfg['snapshot']);identity=sha256(config_path)
    mobile=r['mobile_indices'];frozen=sorted(set(range(len(a)))-set(mobile));a.set_constraint(FixAtoms(indices=frozen))
    a.calc=UmbrellaCalculator(EmbeddedUMAXTB(p,r,platform='CPU'),cfg['coordinates'],cfg['centers'],cfg['springs_kcal_A2'])
    rng=np.random.default_rng(cfg['seed']);done=0;chunk=0
    previous=out/'checkpoint.json'
    producer={name:sha256(Path(__file__).with_name(name)) for name in ['umbrella.py','embedded.py','sampler.py','region.py']}
    if previous.exists():
        cp=json.loads(previous.read_text())
        if cp['config_sha256']!=identity:raise ValueError('Configuration changed; create a versioned window')
        if cp.get('producer_sha256')!=producer:raise ValueError('Producer changed; create a versioned window')
        state=np.load(out/cp['checkpoint_file']);a.positions[:]=state['positions'];a.set_velocities(state['velocities'])
        rng.bit_generator.state=cp['rng_state'];done=cp['completed_steps'];chunk=cp['chunks']
    else:
        if cfg.get('start_positions'):a.positions[:]=np.load(cfg['start_positions'])
        MaxwellBoltzmannDistribution(a,temperature_K=cfg.get('temperature_K',303.15),rng=rng)
    dyn=Langevin(a,cfg.get('dt_fs',.5)*units.fs,temperature_K=cfg.get('temperature_K',303.15),
                 friction=.02,rng=rng,fixcm=False)
    dyn.nsteps=done;t0=time.monotonic();requested=int(cfg['steps']);stride=int(cfg.get('stride',20));initial_chunk=chunk
    while done<requested and time.monotonic()-t0<cfg.get('max_wall_seconds',3600):
        if chunk-initial_chunk>=cfg.get('max_chunks_per_invocation',10**9):break
        end=min(done+int(cfg.get('chunk_steps',1000)),requested)
        saved=[];values=[];energies=[];biases=[];times=[]
        while done<end:
            n=min(stride,end-done);dyn.run(n);done+=n
            a.get_potential_energy()
            if done>cfg.get('equilibration_steps',1000):
                saved.append(a.positions[mobile].copy());values.append(a.calc.cv_values.copy())
                energies.append(a.calc.unbiased_energy_eV);biases.append(a.calc.bias_eV);times.append(done*cfg.get('dt_fs',.5))
        # A chunk becomes reusable only after trajectory AND restart state commit.
        filename=f'chunk{chunk:05d}.npz';tmp=out/(filename+'.tmp')
        checkpoint=f'state{chunk:05d}.npz'
        for artifact in [out/filename,tmp,out/checkpoint,out/(checkpoint+'.tmp')]:
            if artifact.exists():artifact.rename(artifact.with_name(artifact.name+f'.orphan_{time.time_ns()}'))
        with tmp.open('wb') as f:np.savez_compressed(f,positions=np.asarray(saved),cv=np.asarray(values),energy_eV=energies,bias_eV=biases,time_fs=times)
        tmp.replace(out/filename)
        with (out/(checkpoint+'.tmp')).open('wb') as f:np.savez_compressed(f,positions=a.positions,velocities=a.get_velocities())
        (out/(checkpoint+'.tmp')).replace(out/checkpoint)
        chunk+=1
        atomic_json(previous,dict(config_sha256=identity,producer_sha256=producer,checkpoint_file=checkpoint,completed_steps=done,chunks=chunk,rng_state=rng.bit_generator.state))
        print(json.dumps(dict(completed_steps=done,requested_steps=requested,chunks=chunk,wall_s=time.monotonic()-t0)),flush=True)
    result=dict(status='window_complete' if done==requested else 'partial_wall_budget',completed_steps=done,
       requested_steps=requested,chunks=chunk,config_sha256=identity,snapshot_sha256=sha256(Path(cfg['snapshot'])/'input.json'),
       mobile_indices=mobile,temperature_K=cfg.get('temperature_K',303.15),coordinates=cfg['coordinates'],centers=cfg['centers'],
       springs_kcal_A2=cfg['springs_kcal_A2'],quantity='reference_umbrella_samples',validated_activation_free_energy=False,
       limitations=['requires equilibration/decorrelation analysis','requires DFT ensemble corrections and overlap checks'])
    atomic_json(out/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');a=p.parse_args();print(json.dumps(sample(a.config),indent=2))
