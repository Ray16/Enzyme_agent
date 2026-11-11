"""Additive electrostatically embedded DFT/MM, with gradients on BOTH regions.

E = E_MM(remaining bonded + LJ + MM/MM Coulomb) + E_DFT(QM+links; MM charges).
QM MM charges are zero in MM, including exceptions, so Coulomb is counted once.
The finite, fixed embedding selection is part of the Hamiltonian and is recorded.
This backend is a validation/pilot backend, not a claim of validated kinetics.
"""
import time
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr
from .region import capped_coordinates, project_link_forces


def remove_qm_terms(system, indices):
    import openmm as mm
    from openmm import unit
    qm=set(indices); pairs=set()
    for f in system.getForces():
        if isinstance(f,mm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                p=list(f.getBondParameters(k))
                if set(p[:2])<=qm:p[-1]*=0;f.setBondParameters(k,*p)
        elif isinstance(f,mm.HarmonicAngleForce):
            for k in range(f.getNumAngles()):
                p=list(f.getAngleParameters(k))
                if set(p[:3])<=qm:p[-1]*=0;f.setAngleParameters(k,*p)
        elif isinstance(f,mm.PeriodicTorsionForce):
            for k in range(f.getNumTorsions()):
                p=list(f.getTorsionParameters(k))
                if set(p[:4])<=qm:p[-1]*=0;f.setTorsionParameters(k,*p)
        elif isinstance(f,mm.NonbondedForce):
            for i in qm:
                q,s,e=f.getParticleParameters(i); f.setParticleParameters(i,0*unit.elementary_charge,s,e)
            for k in range(f.getNumExceptions()):
                i,j,q,s,e=f.getExceptionParameters(k)
                if i in qm or j in qm:
                    f.setExceptionParameters(k,i,j,0*q,s,0*e if i in qm and j in qm else e)
            ids=sorted(qm)
            for k,i in enumerate(ids):
                for j in ids[k+1:]:f.addException(i,j,0,1,0,replace=True);pairs.add((i,j))
        elif isinstance(f,mm.CMMotionRemover):pass
        elif isinstance(f,mm.CustomNonbondedForce):
            if 'ccoef' not in f.getEnergyFunction():
                raise ValueError('Unknown custom nonbonded potential; explicit QM/MM policy required')
        elif isinstance(f,mm.CMAPTorsionForce):
            for k in range(f.getNumTorsions()):
                if set(f.getTorsionParameters(k)[1:])<=qm:raise ValueError('QM CMAP unsupported')
        else:raise ValueError(f'Unsupported MM force {type(f).__name__}')
    # OpenMM requires the same exclusion lists for all nonbonded forces.
    for f in system.getForces():
        if isinstance(f,mm.CustomNonbondedForce):
            existing={tuple(sorted(f.getExclusionParticles(k))) for k in range(f.getNumExclusions())}
            for i,j in sorted(pairs-existing):f.addExclusion(i,j)
    return system


def _numpy(x):
    return np.asarray(x.get() if hasattr(x,'get') else x)


def configure_mm_switching(system,cutoff_nm=1.2,switch_nm=1.0):
    import openmm as mm
    from openmm import unit
    if switch_nm is None:return system
    if not 0<switch_nm<cutoff_nm:raise ValueError('MM switch must lie inside the cutoff')
    for force in system.getForces():
        if isinstance(force,(mm.NonbondedForce,mm.CustomNonbondedForce)):
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(switch_nm*unit.nanometer)
    return system


def build_mm_system(parm,region,*,mm_cutoff_nm=1.2,mm_switch_nm=1.0,require_magnesium_c4=True):
    """The exact remaining-MM Hamiltonian, inspectable without a QM/GPU context."""
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    from .parameters import require_c4
    if require_magnesium_c4:require_c4(parm)
    system=parm.createSystem(nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=mm_cutoff_nm*unit.nanometer,
                constraints=None,rigidWater=False,removeCMMotion=False)
    configure_mm_switching(system,mm_cutoff_nm,mm_switch_nm)
    for f in system.getForces():
        if isinstance(f,mm.NonbondedForce):
            original=[f.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(f.getNumParticles())]
            updated=original.copy()
            for key,val in region.get('mm_charge_updates',{}).items():
                i=int(key);q,s,e=f.getParticleParameters(i)
                f.setParticleParameters(i,val*unit.elementary_charge,s,e);updated[i]=val
            for k in range(f.getNumExceptions()):
                i,j,q,s,e=f.getExceptionParameters(k)
                if original[i]*original[j]!=0:
                    q=q*(updated[i]*updated[j])/(original[i]*original[j])
                elif updated[i]*updated[j]!=0:
                    close={at.idx for at in parm.atoms[i].bond_partners}
                    close.update(at.idx for neighbor in parm.atoms[i].bond_partners for at in neighbor.bond_partners)
                    if j not in close:
                        raise ValueError('New nonzero 1-4 charge product needs an explicit scaling policy')
                f.setExceptionParameters(k,i,j,q,s,e)
    return remove_qm_terms(system,region['indices'])


class EmbeddedDFTMM(Calculator):
    implemented_properties=['energy','forces']

    def __init__(self, parm, region, *, xc='wb97m-v', basis='def2-tzvpd',
                 engine='gpu', platform=None, grid_level=3, conv_tol=1e-9,
                 max_cycle=150, mm_cutoff_nm=1.2, mm_switch_nm=1.0, require_magnesium_c4=True, grid_response=True, **kwargs):
        super().__init__(**kwargs)
        import openmm as mm
        import openmm.app as app
        from openmm import unit
        if platform is None:platform='CUDA' if engine=='gpu' else 'CPU'
        self.parm,self.region=parm,region
        self.xc,self.basis,self.engine=xc,basis,engine
        self.grid_level,self.conv_tol,self.max_cycle=grid_level,conv_tol,max_cycle
        self.grid_response=bool(grid_response)
        self.mm_platform=platform;self.mm_switch_nm=mm_switch_nm
        self.dm=None;self.last_components={};self.evaluations=0
        # Nonperiodic RF MM environment. QM Coulomb uses the explicit fixed
        # charge sphere instead; its convergence must be checked separately.
        system=build_mm_system(parm,region,mm_cutoff_nm=mm_cutoff_nm,mm_switch_nm=mm_switch_nm,require_magnesium_c4=require_magnesium_c4)
        self.integrator=mm.VerletIntegrator(0.5*unit.femtosecond)
        props={'Precision':'double'} if platform=='CUDA' else ({'Threads':'1'} if platform=='CPU' else {})
        self.context=mm.Context(system,self.integrator,mm.Platform.getPlatformByName(platform),props)

    def high_energy_forces(self,symbols,qpos,mmpos,mmcharges,need_forces):
        from pyscf import gto
        mol=gto.M(atom=list(zip(symbols,qpos)),basis=self.basis,charge=self.region['charge'],
                  spin=self.region['multiplicity']-1,unit='Angstrom',verbose=0,max_memory=16000)
        if self.engine=='gpu':
            from gpu4pyscf import dft
            from gpu4pyscf.qmmm.itrf import mm_charge
        elif self.engine=='cpu':
            from pyscf import dft
            from pyscf.qmmm import mm_charge
        else:raise ValueError('engine must be cpu or gpu')
        mf=dft.RKS(mol).density_fit();mf.xc=self.xc
        mf.grids.level=self.grid_level;mf.conv_tol=self.conv_tol;mf.max_cycle=self.max_cycle
        if len(mmpos):mf=mm_charge(mf,mmpos,mmcharges,unit='Angstrom')
        def progress(env):
            print(f"SCF cycle={env.get('cycle')} E_Ha={env.get('e_tot')} elapsed_s={time.monotonic()-scf_start:.1f}",flush=True)
        scf_start=time.monotonic();mf.callback=progress
        energy=float(mf.kernel(dm0=self.dm))
        if not mf.converged or not np.isfinite(energy):raise RuntimeError('Embedded SCF failed to converge')
        self.dm=mf.make_rdm1()
        qforce=mmforce=None
        if need_forces:
            grad=mf.nuc_grad_method()
            grad.grid_response=self.grid_response
            qforce=-_numpy(grad.kernel())*Hartree/Bohr
            if len(mmpos):
                mmgrad=_numpy(grad.grad_hcore_mm(self.dm))+_numpy(grad.grad_nuc_mm())
                mmforce=-mmgrad*Hartree/Bohr
        return energy*Hartree,qforce,mmforce

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        import openmm.unit as u
        t0=time.monotonic();pos=self.atoms.positions
        self.context.setPositions(pos*0.1*u.nanometer)
        st=self.context.getState(getEnergy=True,getForces='forces' in properties)
        emm=float(st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))/96.4853321233
        idx=self.region['indices']; nq=len(idx);emb=self.region['embedding_indices']
        symbols=[self.parm.atoms[i].element_name for i in idx]+['H']*len(self.region['links'])
        energy,qforce,mmforce=self.high_energy_forces(symbols,capped_coordinates(pos,self.region),pos[emb],self.region['embedding_charges'],'forces' in properties)
        self.results={'energy':energy+emm}
        if 'forces' in properties:
            force=project_link_forces(qforce,self.region,len(pos))
            if emb:force[emb]+=mmforce
            force+=np.asarray(st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole/u.nanometer))/964.853321233
            self.results['forces']=force
        self.evaluations+=1
        self.last_components=dict(high_embedded_eV=energy,mm_remaining_eV=emm,
             wall_s=time.monotonic()-t0,scf_converged=True,n_qm=nq,n_links=len(self.region['links']),
             n_embedding=len(emb),evaluation=self.evaluations,grid_response=self.grid_response,
             mm_platform=self.mm_platform,mm_precision='double' if self.mm_platform=='CUDA' else 'platform-dependent',
             mm_switch_nm=self.mm_switch_nm)
