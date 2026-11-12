"""Partial-Hessian and qRRHO diagnostics on one explicitly pinned potential.

The entropy interpolation follows the existing in-house Grimme implementation.
These frozen-cluster estimates do not establish an enzyme activation free energy.
"""
import numpy as np
from ase.units import invcm, kB


def partial_frequencies(atoms,free_indices,*,step_A=.001):
    """Central force differences using the SAME attached calculator as energies."""
    from ase.vibrations import VibrationsData
    free=np.asarray(free_indices,int)
    if not len(free) or step_A<=0 or not np.isfinite(step_A):
        raise ValueError('Nonempty free region and positive Hessian step required')
    original=atoms.positions.copy();size=3*len(free);hessian=np.empty((size,size))
    try:
        for column,(atom,axis) in enumerate((i,j) for i in free for j in range(3)):
            atoms.positions[:]=original;atoms.positions[atom,axis]+=step_A
            plus=atoms.get_forces(apply_constraint=False)[free].ravel()
            atoms.positions[:]=original;atoms.positions[atom,axis]-=step_A
            minus=atoms.get_forces(apply_constraint=False)[free].ravel()
            hessian[:,column]=-(plus-minus)/(2*step_A)
    finally:
        atoms.positions[:]=original
    if not np.isfinite(hessian).all():raise ValueError('Nonfinite partial Hessian')
    asymmetry=float(np.max(np.abs(hessian-hessian.T)))
    hessian=(hessian+hessian.T)/2
    frequencies=VibrationsData.from_2d(atoms,hessian,indices=free).get_frequencies()
    return frequencies,dict(step_A=step_A,max_asymmetry_eV_A2=asymmetry,
                            free_indices=free.tolist(),force_calls=2*size)


def qrrho_correction(frequencies_cm,temperature_K,*,cutoff_cm=100.):
    """Vibrational correction in eV; real positive modes only, no free T/R terms."""
    if not np.isfinite(temperature_K) or temperature_K<=0 or cutoff_cm<=0:
        raise ValueError('Positive temperature and rotor cutoff required')
    frequencies=np.asarray(frequencies_cm,complex)
    if not np.isfinite(frequencies).all():raise ValueError('Nonfinite frequencies')
    real=frequencies.real[(np.abs(frequencies.imag)<1e-6)&(frequencies.real>1e-6)]
    energy=real*invcm;x=energy/(kB*temperature_K)
    # Stable oscillator populations even at very high frequency.
    population=np.exp(-x)/(-np.expm1(-x))
    harmonic_entropy=kB*(x*population-np.log(-np.expm1(-x)))
    h=6.62607015e-34;kb_si=1.380649e-23;ev_si=1.602176634e-19
    moment=h/(8*np.pi**2*real*2.99792458e10)
    effective=moment*1e-44/(moment+1e-44)
    rotor_entropy=(kb_si/ev_si)*(.5+.5*np.log(8*np.pi**3*effective*kb_si*temperature_K/h**2))
    weight=1/(1+(cutoff_cm/real)**4)
    entropy=float(np.sum(weight*harmonic_entropy+(1-weight)*rotor_entropy))
    zpe=float(.5*np.sum(energy));thermal=float(np.sum(energy*population))
    return dict(correction_eV=zpe+thermal-temperature_K*entropy,zpe_eV=zpe,
                thermal_vibration_eV=thermal,entropy_eV_K=entropy,
                temperature_K=temperature_K,rotor_cutoff_cm=cutoff_cm,
                significant_imaginary_modes=int(np.count_nonzero(frequencies.imag>50)),
                frequencies_real_cm=frequencies.real.tolist(),frequencies_imag_cm=frequencies.imag.tolist(),
                validated_activation_free_energy=False)
