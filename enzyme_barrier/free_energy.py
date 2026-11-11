"""MBAR for common-Hamiltonian umbrellas and explicit target-level reweighting.

Inputs must be equilibrated/decorrelated. Different chemical compositions or
different base Hamiltonians cannot be joined using bias energies alone.
"""
import numpy as np
from scipy.special import logsumexp


def mbar_bias_weights(bias_kcal,counts,temperature=303.15,tolerance=1e-9,max_iterations=10000):
    """bias[k,n] is umbrella k evaluated on EVERY retained configuration n.

    Returns normalized unbiased log weights and the overlap matrix. These weights
    remove the umbrellas; fixed confinement remains unless included as a target
    Hamiltonian difference. N_k must count the supplied decorrelated samples.
    """
    bias=np.asarray(bias_kcal,float);counts=np.asarray(counts,int)
    if bias.ndim!=2 or len(counts)!=len(bias) or counts.sum()!=bias.shape[1] or (counts<=0).any():
        raise ValueError('Invalid MBAR dimensions or empty window')
    if not np.isfinite(bias).all():raise ValueError('Nonfinite reduced potentials')
    kt=0.00198720425864*temperature;u=bias/kt;f=np.zeros(len(counts))
    for iteration in range(max_iterations):
        denom=logsumexp(np.log(counts)[:,None]+f[:,None]-u,axis=0)
        new=-logsumexp(-u-denom[None,:],axis=1);new-=new[0]
        if np.max(np.abs(new-f))<tolerance:f=new;break
        f=new
    else:raise ValueError('MBAR did not converge; inspect overlap')
    denom=logsumexp(np.log(counts)[:,None]+f[:,None]-u,axis=0)
    logw=-denom;logw-=logsumexp(logw)
    state_weights=np.exp(f[:,None]-u-denom[None,:])
    overlap=(state_weights@state_weights.T)*counts[None,:]
    seen={0};queue=[0]
    while queue:
        i=queue.pop()
        for j in range(len(counts)):
            if j not in seen and min(overlap[i,j],overlap[j,i])>0.01:seen.add(j);queue.append(j)
    return dict(log_weights=logw,free_energies_kcal=f*kt,overlap=overlap,
                overlap_connected=len(seen)==len(counts),iterations=iteration+1,
                kish_ess_upper_bound=float(1/np.exp(2*logw).sum()))


def target_weights(reference_log_weights,delta_energy_kcal,*,temperature=303.15,selection_probability=None):
    """Target-level importance weights, including sparse-SP inclusion probabilities.

    Never replace this by a correction on a single optimized R/TS pair. Report
    ESS per window/basin, not only globally. Sparse DFT subsets must be selected
    by a documented random/stratified design with known inclusion probability.
    """
    reference=np.asarray(reference_log_weights,float);delta=np.asarray(delta_energy_kcal,float)
    if reference.shape!=delta.shape or not np.isfinite(delta).all():raise ValueError('Invalid correction ensemble')
    if selection_probability is None:prob=np.ones_like(delta)
    else:prob=np.asarray(selection_probability,float)
    if prob.shape!=delta.shape or (prob<=0).any() or (prob>1).any():raise ValueError('Invalid inclusion probability')
    lw=reference-delta/(0.00198720425864*temperature)-np.log(prob);normalizer=logsumexp(lw);lw-=normalizer
    return dict(log_weights=lw,kish_ess_upper_bound=float(1/np.exp(2*lw).sum()),
                maximum_weight=float(np.exp(lw).max()),log_normalizer=float(normalizer))


def integrated_basin_difference(log_weights,reactant_mask,product_mask,temperature=303.15):
    w=np.asarray(log_weights,float);r=np.asarray(reactant_mask,bool);p=np.asarray(product_mask,bool)
    if not r.any() or not p.any():raise ValueError('Both basins must be sampled')
    return float(-0.00198720425864*temperature*(logsumexp(w[p])-logsumexp(w[r])))
