"""OPES diagnostics with all biases, explicit sampling failures, and basin populations.

The projected minimax height is exploratory until independent mechanistic and
committor checks pass. A sparse/disconnected histogram never yields a barrier.
"""
import heapq
import json
from pathlib import Path
import numpy as np

KB_KCAL=0.00198720425864
EV_KCAL=23.06054783


def minimax(free_energy, reactant, product):
    """Dijkstra bottleneck path on a finite grid, preserving unsampled holes."""
    f=np.asarray(free_energy,float)
    if not np.isfinite(f[reactant]) or not np.isfinite(f[product]):return None
    heap=[(float(f[reactant]),reactant)];cost={reactant:float(f[reactant])}
    while heap:
        value,node=heapq.heappop(heap)
        if node==product:return value-f[reactant]
        if value>cost[node]:continue
        for axis in range(f.ndim):
            for delta in (-1,1):
                nxt=list(node);nxt[axis]+=delta;nxt=tuple(nxt)
                if any(i<0 or i>=n for i,n in zip(nxt,f.shape)) or not np.isfinite(f[nxt]):continue
                c=max(value,float(f[nxt]))
                if c<cost.get(nxt,float('inf')):cost[nxt]=c;heapq.heappush(heap,(c,nxt))
    return None


def analyze_opes(path, *, temperature=303.15, bias_columns=('opes.bias',),
                 wall_specs=(), discard=0.5, energy_unit='eV', bin_width=0.1,
                 min_transitions=6, min_ess=100, blocks=5):
    lines=Path(path).read_text().splitlines()
    if not lines or not lines[0].startswith('#! FIELDS'):raise ValueError('Missing COLVAR schema')
    names=lines[0].split()[2:];cols={n:i for i,n in enumerate(names)}
    data=np.loadtxt(path,ndmin=2)
    if len(data)<20:raise ValueError('Too few saved frames')
    data=data[int(len(data)*discard):]
    if len(data)<10:raise ValueError('Too few retained frames')
    x,y=data[:,cols['d1']],data[:,cols['d2']]
    conversion={'eV':EV_KCAL,'kcal/mol':1.0,'kJ/mol':1/4.184}[energy_unit]
    bias=sum(data[:,cols[n]] for n in bias_columns)
    # Legacy runs did not print walls; reconstruct ONLY explicitly declared
    # harmonic UPPER_WALLS (EPS=1, OFFSET=0), otherwise refuse to guess them.
    for spec in wall_specs:
        if spec.get('kind')!='upper' or spec.get('exponent',2)!=2:raise ValueError('Unsupported wall')
        z=data[:,cols[spec['column']]]
        bias=bias+spec['kappa']*np.maximum(z-spec['at'],0)**2
    kT=KB_KCAL*temperature
    logw=(bias-bias.max())*conversion/kT;w=np.exp(logw)
    r=(x>3.0)&(y<2.2);p=(x<2.2)&(y>2.7)
    states=np.where(r,0,np.where(p,1,-1));visits=states[states>=0]
    switches=int(np.count_nonzero(np.diff(visits))) if len(visits)>1 else 0
    # Kish ESS is an upper bound: also report correlation-aware block stability.
    ess=float(w.sum()**2/np.dot(w,w))
    edges=[np.arange(1.3,4.9+bin_width/2,bin_width),np.arange(1.3,4.0+bin_width/2,bin_width)]
    centers=[(e[:-1]+e[1:])/2 for e in edges]
    R=(centers[0][:,None]>3)&(centers[1][None,:]<2.2)
    P=(centers[0][:,None]<2.2)&(centers[1][None,:]>2.7)
    def estimate(xx,yy,ww):
        h=np.histogram2d(xx,yy,bins=edges,weights=ww)[0]
        f=np.full(h.shape,np.inf);occupied=h>0;f[occupied]=-kT*np.log(h[occupied]);f[occupied]-=f[occupied].min()
        rr=np.where(R,f,np.inf);pp=np.where(P,f,np.inf)
        if not np.isfinite(rr).any() or not np.isfinite(pp).any():return None
        a=np.unravel_index(rr.argmin(),f.shape);b=np.unravel_index(pp.argmin(),f.shape)
        return minimax(f,a,b)
    height=estimate(x,y,w)
    population_delta=None
    if w[r].sum()>0 and w[p].sum()>0:population_delta=float(-kT*np.log(w[p].sum()/w[r].sum()))
    block_heights=[estimate(x[s],y[s],w[s]) for s in np.array_split(np.arange(len(x)),blocks)]
    reasons=[]
    if not r.any() or not p.any():reasons.append('both basins are not populated')
    if switches<min_transitions:reasons.append('insufficient independent basin exchange')
    if ess<min_ess:reasons.append('low reweighting effective sample size')
    if height is None:reasons.append('disconnected sampled free-energy grid')
    valid_blocks=[v for v in block_heights if v is not None]
    if len(valid_blocks)!=blocks:reasons.append('one or more time blocks lack a connected path')
    spread=float(np.ptp(valid_blocks)) if len(valid_blocks)>1 else None
    if spread is not None and spread>2:reasons.append('block barrier drift exceeds 2 kcal/mol')
    out=dict(schema_version=1,source=str(Path(path).resolve()),quantity='projected_PMF_minimax_height',
         validation_status='sampling_failed' if reasons else 'sampling_checks_passed_mechanism_pending',
         validated_activation_free_energy=False,temperature_K=temperature,frames=len(data),
         last_time=float(data[-1,cols['time']]),reactant_frames=int(r.sum()),product_frames=int(p.sum()),
         basin_switches=switches,kish_ess_upper_bound=ess,bias_columns=list(bias_columns),
         reconstructed_walls=list(wall_specs),projected_barrier_kcal=height if not reasons else None,
         diagnostic_projected_height_kcal=height,basin_population_deltaG_kcal=population_delta,
         block_heights_kcal=block_heights,block_range_kcal=spread,failures=reasons,
         limitations=['CV and binning dependent; committor validation required',
                      'Kish ESS does not correct time correlation',
                      'confinement removal cannot recover unsampled states'])
    return out


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('colvar');p.add_argument('--out',required=True)
    p.add_argument('--bias',action='append',required=True)
    p.add_argument('--walls-json');p.add_argument('--energy-unit',default='eV')
    a=p.parse_args(argv)
    walls=json.loads(Path(a.walls_json).read_text()) if a.walls_json else []
    result=analyze_opes(a.colvar,bias_columns=a.bias,wall_specs=walls,energy_unit=a.energy_unit)
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
