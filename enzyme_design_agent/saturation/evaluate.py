"""Separate ensemble geometry and WT deltas with pose-sampling diagnostics."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from .interface import group_coordinates


def geometry(path, selectors):
    donor=group_coordinates(path,selectors['donor'])
    center=group_coordinates(path,selectors['electrophile'])
    leaving=group_coordinates(path,selectors['leaving'])
    if len(center)!=1 or len(leaving)!=1:
        raise ValueError('Electrophile and leaving selectors must each identify one atom')
    center=center[0]; leaving=leaving[0]
    distances=np.linalg.norm(donor-center,axis=1)
    # Atom-selector order resolves exact ties reproducibly.
    index=int(np.argmin(distances)); d=float(distances[index])
    a,b=donor[index]-center,leaving-center
    norm=float(np.linalg.norm(a)*np.linalg.norm(b))
    if norm<=0:raise ValueError('Degenerate reactive geometry')
    angle=math.degrees(math.acos(float(np.clip(np.dot(a,b)/norm,-1,1))))
    return d,angle


def summarize(poses):
    from structural_modeling_agent.proxies.ensemble_geometry import proximity_weighted_angle_and_distance
    return proximity_weighted_angle_and_distance(poses)


def prediction_paths(out, identifier, expect):
    p=Path(out)/'results'/identifier/('boltz_results_'+identifier)/'predictions'/identifier
    paths=[p/f'{identifier}_model_{i}.pdb' for i in range(expect)]
    for i,path in enumerate(paths):
        confidence=p/f'confidence_{identifier}_model_{i}.json'
        if not path.is_file() or path.stat().st_size < 100 or not confidence.is_file():return None
        try:
            data=json.loads(confidence.read_text())
            if not isinstance(data,dict) or not data:return None
        except (ValueError,OSError):return None
    return paths


def verified_paths(out, job, expect):
    marker = Path(out)/'results'/job['id']/'verified.json'
    try:
        data = json.loads(marker.read_text())
        if data.get('sequence_sha256') != job['sequence_sha256'] or data.get('n_models') != expect:
            return None
    except (ValueError, OSError):
        return None
    return prediction_paths(out, job['id'], expect)


def verify_outputs(out, job, candidate, expect):
    """Verify sequence, component identity and required atoms before completion."""
    from ..structure import load_structure
    paths = prediction_paths(out, job['id'], expect)
    if paths is None:
        return False
    source = load_structure(candidate['structure_path'])
    expected_components = {c.id: sorted(c.elements) for c in source.components}
    try:
        for p in paths:
            st = load_structure(p)
            if st.protein_chains() != (candidate['protein_chain'],):
                return False
            if st.sequence(candidate['protein_chain']) != job['sequence']:
                return False
            if {c.id: sorted(c.elements) for c in st.components} != expected_components:
                return False
            if not all(np.all(np.isfinite(r.xyz)) for r in st.residues):
                return False
            geometry(p, candidate['geometry'])
    except (ValueError, OSError):
        return False
    marker = Path(out)/'results'/job['id']/'verified.json'
    marker.write_text(json.dumps({'sequence_sha256': job['sequence_sha256'],
                                 'n_models': expect, 'qc': 'sequence_components_reactive_atoms'})+'\n')
    return True


def evaluate(manifest_path, n_bootstrap=2000):
    m=json.loads(Path(manifest_path).read_text());out=Path(m['output_dir'])
    specs=json.loads((out/'evaluation.json').read_text())
    candidates={c['id']:c for c in m['candidates']}
    poses={}; counts={}; errors={}
    for job in m['jobs']:
        paths=prediction_paths(out,job['id'],m['n_samples'])
        if paths is None:continue
        try:
            from ..structure import load_structure
            for p in paths:
                if load_structure(p).sequence(candidates[job['candidate']]['protein_chain'])!=job['sequence']:
                    raise ValueError('Predicted sequence differs from requested variant')
            poses[job['id']]=np.array([geometry(p,specs[job['candidate']]) for p in paths])
        except (ValueError,OSError) as e:errors[job['id']]=str(e)
    rows=[];raw=[]
    rng=np.random.default_rng(m['seed'])
    for job in m['jobs']:
        key=job['id'];wt=job['candidate']+'_WT'
        row={'id':key,'candidate':job['candidate'],'mutation':job['mutation'],
             'status':'pending','n_models':0}
        if key in errors:row.update(status='qc_failed',error=errors[key])
        elif key in poses:
            x=poses[key];s=summarize(x)
            row.update(status='awaiting_matched_wt',n_models=len(x),angle_deg=float(s[0]),harmonic_distance_A=float(s[1]))
            for i,(d,a) in enumerate(x):raw.append({'id':key,'sample':i,'distance_A':d,'angle_deg':a})
            if wt in poses:
                ref=poses[wt];delta=s-summarize(ref)
                if key==wt:
                    ci=np.zeros((2,2));support=0.
                else:
                    # Independent ensemble resampling: same seed does not establish paired poses.
                    bs=summarize(x[rng.integers(len(x),size=(n_bootstrap,len(x)))])-summarize(ref[rng.integers(len(ref),size=(n_bootstrap,len(ref)))])
                    ci=np.percentile(bs,[5,95],axis=0)
                    support=float(np.mean(bs[:,0]>0))
                row.update(status='evaluated',delta_angle_deg=float(delta[0]),delta_distance_A=float(delta[1]),
                           delta_angle_p05=float(ci[0,0]),delta_angle_p95=float(ci[1,0]),
                           delta_distance_p05=float(ci[0,1]),delta_distance_p95=float(ci[1,1]),
                           angle_improvement_resampling_fraction=support,
                           improves_angle_point_estimate=int(delta[0]>0),
                           distance_increased=int(delta[1]>0))
        rows.append(row);counts[row['status']]=counts.get(row['status'],0)+1
    def table(path,items):
        fields=list(dict.fromkeys(k for r in items for k in r))
        tmp=path.with_suffix('.tmp')
        with tmp.open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(items)
        tmp.replace(path)
    table(out/'variant_metrics.csv',rows);table(out/'pose_geometry.csv',raw)
    status={'counts':counts,'expected_variants':len(m['jobs']),'complete':counts.get('evaluated',0)==len(m['jobs']),
            'interpretation':'Pose-resampling intervals are not biological confidence or multiplicity-adjusted significance. Improvements are structural hypotheses under the specified assembly.'}
    (out/'evaluation_status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('manifest');a=ap.parse_args()
    print(json.dumps(evaluate(a.manifest),indent=2))
