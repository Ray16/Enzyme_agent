"""Enzyme-level exploratory assessment, with fixed primary endpoint and baseline."""
import argparse
import csv
import html
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata, spearmanr
from core import digest, write_json

BASE=Path(__file__).resolve().parents[2]


def save_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0: return None
    return float(spearmanr(x,y).statistic)


def spearman_uncertainty(x,y,seed=20261103):
    rho=correlation(x,y)
    if rho is None: return dict(rho=None,ci95=None,permutation_p_two_sided=None,reason='Insufficient variation or coverage')
    rng=np.random.default_rng(seed)
    a=rankdata(x);a-=a.mean(); a/=np.linalg.norm(a)
    b=rankdata(y);b-=b.mean(); b/=np.linalg.norm(b)
    null=np.array([a@rng.permutation(b) for _ in range(10000)])
    idx=rng.integers(0,len(x),size=(5000,len(x)))
    xx=rankdata(np.asarray(x)[idx],axis=1);xx-=xx.mean(axis=1,keepdims=True)
    yy=rankdata(np.asarray(y)[idx],axis=1);yy-=yy.mean(axis=1,keepdims=True)
    denom=np.linalg.norm(xx,axis=1)*np.linalg.norm(yy,axis=1)
    boot=(xx*yy).sum(axis=1)[denom>0]/denom[denom>0]
    return dict(rho=rho,ci95=np.quantile(boot,[.025,.975]).tolist(),
        permutation_p_two_sided=float((1+(np.abs(null)>=abs(rho)-1e-12).sum())/10001),
        n_enzymes=len(x),bootstrap_replicates=len(boot),permutations=10000,
        limitation='Nominal enzyme-level uncertainty; evolutionary dependence not accounted for.')


def auc(y,score):
    y=np.asarray(y,dtype=bool);score=np.asarray(score)
    n=y.sum();m=(~y).sum()
    if not n or not m:return None
    return float((rankdata(score)[y].sum()-n*(n+1)/2)/(n*m))


def auc_uncertainty(labels,score):
    labels=np.asarray(labels,dtype=bool);score=np.asarray(score)
    pos=np.where(labels)[0];neg=np.where(~labels)[0]
    if not len(pos) or not len(neg): return dict(auroc=None,ci95=None)
    rng=np.random.default_rng(20261103)
    samples=[]
    for _ in range(3000):
        idx=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
        samples.append(auc(labels[idx],score[idx]))
    return dict(auroc=auc(labels,score),ci95=np.quantile(samples,[.025,.975]).tolist(),
        positives=len(pos),negatives=len(neg),
        limitation='Stratified bootstrap conditional on the existing strong-producer set; exploratory.')


def loo_ridge(x,y):
    x=np.asarray(x,float);y=np.log1p(np.asarray(y,float))
    predictions=[]
    for i in range(len(y)):
        mask=np.arange(len(y))!=i
        mean=x[mask].mean(0);sd=x[mask].std(0);sd[sd<1e-12]=1
        train=(x[mask]-mean)/sd;test=(x[i]-mean)/sd
        coef=np.linalg.solve(train.T@train+np.eye(x.shape[1]),train.T@(y[mask]-y[mask].mean()))
        predictions.append(float(test@coef+y[mask].mean()))
    return dict(mean_squared_error_log1p=float(np.mean((np.asarray(predictions)-y)**2)),
        spearman_rho=correlation(predictions,y),predictions_log1p=predictions,
        method='Leave one enzyme out; alpha=1 ridge; standardization fitted on training enzymes only.',
        limitation='Retrospective internal validation; homologs can occur in both training and test sets.')


def summarize(root,manifest,results):
    base=root.parents[2]
    assay={r['id']:r for r in csv.DictReader((base/'assay_data/panel_activity.csv').open())}
    static={r['id']:r for r in csv.DictReader((base/'results/analysis/current_two_step_metrics_20261023/per_enzyme.csv').open())}
    frames=[]
    for r in results:
        metric={k:v for k,v in r['metrics'].items() if not isinstance(v,list)}
        metric.update(id=r['task']['enzyme'],pose=r['task']['pose'],lag_ns=r['task']['lag_ns'],
            seed=r['task']['seed'],task_id=r['task']['id'],
            generation_seconds=r['timing']['generation_entrypoint_seconds'])
        frames.append(metric)
    rows=[]
    feature_names=['native_contact_retention','pocket_centroid_displacement_A','pocket_retained',
        'substrate_ATP_PA_distance_A','substrate_ATP_attack_angle_deg']
    for n in range(1,49):
        e=f'AS{n}'
        baselines=[json.loads((root/f'inputs/{e}/pose_{pose}/mapping.json').read_text())['baseline'] for pose in [0,25]]
        row=dict(id=e,activity_ugml=float(assay[e]['activity_N66L1_ugml_avg']),
            strong_producer=static[e]['strong_producer']=='True',
            static_weighted_attack_angle_deg=float(static[e]['step1_angle_deg']),
            baseline_contact_count=float(np.mean([b['native_contact_count'] for b in baselines])),
            baseline_selected_pose_attack_distance_A=float(np.mean([b['substrate_ATP_PA_distance_A'] for b in baselines])))
        for lag in [1,10]:
            subset=[f for f in frames if f['id']==e and f['lag_ns']==lag]
            valid=[f for f in subset if f['protein_physicality_passed'] and f['substrate_protein_pairs_below_1p5A']==0]
            row[f'n_endpoints_{lag}ns']=len(subset)
            row[f'n_geometry_eligible_{lag}ns']=len(valid)
            for key in feature_names:
                row[f'{key}_{lag}ns']=float(np.mean([f[key] for f in subset])) if len(subset)==2 else None
                row[f'{key}_eligible_{lag}ns']=float(np.mean([f[key] for f in valid])) if len(valid)==2 else None
        rows.append(row)
    save_csv(root/'endpoint_metrics.csv',frames)
    save_csv(root/'per_enzyme.csv',rows)
    eligible=[r for r in rows if r['n_geometry_eligible_10ns']==2]
    raw=[r for r in rows if r['n_endpoints_10ns']==2]
    stats=dict(completed_endpoints=len(results),expected_endpoints=len(manifest['tasks']),
        panel_complete=len(results)==len(manifest['tasks']),
        primary_definition='Mean starting-residue contact retention across the two 10 ns endpoints per enzyme.',
        primary_geometry_eligible_n=len(eligible),geometry_exclusions='Protein physicality failed or any substrate/protein heavy atom pair below 1.5 A.',
        primary=None,all_saved_frames_sensitivity=None,secondary={},incremental_prediction=None,
        limitations=manifest['protocol']['limitations'])
    if len(eligible)>=5:
        y=[r['activity_ugml'] for r in eligible]
        x=[r['native_contact_retention_eligible_10ns'] for r in eligible]
        stats['primary']=spearman_uncertainty(x,y)
        stats['strong_producer_secondary']=auc_uncertainty([r['strong_producer'] for r in eligible],x)
        baseline=[[r['static_weighted_attack_angle_deg'],r['baseline_contact_count'],
            r['baseline_selected_pose_attack_distance_A']] for r in eligible]
        augmented=[b+[r['native_contact_retention_eligible_10ns']] for b,r in zip(baseline,eligible)]
        base_pred=loo_ridge(baseline,y);dynamic_pred=loo_ridge(augmented,y)
        stats['incremental_prediction']=dict(baseline_features=['static_weighted_attack_angle_deg','baseline_contact_count',
                'baseline_selected_pose_attack_distance_A'],
            added_feature='native_contact_retention_eligible_10ns',baseline=base_pred,augmented=dynamic_pred,
            mse_improvement=base_pred['mean_squared_error_log1p']-dynamic_pred['mean_squared_error_log1p'])
    if len(raw)>=5:
        stats['all_saved_frames_sensitivity']=spearman_uncertainty([r['native_contact_retention_10ns'] for r in raw],
            [r['activity_ugml'] for r in raw])
    # Secondary metrics remain exploratory; do not select the best and call it primary.
    for lag in [1,10]:
        selected=[r for r in rows if r[f'n_geometry_eligible_{lag}ns']==2]
        for feature,sign in [('native_contact_retention',1),('pocket_centroid_displacement_A',-1),('pocket_retained',1)]:
            if len(selected)>=5:
                stats['secondary'][f'{feature}_{lag}ns']=dict(n_enzymes=len(selected),
                    oriented_spearman_rho=correlation([sign*r[f'{feature}_eligible_{lag}ns'] for r in selected],
                        [r['activity_ugml'] for r in selected]),direction=sign)
    times=[r['timing']['generation_entrypoint_seconds'] for r in results]
    stats['performance']=dict(generation_seconds_sum=sum(times),median_entrypoint_seconds=float(np.median(times)),
        max_peak_allocated_bytes=max(r['resources']['peak_allocated_bytes'] for r in results),
        max_peak_reserved_bytes=max(r['resources']['peak_reserved_bytes'] for r in results),
        note='Entrypoint durations include embedding on first use; sums are GPU-worker time, not elapsed wall time.')
    write_json(root/'statistics.json',stats)
    make_report(root,rows,stats)
    print(json.dumps(stats,indent=2),flush=True)


def make_report(root,rows,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['n_geometry_eligible_10ns']==2]
    if selected:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for ax,key,label in [(axes[0],'native_contact_retention_eligible_10ns','Starting contacts retained at 10 ns'),
                (axes[1],'pocket_centroid_displacement_A_eligible_10ns','Pocket-aligned substrate displacement (Å)')]:
            ax.scatter([r[key] for r in selected],[r['activity_ugml'] for r in selected],
                c=['#D55E00' if r['strong_producer'] else '#0072B2' for r in selected],alpha=.8)
            for r in selected:
                if r['strong_producer']:ax.annotate(r['id'],(r[key],r['activity_ugml']),xytext=(3,3),textcoords='offset points')
            ax.set_xlabel(label);ax.set_ylabel('Measured N66L1 product (µg/mL)')
            ax.set_yscale('symlog',linthresh=.01);ax.grid(alpha=.15)
        fig.suptitle(f'BioEmu2 substrate-retention pilot: {len(selected)}/48 enzymes eligible')
        fig.savefig(root/'retention_vs_yield.png',dpi=180)
        fig.savefig(root/'retention_vs_yield.svg')
        plt.close(fig)
    primary=stats['primary']
    statement='Panel incomplete; no final conclusion.'
    if stats['panel_complete']:
        if primary is None or primary['rho'] is None:
            statement='This pilot did not provide an assessable retention–yield association.'
        elif primary['rho']>0 and primary['permutation_p_two_sided']<.05:
            statement='A positive exploratory retention–yield association was observed; independent validation is still required.'
        else:
            statement='This pilot did not establish the hypothesized positive retention–yield association.'
    columns=['id','activity_ugml','n_geometry_eligible_10ns','native_contact_retention_eligible_10ns',
             'pocket_centroid_displacement_A_eligible_10ns','static_weighted_attack_angle_deg']
    table='<tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'
    for r in rows:
        table+='<tr>'+''.join('<td>'+html.escape(str(round(r[c],4)) if isinstance(r[c],float) else str(r[c]))+'</td>' for c in columns)+'</tr>'
    content='''<!doctype html><html><meta charset="utf-8"><title>BioEmu2 retention pilot</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#24323c}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}img{width:100%}</style>'''
    content+='<h1>BioEmu2 substrate-retention pilot</h1><p>'+html.escape(statement)+'</p>'
    content+=f'<p>{stats["completed_endpoints"]}/{stats["expected_endpoints"]} endpoints; two starting poses per enzyme, 1 ns and 10 ns independent endpoints, no contact conditioning.</p>'
    if selected:content+='<img src="retention_vs_yield.png" alt="Enzyme-level substrate retention and measured yield">'
    content+='<h2>Fixed primary analysis</h2><pre>'+html.escape(json.dumps(primary,indent=2))+'</pre>'
    content+='<h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in stats['limitations'])+'</ul>'
    content+='<h2>Enzyme-level measurements</h2><table>'+table+'</table>'
    content+='<p><a href="statistics.json">Full statistics</a> · <a href="per_enzyme.csv">Enzyme table</a> · <a href="endpoint_metrics.csv">All endpoint measurements</a></p></html>'
    (root/'report.html').write_text(content)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    p.add_argument('--wait-seconds',type=float,default=21600)
    args=p.parse_args();root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    deadline=time.monotonic()+args.wait_seconds
    while True:
        results=[];terminal=[]
        for task in manifest['tasks']:
            path=root/'tasks'/task['id']/'result.json'
            if path.is_file():
                result=json.loads(path.read_text())
                if result['task']!=task or any(digest(f['path'])!=f['sha256'] for f in result['files']):
                    raise RuntimeError('Result integrity mismatch: '+task['id'])
                results.append(result)
            elif len(list(path.parent.glob('attempt_*/failure.json')))>=2:
                terminal.append(task['id'])
        write_json(root/'progress.json',dict(expected=len(manifest['tasks']),completed=len(results),
            terminal_failures=terminal,updated_epoch=time.time()))
        if not args.wait or len(results)+len(terminal)==len(manifest['tasks']) or time.monotonic()>deadline:break
        time.sleep(30)
    if not results:raise RuntimeError('No completed endpoints to analyze')
    summarize(root,manifest,results)


if __name__=='__main__':
    main()
