"""Rolling ensemble tables and plots from saved family-cofold results, without LLMs."""
import csv
import hashlib
import json
import os
import re
from pathlib import Path

from enzyme_discovery_agent.state_store import atomic_write_json,state_lock
from .proxies.attack_angle import weighted_attack_angle


def refresh(directory):
    directory=Path(directory);out=directory/'analysis';out.mkdir(exist_ok=True)
    files=sorted(directory.glob('jobs/*/complete.json'))
    signature=hashlib.sha256(json.dumps({'files':[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files],
        'analyzer':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}).encode()).hexdigest()
    with state_lock(out/'refresh'):
        status=out/'status.json'
        if status.exists() and json.loads(status.read_text()).get('signature')==signature:return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        poses=[];summaries=[]
        for path in files:
            result=json.loads(path.read_text());job=result['job'];analysis=result['analysis'];raw=analysis['poses']
            score=weighted_attack_angle([p['distance_A'] for p in raw],[p['angle_deg'] for p in raw])
            label=job['enzyme_name']+' ('+job['accession']+')';step=job['step']
            discovery_run_id=job.get('discovery_run_id',directory.parent.name)
            total=sum(1/p['distance_A'] for p in raw)
            for number,pose in enumerate(raw):
                poses.append({'discovery_run_id':discovery_run_id,'enzyme_name':job['enzyme_name'],'accession':job['accession'],'family_id':job['family_id'],
                    'step':step,'pose':int(re.search(r'_model_(\d+)',Path(pose['path']).name)[1]),'distance_A':pose['distance_A'],'angle_deg':pose['angle_deg'],
                    'normalized_inverse_distance_weight':(1/pose['distance_A'])/total,
                    'identity_pass':not pose['identity_issues'],'path':pose['path']})
            summaries.append({'discovery_run_id':discovery_run_id,'enzyme_name':job['enzyme_name'],'accession':job['accession'],'family_id':job['family_id'],
                'step':step,**score,'identity_pass_poses':analysis['identity_pass_poses'],
                'numerical_profile':result['receipt'].get('numerical_profile'),'result':str(path)})
            folder=out/'ensembles'/job['accession'];folder.mkdir(parents=True,exist_ok=True)
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            axes[0].scatter([p['distance_A'] for p in raw],[p['angle_deg'] for p in raw],s=25,alpha=.7,color='#0072B2')
            axes[0].set(xlabel='Attack distance (Å)',ylabel='Attack angle (°)',ylim=(0,180))
            axes[1].hist([p['distance_A'] for p in raw],bins=12,color='#009E73',edgecolor='white')
            axes[1].set(xlabel='Attack distance (Å)',ylabel='Predictions')
            fig.suptitle(f"{label} — step {step}\nWeighted angle {score['weighted_attack_angle_deg']:.2f}°; harmonic distance {score['harmonic_distance_A']:.2f} Å; n={len(raw)}",fontsize=10)
            for extension in ['png','svg']:
                target=folder/('step_'+str(step)+'.'+extension)
                temporary=target.with_name(target.stem+'.'+str(os.getpid())+'.tmp'+target.suffix)
                fig.savefig(temporary,dpi=180);os.replace(temporary,target)
            plt.close(fig)
            view=directory/'results'/job['accession']/('step_'+str(step));view.mkdir(parents=True,exist_ok=True)
            for name,target in [('structures',Path(result['receipt']['prediction_dir'])),('full_result.json',path)]:
                alias=view/name
                if not alias.exists():alias.symlink_to(target.resolve())
            atomic_write_json(view/'analysis.json',analysis)
        for name,rows in [('pose_geometry.csv',poses),('ensemble_summary.csv',summaries)]:
            if not rows:continue
            path=out/name;temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
            with temporary.open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            os.replace(temporary,path)
        for step in sorted({row['step'] for row in summaries}):
            rows=[r for r in summaries if r['step']==step]
            fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
            ax.scatter([r['harmonic_distance_A'] for r in rows],[r['weighted_attack_angle_deg'] for r in rows],color='#0072B2')
            for row in rows:ax.annotate(row['enzyme_name']+' ('+row['accession']+')',(row['harmonic_distance_A'],row['weighted_attack_angle_deg']),fontsize=6,xytext=(4,4),textcoords='offset points')
            ax.set(xlabel='Harmonic attack distance (Å)',ylabel='Proximity-weighted attack angle (°)',ylim=(0,180),title='Step '+str(step)+' — completed ensembles')
            target=out/('step_'+str(step)+'_comparison.png');temporary=target.with_name(target.stem+'.tmp.png')
            fig.savefig(temporary,dpi=180);os.replace(temporary,target);plt.close(fig)
        atomic_write_json(status,{'discovery_run_id':directory.parent.name,'signature':signature,'completed_ensembles':len(summaries),'poses':len(poses),
            'metric':'sum(angle/distance) / sum(1/distance)','scope':'Step-specific structural geometry, not demonstrated activity; no cutoffs fitted.'})


if __name__=='__main__':
    import argparse
    import time
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--watch-seconds',type=int,default=0)
    args=parser.parse_args()
    while True:
        refresh(args.directory)
        if not args.watch_seconds:break
        time.sleep(args.watch_seconds)
