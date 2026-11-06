"""Rolling ensemble tables and plots from saved family-cofold results, without LLMs."""
import csv
import hashlib
import json
import os
import re
import textwrap
from collections import Counter,defaultdict
from pathlib import Path

from enzyme_discovery_agent.state_store import atomic_write_json,state_lock
from .proxies.attack_angle import weighted_attack_angle


def _short_name(raw,genes=(),literature=None):
    """Prefer an explicit enzyme symbol; never turn a family into an enzyme."""
    if literature:return str(literature).strip()
    for gene in genes or []:
        value=gene.get('value','') if isinstance(gene,dict) else str(gene)
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9-]{1,12}',value) and not value.isupper():
            return value[0].upper()+value[1:]
    first=re.split(r'\s+@\s+|,',str(raw or ''))[0].strip()
    if 'family' in first.lower() or 'superfamily' in first.lower():return 'Unassigned enzyme'
    symbols=re.findall(r'\b[A-Z][a-z]{2,5}[A-Z][0-9]*\b',str(raw or ''))
    if symbols:return symbols[0]
    if len(first)>50 and '/' in first:return 'Unassigned enzyme'
    if first.lower().endswith('synthetase component'):return 'Unassigned enzyme'
    first=re.sub(r'\s+monomer$','',first,flags=re.I)
    return first or 'Unnamed enzyme'


def display_labels(directory,jobs):
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
    directory=Path(directory)
    try:registry=load_candidate_registry(directory.parent,strict=True)['candidates']
    except (OSError,ValueError,KeyError):registry={}
    override_path=directory/'analysis/display_names.json'
    overrides=json.loads(override_path.read_text()) if override_path.exists() else {}
    names={};organisms={}
    for job in jobs:
        accession=job['accession'];meta=(registry.get(accession) or {}).get('metadata') or {}
        evidence=meta.get('evaluation_data') or {}
        override=overrides.get(accession)
        name=override.get('name') if isinstance(override,dict) else override
        names[accession]=name or _short_name(job['enzyme_name'],meta.get('gene_names') or evidence.get('gene_names'),meta.get('literature_name'))
        organism=meta.get('organism') or evidence.get('organism') or ''
        if isinstance(organism,dict):organism=organism.get('scientificName','')
        words=str(organism).split()
        if len(words)>=2:organism=words[0][0]+'. '+' '.join(words[1:])
        organisms[accession]=str(organism)
    counts=Counter(names.values());labels={}
    for accession,name in names.items():
        labels[accession]=name+(' · '+organisms[accession] if counts[name]>1 and organisms[accession] else '')
    repeated=defaultdict(list)
    for accession,label in labels.items():repeated[label].append(accession)
    for label,accessions in repeated.items():
        if len(accessions)>1:
            for number,accession in enumerate(sorted(accessions),1):labels[accession]=label+f' · sequence {number}'
    return labels


def comparison_figure(rows,labels,step):
    """One named enzyme per row; no text annotations can shrink the axes."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
    rows=sorted(rows,key=lambda row:(-row['weighted_attack_angle_deg'],labels[row['accession']]))
    count=len(rows);height=max(5,2.5+.43*count)
    fig,(angles,distances)=plt.subplots(1,2,figsize=(14,height),sharey=True,gridspec_kw={'width_ratios':[1,1]})
    fig.subplots_adjust(left=.36,right=.97,top=1-.95/height,bottom=.8/height,wspace=.22)
    colors={'exact':'#0072B2','stock_fp32_serial':'#D55E00'}
    wrapped=['\n'.join(textwrap.wrap(labels[row['accession']],42)) for row in rows]
    angles.set_yticks(range(count),wrapped,fontsize=9)
    distances.tick_params(axis='y',left=False,labelleft=False)
    maximum=max(row['harmonic_distance_A'] for row in rows)
    minimum=min(row['harmonic_distance_A'] for row in rows)
    distances.set_xscale('log');distances.set_xlim(max(.1,minimum*.75),maximum*2.1)
    ticks=[v for v in [1,2,3,5,10,20,50,100,200] if minimum*.75<=v<=maximum*2.1]
    distances.xaxis.set_major_locator(FixedLocator(ticks));distances.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
    distances.xaxis.set_minor_formatter(NullFormatter())
    for y,row in enumerate(rows):
        color=colors.get(row.get('numerical_profile'),'#666666')
        for ax in [angles,distances]:ax.axhspan(y-.42,y+.42,color='#F4F6F8' if y%2==0 else 'white',zorder=0)
        angles.scatter(row['weighted_attack_angle_deg'],y,s=42,color=color,zorder=3)
        distances.scatter(row['harmonic_distance_A'],y,s=42,color=color,zorder=3)
        angles.text(204,y,f"{row['weighted_attack_angle_deg']:.1f}°",ha='right',va='center',fontsize=9)
        distances.text(maximum*1.95,y,f"{row['harmonic_distance_A']:.2f}",ha='right',va='center',fontsize=9)
    angles.set(xlim=(0,210),xticks=[0,30,60,90,120,150,180],xlabel='Proximity-weighted attack angle (°)')
    distances.set_xlabel('Harmonic attack distance (Å; log scale)')
    for ax in [angles,distances]:
        ax.set_ylim(count-.5,-.5);ax.grid(axis='x',color='#DDE2E6',linewidth=.7,zorder=1)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=10)
    fig.suptitle(f'Reaction step {step} — {count} completed enzyme ensembles',y=1-.18/height,fontsize=15,fontweight='bold')
    fig.text(.36,1-.58/height,'Each row is one ensemble; ordered by weighted angle. Distance is shown separately.',fontsize=10,color='#555555')
    profiles=sorted({row.get('numerical_profile') for row in rows if row.get('numerical_profile')})
    if len(profiles)>1:
        from matplotlib.lines import Line2D
        legend=[Line2D([],[],marker='o',linestyle='',color=colors.get(p,'#666666'),label=p.replace('_',' ')) for p in profiles]
        fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.36,.005),ncol=len(legend),frameon=False,fontsize=8)
    return fig


def refresh(directory):
    directory=Path(directory);out=directory/'analysis';out.mkdir(exist_ok=True)
    files=sorted(directory.glob('jobs/*/complete.json'))
    overrides=out/'display_names.json'
    signature=hashlib.sha256(json.dumps({'files':[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files],
        'labels':overrides.read_text() if overrides.exists() else '',
        'analyzer':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}).encode()).hexdigest()
    with state_lock(out/'refresh'):
        status=out/'status.json'
        if status.exists() and json.loads(status.read_text()).get('signature')==signature:return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        poses=[];summaries=[]
        queue=directory/'queue.json'
        jobs=json.loads(queue.read_text())['items'] if queue.exists() else [json.loads(p.read_text())['job'] for p in files]
        known={job['accession'] for job in jobs}
        jobs += [json.loads(p.read_text())['job'] for p in files if json.loads(p.read_text())['job']['accession'] not in known]
        labels=display_labels(directory,jobs)
        for path in files:
            result=json.loads(path.read_text());job=result['job'];analysis=result['analysis'];raw=analysis['poses']
            score=weighted_attack_angle([p['distance_A'] for p in raw],[p['angle_deg'] for p in raw])
            label=labels[job['accession']];step=job['step']
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
            fig=comparison_figure(rows,labels,step)
            for extension in ['png','svg']:
                target=out/('step_'+str(step)+'_comparison.'+extension);temporary=target.with_name(target.stem+'.tmp.'+extension)
                fig.savefig(temporary,dpi=180);os.replace(temporary,target)
            plt.close(fig)
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
