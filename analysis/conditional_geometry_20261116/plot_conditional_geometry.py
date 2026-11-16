"""Separate step-1 proximity frequency, conditional angle, and step-2 pose distances."""
from pathlib import Path
import csv, json, hashlib, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from structural_modeling_agent.paired_analysis import _place_labels
SOURCE=ROOT/'enzyme_discovery_agent/runs/20261106_000850_Find_enzymes_that_ligate_glutarate_and_cadaverine_Production_objective_deliver_a/family_cofold_pocket_guided/analysis'
OUT=Path(__file__).resolve().parent
THRESHOLD=4.0
poses=defaultdict(lambda:defaultdict(list))
for r in csv.DictReader((SOURCE/'pose_geometry.csv').open()):
    poses[r['accession']][int(r['step'])].append(r)
summary=[]
for r in csv.DictReader((SOURCE/'both_steps_summary.csv').open()):
    acc=r['accession']; steps=poses[acc]
    assert len(steps[1])==len(steps[2])==50, (acc,'Incomplete ensemble')
    assert all(p['identity_pass']=='True' for group in steps.values() for p in group)
    close=[p for p in steps[1] if float(p['distance_A'])<THRESHOLD]
    weights=np.array([1/float(p['distance_A']) for p in close])
    angle=float(np.average([float(p['angle_deg']) for p in close],weights=weights)) if close else None
    d2=np.array([float(p['distance_A']) for p in steps[2]])
    summary.append(dict(accession=acc,name=r['enzyme_name'],n_step1=50,n_step2=50,
        n_step1_close=len(close),step1_close_fraction=len(close)/50,
        conditional_angle_deg=angle,step2_median_distance_A=float(np.median(d2)),
        step2_close_fraction=float(np.mean(d2<THRESHOLD)),
        frozen_all_pose_angle_deg=float(r['step_1_weighted_angle_deg'])))
with (OUT/'conditional_geometry_data.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
valid=[r for r in summary if r['conditional_angle_deg'] is not None]
missing=[r for r in summary if r['conditional_angle_deg'] is None]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
fig=plt.figure(figsize=(19,11),facecolor='white')
gs=fig.add_gridspec(1,2,width_ratios=[4.5,1.35],left=.055,right=.985,bottom=.22,top=.86,wspace=.06)
ax=fig.add_subplot(gs[0]); strip=fig.add_subplot(gs[1],sharey=ax)
maxd=max(float(p['distance_A']) for ss in poses.values() for p in ss[2])
for a in (ax,strip):
    a.set_ylim(max(18,maxd+1),-1)
    a.axhspan(0,4,color='#e7f2e9',zorder=0)
    a.axhline(4,color='#b94a48',lw=1.2,ls='--')
    a.grid(axis='y',color='#e5e7eb',lw=.7);a.set_axisbelow(True)
    a.spines[['top','right']].set_visible(False)
ax.set_xlim(min(r['conditional_angle_deg'] for r in valid)-5,180)
ax.set_ylabel('Step 2 — amine N to carbonyl C distance (Å)\nSmaller distance is higher',labelpad=12)
ax.set_xlabel('Step 1 — proximity-weighted attack angle A among poses with distance <4 Å (°)',labelpad=12)
strip.set_xlim(-.65,len(missing)-.35);strip.tick_params(axis='y',left=False,labelleft=False)
strip.set_title('No close step-1 poses\nConditional angle undefined',fontsize=11,pad=14)
colors=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('tab20b').colors)
rng=np.random.default_rng(17)
labels=[]
for i,r in enumerate(valid):
    acc=r['accession'];x=r['conditional_angle_deg'];color='#b83280' if acc=='P33222' else colors[i%len(colors)]
    distances=[float(p['distance_A']) for p in poses[acc][2]]
    ax.scatter(x+rng.uniform(-.28,.28,50),distances,s=13,alpha=.30,color=color,edgecolors='none',zorder=2)
    ax.scatter(x,r['step2_median_distance_A'],s=500*r['step1_close_fraction'],color=color,
               edgecolors='#222222',linewidths=1.1,zorder=4,alpha=.9)
    short=r['name'].split(' · ')[0].split(' (')[0]
    if short.lower().startswith('asparagine'):short='Asn synthetase'
    if short=='Trypanothione synthetase':short='Trypanothione syn.'
    labels.append(dict(x=x,y=r['step2_median_distance_A'],label=short,accession=acc,
                       color='#8f2462' if acc=='P33222' else '#222222',weight='bold' if acc=='P33222' else 'normal'))
_place_labels(ax,labels,fontsize=7.5)
for i,r in enumerate(missing):
    distances=[float(p['distance_A']) for p in poses[r['accession']][2]]
    color='#222222' if r['accession']=='WT_AS9' else '#84919b'
    strip.scatter(i+rng.uniform(-.12,.12,50),distances,s=12,alpha=.4,color=color,edgecolors='none')
    strip.scatter(i,r['step2_median_distance_A'],marker='x',s=45,color=color,zorder=4)
strip.set_xticks(range(len(missing)),[('WT AS9' if r['accession']=='WT_AS9' else r['name'].split(' · ')[0])+'\n'+r['accession'] for r in missing],rotation=90,fontsize=8)
fig.suptitle('Conditional step-1 geometry and step-2 attack distances',x=.055,y=.975,ha='left',fontsize=22,fontweight='bold')
fig.text(.055,.932,f'{len(summary)} candidates · 50 predicted poses per step · distance-only conditioning · no angle-based inclusion cutoff',fontsize=12,color='#53606a')
fig.text(.055,.898,'Small dots: individual step-2 poses    •    Bubble center: median step-2 distance    •    Bubble area: fraction of step-1 poses <4 Å',fontsize=11)
handles=[Line2D([],[],marker='o',ls='',markersize=np.sqrt(500*f),markerfacecolor='#8da5b5',markeredgecolor='#222222',label=f'{100*f:.0f}% ({int(50*f)}/50)') for f in [.1,.3,.6,1]]
fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.055,.078),ncol=4,frameon=False,title='Step-1 close-pose fraction (bubble area)',title_fontsize=11,fontsize=10,columnspacing=2.5)
fig.text(.055,.052,'Step 1: unrestrained predictions. Step 2: pocket-guided localization. The dashed line marks 4 Å; step-2 angle is not assessed here.',fontsize=10,color='#53606a')
fig.text(.055,.027,'Pose fractions describe prediction consistency, not equilibrium populations or measured activity. × in the right strip marks a median, not bubble area.',fontsize=10,color='#53606a')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'conditional_geometry_distance_scatter.{ext}',dpi=180)
plt.close(fig)
meta={'threshold_A':THRESHOLD,'comparison':'strictly less than','conditional_angle':'sum(angle / distance) / sum(1 / distance), using step-1 poses with distance < 4 A only','angle_selection':False,'bubble_area':'500 * step1_close_fraction points squared','y':'all step-2 pose distances; bubble at median','source_files':{str(SOURCE/f):hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in ['pose_geometry.csv','both_steps_summary.csv']},'n_candidates':len(summary),'n_no_close_step1':len(missing)}
(OUT/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
yjf=next(r for r in summary if r['accession']=='P33222')
assert yjf['n_step1_close']==16 and abs(yjf['conditional_angle_deg']-167.77612327325429)<1e-8
print(json.dumps({'candidates':len(summary),'undefined_angle':len(missing),'YjfC':yjf},indent=2))
