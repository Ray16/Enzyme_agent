"""Standalone candidacy evidence plots from saved data; no inference or deck edits."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import json,hashlib,re,textwrap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'exp_data/amide_synthetase'
PROX=BASE/'results/structural_proxies'
MUT=ROOT/'enzyme_design_agent/campaigns/amide_AS9_AS17_AS23_step1_2Mg_saturation_5A_20261019'
BLUE,ORANGE,TEAL,PURPLE,GRAY,INK='#0072B2','#D55E00','#009E73','#8043AD','#A0A5AA','#263746'
STRONG={'AS9','AS17','AS23'}
SOURCE_FILES=[];FIGURES=[];STATS={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.labelsize':17,
 'axes.titlesize':19,'axes.spines.top':False,'axes.spines.right':False,
 'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})


def read(relative):
    p=ROOT/relative;SOURCE_FILES.append(p)
    return pd.read_csv(p) if p.suffix=='.csv' else json.loads(p.read_text())


def finish(fig,stem,title,caption,kind='saved-data analysis'):
    title=textwrap.fill(title,78)
    fig.suptitle(title,fontsize=22,fontweight='bold',color=INK,y=.98)
    fig.tight_layout(rect=[0,.075,1,.85 if '\n' in title else .92])
    fig.text(.025,.015,caption.split('\n')[0],fontsize=10,color='#59636C',va='bottom')
    for ext in ['png','svg','pdf']:
        p=HERE/f'{stem}.{ext}'
        fig.savefig(p,dpi=240,bbox_inches='tight')
        if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    plt.close(fig)
    FIGURES.append({'name':stem,'title':title,'caption':caption,'kind':kind,
                    'files':[f'{stem}.{e}' for e in ['png','svg','pdf']]})


def load():
    global ASSAY,SCORES,POSES,FEATURES,COMPARISON,STEP2,VARIANTS,DISCOVERY
    ASSAY=read('exp_data/amide_synthetase/assay_data/panel_activity.csv').set_index('id')
    SCORES=read('exp_data/amide_synthetase/results/structural_proxies/attempts/simple_geometry_review_20261010/all_groups_scores.csv')
    POSES=read('exp_data/amide_synthetase/results/structural_proxies/tables/continuous_pose_geometry.csv')
    FEATURES=read('exp_data/amide_synthetase/results/structural_proxies/attempts/proxy_performance_20261011/per_enzyme_scores.csv').set_index('id')
    COMPARISON=read('exp_data/amide_synthetase/results/structural_proxies/attempts/proxy_performance_20261011/comparison.csv')
    STEP2=read('exp_data/amide_synthetase/results/analysis/current_two_step_metrics_20261023/per_enzyme.csv').set_index('id')
    VARIANTS=read('enzyme_design_agent/campaigns/amide_AS9_AS17_AS23_step1_2Mg_saturation_5A_20261019/analysis/single_mutants/variant_metrics.csv')
    DISCOVERY=read('slides-candidacy/discovery-ranking-simplified/ranking_and_modeling_details.csv')
    cols=[c for c in ASSAY if c.startswith('rep_')]
    assert ASSAY[cols].notna().all().all() and len(cols)==6
    assert np.allclose(ASSAY[cols].mean(1),ASSAY.activity_N66L1_ugml_avg)
    assert np.allclose(ASSAY[cols].std(1),ASSAY.activity_N66L1_ugml_std)
    panel=SCORES[SCORES.group=='calibration'].set_index('id')
    for ident,g in POSES[POSES.group=='calibration'].groupby('id'):
        w=1/g.distance_A.to_numpy();a=np.dot(g.angle_deg,w)/w.sum()
        assert abs(a-panel.loc[ident,'inverse_distance_angle_deg'])<1e-8
        if ident!='SfaB':assert abs(a-FEATURES.loc[ident,'weighted_angle'])<1e-8
    STATS['checks']={'assay_rows':len(ASSAY),'measurements_per_row':6,
        'summary_mean_and_sd_match':True,'pose_angles_match_saved_current_metric':True,
        'assay_repeat_type':'Not established by this CSV; do not call biological replicates.',
        'retrospective_scope':'Repeatedly inspected homolog panel; no independent validation.'}


def assay_plots():
    reps=[c for c in ASSAY if c.startswith('rep_')]
    for stem,ids in [('assay_full_panel_measurements',ASSAY.sort_values('activity_N66L1_ugml_avg',ascending=False).index.tolist()),
                     ('assay_key_candidates_and_controls',['AS9','AS17','AS23','AS4','SfaB','BL21'])]:
        d=ASSAY.loc[ids];fig,ax=plt.subplots(figsize=(17 if len(ids)>10 else 11,6))
        x=np.arange(len(d))
        for j,c in enumerate(reps):
            ax.scatter(x+(j-2.5)*.075,d[c],s=20 if len(d)>10 else 50,
                       color=BLUE if j<3 else ORANGE,alpha=.7,
                       label='2025-01-24 labels' if j==0 else '2025-01-09 labels' if j==3 else None)
        ax.plot(x,d.activity_N66L1_ugml_avg,'k_',markersize=12,label='Stored mean')
        ax.set_yscale('symlog',linthresh=1e-4)
        ax.set_ylabel('Product concentration (µg/mL)')
        ax.set_xticks(x,ids,rotation=90 if len(d)>10 else 0,fontsize=9 if len(d)>10 else 14)
        ax.legend(frameon=False,fontsize=11,ncol=3,loc='upper right')
        finish(fig,stem,'Stored assay measurements and experimental variability',
          'All six stored measurements shown; black marker: stored mean. Axis is symlog near zero to retain zero readings.\n'
          'Zeros are shown at zero. The CSV does not establish whether repeats are biological or technical. '
          'BL21 is the stored control label; empty-vector/no-enzyme identity is not inferred.')
    a=ASSAY[reps[:3]].mean(1);b=ASSAY[reps[3:]].mean(1)
    fig,ax=plt.subplots(figsize=(9,7))
    for ident in ASSAY.index:
        color=BLUE if ident in STRONG else ORANGE if ident in ['SfaB','BL21'] else GRAY
        ax.scatter(a[ident],b[ident],color=color,s=65 if ident in STRONG|{'SfaB','BL21'} else 25,zorder=3)
        if ident in STRONG|{'SfaB','BL21'}:ax.annotate(ident,(a[ident],b[ident]),xytext=(6,5),textcoords='offset points',fontsize=12)
    ax.plot([0,20],[0,20],ls='--',color='#7D858C',lw=1)
    ax.set_xscale('symlog',linthresh=1e-4);ax.set_yscale('symlog',linthresh=1e-4)
    ax.set_xlabel('Mean under 2025-01-24 labels (µg/mL)');ax.set_ylabel('Mean under 2025-01-09 labels (µg/mL)')
    finish(fig,'assay_date_label_comparison','Measurement variation across recorded date groups',
           'Three stored values under each date label. Dashed line: equal means. This does not establish the cause of date-group differences.')
    STATS['date_group_means']={i:{'2025-01-24':float(a[i]),'2025-01-09':float(b[i])} for i in ['AS9','AS17','AS23','SfaB','BL21']}
    ids=['AS9','AS17','AS23','AS4','SfaB','BL21']
    rows=[[i,'6',f'{ASSAY.loc[i,"activity_N66L1_ugml_avg"]:.6g}',
           f'{ASSAY.loc[i,"activity_N66L1_ugml_std"]:.6g}'] for i in ids]
    fig,ax=plt.subplots(figsize=(11,5));ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Stored ID','Stored measurements','Mean (µg/mL)','SD (µg/mL)'],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(16);table.scale(1,2)
    finish(fig,'assay_summary_mean_sd','Recorded mean and variation for key candidates and controls',
           'Mean and sample SD recomputed from all six stored values and verified against the CSV. No confidence interval or replicate-type claim.')


def angle_activity():
    fig,axs=plt.subplots(1,2,figsize=(13,5.5))
    corr=[]
    for ax,subset,label in [(axs[0],FEATURES,'All 48 homologs'),(axs[1],FEATURES.drop(index=list(STRONG)),'Without the three highest-yielding homologs')]:
        x=subset.weighted_angle;y=subset.activity_ugml
        for ident in subset.index:
            ax.scatter(x[ident],y[ident],color=BLUE if ident in STRONG else GRAY,s=55 if ident in STRONG else 24)
            if ident in STRONG or ident=='AS4':ax.annotate(ident,(x[ident],y[ident]),xytext=(4,5),textcoords='offset points',fontsize=11)
        rho=float(spearmanr(x,y).statistic);corr.append({'subset':label,'n':len(subset),'spearman_rho':rho})
        ax.set_yscale('log');ax.set_xlabel('Proximity-weighted attack angle A (°)');ax.set_ylabel('Mean product concentration (µg/mL)')
        ax.set_title(label,fontsize=14);ax.text(.03,.96,f'n = {len(subset)}; Spearman ρ = {rho:.3f}',transform=ax.transAxes,va='top',fontsize=12)
    finish(fig,'angle_yield_and_influence','Enrichment of strong producers does not imply quantitative yield prediction',
           'Descriptive, retrospective association. The right panel is a post-hoc influence diagnostic, not a separate validation set.')
    STATS['angle_yield_correlations']=corr
    panel=SCORES[SCORES.group=='calibration'].set_index('id').loc[FEATURES.index]
    fig,ax=plt.subplots(figsize=(9,6.5))
    dots=ax.scatter(panel.inverse_distance_angle_deg,panel.harmonic_distance_A,
                    c=np.log10(panel.activity_ugml),cmap='viridis',s=65,edgecolors='white',lw=.5)
    for ident in STRONG|{'AS4'}:
        ax.annotate(ident,(panel.loc[ident,'inverse_distance_angle_deg'],panel.loc[ident,'harmonic_distance_A']),xytext=(5,5),textcoords='offset points',fontsize=12)
    ax.set_xlabel('Proximity-weighted attack angle A (°)');ax.set_ylabel('Harmonic donor–phosphorus distance (Å)')
    fig.colorbar(dots,ax=ax,label='log₁₀ mean product concentration (µg/mL)')
    finish(fig,'joint_angle_distance_activity','Angular alignment and absolute proximity are separate observations',
           '48 homologs; one-Mg historical panel; 50 predictions each. Color is measured product concentration, not predicted activity.')


def ranking_plots():
    names={'weighted_angle':'Weighted angle A','mean_angle':'Mean angle',
           'minimum_distance':'Shortest distance','near_attack_fraction':'Near-attack fraction',
           'interface_confidence':'Interface confidence'}
    fig,axs=plt.subplots(1,2,figsize=(13,5.5));records={}
    for ax,label,positive in [(axs[0],'Three highest-yielding homologs',FEATURES.top3.astype(int)),
                              (axs[1],'Seven homologs above SfaB mean',FEATURES.above_sfab.astype(int))]:
        for j,(metric,name) in enumerate(names.items()):
            ordered=FEATURES.sort_values(metric,ascending=False,kind='mergesort')
            curve=[];previous=0.
            for _,ties in ordered.groupby(metric,sort=False):
                n=len(ties);hits=float(positive.loc[ties.index].sum())
                curve.extend(previous+np.arange(1,n+1)*hits/n);previous+=hits
            curve=np.array(curve);ax.plot(np.arange(1,49),curve,lw=2,label=name)
            records[label+' / '+metric]={'at5':float(curve[4]),'at10':float(curve[9]),'at15':float(curve[14])}
        ax.plot(np.arange(49),np.arange(49)*positive.sum()/48,'--',color='black',lw=1,label='Random expected recovery')
        ax.set_title(label,fontsize=14);ax.set_xlabel('Candidates selected');ax.set_ylabel('Retrospective producers recovered')
        ax.set_xlim(0,20);ax.set_ylim(-.1,positive.sum()+.2);ax.set_yticks(range(positive.sum()+1))
    axs[1].legend(frameon=False,fontsize=10,loc='lower right')
    finish(fig,'current_metric_shortlist_recovery','Shortlist recovery under current geometric and confidence baselines',
           '48 homologs; SfaB excluded from ranking. Groups are retrospective yield definitions, not validated active/inactive labels.\n'
           'All compared metrics use the same saved one-Mg ensembles. Ties use expected recovery within a tied group. Earlier Gaussian-score results do not apply to A.')
    STATS['shortlist_recovery']=records
    comp=COMPARISON.copy();comp['display']=comp.label+'  (n='+comp.defined.astype(str)+')'
    values=np.column_stack([comp.auroc,comp.top3_at10/3,comp.hits_at10/7])
    fig,ax=plt.subplots(figsize=(12,9));im=ax.imshow(values,vmin=0,vmax=1,cmap='cividis',aspect='auto')
    ax.set_yticks(np.arange(len(comp)),comp.display,fontsize=12)
    ax.set_xticks([0,1,2],['AUROC\nabove SfaB','Top-three recovery\n@10 / 3','Above-SfaB recovery\n@10 / 7'],fontsize=13)
    for i in range(len(comp)):
        for j in range(3):
            if np.isfinite(values[i,j]):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]<.5 else 'black',fontsize=13)
    fig.colorbar(im,ax=ax,fraction=.03,pad=.03)
    finish(fig,'all_explored_proxy_comparison','Complete saved comparison of 14 structural descriptors',
           'Exploratory performance on a repeatedly inspected panel; n denotes defined enzyme-level scores. AUROC uses available scores.\n'
           'Definitions and orientation are retained from the saved comparison; no direction was chosen using the plotted activity labels.')


def metric_limit():
    angles=np.array([160.,170.,175.]);dist=np.array([3.,4.,5.]);scales=np.array([1.,2.,5.])
    a=[np.dot(angles,1/(s*dist))/np.sum(1/(s*dist)) for s in scales]
    dh=[3/np.sum(1/(s*dist)) for s in scales]
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    axs[0].plot(scales,a,'o-',color=PURPLE,lw=2);axs[0].set_ylim(150,180)
    axs[0].set_ylabel('Weighted angle A (°)');axs[1].plot(scales,dh,'o-',color=TEAL,lw=2)
    axs[1].set_ylabel('Harmonic distance dH (Å)')
    for ax in axs:ax.set_xlabel('Uniform distance multiplier');ax.set_xticks(scales)
    finish(fig,'weighted_angle_scale_invariance','A weights proximity but does not impose an absolute distance requirement',
           'Mathematical illustration, not enzyme data: angles [160°, 170°, 175°], distances [3, 4, 5] Å before scaling.\n'
           'A = Σ(θᵢ/dᵢ) / Σ(1/dᵢ); multiplying every distance by the same factor cancels. Report dH alongside A.',
           kind='mathematical illustration')


def ensemble_plots():
    p=POSES[POSES.group=='calibration'].copy();p['sample']=p.model.str.extract(r'_model_(\d+)')[0].astype(int)
    ids=FEATURES.sort_values('weighted_angle',ascending=False).index
    distance=p.pivot(index='id',columns='sample',values='distance_A').loc[ids]
    angle=p.pivot(index='id',columns='sample',values='angle_deg').loc[ids]
    fig,axs=plt.subplots(1,2,figsize=(13,13))
    for ax,data,label,cmap,limits in [(axs[0],distance,'Attack distance (Å)','viridis',(2.5,12)),
                                      (axs[1],angle,'Attack angle (°)','cividis',(90,180))]:
        im=ax.imshow(data,aspect='auto',cmap=cmap,vmin=limits[0],vmax=limits[1]);fig.colorbar(im,ax=ax,label=label,fraction=.04,pad=.02,extend='both')
        ax.set_yticks(np.arange(len(ids)),ids,fontsize=9);ax.set_xticks([0,9,19,29,39,49],[1,10,20,30,40,50]);ax.set_xlabel('Saved model index + 1')
    finish(fig,'ensemble_distance_angle_coverage','All 50 saved poses for every assayed homolog',
           'Rows sorted by full-ensemble A. Colors saturate outside the indicated ranges; all models remain present.\n'
           'Sample index is not time. Pose fractions are prediction statistics, not equilibrium populations or biological replicates.')
    ids=list(FEATURES.index);distmat=distance.loc[ids].to_numpy();tmat=angle.loc[ids].to_numpy()
    full=FEATURES.loc[ids,'weighted_angle'].to_numpy();positive=FEATURES.loc[ids,'top3'].to_numpy().astype(bool)
    rng=np.random.default_rng(20261104);sizes=[5,10,20,30,40,50];runs=300;prob5=[];prob10=[];taus=[]
    for n in sizes:
        scores=[]
        for _ in range(runs):
            ix=np.array([rng.choice(50,n,replace=False) for i in ids])
            d=np.take_along_axis(distmat,ix,axis=1);t=np.take_along_axis(tmat,ix,axis=1)
            scores.append(np.sum(t/d,axis=1)/np.sum(1/d,axis=1))
        ss=np.array(scores);order=np.argsort(-ss,axis=1,kind='stable')
        prob5.append(float(np.mean(positive[order[:,:5]].sum(1)==3)))
        prob10.append(float(np.mean(positive[order[:,:10]].sum(1)==3)))
        taus.append([float(kendalltau(full,s).statistic) for s in ss])
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    axs[0].plot(sizes,prob5,'o-',label='All three in top 5');axs[0].plot(sizes,prob10,'s-',label='All three in top 10');axs[0].set_ylabel('Fraction of pose subsamples');axs[0].set_ylim(-.03,1.03);axs[0].legend(frameon=False,fontsize=12)
    q=np.quantile(taus,[.05,.5,.95],axis=1);axs[1].plot(sizes,q[1],'o-',color=TEAL);axs[1].fill_between(sizes,q[0],q[2],color=TEAL,alpha=.2);axs[1].set_ylabel('Kendall τ versus full ensemble')
    for ax in axs:ax.set_xlabel('Saved poses retained per enzyme');ax.set_xticks(sizes)
    finish(fig,'finite_pose_sampling_sensitivity','Sensitivity of current A rankings to finite pose sampling',
           '300 subsamples without replacement per size; same existing 50-pose batch. Shading: 5th–95th percentile.\n'
           'This is conditional technical sensitivity, not independent sampling, biological uncertainty or prospective validation.')
    STATS['finite_pose_sampling']={'sizes':sizes,'resamples':runs,'all_top3_at5':prob5,'all_top3_at10':prob10,'kendall_quantiles':q.tolist()}
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    counts={}
    for ax,ident in zip(axs,['AS30','AS32']):
        g=p[p.id==ident];good=(g.distance_A<4)&(g.angle_deg>150);counts[ident]=int(good.sum())
        ax.scatter(g.distance_A,g.angle_deg,c=np.where(good,BLUE,GRAY),s=40)
        ax.axvline(4,color='black',ls='--',lw=1);ax.axhline(150,color='black',ls='--',lw=1)
        ax.set_xlim(2.7,10);ax.set_ylim(90,183);ax.set_xlabel('Attack distance (Å)');ax.set_ylabel('Attack angle (°)');ax.set_title(f'{ident}: {good.sum()}/50 in illustrative region',fontsize=14)
    finish(fig,'best_pose_vs_pose_consistency','A favorable pose and consistent favorable geometry ask different questions',
           'Development examples from the saved one-Mg panel. Region d < 4 Å and θ > 150° is illustrative and not prospectively validated.\n'
           'Generated-pose fractions do not estimate conformational populations. Independent untested candidates are needed for a prospective test.')
    STATS['consistency_examples']=counts


def second_step():
    d=STEP2.loc[FEATURES.index]
    fig,axs=plt.subplots(1,2,figsize=(12,5.5))
    for ax,col,label in [(axs[0],'step1_angle_deg','Activation angle A (°)'),
                         (axs[1],'step2_neutral_Dh_A','Neutral amine–acyl carbon dH (Å)')]:
        for ident in d.index:
            ax.scatter(d.loc[ident,col],d.loc[ident,'activity_ugml'],s=50 if ident in STRONG else 24,color=BLUE if ident in STRONG else GRAY)
            if ident in STRONG:ax.annotate(ident,(d.loc[ident,col],d.loc[ident,'activity_ugml']),xytext=(5,5),textcoords='offset points',fontsize=11)
        ax.set_yscale('log');ax.set_xlabel(label);ax.set_ylabel('Mean product concentration (µg/mL)')
    finish(fig,'activation_vs_aminolysis_readouts','Aminolysis distance does not reproduce activation-step enrichment in this panel',
           '48 homologs, paired saved one-Mg predictions. Neutral attacking N assigned by saved ligand formal charge and topology.\n'
           'Step 2 uses pocket-localization restraints. Distance alone is not an attack-orientation or catalytic-barrier measurement.')
    fig,ax=plt.subplots(figsize=(10,5.5));order=d.sort_values('closest_N_is_protonated_poses',ascending=False)
    ax.bar(np.arange(len(order)),order.closest_N_is_protonated_poses/50,color=ORANGE)
    ax.set_xticks(np.arange(len(order)),order.index,rotation=90,fontsize=9);ax.set_ylim(0,1);ax.set_ylabel('Fraction where nearest terminal N is protonated')
    finish(fig,'amine_identity_assignment','Selecting the closest terminal nitrogen can select the protonated end',
           'Saved step-2 audit; all 50 poses per enzyme retained. Choose the reactive nitrogen from chemical identity, not nearest-N geometry.')


def mutations():
    d=VARIANTS[VARIANTS.mutation!='WT'].copy();assert len(d)==5586 and (d.n_models==50).all()
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,(parent,g) in zip(axs,d.groupby('parent',sort=False)):
        ax.scatter(g.delta_angle_deg,g.delta_distance_A,s=8,alpha=.35,color=BLUE)
        ax.axvline(0,color='black',lw=1);ax.axhline(0,color='black',lw=1)
        ax.set_title(f'{parent}: {len(g):,} single substitutions',fontsize=14)
        ax.set_xlabel('ΔA versus matched WT (°)');ax.set_ylabel('ΔdH versus matched WT (Å)')
        chosen='G211V' if parent=='AS9' else 'P318R' if parent=='AS23' else None
        if chosen:
            r=g[g.mutation==chosen].iloc[0];ax.scatter(r.delta_angle_deg,r.delta_distance_A,s=70,color=ORANGE,zorder=4);ax.annotate(chosen,(r.delta_angle_deg,r.delta_distance_A),xytext=(6,6),textcoords='offset points',fontsize=11)
    finish(fig,'complete_mutant_geometry_tradeoffs','Full saturation scan: angular gains and distance changes',
           '5,586 single substitutions + 3 matched WT controls; 50 predictions each; separate two-Mg campaign.\n'
           'Changes are predicted geometry, not measured activity. Lower-right quadrant combines higher A and shorter dH.')
    aa=list('ACDEFGHIKLMNPQRSTVWY')
    counts={}
    for parent,g0 in d.groupby('parent'):
        g=g0.copy();parts=g.mutation.str.extract(r'([A-Z])(\d+)([A-Z])');g['pos']=parts[1].astype(int).values;g['to']=parts[2].values
        h=g.pivot(index='to',columns='pos',values='delta_angle_deg').reindex(aa)
        norm=TwoSlopeNorm(vmin=float(np.nanmin(h)),vcenter=0,vmax=float(np.nanmax(h)))
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#DDDDDD')
        fig,ax=plt.subplots(figsize=(max(14,len(h.columns)*.13),6))
        im=ax.imshow(h,aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
        ax.set_yticks(range(20),aa);ticks=np.arange(0,len(h.columns),5);ax.set_xticks(ticks,h.columns[ticks],fontsize=9,rotation=90)
        ax.set_xlabel('WT residue position');ax.set_ylabel('Substituted amino acid');fig.colorbar(im,ax=ax,label='ΔA versus matched WT (°)',pad=.02)
        finish(fig,f'{parent}_all_mutation_effects',f'{parent}: all scanned positions and single substitutions',
               'Gray cells are unmutated WT identities, not zero effect. Each panel uses its own numeric color range, centered on zero.\n'
               'Two-Mg predicted ensembles; 50 samples per variant. No experimental improvement is established.')
        counts[parent]={'positions':len(h.columns),'substitutions':len(g),'higher_A':int((g.delta_angle_deg>0).sum()),
                        'higher_A_shorter_dH':int(((g.delta_angle_deg>0)&(g.delta_distance_A<0)).sum())}
    STATS['mutation_counts']=counts


def discovery():
    fig,ax=plt.subplots(figsize=(10,6));labels=['Evaluated reaction / substrate comparisons','Representative candidates','Modeled representatives','Unmodeled representatives']
    vals=[63,43,36,7];ax.barh(labels,vals,color=[BLUE,TEAL,PURPLE,GRAY]);ax.invert_yaxis();ax.set_xlabel('Recorded count')
    for i,v in enumerate(vals):ax.text(v+.7,i,str(v),va='center',fontsize=17)
    ax.set_xlim(0,70)
    finish(fig,'discovery_candidate_accounting','What the saved discovery case actually produced',
           'September 17 case: 63 evaluated, 43 representatives, 36 modeled. Seven >800-residue representatives were unmodeled in that round.\n'
           'Stages are accounting categories, not biological hit rates; unmodeled candidates were not shown to be inactive.')
    records=read('exp_data/amide_synthetase/results/structural_proxies/tables/model_setup_checks.json')
    mg={r['accession']:r['actual_Mg_atoms'] for r in records}
    fig,axs=plt.subplots(1,2,figsize=(15,11),sharex=True)
    for ax,site,title in zip(axs,['PA','PG'],['AMP coordinate: Pα / O3A','ADP coordinate: Pγ / O3B']):
        g=DISCOVERY[DISCOVERY.stored_geometry_site==site].sort_values('angle_deg')
        labels=[f'{r.display_name} ({r.accession})' for r in g.itertuples()]
        colors=[BLUE if mg.get(a)==1 else ORANGE if mg.get(a)==2 else GRAY for a in g.accession]
        ax.scatter(g.angle_deg,np.arange(len(g)),c=colors,s=45);ax.set_yticks(np.arange(len(g)),labels,fontsize=10)
        ax.set_title(title,fontsize=15);ax.set_xlabel('Recorded weighted attack angle (°)');ax.set_xlim(25,180)
    finish(fig,'discovery_geometry_by_activation_route','Structural annotations separated by the recorded activation coordinate',
           'Saved 36-candidate comparison. Blue: one Mg; orange: two Mg in the saved complex. Geometry does not override native-reaction evidence.\n'
           'Substrate, metal and family differences limit cross-family score comparability; values are not universal predicted activities.')
    fig,ax=plt.subplots(figsize=(9,5))
    names=['DdaG','PuuA','AS9 illustration','AS9 mutant WT'];counts=[1,2,1,2]
    ax.bar(names,counts,color=[BLUE,ORANGE,BLUE,ORANGE]);ax.set_yticks([0,1,2]);ax.set_ylim(0,2.5);ax.set_ylabel('Mg ions in the saved model')
    finish(fig,'metal_count_is_model_specific','Metal stoichiometry is an explicit modeling assumption',
           'Saved coordinate/input counts, not experimentally measured stoichiometries. Two Mg ions are retained in the PuuA figure.\n'
           'Related GS structures support metal-assisted catalysis, but AMP/ADP fate alone does not determine Mg count; PuuA-specific stoichiometry remains unverified here.')


def inventory():
    d=read('exp_data/amide_synthetase/results/structural_proxies/attempts/proxy_inventory_20261011/historical_feature_inventory.csv')
    d['category']=d.feature.str.split(':').str[0]
    counts=d[d.status=='retained'].groupby('category').size().sort_values()
    labels={'elec':'Charge environment','pka':'Protonation','mg':'Metal coordination',
            'bat':'Activation geometry','hyp':'Mechanistic descriptors','esp':'Electrostatic potential / field',
            'psiptm':'Interface confidence','tunnel':'Substrate access','sasa':'Intermediate exposure',
            's2nac':'Amine attack (step 2)','preorg':'Pose consistency','boltz':'Other ensemble summary'}
    fig,ax=plt.subplots(figsize=(11,6));ax.barh([labels.get(x,x) for x in counts.index],counts,color=BLUE);ax.set_xlabel('Retained distinct exported definitions')
    for i,v in enumerate(counts):ax.text(v+.2,i,str(v),va='center',fontsize=13)
    finish(fig,'structural_proxy_search_inventory','Breadth of the historical descriptor exploration',
           '98 exported fields: 88 distinct nonconstant definitions, 5 duplicate exports, 5 constant/undefined fields.\n'
           'Field counts include variants and diagnostics. Later matched formulas overlap this inventory; counts are not additive or independent tests.')
    STATS['inventory_status_counts']=d.status.value_counts().to_dict()


def source_and_metal_audits():
    import openpyxl
    source=BASE/'assay_data/panel_activity_summary.xlsx';SOURCE_FILES.append(source)
    w=openpyxl.load_workbook(source,read_only=True,data_only=True)
    data=list(w['Activity'].values);xls={r[0]:np.array(r[1:7],dtype=float) for r in data[1:] if r[0]}
    reps=[c for c in ASSAY if c.startswith('rep_')]
    assert set(xls)==set(ASSAY.index)
    assert all(np.allclose(xls[i],ASSAY.loc[i,reps].to_numpy(dtype=float)) for i in xls)
    historical=BASE/'assay_data/raw_assay_notes_2023-11-17.xlsx';SOURCE_FILES.append(historical)
    wh=openpyxl.load_workbook(historical,read_only=True,data_only=True)
    STATS['workbook_audit']={'2025_activity_rows':len(xls),'all_six_values_match_csv':True,
        'historical_2023_sheets':wh.sheetnames,
        'historical_limits_not_transferred':'No verified mapping from the 2023 Nyl-labeled workbook to the 2025 AS-labeled panel.'}
    fig,ax=plt.subplots(figsize=(12,5.5));ax.axis('off')
    rows=[['2025 Activity worksheet','50 IDs × 6 stored measurements','All values match panel_activity.csv'],
          ['2025 summary statistics','Average and STD columns','Mean and sample SD match the CSV'],
          ['2023 historical workbook','BioRepData, TechRepData, CAL, Noise','Different date and Nyl-style labels'],
          ['Transfer boundary','No verified link to the 2025 AS panel','Do not import its LOD or replicate definition']]
    t=ax.table(cellText=rows,colLabels=['Source','Recorded content','Verified interpretation'],loc='center',cellLoc='left',colWidths=[.24,.35,.41])
    t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,2.7)
    finish(fig,'assay_workbook_provenance','The plotted panel matches the 2025 source workbook',
           'The separate 2023 workbook contains calibration / repetition notes, but its limits and replicate labels are not assigned to this 2025 panel.')
    p=ROOT/'slides-candidacy/adp-attack-geometry/geometry.json';SOURCE_FILES.append(p)
    m=json.loads(p.read_text());structure=Path(m['source']);SOURCE_FILES.append(structure)
    protein=[]
    for line in structure.read_text().splitlines():
        if line.startswith(('ATOM  ','HETATM')) and line[21]=='A' and line[76:78].strip() in ['O','N']:
            protein.append((f'{line[17:20].strip()} {line[22:26].strip()} {line[12:16].strip()}',
                            np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])))
    mg=np.array([a['xyz'] for a in m['atoms']['Mg']]);rows=[];records=[]
    for i,point in enumerate(mg):
        atp=sorted((float(np.linalg.norm(point-np.array(a['xyz']))),a['name']) for a in m['atoms']['ATP'] if a['element']=='O')
        sub=sorted((float(np.linalg.norm(point-np.array(a['xyz']))),a['name']) for a in m['atoms']['adipate'] if a['element']=='O')
        prot=sorted((float(np.linalg.norm(point-xyz)),name) for name,xyz in protein)
        rows.append([f'Mg {i+1}',f'ATP {atp[0][1]}: {atp[0][0]:.2f} Å',
                     f'{prot[0][1]}: {prot[0][0]:.2f} Å',f'Adipate {sub[0][1]}: {sub[0][0]:.2f} Å'])
        records.append({'Mg':i+1,'nearest_ATP_oxygen':atp[0],'nearest_protein_N_or_O':prot[0],
                        'nearest_adipate_oxygen':sub[0]})
    fig,ax=plt.subplots(figsize=(12,5));ax.axis('off')
    t=ax.table(cellText=rows,colLabels=['Saved ion','Nearest ATP oxygen','Nearest protein N / O','Nearest adipate oxygen'],loc='center',cellLoc='center')
    t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,2.7)
    ax.text(.02,.2,f'Mg–Mg separation in the saved pose: {np.linalg.norm(mg[0]-mg[1]):.2f} Å',transform=ax.transAxes,fontsize=16)
    finish(fig,'PuuA_modeled_metal_distance_audit','Two Mg ions are present in the saved PuuA prediction',
           'Distances are measurements of predicted coordinates, not validated coordination assignments or experimental stoichiometry.\n'
           'Water ligands and a complete native assembly are not established by this monomer prediction; this table does not prove a physiological metal site.')
    STATS['PuuA_metal_audit']={'Mg_Mg_distance_A':float(np.linalg.norm(mg[0]-mg[1])),
                             'ions':records,'experimental_stoichiometry_verified':False}


def main():
    HERE.mkdir(exist_ok=True);load()
    for func in [assay_plots,angle_activity,ranking_plots,metric_limit,ensemble_plots,second_step,mutations,discovery,inventory,source_and_metal_audits]:
        func();print(f'Completed {func.__name__}',flush=True)
    sources=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(SOURCE_FILES)]
    (HERE/'quantitative_results.json').write_text(json.dumps(STATS,indent=2)+'\n')
    (HERE/'quantitative_figures.json').write_text(json.dumps({'figures':FIGURES,'sources':sources,
       'scope':'Saved-data plots and explicit mathematical illustrations only; no new predictor or fleet execution.'},indent=2)+'\n')
    print(f'Wrote {len(FIGURES)} independent quantitative figures to {HERE}',flush=True)


if __name__=='__main__':main()
