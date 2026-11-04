"""Plot a completed expression_screen result; no network or model inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from Bio.Data import CodonTable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig,directory,name):
    paths=[]
    for ext in ['png','svg','pdf']:
        path=directory/f'{name}.{ext}'
        fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');paths.append(path)
    plt.close(fig)
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--reference-id',default='AS9_control')
    parser.add_argument('--highlight',nargs='*',default=[])
    args=parser.parse_args(argv);root=args.results_dir
    rows=json.loads((root/'records.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    dna=json.loads((root/'CDS_provenance.json').read_text())
    by_id={r['id']:r for r in rows};ref=by_id.get(args.reference_id)
    arms=sorted(set(a for r in rows for a in r['arms'].split('+') if a and a!='reference'))
    if not arms:arms=['candidates']
    groups={a:[r for r in rows if r['dna_status']=='verified' and (a=='candidates' or a in r['arms'].split('+')) and r['id']!=args.reference_id] for a in arms}
    groups={a:g for a,g in groups.items() if g};arms=list(groups)
    if not groups:raise ValueError('No verified candidate DNA to plot')
    labels={'esm2':'ESM2-650M','genslm':'GenSLM-ESM 600M'}
    palette={'esm2':'#0072B2','genslm':'#D55E00'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    rng=np.random.default_rng(20261020)
    for ax,key,title,xlabel in zip(axes,['CAI','low_adaptiveness_codon_pct'],
        ['Codon adaptation to BL21(DE3)','Low-adaptiveness codon burden'],
        ['Codon adaptation index (CAI; higher = closer to reference preference)',
         'Codons with reference relative adaptiveness < 0.1 (%)']):
        for y,arm in enumerate(arms,1):
            group=groups[arm];values=np.array([r[key] for r in group]);color=palette.get(arm,'#009E73')
            box=ax.boxplot([values],positions=[y],vert=False,widths=.35,patch_artist=True,showfliers=False,
                medianprops={'color':'black'},boxprops={'facecolor':color,'alpha':.16})
            ax.scatter(values,y+rng.uniform(-.09,.09,len(values)),color=color,s=27,alpha=.85,zorder=3)
            ax.plot(values.mean(),y,'D',ms=7,mfc='white',mec='black',zorder=4)
            ax.text(.02,y+.28,f"Mean {values.mean():.3f}"+('%' if key.endswith('pct') else ''),
                transform=ax.get_yaxis_transform(),fontsize=9,color=color)
            for ident in args.highlight:
                if ident in {r['id'] for r in group}:
                    row=by_id[ident];ax.scatter(row[key],y,s=110,facecolors='none',edgecolors='black',linewidths=1,zorder=5)
        if ref and ref.get(key) is not None:
            value=ref[key];ax.axvline(value,color='#444444',ls='--',lw=1)
            ax.text(.02,.97,f'AS9 native CDS: {value:.3f}'+('%' if key.endswith('pct') else ''),
                transform=ax.transAxes,va='top',fontsize=9,color='#444444')
        ax.set_yticks(range(1,len(arms)+1),[f'{labels.get(a,a)} (n={len(groups[a])})' for a in arms])
        ax.set_ylim(.35,len(arms)+.55);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.set_title(title)
        ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Native coding DNA: GenSLM versus ESM2 candidates' if set(arms)=={'esm2','genslm'} else 'Native coding DNA: candidate comparison',fontsize=14)
    n=len({r['id'] for group in groups.values() for r in group})
    fig.supxlabel(f'{n} candidate CDSs verified. Diamonds: means. Rings: highlighted candidates. DNA metrics do not establish soluble expression.',fontsize=9)
    outputs=save(fig,root,'codon_comparison')

    # The host CAI weights are proportional to counts within each synonymous
    # family (with Biopython's 0.5 pseudocount for unobserved reference codons).
    table=CodonTable.unambiguous_dna_by_id[11].forward_table
    amino_acids=sorted(a for a in set(table.values()) if sum(v==a for v in table.values())>1)
    codons=[c for aa in amino_acids for c in sorted(table) if table[c]==aa]
    row_ids=[i for i in [args.reference_id,*args.highlight] if i in dna]
    if not row_ids:row_ids=[r['id'] for r in rows if r['id'] in dna][:5]
    values=[];row_labels=['BL21 ribosomal reference']
    weights=summary['host_reference']['codon_relative_adaptiveness']
    def normalize(counts):return [counts.get(c,0)/sum(counts.get(x,0) for x in codons if table[x]==table[c]) if sum(counts.get(x,0) for x in codons if table[x]==table[c]) else np.nan for c in codons]
    values.append(normalize(weights))
    for ident in row_ids:
        sequence=dna[ident]['dna'];counts=Counter(sequence[i:i+3] for i in range(0,len(sequence),3))
        values.append(normalize(counts));row_labels.append('AS9 native CDS' if ident==args.reference_id else ident)
    fig,ax=plt.subplots(figsize=(17,max(3.1,.6*len(values))),constrained_layout=True)
    im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
    ax.set_yticks(range(len(row_labels)),row_labels);ax.set_xticks(range(len(codons)),codons,rotation=90,fontsize=8)
    cursor=0
    for aa in amino_acids:
        n=sum(table[c]==aa for c in codons)
        if cursor:ax.axvline(cursor-.5,color='white',lw=1.5)
        ax.text(cursor+(n-1)/2,-.72,aa,ha='center',va='bottom',fontsize=9)
        cursor+=n
    ax.set_title('Synonymous codon preferences: host reference and selected sequences',pad=25)
    fig.colorbar(im,ax=ax,label='Fraction within synonymous codon family',shrink=.75,pad=.015)
    fig.supxlabel('DNA codons grouped by encoded amino acid. Native-CDS rows: observed fractions; host row: normalized CAI reference weights. Stops, Met and Trp omitted.',fontsize=9)
    outputs+=save(fig,root,'codon_preferences')
    (root/'codon_figure_manifest.json').write_text(json.dumps(dict(
        candidate_group_counts={a:len(groups[a]) for a in groups},highlight_ids=args.highlight,
        source_summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}),indent=2)+'\n')
    print('Generated',', '.join(str(p) for p in outputs))


if __name__=='__main__':main()
