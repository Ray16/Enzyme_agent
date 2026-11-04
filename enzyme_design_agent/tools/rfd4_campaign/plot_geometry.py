"""Reusable distributions and distance/angle plots for generated campaigns.

Example: python plot_geometry.py --campaign runs/rfd4_campaigns/CAMPAIGN
An inverse-distance weighted distribution across designs is distinct from a
distribution of independent ensemble scores per sequence. No refold metric is
inferred from generated structures.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--csv', type=Path, help='Default: CAMPAIGN/local_analysis/reactive_geometry.csv')
    parser.add_argument('--audit', type=Path, help='Optional motif_integrity_audit.json')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--title', help='Display title; default is the campaign directory name')
    parser.add_argument('--group-column', default='input_id')
    parser.add_argument('--bin-width', type=float, default=.5, help='Histogram bin width, degrees')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    source = args.csv or campaign/'local_analysis/reactive_geometry.csv'
    audit_path = args.audit or campaign/'local_analysis/motif_integrity_audit.json'
    out = args.outdir or campaign/'local_analysis/figures'
    out.mkdir(exist_ok=True, parents=True)
    title = args.title or campaign.name
    if args.bin_width <= 0:
        parser.error('--bin-width must be positive')
    with source.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty table with unique design IDs')
    passed = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        passed = {r['id']: r['motif_and_assembly_integrity_pass'] for r in audit['rows']}
    colors = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
        'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    groups, summary = {}, {}
    for i, key in enumerate(sorted({r[args.group_column] for r in rows})):
        subset = [r for r in rows if r[args.group_column] == key]
        d = np.array([float(r['distance_A']) for r in subset])
        a = np.array([float(r['angle_deg']) for r in subset])
        if not np.isfinite(a).all() or not np.isfinite(d).all() or (d <= 0).any():
            raise ValueError(f'Invalid geometry in {key}; no silent row exclusion')
        status = np.array([1 if passed.get(r['id']) is True else
                           0 if passed.get(r['id']) is False else -1 for r in subset])
        w = 1/d
        groups[key] = (d, a, status, w, colors[i % len(colors)])
        summary[key] = dict(n=len(a), integrity_pass=int((status == 1).sum()),
            integrity_flag=int((status == 0).sum()), integrity_unknown=int((status == -1).sum()),
            proximity_weighted_angle_deg=float(np.sum(a/d)/np.sum(1/d)),
            harmonic_distance_A=float(len(d)/np.sum(1/d)),
            angle_range_deg=[float(a.min()), float(a.max())],
            distance_range_A=[float(d.min()), float(d.max())])

    all_angles = np.concatenate([v[1] for v in groups.values()])
    lower = np.floor(all_angles.min()/args.bin_width)*args.bin_width
    upper = np.ceil(all_angles.max()/args.bin_width)*args.bin_width + args.bin_width
    bins = np.arange(lower, upper + args.bin_width*.5, args.bin_width)
    fig, ax = plt.subplots(figsize=(10, 5.9))
    ymax = 0
    for key, (d, a, status, w, color) in groups.items():
        weights = 100*w/w.sum()
        heights, _ = np.histogram(a, bins=bins, weights=weights)
        ymax = max(ymax, heights.max())
        mean = summary[key]['proximity_weighted_angle_deg']
        ax.hist(a, bins=bins, weights=weights, color=color, alpha=.27,
                edgecolor=color, linewidth=1.3,
                label=f'{key} · n={len(a)} · weighted mean {mean:.2f}°')
        ax.axvline(mean, color=color, ls='--', lw=1.7)
        ax.scatter(a[status != 0], np.zeros((status != 0).sum())-1.2,
                   marker='|', color=color, s=55, linewidths=1, clip_on=False)
        ax.scatter(a[status == 0], np.zeros((status == 0).sum())-1.2,
                   marker='x', color=color, s=38, linewidths=1.5, clip_on=False)
    ax.set(xlabel='Attack angle (degrees)', ylabel='Inverse-distance weighted fraction (%)',
           ylim=(-2.5, max(10,ymax*1.45)))
    ax.set_title(f'{title}\nProximity-weighted attack-angle distribution',
                 loc='left', fontweight='bold', pad=35)
    ax.text(0, 1.025, 'Each group normalized separately; bin weight = Σ(1/d).',
            transform=ax.transAxes, fontsize=10, color='#555555')
    ax.legend(frameon=False, loc='upper center', fontsize=10)
    ax.grid(axis='y', alpha=.16)
    flags = sum(v['integrity_flag'] for v in summary.values())
    fig.text(.10,.065, f'{len(rows)} generated structures · one pose per design · {flags} integrity flags retained (× on rug)',
             fontsize=9,color='#555555')
    fig.text(.10,.025, 'Conditioned generation; independent refolding / cofold ensemble scores are pending.',
             fontsize=9,color='#555555')
    fig.subplots_adjust(left=.10,right=.98,top=.76,bottom=.22)
    for ext in ['png','svg']:
        fig.savefig(out/f'proximity_weighted_attack_angle_distribution.{ext}',dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5.9))
    for key, (d,a,status,w,color) in groups.items():
        for value, marker, desc in [(1,'o','integrity pass'),(0,'x','integrity flag'),(-1,'o','integrity unknown')]:
            mask = status == value
            if not mask.any():
                continue
            kwargs = dict(marker=marker,s=78 if value==0 else 39,
                          linewidths=1.7 if value==0 else .6,
                          label=f'{key}: {mask.sum()} {desc}')
            if value==-1:
                kwargs.update(facecolors='none',edgecolors=color)
            else:
                kwargs.update(color=color,alpha=.8)
            ax.scatter(d[mask],a[mask],**kwargs)
    ax.set(xlabel='Attack distance (Å)',ylabel='Attack angle (degrees)')
    ax.set_title(f'{title}\nAttack distance vs attack angle',loc='left',fontweight='bold',pad=15)
    ax.grid(alpha=.16)
    ax.legend(frameon=False,loc='best',fontsize=9)
    fig.text(.10,.065,f'All {len(rows)} designs shown · no filtering by requested length or integrity outcome',
             fontsize=9,color='#555555')
    fig.text(.10,.025,'Conditioned generation geometry; each point is one design, not an independent ensemble mean.',
             fontsize=9,color='#555555')
    fig.subplots_adjust(left=.10,right=.98,top=.82,bottom=.22)
    for ext in ['png','svg']:
        fig.savefig(out/f'attack_distance_vs_angle.{ext}',dpi=220)
    plt.close(fig)

    record=dict(campaign=campaign.name,source=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Inverse-distance weighted distributions across generated designs, not per-design independent ensemble scores',
        metric='sum(angle/distance)/sum(1/distance)',harmonic_distance='N/sum(1/distance)',
        group_column=args.group_column, bin_width_deg=args.bin_width,by_starting_motif=summary)
    if audit_path.exists():
        record.update(audit=str(audit_path.resolve()),audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
    (out/'geometry_plot_summary.json').write_text(json.dumps(record,indent=2)+'\n')
    # Compose actual coordinate renders, if the rendering command has run.
    provenance=out/'theozyme_render_provenance.json'
    if provenance.exists():
        from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps
        from matplotlib.font_manager import findfont
        render=json.loads(provenance.read_text())
        inputs=render['inputs']
        cols=min(3,len(inputs)); nrows=(len(inputs)+cols-1)//cols
        panel_width,panel_height=800,820
        canvas=Image.new('RGB',(max(1600,cols*panel_width),nrows*panel_height+210),'white')
        draw=ImageDraw.Draw(canvas)
        font_path=findfont('DejaVu Sans')
        font=lambda size: ImageFont.truetype(font_path,size)
        draw.text((30,15),f'{title}: theozyme inputs',font=font(34),fill='#202020')
        for i,item in enumerate(inputs):
            image=Image.open(out/f"{item['pose']}_theozyme.png").convert('RGB')
            box=ImageChops.difference(image,Image.new('RGB',image.size,'white')).getbbox()
            if box:
                image=ImageOps.expand(image.crop(box),border=28,fill='white')
            image=ImageOps.contain(image,(panel_width-30,panel_height-75),Image.Resampling.LANCZOS)
            x=(i%cols)*panel_width; y=70+(i//cols)*panel_height
            draw.text((x+35,y),f"{item['pose']} · {item['fixed_residues']} retained residues · {item['fixed_protein_atoms']} anchors",font=font(24),fill='#202020')
            canvas.paste(image,(x+(panel_width-image.width)//2,y+50))
        y=nrows*panel_height+80
        if render.get('style_source'):
            captions=[
                'Shared slides-candidacy style: carbon gray; oxygen red; nitrogen blue; phosphorus orange; magnesium green.',
                'Opaque protein spheres mark conditioned atoms; translucent sticks show the retained-residue context.',
                f"Input coordinates unchanged. Source residue numbers retained. Both panels use the same {render.get('camera','shared')} camera."
            ]
        else:
            captions=[render['rendering'],'Input coordinates unchanged. Source residue numbers retained.']
        for line in captions:
            draw.text((30,y),line,font=font(21),fill='#454545');y+=35
        canvas.save(out/'theozyme_comparison.png',dpi=(180,180))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
