"""Rank the 36 modeled discovery candidates by the proximity-weighted angle A.

Reproduces the manuscript descriptor (eq. synthetase_proximity_angle) and marks
the GS-fold candidates, whose active site sits at a subunit interface but which
were cofolded as monomers - their geometry is not interpretable as modeled.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent')
OUT = ROOT / 'structural_modeling/figures'
SCORES = Path(sys.argv[1])          # wt_scores.json from the rescoring sweep
CAND = ROOT / ('structural_modeling/cofold/20260917_135330_find_amide_synthetases_'
               'that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl/candidates.csv')
GS_C, OK_C = '#eb6834', '#2a78d6'
INK, INK2, MUTED, SURFACE = '#0b0b0b', '#52514e', '#b8b7b0', '#fcfcfb'

plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                     'savefig.facecolor': SURFACE, 'font.size': 9,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})

C = {r['accession']: r for r in csv.DictReader(CAND.open())}
W = json.loads(SCORES.read_text())
is_gs = lambda a: 'glutamine synthetase' in C.get(a, {}).get('interpro_family', '').lower()
name = lambda a: (C[a].get('gene') or C[a]['protein_name'][:13]) + f' ({a})'
TARGETS = {'P0DXD6', 'P78061', 'Q46E79'}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 6.0),
                               gridspec_kw={'width_ratios': [1.05, 1]})

# -- left: ranking by A, colour = interpretability
W.sort(key=lambda w: w['angle_deg'])
y = np.arange(len(W))
for w, yy in zip(W, y):
    a = w['accession']
    c = GS_C if is_gs(a) else OK_C
    axL.plot([0, w['angle_deg']], [yy, yy], color=c, lw=1.1, alpha=.35, zorder=2)
    axL.scatter([w['angle_deg']], [yy], s=34, color=c, zorder=3,
                edgecolor=INK if a in TARGETS else 'none', lw=1.3)
axL.axvline(150, color=INK, lw=1.0, ls='--', zorder=4)
axL.text(150.6, -1.4, 'in-line threshold 150°', fontsize=8, color=INK)
axL.set_yticks(y)
axL.set_yticklabels([name(w['accession']) for w in W], fontsize=7.6)
for t, w in zip(axL.get_yticklabels(), W):
    if w['accession'] in TARGETS:
        t.set_fontweight('bold')
axL.set_xlabel('proximity-weighted attack angle  A  (deg)')
axL.set_xlim(0, 185)
axL.set_ylim(-2.2, len(W) - .3)
axL.grid(axis='x', color=MUTED, lw=.5, alpha=.45, zorder=0)
axL.set_axisbelow(True)

# -- right: A vs Dh, the pair the manuscript insists on reading together
axR.add_patch(plt.Rectangle((0, 150), 4, 40, facecolor='#1baf7a', alpha=.12,
                            edgecolor='#1baf7a', lw=1.2, ls='--', zorder=1))
axR.text(3.88, 186, 'interpretable &\nposed', ha='right', va='top',
         fontsize=8.5, color='#0f7a55')
for w in W:
    a = w['accession']
    gsf = is_gs(a)
    axR.scatter([w['harmonic_distance_A']], [w['angle_deg']], s=46,
                color=GS_C if gsf else OK_C, alpha=.85, zorder=3,
                marker='s' if gsf else 'o',
                edgecolor=INK if a in TARGETS else 'none', lw=1.3)
for a, dx, dy in (('P0DXD6', 8, -13), ('P78061', 8, 7), ('Q46E79', 8, -4),
                  ('A1AHS4', -8, -14), ('E2JA32', 8, -4)):
    w = next(x for x in W if x['accession'] == a)
    axR.annotate(name(a), (w['harmonic_distance_A'], w['angle_deg']),
                 textcoords='offset points', xytext=(dx, dy),
                 ha='right' if dx < 0 else 'left', fontsize=8.2,
                 color=GS_C if is_gs(a) else OK_C, fontweight='bold')
axR.set_xscale('log')
axR.set_xlabel('harmonic donor–phosphorus distance  $D_h$  (Å, log)')
axR.set_ylabel('proximity-weighted attack angle  A  (deg)')
axR.grid(color=MUTED, lw=.5, alpha=.45, zorder=0)
axR.set_axisbelow(True)
h = [plt.Line2D([], [], ls='', marker='o', ms=7, color=OK_C,
                label='interpretable monomer model'),
     plt.Line2D([], [], ls='', marker='s', ms=7, color=GS_C,
                label='GS fold — active site at subunit interface,\nmodelled as monomer (not interpretable)'),
     plt.Line2D([], [], ls='', marker='o', ms=8, mfc='none', mec=INK, mew=1.4,
                label='saturation-campaign target')]
axR.legend(handles=h, loc='lower left', fontsize=8, frameon=False)

fig.text(.006, .985, 'Ranking the 36 discovery candidates by the proximity-weighted angle',
         ha='left', va='top', fontsize=12, color=INK, fontweight='bold')
fig.text(.006, .945, 'A is scale-invariant, so a high angle at long range is not a poised complex — '
         '8 of the top 10 by A are GS-fold monomer models.',
         ha='left', va='top', fontsize=8.8, color=INK2)
fig.tight_layout(rect=[0, 0, 1, .915])
fig.savefig(OUT / 'discovery_geometry_ranking.png', dpi=200)
print('wrote', OUT / 'discovery_geometry_ranking.png')
