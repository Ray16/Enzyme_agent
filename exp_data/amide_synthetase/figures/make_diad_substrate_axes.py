"""Which substrate dimension limits diad yield: the diacid or the diamine?

Source: Chen et al., "Improved nylon polymerization using amide diads", Fig 2C,
digitized to individual N=3 replicates in
exp_data/amide_synthetase/literature_activity/fig2c_quantified/diads_fig2c_quantified_yield.csv
(independently audited, all bars within 1 pp).

Panel A plots the measured yields. Panel B converts both axes to a common unit --
fold-change per added methylene -- so the two dimensions are compared like for like:
the diacid series spans two CH2 (C4->C5->C6) and the diamine swap spans one
(cadaverine C5 -> HMD C6).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = (HERE.parent / 'literature_activity' / 'fig2c_quantified'
       / 'diads_fig2c_quantified_yield.csv')

# Validated categorical slots (light mode) - see dataviz references/palette.md.
# Checked with scripts/validate_palette.js: worst adjacent dE 24.7 (protan), all PASS.
ENZ = {'DdaG': '#2a78d6', 'SfaB': '#eb6834'}
INK, INK2, MUTED, SURFACE = '#0b0b0b', '#52514e', '#b8b7b0', '#fcfcfb'
# Diacids in chain-length order - this ordering IS the independent variable.
ACIDS = [('succinic', 'succinate\nC4'), ('glutaric', 'glutarate\nC5'),
         ('adipic', 'adipate\nC6')]
AMINES = [('cadaverine', 'cadaverine C5'), ('HMD', 'HMD C6')]
# No clipping: neither enzyme has a below-LOD replicate and the smallest value is
# 0.3 (DdaG / adipate / cadaverine), so every measured point is drawn as measured.
YMIN = 0.2

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load():
    """means[(enzyme, acid, amine)] = bar mean ; reps[...] = the N=3 dots."""
    means, reps = {}, {}
    for r in csv.DictReader(SRC.open()):
        if r['enzyme'] not in ENZ:
            continue
        k = (r['enzyme'], r['acid_name'], r['amine_name'])
        v = float(r['value'])
        if r['is_mean'] == 'True':
            means[k] = v
        else:
            reps.setdefault(k, []).append(v)
    return means, reps


def panel_a(ax, means, reps):
    x = np.arange(len(ACIDS))
    for enz, col in ENZ.items():
        for amine, _ in AMINES:
            solid = amine == 'HMD'
            y = [means[(enz, a, amine)] for a, _ in ACIDS]
            ax.plot(x, y, color=col, lw=2.0,
                    ls='-' if solid else (0, (4, 2.4)), zorder=4, marker='o',
                    ms=8 if solid else 7,
                    mfc=col if solid else SURFACE, mec=col, mew=1.8)
            for i, (a, _) in enumerate(ACIDS):
                d = reps.get((enz, a, amine), [])
                ax.scatter(np.full(len(d), i) + (0.055 if solid else -0.055), d,
                           s=9, color=col, alpha=.45, lw=0, zorder=3)
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in ACIDS])
    ax.set_xlim(-.30, len(ACIDS) - .55)
    ax.set_ylim(YMIN, 150)
    ax.set_ylabel('diad molar yield (%)')
    ax.set_xlabel('diacid')
    ax.grid(axis='y', color=MUTED, lw=.5, alpha=.45, zorder=0)
    ax.set_axisbelow(True)
    # Split encoding so four series need only two text labels: colour carries the
    # enzyme (direct-labelled once each, on its HMD line), line style carries the
    # diamine (its own neutral-ink legend). Labelling all four collides, because
    # the two same-colour lines sit within 1.3x of each other at every diacid.
    for enz, col in ENZ.items():
        ax.annotate(enz, (0, means[(enz, ACIDS[0][0], 'HMD')]),
                    textcoords='offset points', xytext=(12, 7), ha='left',
                    va='center', fontsize=10, color=col, fontweight='bold')
    style = [plt.Line2D([], [], color=INK2, lw=1.8, ls='-', marker='o', ms=7,
                        mfc=INK2, mec=INK2, label='HMD (C6)'),
             plt.Line2D([], [], color=INK2, lw=1.8, ls=(0, (4, 2.4)), marker='o',
                        ms=6, mfc=SURFACE, mec=INK2, label='cadaverine (C5)')]
    ax.legend(handles=style, frameon=False, loc='lower left', fontsize=8.5,
              handletextpad=.6, labelcolor=INK2)
    ax.set_title('Measured yields', fontsize=10, color=INK,
                 fontweight='bold', loc='left')


def panel_b(ax, means):
    """Fold loss per added methylene, so both substrate axes share one unit.

    The two diacid steps are kept separate: pooling them hides that C4->C5 is
    enzyme-dependent while C5->C6 -- the step that reaches adipate -- is the
    one that is uniformly costly.
    """
    rows = []
    for enz in ENZ:
        for amine, _ in AMINES:
            for (a1, l1), (a2, l2) in zip(ACIDS, ACIDS[1:]):
                lab = f'diacid  {l1.split()[-1]}→{l2.split()[-1]}'
                rows.append((lab, enz,
                             means[(enz, a1, amine)] / means[(enz, a2, amine)]))
        for a, _ in ACIDS:
            rows.append(('diamine  C5→C6', enz,
                         means[(enz, a, 'cadaverine')] / means[(enz, a, 'HMD')]))
    rng = np.random.default_rng(0)
    cats = ['diamine  C5→C6', 'diacid  C4→C5', 'diacid  C5→C6']
    for ci, cat in enumerate(cats):
        sub = [r for r in rows if r[0] == cat]
        for _, enz, v in sub:
            ax.scatter(v, ci + rng.uniform(-.15, .15), s=52, color=ENZ[enz],
                       alpha=.85, lw=0, zorder=4)
        vals = [v for _, _, v in sub]
        ax.plot([np.median(vals)] * 2, [ci - .28, ci + .28], color=INK,
                lw=1.8, zorder=5)
        ax.annotate(f'{np.median(vals):.1f}×', (np.median(vals), ci + .32),
                    ha='center', fontsize=8.5, color=INK, fontweight='bold')
    ax.axvline(1.0, color=INK, lw=1.0, zorder=2)
    ax.set_xscale('log')
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_ylim(-.55, len(cats) - .25)
    ax.set_xlim(0.3, 40)
    ax.set_xlabel('fold loss in yield per added methylene  (right of the line = worse)')
    ax.grid(axis='x', color=MUTED, lw=.5, alpha=.45, zorder=0)
    ax.set_axisbelow(True)
    for enz, col in ENZ.items():
        ax.scatter([], [], s=52, color=col, lw=0, label=enz)
    ax.legend(frameon=False, loc='lower right', fontsize=8.5,
              handletextpad=.3, labelcolor=INK2)
    ax.set_title('Cost of one extra methylene, by substrate',
                 fontsize=10, color=INK, fontweight='bold', loc='left')


def main():
    means, reps = load()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5),
                             gridspec_kw={'width_ratios': [1, 1.12]})
    panel_a(axes[0], means, reps)
    panel_b(axes[1], means)
    fig.text(.006, .975,
             'The diacid, not the diamine, limits nylon-6,6 diad formation',
             ha='left', va='top', fontsize=12.5, color=INK, fontweight='bold')
    fig.text(.006, .934,
             'Both characterised synthetases, all six diad combinations, N=3 replicates (dots). '
             'Reaching adipate costs 6-18x; swapping the diamine costs nothing. '
             'Digitized from Chen et al. Fig 2C.',
             ha='left', va='top', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0.012, 0, 1, .885])
    out = HERE / 'diad_substrate_axes.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f'wrote {out}')
    print('\nmeans used:')
    for enz in ENZ:
        for amine, _ in AMINES:
            s = '  '.join(f'{a[:4]}={means[(enz,a,amine)]:5.1f}' for a, _ in ACIDS)
            print(f'  {enz:5s} {amine:11s} {s}')


if __name__ == '__main__':
    main()
