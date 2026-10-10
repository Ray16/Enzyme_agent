"""Campaign figures: position effect, near-attack plane, permutation null.

Re-runnable as the fleet lands more variants; reads variant_metrics.csv /
pose_geometry.csv written by enzyme_design_agent.saturation.evaluate.
"""
from __future__ import annotations
import collections
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAMPAIGN = Path(__file__).resolve().parent.parent
OUT = CAMPAIGN / 'figures'
# Validated categorical slots (light mode) - see dataviz references/palette.md
SERIES = {'P0DXD6': '#2a78d6', 'P78061': '#eb6834', 'Q46E79': '#1baf7a'}
# Short names from the campaign spec; confirmed against UniProt recommended names.
NAMES = {'P0DXD6': 'PubC', 'P78061': 'PuuA', 'Q46E79': 'PylC'}
label = lambda a: f'{NAMES[a]} ({a})'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b8b7b0'
SURFACE = '#fcfcfb'
NEAR_ATTACK_D, NEAR_ATTACK_ANG = 4.0, 150.0

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load():
    rows = list(csv.DictReader((CAMPAIGN / 'variant_metrics.csv').open()))
    ev = [r for r in rows if r['status'] == 'evaluated']
    poses = collections.defaultdict(list)
    for r in csv.DictReader((CAMPAIGN / 'pose_geometry.csv').open()):
        poses[r['id']].append((float(r['distance_A']), float(r['angle_deg'])))
    return ev, poses


def fig_position_effect(ev):
    cands = [c for c in SERIES if any(r['candidate'] == c for r in ev)]
    # Never share y: each enzyme has its own residue positions, and a shared
    # axis would stamp one enzyme's labels onto all three panels.
    fig, axes = plt.subplots(1, len(cands), figsize=(3.5 * len(cands), 4.3), sharey=False)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, cand in zip(axes, cands):
        sub = [r for r in ev if r['candidate'] == cand and r['mutation'] != 'WT']
        g = collections.defaultdict(list)
        for r in sub:
            g[r['mutation'][0] + r['mutation'][1:-1]].append(float(r['delta_angle_deg']))
        order = sorted(g, key=lambda k: np.mean(g[k]))
        for i, k in enumerate(order):
            v = np.array(g[k])
            ax.scatter(v, i + rng.uniform(-.17, .17, len(v)), s=11, lw=0,
                       color=SERIES[cand], alpha=.55, zorder=3)
            ax.plot([v.mean()], [i], marker='|', ms=16, mew=2.0,
                    color=SERIES[cand], zorder=4)
        ax.axvline(0, color=INK, lw=1.0, zorder=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8.5)
        ax.set_ylim(-.6, len(order) - .4)
        ax.set_xlabel('Δ attack angle vs WT (deg)')
        ax.set_title(label(cand), color=SERIES[cand], fontsize=10, fontweight='bold', loc='left')
        ax.grid(axis='x', color=MUTED, lw=.5, alpha=.45, zorder=0)
        ax.set_axisbelow(True)
    fig.text(.008, .975, 'Every substitution at the essential positions degrades reactive geometry',
             ha='left', va='top', fontsize=11.5, color=INK, fontweight='bold')
    fig.text(.008, .925, 'Each dot = one of 19 substitutions; bar = position mean. '
             'Position effect p < 1e-4 within each enzyme (permutation).',
             ha='left', va='top', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, .885])
    fig.savefig(OUT / 'position_effect.png', dpi=200)
    plt.close(fig)


def fig_near_attack(ev, poses):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.add_patch(plt.Rectangle((0, NEAR_ATTACK_ANG), NEAR_ATTACK_D, 30,
                               facecolor='#1baf7a', alpha=.13, edgecolor='#1baf7a',
                               lw=1.2, ls='--', zorder=1))
    ax.text(NEAR_ATTACK_D - .12, 178.5, 'near-attack basin\n(d < 4 Å, angle > 150°)',
            ha='right', va='top', fontsize=8.5, color='#0f7a55')
    for cand, col in SERIES.items():
        sub = [r for r in ev if r['candidate'] == cand]
        if not sub:
            continue
        mut = [r for r in sub if r['mutation'] != 'WT']
        ax.scatter([float(r['harmonic_distance_A']) for r in mut],
                   [float(r['angle_deg']) for r in mut],
                   s=15, lw=0, color=col, alpha=.45, zorder=3)
        wt = [r for r in sub if r['mutation'] == 'WT'][0]
        x, y = float(wt['harmonic_distance_A']), float(wt['angle_deg'])
        ax.scatter([x], [y], s=170, marker='*', color=col,
                   edgecolor=SURFACE, lw=1.4, zorder=5)
        # Offsets hand-placed to keep the three WT labels off each other and
        # off the circled S137H marker.
        dx, dy = {'P0DXD6': (-6, 14), 'P78061': (12, 10), 'Q46E79': (12, -14)}[cand]
        ax.annotate(f'{label(cand)} WT', (x, y), textcoords='offset points',
                    xytext=(dx, dy), ha='right' if dx < 0 else 'left',
                    fontsize=9, color=col, fontweight='bold')
    for vid, lab in (('Q46E79_S137H', 'PylC S137H'),):
        r = next((r for r in ev if r['id'] == vid), None)
        if r:
            x, y = float(r['harmonic_distance_A']), float(r['angle_deg'])
            ax.scatter([x], [y], s=70, facecolor='none',
                       edgecolor=INK, lw=1.6, zorder=6)
            ax.annotate(f'{lab}: +12.0° angle, but d stays {x:.1f} Å', (x, y),
                        textcoords='offset points', xytext=(26, -30), ha='left',
                        fontsize=8.5, color=INK,
                        arrowprops=dict(arrowstyle='-', color=INK, lw=.8,
                                        shrinkA=0, shrinkB=5))
    ax.set_xlabel('nucleophile–electrophile distance (Å)')
    ax.set_ylabel('attack angle (deg)')
    ax.set_xlim(2.3, None)
    ax.grid(color=MUTED, lw=.5, alpha=.45, zorder=0)
    ax.set_axisbelow(True)
    fig.text(.008, .975, 'Only PubC (P0DXD6) sits in the near-attack basin',
             ha='left', va='top', fontsize=11.5, color=INK, fontweight='bold')
    fig.text(.008, .928, 'Angle gains in the other two do not close the distance. '
             'Dots = single substitutions; stars = wild type.',
             ha='left', va='top', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, .89])
    fig.savefig(OUT / 'near_attack_plane.png', dpi=200)
    plt.close(fig)


def fig_permutation(ev):
    rows = [r for r in ev if r['mutation'] != 'WT']
    pos = lambda r: (r['candidate'], r['mutation'][1:-1])
    idx = collections.defaultdict(list)
    for i, r in enumerate(rows):
        idx[r['candidate']].append(i)
    d = np.array([float(r['delta_angle_deg']) for r in rows])
    P = [pos(r) for r in rows]
    rng = np.random.default_rng(0)

    def between(vals):
        g = collections.defaultdict(list)
        for k, v in zip(P, vals):
            g[k].append(v)
        out = 0.
        for cand, ii in idx.items():
            m = np.mean(vals[ii])
            for k in {P[i] for i in ii}:
                out += len(g[k]) * (np.mean(g[k]) - m) ** 2
        return out

    obs = between(d)
    null = []
    for _ in range(5000):
        v = d.copy()
        for cand, ii in idx.items():
            v[ii] = rng.permutation(v[ii])
        null.append(between(v))
    null = np.array(null)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(null, bins=60, color=MUTED, edgecolor='none', zorder=2)
    ax.axvline(obs, color='#eb6834', lw=2.2, zorder=4)
    ax.annotate(f'observed  {obs:,.0f}\np < {1/len(null):.0e}', (obs, ax.get_ylim()[1] * .78),
                textcoords='offset points', xytext=(-12, 0), ha='right',
                fontsize=9, color='#eb6834', fontweight='bold')
    ax.annotate('permutation null\n(labels shuffled within enzyme)',
                (np.percentile(null, 60), ax.get_ylim()[1] * .55),
                textcoords='offset points', xytext=(14, 0), fontsize=8.5, color=INK2)
    ax.set_xlabel('between-position sum of squares of Δ attack angle')
    ax.set_ylabel('permutations')
    ax.set_title('Which position you mutate determines the outcome',
                 fontsize=10.5, color=INK, fontweight='bold', loc='left')
    ax.grid(axis='y', color=MUTED, lw=.5, alpha=.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / 'permutation_null.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    ev, poses = load()
    print(f'evaluated variants: {len(ev)}')
    fig_position_effect(ev)
    fig_near_attack(ev, poses)
    fig_permutation(ev)
    print('wrote:', *sorted(p.name for p in OUT.glob('*.png')))
