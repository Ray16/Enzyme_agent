"""Compare saved step-1 geometry; run from anywhere. No trajectory resampling."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
CAMP = ROOT / 'structural_modeling_agent/runs/glutarate_cadaverine_step1_multipose_md_20261117'
GROUPS = [('GlnA3', 'GlnA3_*', '#0072B2'), ('BacD', 'BacD_*', '#009E73'),
          ('AS9', None, '#D55E00'), ('PubC1', 'PubC1_*', '#CC79A7'),
          ('PubC-Q0HWI9', 'PubCorth_*', '#E69F00'), ('AsbA', 'AsbA_*', '#555555')]
records = []
for name, pattern, color in GROUPS:
    files = sorted(CAMP.glob(pattern + '/md*/geometry.jsonl')) if pattern else [ROOT / 'runs/as9_step1_attackgeom_md_20261111/WT_glutarate_md_s17/geometry.jsonl']
    for pose, path in enumerate(files, 1):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in rows if 0 < r['time_ps'] <= 500]
        assert len(rows) == 50, (path, len(rows))
        for r in rows:
            records.append(dict(candidate=name, pose=pose, source=str(path.relative_to(ROOT)), **r))
(OUT / 'frames.json').write_text(json.dumps(records, indent=2) + '\n')
plt.rcParams.update({'font.size': 18, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(20, 16), facecolor='white')
gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=.35, wspace=.23)
ax = fig.add_subplot(gs[0, :]); dx = fig.add_subplot(gs[1, 0]); ay = fig.add_subplot(gs[1, 1])
markers = ['o', '^', 's']; summary = []
for j, (name, pattern, color) in enumerate(GROUPS):
    rr = [r for r in records if r['candidate'] == name]
    for pose in sorted({r['pose'] for r in rr}):
        r = [r for r in rr if r['pose'] == pose]
        d = np.array([x['forming_A'] for x in r]); a = np.array([x['attack_angle_deg'] for x in r])
        ax.scatter(d, a, s=23, marker=markers[pose-1], color=color, alpha=.42, linewidths=0)
        for target, values in [(dx, d), (ay, a)]:
            pos = j + (pose - 2)*.19 if name != 'AS9' else j
            parts = target.violinplot([values], positions=[pos], widths=.17, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(color); body.set_alpha(.38)
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            target.plot([pos, pos], [q1, q3], color=color, lw=3)
            target.scatter(pos, med, marker=markers[pose-1], s=27, color=color, edgecolors='white', linewidths=.5, zorder=4)
        summary.append(dict(candidate=name, pose=pose, frames=len(r), harmonic_distance_A=float(len(d)/sum(1/d)), weighted_angle_deg=float(sum(a/d)/sum(1/d)), fraction_distance_lt4=float(np.mean(d<4)), fraction_distance_lt4_angle_gt150=float(np.mean((d<4)&(a>150)))))
ax.axvline(4, color='#999999', ls='--', lw=1); ax.axhline(150, color='#999999', ls='--', lw=1)
ax.set(xlabel='Attack distance (Å)', ylabel='Attack angle (°)', xlim=(2.9, 9.1), ylim=(65, 182))
ax.set_title('A   Joint distribution • one point per saved MD frame', loc='left', fontsize=18, weight='bold')
legend = [Line2D([], [], marker='o', color=c, linestyle='', label=n + (' (1 pose)' if n=='AS9' else ' (3 poses)')) for n,p,c in GROUPS]
ax.legend(handles=legend, ncol=3, loc='lower right', frameon=True, fontsize=18)
for target, title, ylabel, threshold in [(dx, 'B   Distance distributions by pose', 'Attack distance (Å)', 4), (ay, 'C   Angle distributions by pose', 'Attack angle (°)', 150)]:
    target.set_title(title, loc='left', fontsize=18, weight='bold')
    target.set_xticks(range(len(GROUPS)), [x[0] for x in GROUPS], rotation=20, ha='right')
    target.set_ylabel(ylabel); target.axhline(threshold, color='#999999', ls='--', lw=1)
    target.grid(axis='y', alpha=.12)
dx.set_ylim(2.9,9.1); ay.set_ylim(65,182)
fig.suptitle('Step-1 MD: GlnA3 retains close geometry across all three starting poses', fontsize=18, weight='bold', y=.98)
fig.text(.5,.938,'Glutarate + ATP • matched production window: 10–500 ps • 50 frames per trajectory',ha='center',fontsize=18)
fig.text(.08,.055,'Pose symbols:  ○ 1    △ 2    □ 3     |     Violin: smoothed distribution; bar: middle 50%; symbol: median.',fontsize=18)
fig.text(.08,.033,'Dashed lines: historical 4 Å / 150° cutoffs, not validated activity thresholds. Frames are time-correlated.',fontsize=18)
fig.text(.08,.011,'AS9: existing WT benchmark; preparation/mechanism differ across enzymes. SfaB MD pending. Geometry does not establish activity.',fontsize=18)
fig.subplots_adjust(top=.89,bottom=.15,left=.08,right=.98)
for ext in ['png','svg']:
    fig.savefig(OUT / ('step1_md_comparison.'+ext), dpi=300)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('Saved figure and data:', OUT)
