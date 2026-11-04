"""Annotate an unchanged PuuA pose with independently calibrated projections."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
import math
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent/'enzyme_visualization'))
from style import load_style, adjacent_label
STYLE = load_style()
M = json.loads((OUT/'geometry.json').read_text())
ROT = np.array(M['view'][:9]).reshape(3, 3)
arr = np.asarray(Image.open(OUT/'projection_fiducials.png').convert('RGB')).astype(float)
centers = []
for i in range(3):
    mask = (arr[:, :, i] > np.delete(arr, i, axis=2).max(2)+60) & (arr[:, :, i] > 100)
    ys, xs = np.where(mask)
    assert len(xs) > 50
    centers.append([xs.mean(), ys.mean()])
fp = np.array(M['fiducials']) @ ROT
AFFINE = np.linalg.solve(np.column_stack([fp[:, :2], np.ones(3)]), centers)


def project(xyz):
    xyz = np.atleast_2d(xyz) @ ROT
    return np.column_stack([xyz[:, :2], np.ones(len(xyz))]) @ AFFINE


P = {k: project(v)[0] for k, v in M['points'].items()}
ARC = project(M['angle_arc'])
INK, MUTED = '#263746', '#5E6C78'
TEAL, PURPLE = STYLE['distance_color'], STYLE['angle_color']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':16,
                     'svg.fonttype':'none', 'pdf.fonttype':42,
                     'mathtext.fontset':'dejavusans'})
fig = plt.figure(figsize=(11, 9), facecolor='white')
ax = fig.add_axes([.01, .01, .98, .98])
ax.imshow(Image.open(OUT/'PuuA_site_render.png'))
all_xy = project([a['xyz'] for atoms in M['atoms'].values() for a in atoms])
left, top = all_xy.min(0)-[185, 155]
right, bottom = all_xy.max(0)+[185, 220]
# Preserve a useful aspect ratio while retaining all ligand and metal atoms.
cx, cy = (left+right)/2, (top+bottom)/2
width = max(right-left, (bottom-top)*11/9)
height = width*9/11
ax.set_xlim(cx-width/2, cx+width/2)
ax.set_ylim(cy+height/2, cy-height/2)
ax.axis('off')
pair = np.array([P['donor'], P['P_gamma']])
ax.plot(pair[:, 0], pair[:, 1], color='white', lw=5.2, zorder=7)
ax.plot(pair[:, 0], pair[:, 1], color=TEAL, lw=2.5, dashes=(4,3), zorder=8)
ax.plot(ARC[:, 0], ARC[:, 1], color=PURPLE, lw=3.1,
        path_effects=[pe.Stroke(linewidth=5.8, foreground='white'),pe.Normal()],zorder=9)


def tag(value, xy, color=INK, size=20, align='left', bold=False):
    return adjacent_label(ax, value, xy, color=color, size=size, align=align, bold=bold)


ad = project([a['xyz'] for a in M['atoms']['adipate']])
atp_ring = project([a['xyz'] for a in M['atoms']['ATP'] if a['name'] in ['N1','N3','N6','N7','N9']])
tag('Adipate', [ad[:,0].mean(), ad[:,1].min()-65], size=20, bold=True, align='center')
tag('ATP', atp_ring.mean(0)+[110,-20], size=20, bold=True)
mg = project([a['xyz'] for a in M['atoms']['Mg']])
for i, point in enumerate(mg[np.argsort(mg[:, 0])]):
    tag(r'Mg$^{2+}$', point+([-60,65] if i == 0 else [0,105]),
        size=16, align='right' if i == 0 else 'center')
tag(r'$O_{\mathrm{nuc}}$', P['donor']+[55,-45], TEAL, 21, 'left')
tag(r'$d$', pair.mean(0)+[-45,16], TEAL, 23, 'center')
tag(r'$P_\gamma$', P['P_gamma']+[0,-100], INK, 22, 'center')
tag(r'$\theta$', ARC[len(ARC)//2]+[55,-35], PURPLE, 23, 'center')
tag(r'$O_{\mathrm{lg}}$ (O3B)', P['O3B']+[30,75], INK, 17, 'center')
ax.text(.025,.045,'PuuA (P78061) • one predicted activation-step pose',
        transform=ax.transAxes, fontsize=12, color=MUTED,
        bbox=dict(boxstyle='round,pad=.19',facecolor='white',edgecolor='none',alpha=.94),zorder=30)
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'PuuA_ADP_attack_geometry.{ext}',dpi=240,facecolor='white')
svg_path = OUT/'PuuA_ADP_attack_geometry.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
fig.savefig(OUT/'PuuA_ADP_attack_geometry_preview.png',dpi=110,facecolor='white')
plt.close(fig)
u=P['donor']-P['P_gamma'];v=P['O3B']-P['P_gamma']
angle=math.degrees(math.acos(float(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))
error=float(np.max(np.linalg.norm(project(M['fiducials'])-np.array(centers),axis=1)))
assert abs(angle-M['angle_deg'])<.05
assert error<.1
report={'source_coordinates_changed':False,'source':M['source'],
        'source_sha256':M['source_sha256'],'selection':'Saved model 0; not selected for optimized geometry.',
        'electrophile':'chain B / PG','leaving_oxygen':'chain B / O3B',
        'donor':'chain E / '+M['donor_atom'],'metals':len(M['atoms']['Mg']),
        'distance_A':M['distance_A'],'angle_3d_deg':M['angle_deg'],'angle_visible_2d_deg':angle,
        'projection_fiducial_error_pixels':error,'scope':M['scope']}
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
