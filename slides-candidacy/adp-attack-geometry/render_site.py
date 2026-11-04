"""Coordinate-based ADP-forming PuuA render; the supplied image is a style reference.

Run with /homes/rzhu/miniforge3/envs/pymol/bin/python render_site.py.
Only the camera and display change. Molecular coordinates remain unchanged.
"""
from pathlib import Path
import os
import sys
for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import math
import numpy as np
import pymol

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1] / 'structural_modeling_agent/campaigns/amide_activation/modeling/results/output_configs/step1_P78061/boltz_results_step1_P78061/predictions/step1_P78061/step1_P78061_model_0.pdb'
sys.path.insert(0, str(OUT.parent/'enzyme_visualization'))
from style import apply_pymol_style, load_style
STYLE = load_style()
W, H = 2200, 1800
pymol.finish_launching(['pymol', '-cq'])
cmd = pymol.cmd
cmd.set('max_threads', 2)
cmd.load(str(SOURCE), 'PuuA')
cmd.select('near_donor', '(chain A and not name N+CA+C+O) within 3.8 of chain E')
cmd.select('near_gamma', '(chain A and not name N+CA+C+O) within 3.6 of (chain B and name PG+O1G+O2G+O3B)')
cmd.select('site', '(byres (near_donor or near_gamma)) and not name N+C+O')
apply_pymol_style(cmd, protein='chain A', ligands='chain B or chain E',
                  magnesium='chain C or chain D', sidechains='site', style=STYLE)

p = cmd.get_coords('chain B and name PG')[0].astype(float)
leaving = cmd.get_coords('chain B and name O3B')[0].astype(float)
donors = cmd.get_model('chain E and elem O').atom
donor = min(donors, key=lambda a: np.linalg.norm(np.array(a.coord)-p))
o = np.array(donor.coord)
u, v = o-p, leaving-p
d = float(np.linalg.norm(u))
theta = math.acos(float(np.clip(u@v / (np.linalg.norm(u)*np.linalg.norm(v)), -1, 1)))
# Face the actual O--P--O plane so the displayed angle equals the 3D angle.
ex = u / np.linalg.norm(u)
ey = v - (v@ex)*ex
ey /= np.linalg.norm(ey)
phi = math.radians(135)
right = math.cos(phi)*ex + math.sin(phi)*ey
up = math.sin(phi)*ex - math.cos(phi)*ey
normal = np.cross(right, up)
rotation = np.column_stack([right, up, normal])
view = list(cmd.get_view())
view[:9] = rotation.ravel().tolist()
cmd.set_view(view)
cmd.viewport(W, H)
cmd.zoom('chain B or chain E', buffer=3.8, complete=1)
cmd.deselect()
view = list(cmd.get_view())
cmd.png(str(OUT / 'PuuA_site_render.png'), width=W, height=H, dpi=300, ray=1)
cmd.save(str(OUT / 'PuuA_ADP_attack_geometry.pse'))

# Independent rendered fiducials calibrate exact pixel positions, including
# PyMOL matrix/storage conventions and orthographic zoom.
focus = (p+o+leaving)/3
points = [focus, focus+4*right, focus+4*up]
cmd.disable('PuuA')
for i, (point, color) in enumerate(zip(points, ['red', 'green', 'blue'])):
    cmd.pseudoatom(f'fiducial_{i}', pos=point.tolist(), elem='C')
    cmd.show('spheres', f'fiducial_{i}')
    cmd.color(color, f'fiducial_{i}')
    cmd.set('sphere_scale', .15, f'fiducial_{i}')
cmd.set_view(view)
cmd.png(str(OUT/'projection_fiducials.png'), width=W, height=H, ray=1)
arc = [p + 1.15*(math.cos(t)*ex + math.sin(t)*ey)
       for t in np.linspace(0, theta, 100)]
atoms = {}
for selection, key in [('chain B', 'ATP'), ('chain E', 'adipate'), ('chain C or chain D', 'Mg')]:
    atoms[key] = [{'name': a.name, 'element': a.symbol, 'xyz': list(a.coord)}
                  for a in cmd.get_model(f'PuuA and ({selection})').atom]
metadata = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                size=[W, H], view=view, fiducials=[x.tolist() for x in points],
                points={'donor': o.tolist(), 'P_gamma': p.tolist(), 'O3B': leaving.tolist()},
                donor_atom=donor.name, distance_A=d, angle_deg=math.degrees(theta),
                angle_arc=[a.tolist() for a in arc], atoms=atoms,
                scope='One unmodified Boltz-2 pose; d and theta are per-pose values, not the ensemble A.',
                camera='Face-on to O_nuc--P_gamma--O3B plane; coordinates unchanged.',
                rendering='CPU PyMOL; two threads; element colors; gray protein cartoon; 94% cartoon transparency; 70% protein side-chain transparency.')
(OUT/'geometry.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps({k: metadata[k] for k in ['donor_atom', 'distance_A', 'angle_deg', 'camera']}), flush=True)
cmd.quit()
