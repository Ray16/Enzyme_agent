"""Render unchanged M0/M1 mmCIFs, distinguishing anchors from residue context."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--panels-only', action='store_true',
                        help='Compose existing PNGs in an environment with Pillow')
    args = parser.parse_args()
    root = args.campaign.resolve()
    out = root/'figures'
    out.mkdir(exist_ok=True)
    if args.panels_only:
        compose_panels(root)
        return
    import pymol2
    records = []
    with pymol2.PyMOL() as pm:
        c = pm.cmd
        for pose in (15,6):
            shared_view = None
            for arm in ('M1','M0'):
                iid = f'{arm}_pose{pose}'
                source = root/'inputs'/(iid+'_view.cif')
                meta = json.loads((root/'inputs'/(iid+'.metadata.json')).read_text())
                c.reinitialize()
                c.set('max_threads',1)
                c.load(str(source),'motif')
                original = c.get_coords('motif').copy()
                fixed = ' or '.join(f'(chain A and resi {r} and name '+ '+'.join(d['fixed_atoms'])+')'
                    for r,d in meta['motif_fixed_atoms'].items())
                c.select('protein_anchors',fixed)
                assert len(c.get_model('protein_anchors').atom)==meta['fixed_protein_atom_count']
                c.hide('everything')
                c.bg_color('white')
                c.set('orthoscopic',1)
                c.set('ray_shadows',0)
                c.set('specular',0.1)
                c.set('antialias',2)
                c.set('stick_radius',0.13)
                c.set('sphere_scale',0.23)
                c.show('sticks','motif')
                c.color('gray70','chain A')
                c.color('cyan','chain B')
                c.color('orange','chain D')
                for element,color in [('O','red'),('N','blue'),('P','yelloworange')]:
                    c.color(color,'elem '+element)
                c.set('stick_transparency',.65,'chain A and not protein_anchors')
                c.show('spheres','protein_anchors or chain C')
                c.color('green','chain C')
                c.set('sphere_scale',.5,'chain C')
                if arm=='M1':
                    c.hide('sticks','chain A and resi 310+311 and not name N+CA+C+O')
                    c.color('violetpurple','chain A and resi 310+311 and elem C')
                c.set('label_size',13)
                c.set('label_color','black')
                c.set('label_outline_color','white')
                c.set('float_labels',1)
                c.set('label_position',(1,1,1))
                for rid in (173,301,392,498):
                    c.label(f'chain A and resi {rid} and name CA','resn+resi')
                c.distance('activation_distance','chain D and name O14','chain B and name PA')
                c.hide('labels','activation_distance')
                c.set('dash_color','black','activation_distance')
                c.set('dash_width',2)
                if arm=='M1':
                    distal = 'O11' if pose==15 else 'O13'
                    c.distance('distal_contact','chain A and resi 311 and name N',f'chain D and name {distal}')
                    c.set('dash_color','violetpurple','distal_contact')
                    c.hide('labels','distal_contact')
                    c.label('chain A and resi 311 and name CA','"310-311 backbone"')
                if shared_view is None:
                    # The same camera is applied to M0/M1 within a source pose.
                    center = c.get_coords('chain B and name PA')[0]
                    attack = c.get_coords('chain D and name O14')[0]-center
                    attack /= np.linalg.norm(attack)
                    leave = c.get_coords('chain B and name O3A')[0]-center
                    leave -= np.dot(leave,attack)*attack
                    leave /= np.linalg.norm(leave)
                    phi = np.radians(135)
                    right = np.cos(phi)*attack+np.sin(phi)*leave
                    up = np.sin(phi)*attack-np.cos(phi)*leave
                    view = list(c.get_view())
                    view[:9] = np.column_stack([right,up,np.cross(right,up)]).ravel().tolist()
                    c.set_view(view)
                    c.zoom('motif',buffer=2.5,complete=1)
                    shared_view = c.get_view()
                else:
                    c.set_view(shared_view)
                c.deselect()
                c.png(str(out/(iid+'.png')),width=1500,height=1250,ray=1)
                c.save(str(out/(iid+'.pse')))
                assert np.array_equal(original,c.get_coords('motif'))
                records.append(dict(id=iid,input=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    coordinates_unchanged=True,view=list(c.get_view())))
    (out/'render_provenance.json').write_text(json.dumps(records,indent=2)+'\n')


def compose_panels(root):
    # Layout only: molecular pixels come from PyMOL, with no geometry editing.
    out = root/'figures'
    from PIL import Image, ImageDraw, ImageFont
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    for pose in (15,6):
        canvas = Image.new('RGB',(1600,850),'white')
        draw = ImageDraw.Draw(canvas)
        title = ImageFont.truetype(font,29)
        small = ImageFont.truetype(font,20)
        for x,arm in [(0,'M0'),(800,'M1')]:
            pic = Image.open(out/f'{arm}_pose{pose}.png').convert('RGB')
            pic.thumbnail((800,670))
            canvas.paste(pic,(x+(800-pic.width)//2,70))
            draw.text((x+25,20),f'{arm}: '+('core control' if arm=='M0' else 'core + distal backbone contact'),font=title,fill='#20252b')
        draw.text((25,750),'ATP carbon: cyan  |  Adipate carbon: orange  |  Mg: green  |  Added backbone carbon: purple',font=small,fill='#333333')
        draw.text((25,782),'Spheres mark protein coordinate anchors. Residues 310/311 have free identities. Source pose '+str(pose)+'.',font=small,fill='#333333')
        draw.text((25,814),'Unoptimized input coordinates; dashed lines mark measured proximity, not demonstrated catalysis.',font=small,fill='#333333')
        canvas.save(out/f'pose{pose}_comparison.png')


if __name__ == '__main__':
    main()
