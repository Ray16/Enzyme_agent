"""Render and measure unchanged AS9 motif inputs alongside their source pocket.

Run in the PyMOL environment. Contacts are distances, not assigned hydrogen bonds.
The supplied full source must match every extracted motif/ligand coordinate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pymol2


def key(a):
    return (a.chain, a.resi, a.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--source-inputs', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    with pymol2.PyMOL() as pm:
        c = pm.cmd
        for pose in ('pose15', 'pose6'):
            c.reinitialize()
            c.set('max_threads', 1)
            motif = args.campaign / 'inputs' / f'{pose}_view.cif'
            source = args.source_inputs / f'{pose}.cif'
            meta = json.loads((args.campaign / 'inputs' / f'{pose}.metadata.json').read_text())
            c.load(str(motif), 'motif')
            c.load(str(source), 'native_context')
            ma = c.get_model('motif').atom
            na = c.get_model('native_context').atom
            native = {key(a): a for a in na}
            deviations = [np.linalg.norm(np.array(a.coord)-native[key(a)].coord) for a in ma]
            if max(deviations) > 0.002:
                raise ValueError('Full source does not match extracted coordinates')
            original = c.get_coords('motif').copy()
            retained = set(meta['motif_fixed_atoms'])
            fixed = {(str(r), n) for r, d in meta['motif_fixed_atoms'].items() for n in d['fixed_atoms']}
            c.select('conditioned_atoms', 'motif and (' + ' or '.join(
                f'(chain A and resi {r} and name {n})' for r, n in sorted(fixed)) + ')')
            protein = [a for a in na if a.chain == 'A' and a.symbol != 'H']
            acid = [a for a in ma if a.chain == 'D']
            phosphorus = next(a for a in ma if a.chain == 'B' and a.name == 'PA')
            leaving = next(a for a in ma if a.chain == 'B' and a.name == 'O3A')
            oxygen = min((a for a in acid if a.symbol == 'O'), key=lambda a: np.linalg.norm(np.array(a.coord)-phosphorus.coord))
            u = np.array(oxygen.coord)-phosphorus.coord
            v = np.array(leaving.coord)-phosphorus.coord
            angle = float(np.degrees(np.arccos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
            contacts = []
            for a in acid:
                for b in protein:
                    distance = float(np.linalg.norm(np.array(a.coord)-b.coord))
                    if distance <= 4:
                        contacts.append(dict(substrate_atom=a.name, protein_residue=f'{b.resn}{b.resi}',
                            protein_atom=b.name, distance_A=distance, retained_residue=b.resi in retained,
                            conditioned_protein_atom=(b.resi,b.name) in fixed))
            contacts.sort(key=lambda r: r['distance_A'])
            mg = next(a for a in ma if a.chain == 'C')
            metal = []
            for a in na:
                if a.symbol in ('O','N','S'):
                    distance = float(np.linalg.norm(np.array(a.coord)-mg.coord))
                    if distance <= 3:
                        metal.append(dict(atom=':'.join(key(a)), residue=a.resn, distance_A=distance,
                            present_in_extracted_motif=key(a) in {key(x) for x in ma}))
            metal.sort(key=lambda r:r['distance_A'])
            for obj in ('motif', 'native_context'):
                c.hide('everything', obj)
            c.bg_color('white')
            c.set('orthoscopic', 1)
            c.set('ray_shadows', 0)
            c.set('specular', 0.15)
            c.set('antialias', 2)
            c.set('stick_radius', 0.13)
            c.set('sphere_scale', 0.23)
            c.set('label_size', 12)
            c.set('label_color', 'black')
            c.set('label_outline_color', 'white')
            c.set('float_labels', 1)
            c.set('label_position', (1,1,1))
            c.set('dash_color', 'black')
            c.set('dash_width', 2)
            c.set('dash_gap', 0.25)
            c.show('sticks', 'motif')
            c.color('gray70', 'motif and chain A')
            c.color('cyan', 'motif and chain B')
            c.color('orange', 'motif and chain D')
            for el, color in [('O','red'),('N','blue'),('P','yelloworange')]:
                c.color(color, f'motif and elem {el}')
            c.show('spheres', 'conditioned_atoms or (motif and chain C)')
            c.color('green', 'motif and chain C')
            c.set('sphere_scale', 0.5, 'motif and chain C')
            c.select('omitted_pocket', 'byres ((native_context and chain A and not resi '+
                '+'.join(sorted(retained)) + ') within 5 of (motif and chain D))')
            c.color('violetpurple', 'omitted_pocket')
            for el, color in [('O','red'),('N','blue')]:
                c.color(color, f'omitted_pocket and elem {el}')
            c.distance('attack_distance', f'motif and chain D and name {oxygen.name}', 'motif and chain B and name PA')
            c.distance('leaving_axis', 'motif and chain B and name PA', 'motif and chain B and name O3A')
            c.hide('labels', 'leaving_axis')
            for rid in ['173','301','392','498']:
                c.label(f'motif and chain A and resi {rid} and name CA', 'resn+resi')
            # Reuse the exact shared overview camera for visual comparison.
            provenance = json.loads((args.campaign/'local_analysis/figures/theozyme_render_provenance.json').read_text())
            c.set_view(next(r['view'] for r in provenance['inputs'] if r['pose']==pose))
            c.zoom('motif', buffer=2.5, complete=1)
            c.scene('01_extracted_motif', 'store')
            c.png(str(args.outdir/f'{pose}_motif.png'), width=1500, height=1250, ray=1)
            c.hide('labels', 'motif and chain A')
            c.hide('spheres', 'conditioned_atoms')
            c.set('stick_transparency', 0.55, 'motif and chain A')
            c.show('sticks', 'omitted_pocket')
            c.label('omitted_pocket and resi 311+279+210+304 and name CA', 'resn+resi')
            c.label('motif and chain A and resi 498 and name NZ', '"LYS498"')
            if c.count_atoms('native_context and chain A and resi 311 and name N'):
                s311 = next(a for a in protein if a.resi=='311' and a.name=='N')
                near = min((a for a in acid if a.symbol=='O'),key=lambda a:np.linalg.norm(np.array(a.coord)-s311.coord))
                c.distance('S311_backbone_contact', 'native_context and chain A and resi 311 and name N', f'motif and chain D and name {near.name}')
                c.set('dash_color','purple','S311_backbone_contact')
            c.zoom('motif or omitted_pocket', buffer=2, complete=1)
            c.scene('02_omitted_native_pocket', 'store')
            c.png(str(args.outdir/f'{pose}_native_pocket.png'),width=1500,height=1250,ray=1)
            c.scene('01_extracted_motif', 'recall')
            c.save(str(args.outdir/f'{pose}_inspection.pse'))
            if not np.array_equal(original,c.get_coords('motif')):
                raise ValueError('Rendering changed motif coordinates')
            records.append(dict(pose=pose, motif=str(motif.resolve()),source=str(source.resolve()),
                motif_sha256=hashlib.sha256(motif.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_match_max_deviation_A=float(max(deviations)),conditioned_atoms=len(fixed),
                attack_oxygen=oxygen.name,attack_distance_A=float(np.linalg.norm(u)),attack_angle_deg=angle,
                adipate_contacts_within_4A=contacts,metal_contacts_within_3A=metal,
                coordinates_unchanged=True))
            print(pose, 'rendered and measured', flush=True)
    (args.outdir/'inspection.json').write_text(json.dumps(dict(
        scope='Unmodified starting inputs and matching source pocket. Distance contacts are not hydrogen-bond or catalytic assignments.',
        scenes=['01_extracted_motif','02_omitted_native_pocket'],records=records),indent=2)+'\n')


if __name__ == '__main__':
    main()
