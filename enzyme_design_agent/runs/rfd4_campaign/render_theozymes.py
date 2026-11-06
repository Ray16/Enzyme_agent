"""CPU PyMOL renders from a campaign's viewing CIFs and motif metadata.

Only display settings and camera are changed. No input file is modified.
Run in an environment with PyMOL and NumPy installed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pymol2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--poses',nargs='+',help='Input IDs; default: all *_view.cif inputs')
    parser.add_argument('--outdir',type=Path)
    parser.add_argument('--protein-chain',default='A')
    parser.add_argument('--atp-chain',default='B')
    parser.add_argument('--substrate-chain',default='D')
    parser.add_argument('--metal-chain',default='C')
    parser.add_argument('--style',type=Path,help='Override the shared slides-candidacy style JSON')
    parser.add_argument('--style-dir',type=Path,
                        default=Path(__file__).resolve().parents[3]/'slides-candidacy/enzyme_visualization',
                        help='Directory containing the shared style.py and style.json')
    parser.add_argument('--camera',choices=['attack-plane','adenine-plane'],default='attack-plane')
    parser.add_argument('--center-atom',default='PA',help='ATP atom used for the attack-plane camera')
    parser.add_argument('--leaving-atom',default='O3A',help='ATP leaving atom used for the attack-plane camera')
    parser.add_argument('--label-residues',nargs='+',type=int,
                        help='Default: endpoints of each contiguous motif segment')
    parser.add_argument('--width',type=int,default=1500)
    parser.add_argument('--height',type=int,default=1300)
    args=parser.parse_args()
    root=args.campaign.resolve()
    out=args.outdir or root/'local_analysis/figures'
    out.mkdir(exist_ok=True,parents=True)
    poses=args.poses or sorted(p.name.removesuffix('_view.cif') for p in (root/'inputs').glob('*_view.cif'))
    if not poses:
        parser.error('No *_view.cif inputs; specify a campaign with motif viewing copies')
    style_module=args.style_dir.resolve()/'style.py'
    style_path=(args.style or args.style_dir/'style.json').resolve()
    module_spec=importlib.util.spec_from_file_location('candidacy_molecular_style',style_module)
    shared_style=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(shared_style)
    style=shared_style.load_style(style_path)
    pymol=pymol2.PyMOL()
    pymol.start()
    cmd=pymol.cmd
    records=[]
    camera=None
    for pose in poses:
        cmd.reinitialize()
        cmd.set('max_threads',1)
        source=root/'inputs'/f'{pose}_view.cif'
        metadata=root/'inputs'/f'{pose}.metadata.json'
        meta=json.loads(metadata.read_text())
        definitions=meta['motif_fixed_atoms']
        cmd.load(str(source),'motif')
        original=cmd.get_coords('motif').copy()
        protein=f'chain {args.protein_chain}'
        atp=f'chain {args.atp_chain}'
        substrate=f'chain {args.substrate_chain}'
        metal=f'chain {args.metal_chain}'
        # get_model excludes absent-coordinate guide atoms synthesized by PyMOL
        # for nonstandard polymer numbering in an mmCIF sequence table.
        if len(cmd.get_model(f'{protein} and name CA').atom)!=len(definitions):
            raise ValueError('Viewing CIF protein residues disagree with motif metadata')
        for sel in [atp,substrate,metal]:
            if not cmd.get_model(sel).atom:
                raise ValueError('Empty component selection '+sel)
        fixed=' or '.join(f'({protein} and resi {rid} and name '+ '+'.join(row['fixed_atoms'])+')'
                           for rid,row in definitions.items())
        cmd.select('conditioned',fixed)
        fixed_count=len(cmd.get_model('conditioned').atom)
        if fixed_count!=sum(len(v['fixed_atoms']) for v in definitions.values()):
            raise ValueError('Viewing CIF lacks some conditioned motif atoms')
        shared_style.apply_pymol_style(cmd,protein=protein,ligands=f'{atp} or {substrate}',
                                      magnesium=metal,sidechains=protein,style=style,threads=1)
        # Retained-residue context uses the slide side-chain transparency.
        # Conditioned protein atoms are opaque ball-and-stick in the same palette.
        cmd.show('sticks','conditioned')
        cmd.show('spheres','conditioned')
        cmd.set('sphere_scale',style['ligand_sphere_scale'],'conditioned')
        cmd.set('stick_radius',style['ligand_stick_radius_A'],'conditioned')
        cmd.set('stick_transparency',0,'conditioned')
        for element in style['elements']:
            cmd.color('enzyme_'+element,f'conditioned and elem {element}')
        cmd.set('label_color','black')
        cmd.set('label_outline_color','white')
        cmd.set('label_font_id',7)
        cmd.set('label_shadow_mode',0)
        cmd.set('label_size',style['label_size'])
        cmd.set('label_position',(1.,1.,1.))
        residues=sorted(map(int,definitions))
        labels=args.label_residues or [r for r in residues if r-1 not in residues or r+1 not in residues]
        cmd.viewport(args.width,args.height)
        if camera is None:
            if args.camera=='attack-plane':
                center=cmd.get_coords(f'{atp} and name {args.center_atom}')
                leaving=cmd.get_coords(f'{atp} and name {args.leaving_atom}')
                oxygens=cmd.get_coords(f'{substrate} and elem O')
                if center is None or len(center)!=1 or leaving is None or len(leaving)!=1 or oxygens is None:
                    raise ValueError('Attack-plane camera requires unique center/leaving atoms and substrate oxygens')
                p=center[0];o=oxygens[np.argmin(np.linalg.norm(oxygens-p,axis=1))]
                ex=o-p;ex/=np.linalg.norm(ex)
                ey=leaving[0]-p;ey-=np.dot(ey,ex)*ex
                if np.linalg.norm(ey)<1e-7:
                    axis=np.eye(3)[np.argmin(np.abs(ex))];ey=axis-np.dot(axis,ex)*ex
                ey/=np.linalg.norm(ey)
                phi=np.radians(135)
                right=np.cos(phi)*ex+np.sin(phi)*ey
                up=np.sin(phi)*ex-np.cos(phi)*ey
                rotation=np.column_stack([right,up,np.cross(right,up)])
                view=list(cmd.get_view());view[:9]=rotation.ravel().tolist();cmd.set_view(view)
            else:
                ring=cmd.get_coords(f'{atp} and name N9+C8+N7+C5+C6+N1+C2+N3+C4')
                if ring is None or len(ring)<3:
                    raise ValueError('Adenine-plane camera requires ATP ring atoms')
                _,_,v=np.linalg.svd(ring-ring.mean(axis=0));z=v[-1]
                x=cmd.get_coords(substrate).mean(axis=0)-cmd.get_coords(atp).mean(axis=0)
                x-=z*np.dot(x,z);x/=np.linalg.norm(x);y=np.cross(z,x)
                view=list(cmd.get_view());view[:9]=np.array([x,y,z]).ravel().tolist();cmd.set_view(view)
            cmd.zoom('motif',buffer=1.8,complete=1)
            camera=cmd.get_view()
        else:
            cmd.set_view(camera)
        # Adjacent text in the camera plane, as in the candidacy renderer.
        # Label-only pseudoatoms do not alter the input molecular object.
        rotation=np.array(cmd.get_view()[:9]).reshape(3,3)
        right,up=rotation[:,0],rotation[:,1]
        center=cmd.get_coords(f'{atp} or {substrate}').mean(axis=0)
        amino={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
               'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
               'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
        for rid in labels:
            residue=definitions[str(rid)]['res_name']
            atoms=cmd.get_model(f'{protein} and resi {rid} and name CA').atom
            if len(atoms)!=1:
                raise ValueError('Ambiguous residue label anchor '+str(rid))
            xyz=np.array(atoms[0].coord)
            dx=1. if np.dot(xyz-center,right)>=0 else -1.4
            dy=.7 if np.dot(xyz-center,up)>=0 else -.7
            name=f'annotation_{rid}'
            cmd.pseudoatom(name,pos=(xyz+dx*right+dy*up).tolist(),label=f'{amino.get(residue,residue)}{rid}')
            cmd.hide('nonbonded',name)
            cmd.set('float_labels',1,name)
        if not np.array_equal(original,cmd.get_coords('motif'),equal_nan=True):
            raise ValueError('Source coordinates changed during rendering')
        cmd.deselect()
        cmd.png(str(out/f'{pose}_theozyme.png'),width=args.width,height=args.height,dpi=200,ray=1)
        cmd.save(str(out/f'{pose}_theozyme.pse'))
        records.append(dict(pose=pose,source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            metadata=str(metadata),metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
            fixed_residues=len(definitions),fixed_protein_atoms=fixed_count,view=list(cmd.get_view()),
            coordinates_unchanged=True))
        print('Rendered',pose,flush=True)
    (out/'theozyme_render_provenance.json').write_text(json.dumps(dict(
        scope='Actual source CIF coordinates; no minimization or generated molecular geometry',
        rendering='Shared candidacy element colors and ball-and-stick style; translucent retained-residue context; opaque conditioned atoms',
        camera=args.camera,
        camera_reference=str(args.style_dir.resolve().parent/'attack-angle/render_site.py'),
        style_source=str(style_path),style_sha256=hashlib.sha256(style_path.read_bytes()).hexdigest(),
        style_module=str(style_module),style_module_sha256=hashlib.sha256(style_module.read_bytes()).hexdigest(),
        style=style,
        inputs=records),indent=2)+'\n')
    pymol.stop()


if __name__=='__main__':
    main()
