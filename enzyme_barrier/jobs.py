"""Reproducible embedded pilots. All outputs are provisional until validation passes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def atomic_json(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(prmtop,coordinates,spec_path,out,*,radius=3.0,embedding_radius=20.0,fixed_region=None,force_residues=()):
    import parmed as pmd
    from .carve import _resolve
    from .spec import ReactionSpec
    from .region import build_region
    from .parameters import require_c4
    from .coordinates import load_system,reimage_molecules
    p=load_system(prmtop,coordinates);require_c4(p)
    spec=ReactionSpec.load(spec_path);x=p.coordinates.copy()
    j=_resolve(p,spec.electrophile);k=_resolve(p,spec.leaving)
    x=reimage_molecules(p,x,j)
    i=_resolve(p,spec.nucleophile,near_xyz=x[j],xyz=x)
    for b in p.bonds:
        if {b.atom1.idx,b.atom2.idx}=={j,k}:continue # explicitly reactive breaking bond
        if np.linalg.norm(x[b.atom1.idx]-x[b.atom2.idx])>2.2:
            raise ValueError(f'Long nonreactive covalent bond {b.atom1.idx}-{b.atom2.idx}; unwrap or review chemistry')
    region=fixed_region or build_region(p,x,[i,j,k],dict(spec.ligands)|dict.fromkeys(spec.metals,2),
                                       radius=radius,embedding_radius=embedding_radius,force_residues=force_residues)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    np.save(out/'positions.npy',x)
    atomic_json(out/'region.json',region)
    manifest=dict(schema_version=1,prmtop=str(Path(prmtop).resolve()),
      coordinates=str(Path(coordinates).resolve()),spec=str(Path(spec_path).resolve()),
      input_sha256={str(Path(f).resolve()):sha256(f) for f in [prmtop,coordinates,spec_path]},
      region_sha256=sha256(out/'region.json'),positions_sha256=sha256(out/'positions.npy'),
      n_atoms=len(p.atoms),n_qm=len(region['indices']),n_links=len(region['links']),
      n_embedding=len(region['embedding_indices']),reaction_id=spec.reaction_id,
      assembly_validation='pending',protonation_validation='pending',
      method_status='embedded_pilot_not_validated',source_revision=os.popen('git rev-parse HEAD').read().strip())
    manifest['box_lengths_angles']=p.box.tolist() if p.box is not None else None
    atomic_json(out/'input.json',manifest)
    return manifest


def load_snapshot(path):
    import parmed as pmd
    from ase import Atoms
    root=Path(path);d=json.loads((root/'input.json').read_text())
    for f,h in d['input_sha256'].items():
        if sha256(f)!=h:raise ValueError(f'Input changed: {f}')
    if sha256(root/'region.json')!=d['region_sha256'] or sha256(root/'positions.npy')!=d['positions_sha256']:
        raise ValueError('Snapshot changed')
    p=pmd.load_file(d['prmtop']);region=json.loads((root/'region.json').read_text());x=np.load(root/'positions.npy')
    a=Atoms(numbers=[at.atomic_number for at in p.atoms],positions=x)
    return p,region,a,d


def import_cluster_path(prmtop,full_coords,atom_map,geometries,spec_path,out,*,radius=4.0,embedding_radius=20.0):
    """Re-evaluate a legacy path in one fixed QM/MM partition; never call it a TS.

    The original cluster caps are discarded. New, topology-defined sidechain
    link atoms are built by the embedded backend. All environment coordinates
    displaced along the old path are retained through the global atom map.
    """
    import parmed as pmd
    from ase.io import read
    p=pmd.load_file(str(prmtop),str(full_coords));mapping=json.loads(Path(atom_map).read_text())['atoms']
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    snapshots={};reference_region=None
    for label,file in geometries.items():
        xyz=read(file).positions;x=p.coordinates.copy()
        for row in mapping:x[row['gi']]=xyz[row['li']]
        pdb=out/f'{label}.pdb';pmd.formats.PDBFile.write(p,str(pdb),coordinates=x)
        snap=out/label
        d=snapshot(prmtop,pdb,spec_path,snap,radius=radius,embedding_radius=embedding_radius)
        if reference_region is None:reference_region=json.loads((snap/'region.json').read_text())
        else:
            atomic_json(snap/'region.json',reference_region);d['region_sha256']=sha256(snap/'region.json')
        d['source_scope']='legacy imposed cluster path re-evaluated with embedded DFT/MM'
        d['input_sha256'].update({str(Path(f).resolve()):sha256(f) for f in [full_coords,atom_map,file]})
        atomic_json(snap/'input.json',d);snapshots[label]=str(snap.resolve())
    atomic_json(out/'path.json',dict(snapshots=snapshots,radius_A=radius,embedding_radius_A=embedding_radius,
                     quantity='embedded_energy_gaps_on_legacy_geometries',validated_activation_free_energy=False))
    return snapshots


def evaluate(path,out,*,xc='wb97m-v',basis='def2-tzvpd',engine='gpu',platform='CPU',forces=True,potential='dft'):
    from .embedded import EmbeddedDFTMM
    p,r,a,d=load_snapshot(path)
    if potential=='dft':factory=EmbeddedDFTMM
    elif potential=='uma-xtb':
        from .sampler import EmbeddedUMAXTB
        factory=EmbeddedUMAXTB
    else:raise ValueError('Unknown potential')
    a.calc=factory(p,r,xc=xc,basis=basis,engine=engine,platform=platform)
    t0=time.monotonic()
    if forces:
        f=a.get_forces();np.save(str(out)+'.forces.npy',f)
    e=a.get_potential_energy()
    result=dict(energy_eV=float(e),components=a.calc.last_components,
        snapshot=str(Path(path).resolve()),input_sha256=sha256(Path(path)/'input.json'),
        xc=xc,basis=basis,engine=engine,platform=platform,wall_s=time.monotonic()-t0,
        potential=potential,high_diagnostics=getattr(a.calc,'high_diagnostics',{}),
        quantity='embedded_electronic_energy',validated_activation_free_energy=False)
    if forces:
        result['max_force_eV_A']=float(np.linalg.norm(f,axis=1).max())
        result['max_qm_force_eV_A']=float(np.linalg.norm(f[r['indices']],axis=1).max())
    atomic_json(out,result);return result


def evaluate_geometries(config_path):
    """DFT diagnostics at named reference geometries, without cross-model subtraction."""
    from .embedded import EmbeddedDFTMM
    cfg=json.loads(Path(config_path).read_text());out=Path(cfg['out'])
    p,r,a,d=load_snapshot(cfg['snapshot']);rows=[];labels=set()
    if not cfg['geometries']:raise ValueError('No diagnostic geometries')
    for point in cfg['geometries']:
        label=point['label']
        if not label or not all(c.isalnum() or c in '_-' for c in label) or label in labels:raise ValueError('Invalid or repeated geometry label')
        labels.add(label)
        if sha256(point['positions'])!=point['sha256']:raise ValueError('Geometry changed')
        positions=np.load(point['positions'])
        if positions.shape!=a.positions.shape or not np.isfinite(positions).all():raise ValueError('Invalid geometry')
    out.mkdir(parents=True,exist_ok=False)
    a.calc=EmbeddedDFTMM(p,r,xc=cfg.get('xc','wb97m-v'),basis=cfg.get('basis','def2-svp'),
        engine=cfg.get('engine','gpu'),platform='CPU')
    for point in cfg['geometries']:
        if sha256(point['positions'])!=point['sha256']:raise ValueError('Geometry changed')
        positions=np.load(point['positions'])
        if positions.shape!=a.positions.shape or not np.isfinite(positions).all():raise ValueError('Invalid geometry')
        a.positions[:]=positions;t0=time.monotonic()
        force=a.get_forces();energy=float(a.get_potential_energy())
        np.save(out/(point['label']+'.forces.npy'),force)
        row=dict(point,energy_eV=energy,components=a.calc.last_components.copy(),wall_s=time.monotonic()-t0,
            max_qm_force_eV_A=float(np.linalg.norm(force[r['indices']],axis=1).max()),
            max_mobile_force_eV_A=float(np.linalg.norm(force[r['mobile_indices']],axis=1).max()))
        rows.append(row);atomic_json(out/'progress.json',dict(points=rows,requested=len(cfg['geometries'])))
        print(json.dumps(row),flush=True)
    result=dict(status='geometry_diagnostics_complete',points=rows,config_sha256=sha256(config_path),
        snapshot=str(Path(cfg['snapshot']).resolve()),snapshot_sha256=sha256(Path(cfg['snapshot'])/'input.json'),
        xc=cfg.get('xc','wb97m-v'),basis=cfg.get('basis','def2-svp'),
        quantity='finite_embedding_DFT_MM_diagnostics_on_reference_geometries',validated_activation_free_energy=False,
        limitations=['Reference geometries are not stationary on the DFT/MM surface',
            'SVP pilot requires basis and QM-region/embedding sensitivity checks',
            'Periodic native reference and finite DFT/MM must not be subtracted without matched decomposition'])
    atomic_json(out/'result.json',result);return result


def gpu_smoke(out,engine='gpu',platform='CPU',grid_response=False):
    from ase import Atoms
    from .fixtures import water_pair
    from .embedded import EmbeddedDFTMM
    p=water_pair();r=dict(indices=[0,1,2],links=[],charge=0,multiplicity=1,
                           embedding_indices=[3,4,5],embedding_charges=[-.834,.417,.417])
    a=Atoms(numbers=[v.atomic_number for v in p.atoms],positions=p.coordinates)
    a.calc=EmbeddedDFTMM(p,r,engine=engine,platform=platform,
                  xc='wb97m-v',basis='def2-svp',grid_level=2,conv_tol=1e-10,require_magnesium_c4=False,grid_response=grid_response)
    f=a.get_forces();energy=a.get_potential_energy();errors=[]
    for i,k in [(0,2),(3,2),(4,0)]:
        x=a.positions.copy();a.positions[i,k]+=1e-4;ep=a.get_potential_energy()
        a.positions[:]=x;a.positions[i,k]-=1e-4;em=a.get_potential_energy();a.positions[:]=x
        errors.append(abs(float(f[i,k]+(ep-em)/2e-4)))
    result=dict(engine=engine,platform=platform,grid_response=grid_response,energy_eV=energy,finite_difference_errors_eV_A=errors,
                passed=max(errors)<0.002,components=a.calc.last_components)
    atomic_json(out,result)
    if not result['passed']:raise RuntimeError(f'Energy/force consistency failed: {errors}')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    b=s.add_parser('snapshot');b.add_argument('--prmtop',required=True);b.add_argument('--coords',required=True)
    b.add_argument('--spec',required=True);b.add_argument('--out',required=True)
    b.add_argument('--radius',type=float,default=3.0);b.add_argument('--embedding-radius',type=float,default=20)
    e=s.add_parser('evaluate');e.add_argument('--snapshot',required=True);e.add_argument('--out',required=True)
    e.add_argument('--basis',default='def2-tzvpd');e.add_argument('--xc',default='wb97m-v')
    e.add_argument('--engine',default='gpu',choices=['gpu','cpu']);e.add_argument('--energy-only',action='store_true')
    e.add_argument('--platform',default='CPU')
    e.add_argument('--potential',default='dft',choices=['dft','uma-xtb'])
    b=s.add_parser('geometry-diagnostics');b.add_argument('config')
    g=s.add_parser('smoke');g.add_argument('--out',required=True);g.add_argument('--engine',default='gpu');g.add_argument('--platform',default='CPU')
    g.add_argument('--grid-response',action='store_true')
    a=p.parse_args(argv)
    if a.cmd=='snapshot':v=snapshot(a.prmtop,a.coords,a.spec,a.out,radius=a.radius,embedding_radius=a.embedding_radius)
    elif a.cmd=='evaluate':v=evaluate(a.snapshot,a.out,xc=a.xc,basis=a.basis,engine=a.engine,platform=a.platform,forces=not a.energy_only,potential=a.potential)
    elif a.cmd=='geometry-diagnostics':v=evaluate_geometries(a.config)
    else:v=gpu_smoke(a.out,a.engine,a.platform,a.grid_response)
    print(json.dumps(v,indent=2),flush=True)


if __name__=='__main__':main()
