"""Reproducible embedded pilots. All outputs are provisional until validation passes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np


def atomic_json(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(prmtop,coordinates,spec_path,out,*,radius=3.0,embedding_radius=20.0,fixed_region=None,force_residues=(),cap_charge_delete_A=None):
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
    region=fixed_region or build_region(p,x,[i,j,k],spec.qm_residue_charges(),
                                       radius=radius,embedding_radius=embedding_radius,force_residues=force_residues,
                                       cap_charge_delete_A=cap_charge_delete_A,multiplicity=spec.multiplicity,
                                       metal_shell_A=spec.metal_shell_A)
    if fixed_region is None:
        region['path_definition']=spec.path_definition([a.atomic_number for a in p.atoms],x,[i,j,k])
        region['state_classifier']=spec.state_classifier
    elif region.get('reaction_atoms') != [i,j,k]:
        raise ValueError('Fixed partition reaction atoms differ from the reaction specification')
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
      method_status='embedded_pilot_not_validated')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],
                            capture_output=True,text=True,check=False)
    manifest['source_revision']=revision.stdout.strip() if revision.returncode==0 else None
    source_root=Path(__file__).resolve().parent
    manifest['producer_sha256']={name:sha256(source_root/name) for name in
        ('jobs.py','spec.py','region.py','coordinates.py','carve.py')}
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


def _box_matrix(box):
    """Lattice-vector rows (row-major) from a box [a,b,c,alpha,beta,gamma] (or a
    ParmEd-like object exposing .box), or None if non-periodic. Handles
    truncated-octahedron cells."""
    if box is None:return None
    if hasattr(box,'box'):box=box.box
    if box is None:return None
    a,b,c,al,be,ga=box
    al,be,ga=np.radians([al,be,ga])
    v1=np.array([a,0.0,0.0])
    v2=np.array([b*np.cos(ga),b*np.sin(ga),0.0])
    cx=c*np.cos(be);cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/np.sin(ga)
    cz=np.sqrt(max(c*c-cx*cx-cy*cy,0.0))
    return np.vstack([v1,v2,np.array([cx,cy,cz])])


def _recorded_box(snapshot_dir,manifest):
    """The coordinate cell recorded WITH a snapshot (not the prep topology):
    input.json 'box' [a,b,c,al,be,ga] if present, else a CRYST1 line in the
    snapshot's or its parent's <label>.pdb. None if nothing is recorded."""
    for key in ('box_lengths_angles','box'):
        if manifest.get(key) is not None:
            return [float(v) for v in manifest[key]]
    sp=Path(snapshot_dir)
    for pdb in [sp/f'{sp.name}.pdb',sp.parent/f'{sp.name}.pdb',*sp.glob('*.pdb')]:
        if pdb.exists():
            for line in pdb.read_text().splitlines():
                if line.startswith('CRYST1'):
                    try:return [float(line[6:15]),float(line[15:24]),float(line[24:33]),
                                float(line[33:40]),float(line[40:47]),float(line[47:54])]
                    except ValueError:pass
    return None


def _min_image(delta,H):
    """Nearest periodic image of displacement rows `delta` under lattice `H`
    (searching +-1 in each lattice vector, which spans the truncated-octahedron
    Wigner-Seitz cell). `H=None` -> identity (non-periodic)."""
    if H is None:return delta
    import itertools
    delta=np.asarray(delta,float)
    delta=delta-np.round(delta@np.linalg.inv(H))@H
    shifts=np.array([np.array(t)@H for t in itertools.product((-1,0,1),repeat=3)])
    cand=delta[:,None,:]+shifts[None,:,:]
    j=np.argmin(np.linalg.norm(cand,axis=2),axis=1)
    return cand[np.arange(len(delta)),j]


def consistent_path_positions(path_dir,*,transplant_radius=None,guard_radius=4.0,guard_tol=0.3):
    """Positions for every point of a barrier path sharing ONE environment.

    A barrier must isolate the reaction coordinate (and the enzyme's local
    response to it) against a single fixed environment. Legacy multi-frame paths
    can carry inconsistent environments: periodic re-imaging (atoms wrapped to
    different images, tens of A apart) and frame-to-frame bulk differences. Under
    the backend's non-periodic reaction-field MM (or a finite explicit cluster)
    those corrupt the energy difference even though the chemistry is fine -- the
    AS9 pilot blew up to +154 kcal/mol from exactly this, entirely from bulk
    beyond ~33 A, while the active site was identical across R/TS/P.

    Construction: keep each frame's own ACTIVE-SITE geometry (the atoms within
    `transplant_radius` of the QM region -- the real reaction coordinate plus the
    enzyme's local relaxation, min-imaged onto the reference), and replace all
    the distant bulk with the single reference frame. Frozen bulk is then
    identical across points and cancels exactly in dE. By default the transplant
    set is the region's methodology-defined `mobile_indices`; pass a radius to
    override.

    Safety: it REFUSES if any atom OUTSIDE the transplant set but within
    `guard_radius` of it genuinely moves (after min-imaging) by more than
    `guard_tol` -- that means real motion spills past the transplant shell and
    freezing it would discard genuine active-site response. Re-imaging jumps and
    distant bulk differences do not trip this (they are far from the shell and
    intended to be frozen).

    Returns (parm, region, reference_label, {label: positions (natom,3)}).
    Generalizable: no enzyme-specific assumptions; reference is 'R' if present,
    else the first snapshot in path.json.
    """
    import parmed as pmd
    root=Path(path_dir)
    path=json.loads((root/'path.json').read_text())
    snaps=path['snapshots']
    if not snaps:raise ValueError('Path has no snapshots')
    labels=list(snaps);ref=labels[0] if 'R' not in snaps else 'R'
    # verify every snapshot's recorded hashes before trusting its files
    for label in labels:
        sp=Path(snaps[label]);d=json.loads((sp/'input.json').read_text())
        if d.get('region_sha256') and sha256(sp/'region.json')!=d['region_sha256']:
            raise ValueError(f'Snapshot {label}: region.json hash mismatch')
        if d.get('positions_sha256') and sha256(sp/'positions.npy')!=d['positions_sha256']:
            raise ValueError(f'Snapshot {label}: positions.npy hash mismatch')
        for f,h in d.get('input_sha256',{}).items():
            if sha256(f)!=h:raise ValueError(f'Snapshot {label}: input {f} changed')
    region=json.loads((Path(snaps[ref])/'region.json').read_text())
    man=json.loads((Path(snaps[ref])/'input.json').read_text())
    parm=pmd.load_file(man['prmtop'])
    topology_hash=sha256(man['prmtop'])
    for label in labels:
        sp=Path(snaps[label]);point_manifest=json.loads((sp/'input.json').read_text())
        if sha256(point_manifest['prmtop'])!=topology_hash:
            raise ValueError(f'Snapshot {label}: topology differs from reference')
        point_region=json.loads((sp/'region.json').read_text())
        if point_region!=region:
            raise ValueError(f'Snapshot {label}: QM/MM partition differs from reference')
        point_positions=np.load(sp/'positions.npy')
        if point_positions.shape!=(len(parm.atoms),3) or not np.isfinite(point_positions).all():
            raise ValueError(f'Snapshot {label}: invalid positions')
        point_box=_recorded_box(sp,point_manifest)
        reference_box=_recorded_box(Path(snaps[ref]),man)
        if (point_box is None)!=(reference_box is None) or (point_box is not None and
                not np.allclose(point_box,reference_box,atol=1e-6,rtol=0)):
            raise ValueError(f'Snapshot {label}: coordinate cell differs from reference')
    qm=np.array(sorted(set(region['indices'])))
    natom=len(parm.atoms)
    # Use the snapshot's RECORDED coordinate cell, not the prep topology (an
    # NPT-equilibrated frame's box differs: AS9 records 100.53 A vs prmtop 104.14).
    box=_recorded_box(Path(snaps[ref]),man)
    if box is None:
        box=parm.box
        if box is not None:print(f'consistent_path_positions: WARNING no recorded cell; '
                                 f'falling back to topology box {np.asarray(box)[:3]}',flush=True)
    elif parm.box is not None and max(abs(np.asarray(box)[:3]-np.asarray(parm.box)[:3]))>0.01*box[0]:
        print(f'consistent_path_positions: using recorded cell {np.asarray(box)[:3]} '
              f'(topology differs: {np.asarray(parm.box)[:3]})',flush=True)
    H=_box_matrix(box)
    xref=np.load(Path(snaps[ref])/'positions.npy')
    if transplant_radius is None and region.get('mobile_indices'):
        transplant=np.array(sorted(set(region['mobile_indices'])|set(qm.tolist())))
    else:
        r=float(transplant_radius if transplant_radius is not None else region.get('mobile_radius_A',8.0))
        dq=np.linalg.norm(xref[:,None,:]-xref[qm][None,:,:],axis=2).min(axis=1)
        transplant=np.where(dq<=r)[0]
    tset=set(transplant.tolist())
    # guard shell: atoms just outside the transplant set, near it
    others=np.array([i for i in range(natom) if i not in tset],dtype=int)
    dshell=np.linalg.norm(xref[others][:,None,:]-xref[transplant][None,:,:],axis=2).min(axis=1)
    guard=others[dshell<=guard_radius]
    frames={}
    for label in labels:
        x=np.load(Path(snaps[label])/'positions.npy')
        if label==ref:frames[label]=xref.copy();continue
        dg=np.linalg.norm(_min_image(x[guard]-xref[guard],H),axis=1) if len(guard) else np.zeros(0)
        bad=guard[dg>guard_tol] if len(guard) else np.array([],int)
        if len(bad):
            res=sorted({f"{parm.atoms[i].residue.name}{parm.atoms[i].residue.idx}" for i in bad})
            raise ValueError(
                f"Frame {label!r}: {len(bad)} atoms just outside the transplant "
                f"shell moved >{guard_tol} A vs {ref!r} (max {dg.max():.2f} A); "
                f"residues {res[:12]}. Genuine motion spills past the transplant "
                "radius -- widen transplant_radius or rebuild the path; do not freeze.")
        out=xref.copy()
        out[transplant]=xref[transplant]+_min_image(x[transplant]-xref[transplant],H)
        frames[label]=out
    region=dict(region,consistent_env_transplant=sorted(int(i) for i in transplant),
                consistent_env_reference=ref)
    return parm,region,ref,frames


def validate_geometry(parm,positions,*,reaction_atoms=(),bond_max=2.2,bond_min=0.4,
                      clash=0.9,raise_on_fail=True):
    """Structural sanity for an input geometry before any (expensive) energy.

    The barrier is only as good as the structure, so every geometry fed to a
    backend should pass: finite coordinates, an (approximately) integer total
    charge, no over-stretched or collapsed covalent bonds, and no non-bonded
    steric clashes. A bond may exceed `bond_max` (or fall below `bond_min`) only
    if BOTH its atoms are reaction atoms -- that is the bond being made/broken.
    This catches periodic re-imaging that tears molecules, bad transplants, and
    overlapping atoms, none of which the MLIP/DFT would flag on its own.

    Returns a report dict; raises ValueError on any violation when
    raise_on_fail (default). Generalizable: reaction_atoms is the only
    chemistry-specific input and defaults to none.
    """
    from scipy.spatial import cKDTree
    x=np.asarray(positions,float);rset=set(int(i) for i in reaction_atoms);fails=[]
    if not np.isfinite(x).all():fails.append('non-finite coordinates')
    qtot=float(sum(a.charge for a in parm.atoms))
    if abs(qtot-round(qtot))>0.02:fails.append(f'total charge {qtot:.3f} is not integer')
    bonded=set()
    bad_bonds=[]
    for b in parm.bonds:
        i,j=b.atom1.idx,b.atom2.idx;bonded.add((min(i,j),max(i,j)))
        d=float(np.linalg.norm(x[i]-x[j]))
        if (d>bond_max or d<bond_min) and not({i,j}<=rset):
            bad_bonds.append((i,j,round(d,2)))
    if bad_bonds:
        ex=[f"{parm.atoms[i].residue.name}{parm.atoms[i].residue.idx}:{parm.atoms[i].name}-"
            f"{parm.atoms[j].name}={d}A" for i,j,d in sorted(bad_bonds,key=lambda t:-t[2])[:8]]
        fails.append(f'{len(bad_bonds)} covalent bond(s) out of [{bond_min},{bond_max}]A '
                     f'not at the reaction center: {ex}')
    pairs=cKDTree(x).query_pairs(clash)
    clashes=[(i,j) for i,j in pairs if (min(i,j),max(i,j)) not in bonded and not({i,j}<=rset)]
    if clashes:
        ex=[f"{parm.atoms[i].residue.name}{parm.atoms[i].residue.idx}:{parm.atoms[i].name}.."
            f"{parm.atoms[j].residue.name}{parm.atoms[j].residue.idx}:{parm.atoms[j].name}="
            f"{np.linalg.norm(x[i]-x[j]):.2f}A" for i,j in clashes[:8]]
        fails.append(f'{len(clashes)} non-bonded clash(es) <{clash}A: {ex}')
    report=dict(n_atoms=len(x),total_charge=qtot,n_bad_bonds=len(bad_bonds),
                n_clashes=len(clashes),passed=not fails,failures=fails)
    if fails and raise_on_fail:
        raise ValueError('Geometry validation failed: '+'; '.join(fails))
    return report


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
        d=snapshot(prmtop,pdb,spec_path,snap,radius=radius,embedding_radius=embedding_radius,
                   fixed_region=reference_region)
        if reference_region is None:reference_region=json.loads((snap/'region.json').read_text())
        else:
            atomic_json(snap/'region.json',reference_region);d['region_sha256']=sha256(snap/'region.json')
        d['source_scope']='legacy imposed cluster path re-evaluated with embedded DFT/MM'
        d['input_sha256'].update({str(Path(f).resolve()):sha256(f) for f in [full_coords,atom_map,file]})
        atomic_json(snap/'input.json',d);snapshots[label]=str(snap.resolve())
    atomic_json(out/'path.json',dict(snapshots=snapshots,radius_A=radius,embedding_radius_A=embedding_radius,
                     quantity='embedded_energy_gaps_on_legacy_geometries',validated_activation_free_energy=False))
    return snapshots


def evaluate(path,out,*,xc='wb97m-v',basis='def2-tzvpd',engine='gpu',platform=None,forces=True,potential='dft'):
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
        xc=xc,basis=basis,engine=engine,platform=getattr(a.calc,'mm_platform',platform),wall_s=time.monotonic()-t0,
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
        engine=cfg.get('engine','gpu'),platform=cfg.get('mm_platform','CUDA'))
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
        mm_platform=cfg.get('mm_platform','CUDA'),
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
    e.add_argument('--platform',default=None)
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
