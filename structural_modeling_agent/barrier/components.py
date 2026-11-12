"""Multi-ligand Amber preparation from explicit mapped molecular graphs.

The ligand exporter supplies SDF bond orders, formal charges and atom identities.
Parameterization must retain atom order, coordinates and connectivity. This
operator supports activated intermediates plus amines/byproducts; it does not
infer a mechanism or certify intermediate stability.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import numpy as np
from .jobs import atomic_json,sha256


def mol2_tables(path):
    sections={};section=None
    for line in Path(path).read_text().splitlines():
        if line.startswith('@<TRIPOS>'):
            section=line[len('@<TRIPOS>'):];sections[section]=[]
        elif section and line.strip():sections[section].append(line.split())
    return sections


def verify_parameterization(component,mol2):
    sdf=Path(component['sdf']).read_text().splitlines();natoms=int(sdf[3][:3]);nbonds=int(sdf[3][3:6])
    source=[line.split() for line in sdf[4:4+natoms]]
    bonds={tuple(sorted((int(line[:3])-1,int(line[3:6])-1))):int(line[6:9]) for line in sdf[4+natoms:4+natoms+nbonds]}
    table=mol2_tables(mol2);atoms=table['ATOM']
    if len(atoms)!=natoms or len(component['atoms'])!=natoms:raise ValueError('Parameterized atom count changed')
    indices=[int(a[0]) for a in atoms]
    if indices!=list(range(1,natoms+1)):raise ValueError('MOL2 atom order is not consecutive')
    if len({a[1] for a in atoms})!=natoms:raise ValueError('Duplicate MOL2 atom names')
    for original,atom in zip(source,atoms):
        atom_type=atom[5].lower();element=atom_type.capitalize() if atom_type in {'cl','br'} else atom_type[0].upper()
        if element!=original[3]:raise ValueError('Parameterization changed atomic element/order')
    x=np.array([[float(v) for v in a[2:5]] for a in atoms]);reference=np.array([[float(v) for v in a[:3]] for a in source])
    if not np.isfinite(x).all() or not np.isfinite(reference).all():raise ValueError('Nonfinite ligand coordinates')
    if np.max(np.abs(x-reference))>.001:raise ValueError('Parameterization moved or reordered the ligand')
    connectivity={tuple(sorted((int(b[1])-1,int(b[2])-1))):b[3] for b in table['BOND']}
    if set(connectivity)!=set(bonds):raise ValueError('Parameterization changed molecular connectivity')
    resonance_pairs=set()
    for center in component['atoms']:
        if center['element'] not in {'C','P'}:continue
        terminal=[component['atoms'][i] for i in center['neighbors'] if component['atoms'][i]['element']=='O' and len(component['atoms'][i]['neighbors'])==1]
        if len(terminal)<2 or sum(a['formal_charge'] for a in terminal)>=0:continue
        expected_type='c' if center['element']=='C' else 'p5'
        if atoms[center['index']][5]!=expected_type or any(atoms[a['index']][5]!='o' for a in terminal):
            raise ValueError('Anionic resonance group lost its GAFF carbonyl/phosphate types')
        resonance_pairs.update(tuple(sorted((center['index'],a['index']))) for a in terminal)
    adjustments=[]
    # Kekule/aromatic representation can differ; preserve explicit non-aromatic
    # carbonyl/phosphoryl bond orders that define the reaction chemistry.
    for pair,order in bonds.items():
        if pair in resonance_pairs and connectivity[pair] in {'1','2','ar'}:
            if connectivity[pair]!=str(order):adjustments.append(dict(atoms=list(pair),source_order=order,mol2_order=connectivity[pair],interpretation='GAFF delocalized anionic group; source Lewis graph remains authoritative'))
            continue
        if connectivity[pair]=='ar':
            # Older exported graphs omit aromatic flags: require appropriate
            # GAFF aromatic types as well. Never accept an arbitrary 'ar' bond.
            aromatic_types={'ca','cp','cq','cc','cd','ce','cf','na','nb','nc','nd','ne','nf','os','ss'}
            if not all(component['atoms'][i].get('aromatic',atoms[i][5] in aromatic_types) for i in pair):
                raise ValueError('Aromatic bond notation outside a declared aromatic group')
        if order in (1,2,3) and connectivity[pair] not in {str(order),'ar','am' if order==1 else str(order)}:
            raise ValueError(f'Bond order changed at {pair}: {order} -> {connectivity[pair]}')
        elements={source[i][3] for i in pair}
        if order==2 and 'O' in elements and connectivity[pair]!='2':raise ValueError('Carbonyl/phosphoryl double bond changed')
    q=sum(float(a[8]) for a in atoms)
    if not np.isfinite(q) or abs(q-component['charge'])>.005:raise ValueError('Parameterized ligand charge changed or nonfinite')
    return dict(resname=component['resname'],charge=component['charge'],charge_sum=q,
          atoms=[dict(a,name=atoms[a['index']][1]) for a in component['atoms']],
          reactive_atoms={k:atoms[i][1] for k,i in component.get('reactive_atoms',{}).items()},
          neutral_nitrogens=[atoms[i][1] for i in component.get('neutral_nitrogens',[])],
          resonance_representation_adjustments=adjustments,
          mol2_sha256=sha256(mol2),maximum_coordinate_difference_A=float(np.max(np.abs(x-reference))))


def prepare(graphs,protein_pose,out,polarizabilities,*,overrides=None,resume_from=None):
    from .parameters import install_c4
    from .prep_cofold import _read_pdb_atoms
    graph_file=Path(graphs)/'components.json';data=json.loads(graph_file.read_text())
    for component in data['components']:
        if sha256(component['sdf'])!=component['sdf_sha256']:raise ValueError('Exported ligand SDF changed')
    identity=dict(graphs=str(graph_file.resolve()),graphs_sha256=sha256(graph_file),protein_pose=str(Path(protein_pose).resolve()),protein_pose_sha256=sha256(protein_pose))
    previous=Path(resume_from) if resume_from else None
    if previous and json.loads((previous/'inputs.json').read_text())!=identity:raise ValueError('Preparation cache has different inputs')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'inputs.json',identity)
    rows=_read_pdb_atoms(protein_pose);protein=[r for r in rows if r['rec']=='ATOM  ']
    metals=[r for r in rows if r['rec']=='HETATM' and r['elem']=='MG']
    if not protein or not metals:raise ValueError('Protein or magnesium missing')
    lines=[];previous_chain=None
    for row in protein:
        if previous_chain is not None and row['chain']!=previous_chain:lines.append('TER')
        lines.append(row['raw']);previous_chain=row['chain']
    (out/'protein_raw.pdb').write_text('\n'.join(lines+['TER','END'])+'\n')
    def run(argv,log):
        result=subprocess.run(argv,cwd=out,text=True,capture_output=True)
        (out/log).write_text(result.stdout+'\n'+result.stderr);result.check_returncode()
    run(['pdb4amber','-i','protein_raw.pdb','-o','protein_clean.pdb','--dry'],'pdb4amber.log')
    mlines=[]
    for i,row in enumerate(metals,1):
        mlines.append(f"HETATM{i:5d} MG    MG M{i:4d}    {row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}  1.00  0.00          MG")
    (out/'metal.pdb').write_text('\n'.join(mlines+['TER','END'])+'\n')
    maps=[]
    for component in data['components']:
        name=component['resname']
        if not name.isalnum() or len(name)>3:raise ValueError('Amber residue name must be at most 3 alphanumeric characters')
        if previous and (previous/(name+'.mol2')).is_file():
            verify_parameterization(component,previous/(name+'.mol2'))
            shutil.copy2(previous/(name+'.mol2'),out/(name+'.mol2'))
            atomic_json(out/(name+'.cache.json'),dict(source=str((previous/(name+'.mol2')).resolve()),sha256=sha256(previous/(name+'.mol2')),reason='same hashed graphs/pose; atom order, coordinates, connectivity, types and charge revalidated'))
        else:
            run(['antechamber','-i',component['sdf'],'-fi','mdl','-o',name+'.mol2','-fo','mol2','-c','bcc','-nc',str(component['charge']),'-at','gaff2','-rn',name,'-dr','no','-j','1'],name+'.antechamber.log')
            for filename in ['sqm.in','sqm.out']:
                if (out/filename).exists():shutil.copy2(out/filename,out/(name+'.'+filename))
        mapping=verify_parameterization(component,out/(name+'.mol2'));maps.append(mapping)
        run(['parmchk2','-i',name+'.mol2','-f','mol2','-o',name+'.frcmod','-s','gaff2'],name+'.parmchk.log')
        atomic_json(out/(name+'.atom_map.json'),mapping)
    setup=['source leaprc.protein.ff14SB','source leaprc.gaff2','source leaprc.water.tip3p','loadamberparams frcmod.ions234lm_1264_tip3p']
    for component in data['components']:
        n=component['resname'];setup += [f'{n} = loadmol2 {n}.mol2',f'loadamberparams {n}.frcmod']
    setup += ['prot = loadpdb protein_clean.pdb','mgu = loadpdb metal.pdb','cmplx = combine {prot '+' '.join(c['resname'] for c in data['components'])+' mgu}',
              'solvateOct cmplx TIP3PBOX 12.0','addIonsRand cmplx Na+ 0','addIonsRand cmplx Cl- 0','charge cmplx','saveamberparm cmplx complex.prmtop complex.inpcrd','quit']
    (out/'build.leap').write_text('\n'.join(setup)+'\n');run(['tleap','-f','build.leap'],'build.log')
    if not (out/'complex.prmtop').exists():raise ValueError('tleap failed to write topology')
    install_c4(out/'complex.prmtop',out/'complex_1264.prmtop',polarizabilities,overrides=overrides)
    # The source graph, not nearest arbitrary C/O, identifies the aminolysis atoms.
    acid=next(c for c in data['components'] if c.get('role')=='activated_acid')
    amine=next(c for c in data['components'] if c.get('role')=='amine')
    amap=next(m for m in maps if m['resname']==acid['resname']);nmap=next(m for m in maps if m['resname']==amine['resname'])
    source_mol2=mol2_tables(out/(amine['resname']+'.mol2'))['ATOM'];acid_mol2=mol2_tables(out/(acid['resname']+'.mol2'))['ATOM']
    carbon_name=amap['reactive_atoms']['carbonyl'];cx=np.array([float(v) for a in acid_mol2 if a[1]==carbon_name for v in a[2:5]])
    neutral=amine.get('neutral_nitrogens',[])
    if len(neutral)==1:nuc=nmap['atoms'][neutral[0]]
    else:
        protonated=[x['atom'] for x in amine['transformations'] if x.get('action')=='reactive amine protonation']
        if len(protonated)!=1:raise ValueError('Reactive amine atom requires an explicit graph assignment')
        nuc=next(a for a in nmap['atoms'] if a['source_name']==protonated[0])
    spec=dict(reaction_id=acid['resname']+'_'+amine['resname']+'_aminolysis',ligands=[[c['resname'],c['charge']] for c in data['components']],metals=['MG'],
      nucleophile=dict(resname=amine['resname'],atom=nuc['name']),electrophile=dict(resname=acid['resname'],atom=carbon_name),
      leaving=dict(resname=acid['resname'],atom=amap['reactive_atoms']['leaving']),multiplicity=1,r_lig=6.0,free_r=6.0)
    spec['carbonyl_oxygen']=dict(resname=acid['resname'],atom=amap['reactive_atoms']['carbonyl_oxygen'])
    atomic_json(out/'reaction.json',spec)
    donor_h=[a['name'] for a in nmap['atoms'] if a['atomic_number']==1 and nuc['index'] in a['neighbors']]
    atomic_json(out/'proton_setup.json',dict(reactive_nitrogen=nuc['name'],donor_hydrogens=donor_h,declared_amine_charge=amine['charge'],carbonyl_oxygen=amap['reactive_atoms']['carbonyl_oxygen'],status='acceptor/relay pathways require chemical review',scope=data['scope']))
    result=dict(status='parameterized_requires_protonation_assembly_and_chemical_review',magnesium_count=len(metals),components=maps,
        prmtop=str((out/'complex_1264.prmtop').resolve()),coordinates=str((out/'complex.inpcrd').resolve()),reaction=str((out/'reaction.json').resolve()),
        graph_manifest_sha256=sha256(graph_file),scope=data['scope'],validated_activation_free_energy=False)
    atomic_json(out/'preparation.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graphs',required=True);p.add_argument('--protein-pose',required=True);p.add_argument('--out',required=True);p.add_argument('--polarizabilities',required=True);p.add_argument('--overrides-json')
    p.add_argument('--resume-from')
    a=p.parse_args();overrides=json.loads(Path(a.overrides_json).read_text()) if a.overrides_json else None
    print(json.dumps(prepare(a.graphs,a.protein_pose,a.out,a.polarizabilities,overrides=overrides,resume_from=a.resume_from),indent=2))
