"""Heavy-atom physical-sanity screen, compared against matched input structures.

Clash counts are geometric diagnostics, not MolProbity clashscores or energies.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,time
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from biotite.structure.io import pdbx
from biotite.structure import BondList
from biotite.sequence import ProteinSequence
from rdkit import Chem
from rdkit.Geometry import Point3D

VDW={'C':1.70,'N':1.55,'O':1.52,'S':1.80,'P':1.80}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 f=pdbx.CIFFile.read(p);a=pdbx.get_structure(f,model=1);cat=f.block['chem_comp_bond'];templates={}
 for comp,n1,n2,order,arom in zip(cat['comp_id'].as_array(str),cat['atom_id_1'].as_array(str),cat['atom_id_2'].as_array(str),cat['value_order'].as_array(str),cat['pdbx_aromatic_flag'].as_array(str)):
  t={'SING':1,'DOUB':2,'TRIP':3}[order.upper()]
  if arom.upper()=='Y':t={1:5,2:6,3:7}[t]
  templates.setdefault(comp,[]).append((n1,n2,t))
 lookup={(str(c),int(r),str(n)):i for i,(c,r,n) in enumerate(zip(a.chain_id,a.res_id,a.atom_name))};edges=[]
 for c,r,res in dict.fromkeys((str(c),int(r),str(res)) for c,r,res in zip(a.chain_id,a.res_id,a.res_name)):
  for n1,n2,t in templates.get(res,[]):
   i,j=lookup.get((c,r,n1)),lookup.get((c,r,n2))
   if i is not None and j is not None:edges.append((i,j,t))
 # Consecutive protein residues define peptide neighbors; do not add metal or ligand links.
 for c,r,n in lookup:
  if c=='A' and n=='C' and ('A',r+1,'N') in lookup:edges.append((lookup[(c,r,n)],lookup[('A',r+1,'N')],1))
 a.bonds=BondList(len(a),np.asarray(edges,dtype=np.uint32));return f,a

def graph_ligand(block,atoms,chain):
 sub=atoms[atoms.chain_id==chain];site=block['atom_site'];names=site['label_atom_id'].as_array(str);chains=site['auth_asym_id'].as_array(str);charge=site['pdbx_formal_charge'].as_array(str)
 charges={str(n):int(q) if q not in ('.','?','') else 0 for n,c,q in zip(names,chains,charge) if c==chain}
 rw=Chem.RWMol();indices={}
 for n,e in zip(sub.atom_name,sub.element):
  at=Chem.Atom(str(e).title());at.SetFormalCharge(charges[str(n)]);indices[str(n)]=rw.AddAtom(at)
 for i,j,t in sub.bonds.as_array():
  bt={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,5:Chem.BondType.AROMATIC,6:Chem.BondType.AROMATIC,7:Chem.BondType.AROMATIC}.get(int(t))
  if bt is None:raise ValueError('Unsupported ligand bond type '+str(t))
  rw.AddBond(int(i),int(j),bt)
 mol=rw.GetMol();Chem.SanitizeMol(mol);conf=Chem.Conformer(len(sub))
 for i,x in enumerate(sub.coord):conf.SetAtomPosition(i,Point3D(*map(float,x)))
 mol.AddConformer(conf);Chem.AssignStereochemistryFrom3D(mol,confId=0,replaceExistingTags=True);Chem.AssignStereochemistry(mol,force=True,cleanIt=True)
 for atom in mol.GetAtoms():
  if atom.GetSymbol()!='C':atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
 return Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True),sum(charges.values())

def inspect(path,reference=None,compare_protein_positions=True):
 file,a=read(path);protein=a.chain_id=='A';n=len(a);coords=a.coord.astype(float);lookup={(str(c),int(r),str(n)):i for i,(c,r,n) in enumerate(zip(a.chain_id,a.res_id,a.atom_name))};edges=a.bonds.as_array();adj=[set() for _ in range(n)]
 for i,j,t in edges:adj[int(i)].add(int(j));adj[int(j)].add(int(i))
 excluded=set()
 for i in range(n):
  seen={i};front={i}
  for _ in range(3):front={v for u in front for v in adj[u]}-seen;seen|=front
  excluded.update((i,j) for j in seen if j>i)
 nonbond=[];clashes=Counter();worst=[]
 for i,j in cKDTree(coords).query_pairs(3.65):
  if (i,j) in excluded or 'MG' in (str(a.element[i]).upper(),str(a.element[j]).upper()):continue
  d=float(np.linalg.norm(coords[i]-coords[j]));r1=VDW.get(str(a.element[i]));r2=VDW.get(str(a.element[j]))
  if r1 is None or r2 is None:continue
  overlap=r1+r2-d;category='protein_ligand' if protein[i]!=protein[j] else ('protein' if protein[i] else 'ligand_ligand')
  if overlap>.4:clashes[category+'_overlap_gt_0_4A']+=1
  if overlap>.8:clashes[category+'_overlap_gt_0_8A']+=1
  if d<2.0:clashes[category+'_distance_lt_2A']+=1
  if overlap>.4:
   worst.append(dict(atom1=f'{a.chain_id[i]}:{a.res_id[i]}:{a.res_name[i]}:{a.atom_name[i]}',atom2=f'{a.chain_id[j]}:{a.res_id[j]}:{a.res_name[j]}:{a.atom_name[j]}',distance_A=d,vdw_overlap_A=overlap,category=category))
 covalent=[];bond_lengths={}
 for i,j,t in edges:
  i,j=int(i),int(j)
  if 'MG' in (str(a.element[i]).upper(),str(a.element[j]).upper()):continue
  d=float(np.linalg.norm(coords[i]-coords[j]));bond_lengths[(i,j)]=d
  if d<1.0 or d>1.9:covalent.append(dict(atom1=f'{a.chain_id[i]}:{a.res_id[i]}:{a.atom_name[i]}',atom2=f'{a.chain_id[j]}:{a.res_id[j]}:{a.atom_name[j]}',distance_A=d))
 ca=a[protein&(a.atom_name=='CA')];seq=''.join(ProteinSequence.convert_letter_3to1(str(x)) for _,x in sorted(zip(ca.res_id,ca.res_name)));resids=sorted(map(int,ca.res_id));backbone=[];chirality=[];min_pl=999;local=[]
 for r in resids:
  ids={name:lookup.get(('A',r,name)) for name in ('N','CA','C','O','CB')}
  if any(ids[k] is None for k in ('N','CA','C','O')):backbone.append(dict(residue=r,problem='missing backbone'));continue
  for left,right,lo,hi in [('N','CA',1.35,1.60),('CA','C',1.40,1.65),('C','O',1.15,1.35)]:
   d=float(np.linalg.norm(coords[ids[left]]-coords[ids[right]]))
   if not lo<=d<=hi:backbone.append(dict(residue=r,bond=left+'-'+right,distance_A=d,screen=[lo,hi]))
  if r<max(resids):
   nxt=lookup.get(('A',r+1,'N'))
   if nxt is not None:
    d=float(np.linalg.norm(coords[ids['C']]-coords[nxt]))
    if not 1.20<=d<=1.45:backbone.append(dict(residue=r,bond='C-N(next)',distance_A=d,screen=[1.20,1.45]))
  if ids['CB'] is not None:
   vol=float(np.linalg.det(coords[[ids['N'],ids['C'],ids['CB']]]-coords[ids['CA']]))
   chirality.append(dict(residue=r,res_name=str(a.res_name[ids['CA']]),signed_volume=vol))
 ligands={};ligand_errors={}
 for ch in ('B','D'):
  try:
   smiles,charge=graph_ligand(file.block,a,ch);ligands[ch]=dict(coordinate_derived_carbon_stereo_smiles=smiles,formal_charge=charge)
  except Exception as e:ligand_errors[ch]=repr(e)
 metals=[]
 for i in np.flatnonzero(np.char.upper(a.element.astype(str))=='MG'):
  ds=np.linalg.norm(coords-coords[i],axis=1);ix=np.flatnonzero((a.element=='O')&(ds<3.0));metals.append(dict(chain=str(a.chain_id[i]),oxygen_contacts=[dict(atom=f'{a.chain_id[j]}:{a.res_id[j]}:{a.atom_name[j]}',distance_A=float(ds[j])) for j in sorted(ix,key=lambda j:ds[j])],oxygen_contacts_within_2_6A=int(((a.element=='O')&(ds<=2.6)).sum()),oxygen_contacts_under_1_8A=int(((a.element=='O')&(ds<1.8)).sum())))
 acid=a[a.chain_id=='D'];atp=a[a.chain_id=='B'];pc=coords[protein]
 for ch,sub in [('B',atp),('D',acid)]:
  ds=np.linalg.norm(sub.coord[:,None,:]-pc[None,:,:],axis=2);min_pl=min(min_pl,float(ds.min()));local.append(dict(chain=ch,min_protein_distance_A=float(ds.min()),ligand_atoms_with_protein_within_4A=int((ds.min(axis=1)<4).sum())))
 result=dict(path=str(path),sha256=sha(path),protein_length=len(ca),sequence=seq,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),finite_coordinates=bool(np.isfinite(coords).all()),covalent_bond_abnormalities=covalent,backbone_bond_screen_flags=backbone,clash_counts=dict(clashes),worst_nonbonded_contacts=sorted(worst,key=lambda r:-r['vdw_overlap_A'])[:25],alpha_carbon_signed_volumes=chirality,ligands=ligands,ligand_parse_errors=ligand_errors,metals=metals,ligand_protein_contacts=local,minimum_protein_ligand_distance_A=min_pl)
 if reference:
  ref=reference;_,ar=read(ref['path']);reflookup={(str(c),int(r),str(n)):i for i,(c,r,n) in enumerate(zip(ar.chain_id,ar.res_id,ar.atom_name))};refchir={c['residue']:c['signed_volume'] for c in ref['alpha_carbon_signed_volumes']};mut=[]
  if compare_protein_positions:
   for r in resids:
    i=lookup[('A',r,'CA')];j=reflookup[('A',r,'CA')]
    if a.res_name[i]!=ar.res_name[j]:mut.append(f'{ProteinSequence.convert_letter_3to1(str(ar.res_name[j]))}{r}{ProteinSequence.convert_letter_3to1(str(a.res_name[i]))}')
   flips=[c['residue'] for c in chirality if c['residue'] in refchir and c['signed_volume']*refchir[c['residue']]<=0]
  else:
   signs=np.sign(list(refchir.values()))
   if not len(signs) or not np.all(signs==signs[0]):raise ValueError('Reference motif does not define a consistent alpha-carbon handedness')
   flips=[c['residue'] for c in chirality if c['signed_volume']*signs[0]<=0]
  ligand_same={c:(c in ref['ligands'] and v==ref['ligands'][c]) for c,v in ligands.items()}
  changes=[]
  for ch in ('B','D'):
   sub=a[a.chain_id==ch]
   for i,j,t in sub.bonds.as_array():
    ni,nj=str(sub.atom_name[i]),str(sub.atom_name[j]);ri=reflookup.get((ch,1,ni));rj=reflookup.get((ch,1,nj))
    if ri is not None and rj is not None:changes.append(abs(float(np.linalg.norm(sub.coord[i]-sub.coord[j]))-float(np.linalg.norm(ar.coord[ri]-ar.coord[rj]))))
  result.update(mutations=mut if compare_protein_positions else None,protein_comparison='matched native positions' if compare_protein_positions else 'de novo; alpha-carbon handedness compared with source motif, no position-wise mutation comparison',alpha_carbon_chirality_inversions=flips,ligand_graph_charge_carbon_stereo_matches_input=ligand_same,max_ligand_bond_length_change_A=max(changes) if changes else None)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--motif-only',action='store_true',help='Use sentinel-free motif views and avoid native-position comparisons');a=p.parse_args();root=a.campaign.resolve();out=root/'physical_analysis';out.mkdir(exist_ok=True);manifest=json.loads((root/'manifest.json').read_text());refs={i['id']:inspect((root/i['runtime_input']).with_name(Path(i['runtime_input']).stem+'_view.cif') if a.motif_only else root/i['runtime_input']) for i in manifest['inputs']};(out/'input_baselines.json').write_text(json.dumps(refs,indent=2)+'\n');rows=[];errors=[]
 with (root/'local_analysis/reactive_geometry.csv').open() as f:records=list(csv.DictReader(f))
 for record in records:
  try:
   result=inspect(Path(record['path']),refs[record['input_id']],compare_protein_positions=not a.motif_only);result.update(id=record['id'],input_id=record['input_id'],attack_distance_A=float(record['distance_A']),attack_angle_deg=float(record['angle_deg']));(out/(record['id'].replace('/','__')+'.json')).write_text(json.dumps(result,indent=2)+'\n');rows.append(result)
  except Exception as e:errors.append(dict(id=record['id'],error=repr(e)))
 summary=dict(structures=len(rows),unique_sequences=len({r['sequence_sha256'] for r in rows}),errors=errors,scope='geometric physical-sanity screen on conditioned, unrelaxed heavy-atom structures; no energy calculation or independent refolding',rows=rows,updated=time.time());(out/'all_results.json').write_text(json.dumps(summary,indent=2)+'\n')
 cols=['id','input_id','protein_length','sequence_sha256','mutations','attack_distance_A','attack_angle_deg','minimum_protein_ligand_distance_A','n_covalent_flags','n_backbone_flags','n_CA_inversions','ligand_identity_ok','max_ligand_bond_length_change_A','protein_overlap_gt_0_8A','protein_ligand_overlap_gt_0_8A','protein_ligand_distance_lt_2A']
 with (out/'physical_sanity.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in rows:
   d={k:r.get(k) for k in cols};d.update(mutations=','.join(r['mutations']) if r['mutations'] is not None else '',n_covalent_flags=len(r['covalent_bond_abnormalities']),n_backbone_flags=len(r['backbone_bond_screen_flags']),n_CA_inversions=len(r['alpha_carbon_chirality_inversions']),ligand_identity_ok=all(r['ligand_graph_charge_carbon_stereo_matches_input'].values()) and not r['ligand_parse_errors'])
   for k in cols[-3:]:d[k]=r['clash_counts'].get(k,0)
   w.writerow(d)
 print(json.dumps(dict(structures=len(rows),errors=errors,unique_sequences=summary['unique_sequences'])))
if __name__=='__main__':main()
