"""Prepare independent sequence/ligand inputs and explicit AME atom mappings.

No structural coordinates are supplied to either predictor. Coordinate references
are separate scoring artifacts. Missing searched MSAs remain pending, never
replaced with a native MSA or a fabricated query-only search result.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from biotite.structure.io import pdbx

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args();root=a.campaign.resolve();out=a.evaluation.resolve();jobs=json.loads((out/'design_index.json').read_text());cfg=json.loads((out/'control.json').read_text());remote=cfg['remote_root'];msa_dir=out/'msa';reference_cache={};rows=[]
 for job in jobs:
  ident=job['id'];path=Path(job['generated_cif']);assert sha(path)==job['generated_sha256'];file=pdbx.CIFFile.read(path);atoms=pdbx.get_structure(file,model=1);site=file.block['atom_site'];protein=atoms.chain_id=='A';ca=atoms[protein&(atoms.atom_name=='CA')];numbering={int(r):i+1 for i,r in enumerate(sorted(map(int,ca.res_id)))};labels=site['annotation_src-coordinate-conditioned-region_1_atom'].as_array(str)
  pose=job['input_id']
  if pose not in reference_cache:
   src=pdbx.get_structure(pdbx.CIFFile.read(root/'inputs'/f'{pose}_view.cif'),model=1);reference_cache[pose]={(str(c),int(r),str(n)):x.tolist() for c,r,n,x in zip(src.chain_id,src.res_id,src.atom_name,src.coord)}
  backbone=[];motif=[]
  for i in np.flatnonzero(protein):
   rid=int(atoms.res_id[i]);name=str(atoms.atom_name[i]);pos=numbering[rid]
   if name in ('N','CA','C','O'):backbone.append(dict(position=pos,atom=name,xyz=atoms.coord[i].astype(float).tolist()))
   if labels[i]!='.':
    label=str(labels[i]);original=int(label[1:]);target=reference_cache[pose][('A',original,name)];motif.append(dict(position=pos,atom=name,res_name=str(atoms.res_name[i]),source_residue=original,target_xyz=target,generated_xyz=atoms.coord[i].astype(float).tolist()))
  assert len(motif)==31 and len(backbone)==len(ca)*4
  ligands=~protein;d=np.linalg.norm(atoms.coord[protein,None,:]-atoms.coord[None,ligands,:],axis=-1);ip,il=np.where(d<1.5);pindices=np.flatnonzero(protein)[ip];lindices=np.flatnonzero(ligands)[il];fixed=site['condition_CoordinateGroup_1_atom'].as_array(int)>=0;keep=~(fixed[pindices]&fixed[lindices]);clash_atoms=np.unique(np.r_[pindices[keep],lindices[keep]])
  ref=dict(id=ident,length=len(ca),sequence=job['sequence'],source_generated_sha256=job['generated_sha256'],backbone=backbone,motif=motif,clash_metric='clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A',generation_clash_fraction=float(len(clash_atoms)/len(atoms)),generation_clash_atom_count=len(clash_atoms),clash_definition='Fraction of all atoms in protein-ligand pairs closer than1.5A; pairs with both atoms coordinate-conditioned excluded, matching upstream design metric.')
  write(out/'references'/f'{ident}.json',ref)
  h=hashlib.sha1(job['sequence'].encode()).hexdigest()[:16];msa=msa_dir/f'{h}.csv';msa_ready=False;depth=None
  if msa.exists():
   with msa.open() as f:records=list(csv.DictReader(f))
   assert records and records[0]['sequence']==job['sequence'],f'MSA query mismatch {ident}';depth=len(records);msa_ready=True
  boltz=dict(version=1,sequences=[dict(protein=dict(id='A',sequence=job['sequence'],msa=f'{remote}/msa/{h}.csv')),dict(ligand=dict(id='B',ccd='ATP')),dict(ligand=dict(id='C',ccd='MG')),dict(ligand=dict(id='D',smiles='O=C(O)CCCCC(=O)O'))])
  write(out/'inputs_boltz'/f'{ident}.yaml',boltz)
  af3=dict(name=ident,modelSeeds=[20261101],sequences=[dict(protein=dict(id='A',sequence=job['sequence'],pairedMsa='',templates=[])),dict(ligand=dict(id='B',ccdCodes=['ATP'])),dict(ligand=dict(id='C',ccdCodes=['MG'])),dict(ligand=dict(id='D',smiles='O=C(O)CCCCC(=O)O'))],dialect='alphafold3',version=1)
  a3m=msa_dir/f'{h}.a3m'
  if msa_ready and a3m.exists():af3['sequences'][0]['protein']['unpairedMsa']=a3m.read_text()
  write(out/'inputs_af3_pending_msa'/f'{ident}.json',af3)
  rows.append(dict(id=ident,source_id=job['source_id'],sequence_sha256=job['sequence_sha256'],length=len(ca),fixed_motif_identities=12,motif_atoms=len(motif),fresh_searched_msa_ready=msa_ready,msa_depth=depth,msa_hash=h,generation_clash_fraction=ref['generation_clash_fraction'],generation_clash_atom_count=len(clash_atoms),reference_sha256=sha(out/'references'/f'{ident}.json')))
 write(out/'preparation_status.json',dict(designs=len(rows),searched_msas_ready=sum(x['fresh_searched_msa_ready'] for x in rows),generation_AME_clash_gate_pass=sum(x['generation_clash_fraction']==0 for x in rows),prediction_inputs='Sequence and ATP/Mg/adipic acid only; no templates, restraints or designed coordinates',official_AME_status='pending independent refolding',rows=rows))
 print(json.dumps(dict(designs=len(rows),searched_msas_ready=sum(x['fresh_searched_msa_ready'] for x in rows),generation_clash_gate_pass=sum(x['generation_clash_fraction']==0 for x in rows))))

if __name__=='__main__':main()
