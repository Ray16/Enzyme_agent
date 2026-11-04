"""Compare an unrestrained refold with generated backbone and tagged motif targets."""
from pathlib import Path
import argparse,hashlib,json,itertools
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure.io import pdbx
from geometry import evaluate
from ame_filter import gate

def rmsd(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);x=x-x.mean(0);y=y-y.mean(0)
 u,_,vh=np.linalg.svd(x.T@y);sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@vh)
 return float(np.sqrt(np.mean(np.sum((x@u@sign@vh-y)**2,axis=1))))

def motif_rmsd(x,y,description):
 """Resolve the equivalent ASP oxygens present in this motif atom selection."""
 groups={}
 for i,atom in enumerate(description):
  if atom['res_name']=='ASP' and atom['atom'] in ('OD1','OD2'):groups.setdefault(atom['position'],{})[atom['atom']]=i
 pairs=[(g['OD1'],g['OD2']) for g in groups.values() if set(g)=={'OD1','OD2'}]
 scores=[]
 for flags in itertools.product((False,True),repeat=len(pairs)):
  target=np.asarray(y,float).copy()
  for flag,(i,j) in zip(flags,pairs):
   if flag:target[[i,j]]=target[[j,i]]
  scores.append(rmsd(x,target))
 return min(scores),len(scores)

def score(reference,model):
 ref=json.loads(Path(reference).read_text());atoms=pdbx.get_structure(pdbx.CIFFile.read(model),model=1,extra_fields=['b_factor'])
 if not np.isfinite(atoms.coord).all():raise ValueError('Nonfinite coordinates')
 if set(atoms.chain_id)!={'A','B','C','D'}:raise ValueError('Independent assembly mismatch')
 protein=atoms[atoms.chain_id=='A'];ca=protein[protein.atom_name=='CA'];resids=sorted(set(map(int,ca.res_id)))
 seq=''.join(ProteinSequence.convert_letter_3to1(str(protein.res_name[(protein.res_id==r)&(protein.atom_name=='CA')][0])) for r in resids)
 if seq!=ref['sequence'] or len(ca)!=ref['length']:raise ValueError('Independent sequence/length mismatch')
 pos={rid:i+1 for i,rid in enumerate(resids)};lookup={(pos[int(r)],str(n)):xyz for r,n,xyz in zip(protein.res_id,protein.atom_name,protein.coord)}
 bb=[lookup[(x['position'],x['atom'])] for x in ref['backbone']];bb_target=[x['xyz'] for x in ref['backbone']]
 motif=[lookup[(x['position'],x['atom'])] for x in ref['motif']];ideal=[x['target_xyz'] for x in ref['motif']];generated=[x['generated_xyz'] for x in ref['motif']]
 motif_value,n_permutations=motif_rmsd(motif,ideal,ref['motif'])
 metrics={'refolded_backbone_rmsd.rmsd':rmsd(bb,bb_target),'refolded_motif_allatom_rmsd.rmsd':motif_value,'clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A':ref['generation_clash_fraction']}
 status,reason=gate(metrics)
 # Each ligand occupies its own chain in this assay. Select its unique atom
 # identity without assuming the predictor's author-residue numbering origin.
 geom={'center':{'chain':'B','atom':'PA'},'angle_reference':{'chain':'B','atom':'O3A'},'attack_candidates':{'chain':'D','element':'O'},'expected_attack_candidates':4}
 geometry=evaluate(atoms,geom)
 p=atoms.chain_id=='A';d=np.linalg.norm(atoms.coord[p,None,:]-atoms.coord[None,~p,:],axis=-1);ip,il=np.where(d<1.5);fraction=len(np.unique(np.r_[np.flatnonzero(p)[ip],np.flatnonzero(~p)[il]]))/len(atoms)
 plddt=float(np.mean(protein.b_factor)) if np.isfinite(protein.b_factor).all() else None
 return dict(id=ref['id'],independent_model=str(model),independent_model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),protein_length=len(ca),**metrics,ame_gate=status,ame_gate_reason=reason,motif_rmsd_to_generated_A=motif_rmsd(motif,generated,ref['motif'])[0],motif_named_atom_rmsd_A=rmsd(motif,ideal),motif_symmetry_permutations_considered=n_permutations,refold_protein_ligand_clash_fraction_1p5A=fraction,mean_protein_plddt=plddt,scoring_version='motif-symmetry-v1',geometry_scope='One independent unrestrained refold; not a 50-pose ensemble rank',motif_metric_definition='31 coordinate-conditioned protein atoms aligned to input targets; equivalent ASP OD1/OD2 labels resolved by minimum aligned RMSD; backbone independently aligned on N/CA/C/O',**geometry)

def main():
 p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('model',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=score(a.reference,a.model);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
