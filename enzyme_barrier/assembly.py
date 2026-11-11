"""Template-supported local oligomer construction, retaining exact target sequence.

This creates a structural hypothesis, not experimental evidence of assembly.
All copies are fitted with the same sequence correspondence and trimmed core.
"""
import json
from pathlib import Path
import numpy as np
from .jobs import atomic_json,sha256


def build_assembly(monomer,template,out,*,max_rmsd=3.0,min_aligned=200,neighbor_distance=8.0):
    from Bio import Align
    from Bio.Align import substitution_matrices
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    parser=PDBParser(QUIET=True)
    target=parser.get_structure('target',monomer)
    source=parser.get_structure('template',template)
    def residues(chain):return [r for r in chain if r.id[0]==' ' and 'CA' in r]
    target_chains=[c for c in target[0] if len(residues(c))>100]
    if len(target_chains)!=1:raise ValueError('Expected one target protein chain')
    tr=residues(target_chains[0]);seq=''.join(seq1(r.resname) for r in tr)
    copies=[c for model in source for c in model if len(residues(c))>100]
    rr=residues(copies[0]);refseq=''.join(seq1(r.resname) for r in rr)
    aligner=Align.PairwiseAligner();aligner.substitution_matrix=substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score=-10;aligner.extend_gap_score=-0.5
    alignment=aligner.align(seq,refseq)[0]
    pairs=[(i,j) for ab,cd in zip(*alignment.aligned) for i,j in zip(range(*ab),range(*cd))]
    tx=np.array([tr[i]['CA'].coord for i,j in pairs]);ry=np.array([rr[j]['CA'].coord for i,j in pairs])
    def fit(x,y):
        xc=x.mean(axis=0);yc=y.mean(axis=0);u,_,vt=np.linalg.svd((x-xc).T@(y-yc))
        sign=np.eye(3);sign[2,2]=np.linalg.det(u@vt);rot=u@sign@vt
        return rot,yc-xc@rot
    keep=np.ones(len(pairs),bool)
    for _ in range(8):
        rot,shift=fit(tx[keep],ry[keep]);dist=np.linalg.norm(tx@rot+shift-ry,axis=1)
        new=dist<3.0
        if np.array_equal(new,keep):break
        if new.sum()<min_aligned:raise ValueError('Insufficient conserved core for assembly transfer')
        keep=new
    rmsd=float(np.sqrt(np.mean(dist[keep]**2)))
    if keep.sum()<min_aligned or rmsd>max_rmsd:raise ValueError('Assembly transfer fit failed')
    r0,t0=rot,shift
    lines=Path(monomer).read_text().splitlines()
    protein=[ln for ln in lines if ln.startswith('ATOM  ')]
    ligands=[ln for ln in lines if ln.startswith('HETATM')]
    ligand_xyz=np.array([[float(ln[30:38]),float(ln[38:46]),float(ln[46:54])] for ln in ligands])
    prot_xyz=np.array([[float(ln[30:38]),float(ln[38:46]),float(ln[46:54])] for ln in protein])
    chain_labels='AFGHIJKLMNOPQRSTUVWXYZ' # preserve original ligand chains B--E
    result=[];records=[];serial=0
    for k,copy in enumerate(copies):
        cr=residues(copy)
        if len(cr)!=len(rr):raise ValueError('Template copies have different residue maps')
        cy=np.array([cr[j]['CA'].coord for i,j in pairs])
        rk,tk=fit(tx[keep],cy[keep]);xyz=(prot_xyz@rk+tk-t0)@r0.T
        distances=np.linalg.norm(xyz[:,None]-ligand_xyz[None,:],axis=2)
        near=float(distances.min())
        include=k==0 or near<neighbor_distance
        records.append(dict(copy=k,min_ligand_distance_A=near,included=include))
        if not include:continue
        chain=chain_labels[k]
        for ln,v in zip(protein,xyz):
            serial+=1;result.append(f'{ln[:6]}{serial:5d}{ln[11:21]}{chain}{ln[22:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{ln[54:]}')
        result.append('TER')
    for ln in ligands:
        serial+=1;result.append(f'{ln[:6]}{serial:5d}{ln[11:]}')
    result.append('END');out=Path(out);out.write_text('\n'.join(result)+'\n')
    d=dict(target=str(Path(monomer).resolve()),template=str(Path(template).resolve()),
        target_sha256=sha256(monomer),template_sha256=sha256(template),output_sha256=sha256(out),
        target_sequence_sha256=__import__('hashlib').sha256(seq.encode()).hexdigest(),
        n_aligned=len(pairs),n_core=int(keep.sum()),core_rmsd_A=rmsd,template_copies=len(copies),copies=records,
        template_to_target_rotation=r0.T.tolist(),template_to_target_translation=(-t0@r0.T).tolist(),
        status='template_supported_assembly_hypothesis',
        limitations=['local neighbor subset of template assembly','requires interface and clash validation',
                     'sequence identity does not establish oligomeric state'])
    atomic_json(out.with_suffix('.assembly.json'),d);return d


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--monomer',required=True);p.add_argument('--template',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(build_assembly(a.monomer,a.template,a.out),indent=2))
