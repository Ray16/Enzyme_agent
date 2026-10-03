#!/usr/bin/env python3
"""Validate a GS dimer cofold: (1) did the 2nd subunit form the interfacial active site,
i.e. does the NEIGHBOR chain contribute residues to the reactive center the way 1FPY's
neighbor does (res47-53, Ser53' at ~4.3A)?  (2) what is the step-1 in-line attack geometry
now that the pocket is complete, vs the monomer's 152 deg?

Usage: validate_dimer.py <predictions_dir> <pa|pg>   (pg for 2-Mg GS)"""
import sys, os, glob
import numpy as np
sys.path.insert(0, "/homes/rzhu/Enzyme_agent/structural_modeling/scripts")
sys.path.insert(0, "/homes/rzhu/Enzyme_agent/analysis/common")
import step1_nac as s

pred, tag = sys.argv[1], sys.argv[2]

def parse(pdb):
    prot={}; lig=[]
    for ln in open(pdb):
        if ln[:4] not in ("ATOM","HETA"): continue
        rec=ln[:6].strip(); name=ln[12:16].strip(); resn=ln[17:20].strip()
        ch=ln[21]; xyz=np.array([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])])
        elem=ln[76:78].strip() or name[0]
        if resn=="LIG": lig.append((ch,name,elem,xyz))
        else: prot.setdefault(ch,[]).append((name,resn,xyz))
    return prot, lig

rows=[]
for pdb in sorted(glob.glob(f"{pred}/*model_*.pdb")):
    prot,lig=parse(pdb)
    if not lig: continue
    # reactive center = all ligand atoms (ATP + Mg + adipate)
    rc=np.array([a[3] for a in lig])
    chains=list(prot.keys())
    # host chain = protein chain with most atoms within 6A of ligands; neighbor = the other(s)
    def nclose(ch,cut=6.0):
        P=np.array([a[2] for a in prot[ch]])
        return int((np.sqrt(((P[:,None,:]-rc[None,:,:])**2).sum(-1)).min(1)<cut).sum())
    host=max(chains,key=nclose)
    nbrs=[c for c in chains if c!=host]
    # neighbor contribution to reactive center
    def mind_and_count(chs):
        if not chs: return None,0,0,[]
        P=[];meta=[]
        for c in chs:
            for nm,rn,xy in prot[c]: P.append(xy); meta.append((c,rn,nm))
        P=np.array(P); D=np.sqrt(((P[:,None,:]-rc[None,:,:])**2).sum(-1)); m=D.min(1)
        res5=sorted(set((meta[i][0],meta[i][1]) for i in range(len(m)) if m[i]<5))
        return float(m.min()), int((m<8).sum()), int((m<5).sum()), res5
    dmin,n8,n5,res5=mind_and_count(nbrs)
    g=s.geom_model(pdb)
    ang=g.get(f"ang_{tag}") if g else None
    d=g.get(f"d_{tag}") if g else None
    rows.append((os.path.basename(pdb),host,dmin,n8,n5,ang,d,res5))

print(f"{'model':30} {'host':4} {'nbr_dmin':>8} {'n8':>4} {'n5':>4} {'ang':>6} {'dist':>6}")
angs=[]; d_mins=[]; n8s=[]
for r in rows:
    nm,host,dmin,n8,n5,ang,d,res5=r
    print(f"{nm:30} {host:4} {('%.2f'%dmin) if dmin else 'NA':>8} {n8:>4} {n5:>4} "
          f"{('%.1f'%ang) if ang else 'NA':>6} {('%.2f'%d) if d else 'NA':>6}")
    if ang: angs.append(ang)
    if dmin: d_mins.append(dmin); n8s.append(n8)
print(f"\nN poses={len(rows)}")
if d_mins:
    print(f"INTERFACE FORMED? neighbor->reactive-center min dist: median {np.median(d_mins):.2f} A, "
          f"best {min(d_mins):.2f} A; median n8={np.median(n8s):.0f} atoms<8A")
    print(f"  (1FPY reference: 3.5-5.1 A, 47-58 atoms<8A -> native interface)")
if angs:
    ga=[a for a in angs]
    print(f"STEP-1 angle (all poses): mean {np.mean(ga):.1f} deg (monomer panel PuuA was ~152.0)")
# show the neighbor residues that intrude (first pose with contacts)
for r in rows:
    if r[4]>0: print(f"neighbor residues <5A of reactive center ({r[0]}): {r[7]}"); break
