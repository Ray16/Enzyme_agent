"""Fixed QM selection and conservative link coordinates for embedded pilots.

Protein sidechains are cut only across CA--CB single bonds. Proline, disulfides,
backbone-coordinated metals, and unknown residues require explicit review.
"""
import numpy as np

BACKBONE = {'N','H','H1','H2','H3','CA','HA','HA2','HA3','C','O','OXT'}
NEUTRAL = {'ALA','ASN','CYS','GLN','GLY','HID','HIE','ILE','LEU','MET','PHE','SER','THR','TRP','TYR','VAL','ASH','GLH','LYN'}
FORMAL = {**dict.fromkeys(NEUTRAL, 0), 'ARG':1, 'LYS':1, 'HIP':1, 'ASP':-1, 'GLU':-1}
WATER = {'WAT','HOH','TIP3'}


def build_region(parm, coordinates, reaction_atoms, ligand_charges, radius=4.0,
                 embedding_radius=20.0, mobile_radius=8.0,force_residues=(),
                 cap_charge_delete_A=None):
    xyz = np.asarray(coordinates, float)
    center = xyz[list(reaction_atoms)]
    # Whole ligands + complete metal coordination shells, then catalytic sidechains.
    selected, entries = set(), []
    forced=set(force_residues)
    if not forced<=set(range(len(parm.residues))):raise ValueError('Forced QM residue index out of range')
    metals = [a.idx for a in parm.atoms if a.atomic_number == 12]
    shell_res = set()
    for m in metals:
        for a in parm.atoms:
            if a.atomic_number in (7,8,16) and np.linalg.norm(xyz[a.idx]-xyz[m]) < 2.7:
                shell_res.add(a.residue.idx)
    for r in parm.residues:
        heavy = [a.idx for a in r.atoms if a.atomic_number > 1]
        if not heavy: continue
        near = np.linalg.norm(xyz[heavy][:,None]-center[None,:],axis=2).min() <= radius
        if r.name in ligand_charges:
            ids = [a.idx for a in r.atoms]; charge = ligand_charges[r.name]; mode='whole'
        elif r.name in WATER and (near or r.idx in shell_res):
            ids = [a.idx for a in r.atoms]; charge=0; mode='whole'
        elif r.name in FORMAL and (near or r.idx in shell_res or r.idx in forced):
            ids = [a.idx for a in r.atoms if a.name not in BACKBONE]
            charge=FORMAL[r.name]; mode='sidechain'
            if r.idx in shell_res:
                for a in r.atoms:
                    if a.name in BACKBONE and a.atomic_number in (7,8) and any(np.linalg.norm(xyz[a.idx]-xyz[m]) < 2.7 for m in metals):
                        raise ValueError(f'Backbone metal ligand in residue {r.idx}: review QM boundary')
        elif r.idx in shell_res or r.idx in forced or (near and r.name in {'PRO','CYX','HIS'}):
            raise ValueError(f'Residue {r.idx}:{r.name} needs an explicit QM boundary/protonation decision')
        else: continue
        if not ids: continue
        if any(parm.atoms[i].atomic_number == 0 for i in ids):
            raise ValueError('Virtual sites in QM region are unsupported')
        selected.update(ids); entries.append(dict(resid=r.idx,resname=r.name,mode=mode,charge=int(charge),indices=ids))
    indices=sorted(selected); local={i:j for j,i in enumerate(indices)}
    if not set(reaction_atoms) <= selected: raise ValueError('Reaction atoms missing from QM selection')
    links=[]
    for b in parm.bonds:
        i,j=b.atom1.idx,b.atom2.idx
        if (i in selected)==(j in selected): continue
        q,m=(i,j) if i in selected else (j,i)
        if parm.atoms[q].name!='CB' or parm.atoms[m].name!='CA':
            raise ValueError(f'Unsupported covalent QM boundary {q}-{m}; only CB--CA cuts allowed')
        links.append(dict(qm=q,mm=m,local=local[q],ratio=float(1.09/np.linalg.norm(xyz[q]-xyz[m]))))
    qtot=sum(e['charge'] for e in entries)
    nelec=sum(parm.atoms[i].atomic_number for i in indices)+len(links)-qtot
    if nelec%2: raise ValueError('Odd-electron QM region at singlet multiplicity')
    # MM charges: conserve the original system charge after replacing partial QM
    # charges by integer QM charge. Zero the boundary CA charge and move it onto
    # the residue's remaining MM backbone sites, away from the link hydrogen.
    charges=np.array([a.charge for a in parm.atoms],float)
    for e in entries:
        ids=e['indices']; delta=float(charges[ids].sum()-e['charge'])
        charges[ids]=0
        boundary=[l['mm'] for l in links if l['qm'] in ids]
        delta+=float(charges[boundary].sum());charges[boundary]=0
        recipients=[a.idx for a in parm.residues[e['resid']].atoms if a.idx not in selected and a.idx not in boundary]
        if recipients: charges[recipients]+=delta/len(recipients)
        elif abs(delta)>0.005: raise ValueError(f'Noninteger whole-residue charge: {e}')
    # Amber/BCC libraries may have millielectron decimal-rounding residuals;
    # integer molecular charges are authoritative, and the discrepancy is logged.
    residual=float(charges.sum()+qtot-sum(a.charge for a in parm.atoms))
    if abs(residual)>0.005:
        raise ValueError('QM/MM boundary charge conservation failed')
    # Optional boundary safety for diffuse-basis energies: the Z1 scheme above
    # zeroes only the M1 (CA) charge, leaving the CA-bonded backbone N (~-0.45 e)
    # and C (~+0.6 e) ~1.7 A from the link-H cap. A diffuse QM basis spills
    # electron density onto such close point charges. When cap_charge_delete_A is
    # set, delete every MM point charge within that distance of any cap and spread
    # the removed charge uniformly over the remaining non-QM atoms, conserving the
    # total. This presumes the full neutral cell is embedded so the redistributed
    # charge is retained; it is logged, not silently applied (default: off).
    cap_deleted=[]
    if cap_charge_delete_A:
        caps=capped_coordinates(xyz, dict(indices=indices, links=links))[len(indices):]
        if len(caps):
            nonqm=np.array([a.idx for a in parm.atoms if a.idx not in selected])
            dcap=np.linalg.norm(xyz[nonqm][:,None,:]-caps[None,:,:],axis=2).min(axis=1)
            near=nonqm[dcap<=cap_charge_delete_A]
            removed=float(charges[near].sum()); charges[near]=0.0
            cap_deleted=sorted(int(i) for i in near)
            pool=np.array([i for i in nonqm if i not in set(cap_deleted)])
            if len(pool): charges[pool]+=removed/len(pool)
            if abs(float(charges.sum()+qtot-sum(a.charge for a in parm.atoms)))>0.005:
                raise ValueError('cap-charge deletion broke charge conservation')
    emb=[];mobile=set(indices)
    qxyz=xyz[indices]
    for r in parm.residues:
        ids=[a.idx for a in r.atoms if a.idx not in selected]
        if not ids:continue
        distance=np.linalg.norm(xyz[ids][:,None]-qxyz[None,:],axis=2).min()
        if distance<=embedding_radius: emb.extend(ids)
        if distance<=mobile_radius: mobile.update(ids)
    emb=sorted(i for i in emb if abs(charges[i])>1e-12)
    return dict(schema_version=1,indices=indices,links=links,charge=qtot,multiplicity=1,
                entries=entries,embedding_indices=emb,embedding_charges=charges[emb].tolist(),
                mobile_indices=sorted(mobile),radius_A=radius,embedding_radius_A=embedding_radius,
                mobile_radius_A=mobile_radius,reaction_atoms=list(reaction_atoms),
                forced_protein_residues_zero_based=sorted(forced),
                cap_charge_delete_A=cap_charge_delete_A,
                cap_deleted_mm_indices=cap_deleted,
                mm_charge_updates={str(i):float(charges[i]) for i,a in enumerate(parm.atoms) if abs(charges[i]-a.charge)>1e-12},
                net_mm_charge=float(charges.sum()),original_net_charge=float(sum(a.charge for a in parm.atoms)),
                charge_rounding_residual_e=residual,
                limitations=['fixed embedding membership; require radius sensitivity',
                             'finite nonperiodic electrostatic QM environment'])


def capped_coordinates(positions, region):
    p=np.asarray(positions,float); ids=region['indices']
    caps=[(1-l['ratio'])*p[l['qm']]+l['ratio']*p[l['mm']] for l in region['links']]
    return np.vstack([p[ids], np.asarray(caps).reshape(-1,3)]) if caps else p[ids].copy()


def project_link_forces(forces, region, natoms):
    out=np.zeros((natoms,3)); n=len(region['indices'])
    out[region['indices']]+=np.asarray(forces)[:n]
    for k,l in enumerate(region['links']):
        out[l['qm']]+=(1-l['ratio'])*forces[n+k]
        out[l['mm']]+=l['ratio']*forces[n+k]
    return out
