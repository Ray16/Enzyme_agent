"""Authoritative molecular graphs plus named cofold coordinates -> mapped SDF.

Run in the RDKit environment. No bonds are inferred from predicted distances.
Cached RDKit pickles are trusted outputs of this workspace's Boltz jobs, not
arbitrary uploaded files. Each input and resulting chemical state is hashed.
"""
import json
import pickle
from pathlib import Path
import numpy as np
from .jobs import atomic_json,sha256


def reactive_atoms(mol):
    """Activated acyl carbon, carbonyl O and O--P leaving atom, by graph."""
    matches=[]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()!=6:continue
        ox=[bond.GetOtherAtom(atom) for bond in atom.GetBonds() if bond.GetBondTypeAsDouble()==2 and bond.GetOtherAtom(atom).GetAtomicNum()==8]
        bridge=[a for a in atom.GetNeighbors() if a.GetAtomicNum()==8 and any(b.GetAtomicNum()==15 for b in a.GetNeighbors())]
        if len(ox)==1 and len(bridge)==1:matches.append((atom.GetIdx(),ox[0].GetIdx(),bridge[0].GetIdx()))
    if len(matches)!=1:raise ValueError(f'Expected one activated carbonyl; found {len(matches)}')
    return dict(carbonyl=matches[0][0],carbonyl_oxygen=matches[0][1],leaving=matches[0][2])


def deprotonate_adp(mol):
    """Explicit ADP3- model: remove the three terminal phosphate OH protons."""
    from rdkit import Chem
    result=Chem.Mol(mol);sites=[]
    for atom in result.GetAtoms():
        neighbors=list(atom.GetNeighbors())
        if atom.GetAtomicNum()==8 and len(neighbors)==1 and neighbors[0].GetAtomicNum()==15:
            if result.GetBondBetweenAtoms(atom.GetIdx(),neighbors[0].GetIdx()).GetBondTypeAsDouble()==1:
                atom.SetFormalCharge(-1);atom.SetNumExplicitHs(0);atom.SetNoImplicit(True);sites.append(atom.GetProp('name'))
    if len(sites)!=3:raise ValueError(f'ADP phosphate protonation mapping is ambiguous: {sites}')
    Chem.SanitizeMol(result)
    if Chem.GetFormalCharge(result)!=-3:raise ValueError('ADP3- charge not obtained')
    return result,sites


def native_glutamyl_phosphate(mol):
    """Construct the native L-glutamate analogue while retaining the acyl-P pose.

    Replace one alpha-C hydrogen at the free carboxylate end by NH3+, choosing
    the L (S) configuration. This is a pose hypothesis, not measured binding.
    """
    from rdkit import Chem
    hmol=Chem.AddHs(mol,addCoords=True);activated=reactive_atoms(hmol)['carbonyl']
    free=[a for a in hmol.GetAtoms() if a.GetAtomicNum()==6 and a.GetIdx()!=activated and sum(n.GetAtomicNum()==8 for n in a.GetNeighbors())==2]
    if len(free)!=1:raise ValueError('Native analogue requires one free terminal carboxylate')
    alpha=next(a.GetIdx() for a in free[0].GetNeighbors() if a.GetAtomicNum()==6)
    hydrogens=[a.GetIdx() for a in hmol.GetAtomWithIdx(alpha).GetNeighbors() if a.GetAtomicNum()==1]
    if len(hydrogens)!=2:raise ValueError('Expected a terminal alpha CH2 for glutarate analogue')
    for hydrogen in hydrogens:
        candidate=Chem.Mol(hmol);nitrogen=candidate.GetAtomWithIdx(hydrogen)
        nitrogen.SetAtomicNum(7);nitrogen.SetFormalCharge(1);nitrogen.SetNoImplicit(False);nitrogen.SetNumExplicitHs(0);nitrogen.SetProp('name','NAA')
        conf=candidate.GetConformer();center=np.array(conf.GetAtomPosition(alpha));v=np.array(conf.GetAtomPosition(hydrogen))-center
        conf.SetAtomPosition(hydrogen,center+1.49*v/np.linalg.norm(v))
        Chem.SanitizeMol(candidate);Chem.AssignAtomChiralTagsFromStructure(candidate,replaceExistingTags=True);Chem.AssignStereochemistry(candidate,cleanIt=True,force=True)
        if candidate.GetAtomWithIdx(alpha).GetProp('_CIPCode')=='S':
            result=Chem.RemoveHs(candidate);Chem.SanitizeMol(result)
            if Chem.GetFormalCharge(result)!=Chem.GetFormalCharge(mol)+1:raise ValueError('Native analogue charge mismatch')
            return result,dict(alpha_atom=alpha,configuration='S',construction='alpha H replaced by NH3+; original activated-acyl heavy coordinates retained')
    raise ValueError('Could not construct L-glutamyl-phosphate stereochemistry')


def export_components(config_path,out):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    cfg=json.loads(Path(config_path).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
    pose=Path(cfg['pose']);rows={}
    for line in pose.read_text().splitlines():
        if not line.startswith(('ATOM  ','HETATM')):continue
        if line[16].strip():raise ValueError('Resolve alternate atom positions before export')
        key=(line[21],line[12:16].strip())
        if line.startswith('HETATM'):
            if key in rows:raise ValueError('Duplicate ligand atom name within chain')
            rows[key]=(line[76:78].strip().upper(),np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]))
    graphs=pickle.load(Path(cfg['molecules']).open('rb'));ccd=None;reports=[]
    for component in cfg['components']:
        name=component['resname'];key=component['graph'];chain=component['chain']
        if component.get('source')=='ccd':
            if ccd is None:ccd=pickle.load(Path(cfg['ccd']).open('rb'))
            mol=Chem.RemoveHs(Chem.Mol(ccd[key]))
        else:mol=Chem.RemoveHs(Chem.Mol(graphs[key]))
        original_names=[a.GetProp('name') for a in mol.GetAtoms()]
        conf=Chem.Conformer(mol.GetNumAtoms())
        for atom in mol.GetAtoms():
            label=atom.GetProp('name');element,position=rows[(chain,label)]
            if element!=atom.GetSymbol().upper():raise ValueError(f'Element mismatch for {chain}:{label}')
            conf.SetAtomPosition(atom.GetIdx(),position)
        mol.RemoveAllConformers();mol.AddConformer(conf)
        transformations=[]
        if component.get('adp3_minus'):
            mol,sites=deprotonate_adp(mol);transformations.append(dict(action='ADP3- phosphate deprotonation',atoms=sites))
        if component.get('native_glutamate'):
            mol,record=native_glutamyl_phosphate(mol);transformations.append(record)
        if component.get('protonate_reactive_amine'):
            amines=[a for a in mol.GetAtoms() if a.GetAtomicNum()==7 and a.GetFormalCharge()==0]
            if len(amines)!=1:raise ValueError('Reactive amine protonation is ambiguous')
            amines[0].SetFormalCharge(1);mol.UpdatePropertyCache();Chem.SanitizeMol(mol)
            transformations.append(dict(action='reactive amine protonation',atom=amines[0].GetProp('name')))
        if Chem.GetFormalCharge(mol)!=component['charge']:raise ValueError(f'Formal charge mismatch for {name}')
        mol=Chem.AddHs(mol,addCoords=True)
        atoms=[]
        for atom in mol.GetAtoms():
            if not atom.HasProp('name'):
                parent=atom.GetNeighbors()[0].GetProp('name');atom.SetProp('name',f'H{atom.GetIdx()+1}_{parent}')
            atoms.append(dict(index=atom.GetIdx(),source_name=atom.GetProp('name'),element=atom.GetSymbol(),atomic_number=atom.GetAtomicNum(),formal_charge=atom.GetFormalCharge(),aromatic=atom.GetIsAromatic(),neighbors=[n.GetIdx() for n in atom.GetNeighbors()]))
        Chem.CreateAtomStringPropertyList(mol,'name');mol.SetProp('_Name',name)
        writer=Chem.SDWriter(str(out/(name+'.sdf')));writer.write(mol);writer.close()
        reported=dict(component)|dict(sdf=str((out/(name+'.sdf')).resolve()),sdf_sha256=sha256(out/(name+'.sdf')),formula=rdMolDescriptors.CalcMolFormula(mol),canonical_smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)),atoms=atoms,transformations=transformations)
        if component.get('role')=='activated_acid':reported['reactive_atoms']=reactive_atoms(mol)
        if component.get('role')=='amine':
            reported['neutral_nitrogens']=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==7 and a.GetFormalCharge()==0]
        reports.append(reported)
    report=dict(status='molecular_graphs_and_pose_coordinates_exported',config=str(Path(config_path).resolve()),config_sha256=sha256(config_path),pose_sha256=sha256(pose),molecules_sha256=sha256(cfg['molecules']),components=reports,scope=cfg.get('scope','conditional declared microstate; not a pKa calculation'))
    if cfg.get('ccd'):report['ccd_sha256']=sha256(cfg['ccd'])
    atomic_json(out/'components.json',report);return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(export_components(a.config,a.out),indent=2))
