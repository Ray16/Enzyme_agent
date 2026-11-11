import numpy as np
import pytest

Chem=pytest.importorskip('rdkit.Chem')
from rdkit.Chem import AllChem
from enzyme_barrier.ligands import reactive_atoms,deprotonate_adp,native_glutamyl_phosphate
from enzyme_barrier.components import verify_parameterization


def molecule(smiles):
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol,randomSeed=117)==0
    mol=Chem.RemoveHs(mol)
    for a in mol.GetAtoms():a.SetProp('name',f'{a.GetSymbol()}{a.GetIdx()+1}')
    return mol


def test_native_analogue_preserves_activated_pose_and_L_configuration():
    mol=molecule('O=C([O-])CCCC(=O)OP(=O)([O-])[O-]')
    reactive=reactive_atoms(mol);native,record=native_glutamyl_phosphate(mol)
    assert Chem.GetFormalCharge(native)==-2
    assert record['configuration']=='S'
    Chem.AssignStereochemistry(native,cleanIt=True,force=True)
    assert native.GetAtomWithIdx(record['alpha_atom']).GetProp('_CIPCode')=='S'
    assert np.array_equal(native.GetConformer().GetPositions()[:mol.GetNumAtoms()],mol.GetConformer().GetPositions())
    assert reactive_atoms(native)==reactive


def test_adp_deprotonation_is_explicit_and_charge_checked():
    mol=molecule('OP(=O)(O)OP(=O)(O)OC')
    adp,sites=deprotonate_adp(mol)
    assert len(sites)==3 and Chem.GetFormalCharge(adp)==-3
    assert len([a for a in adp.GetAtoms() if a.GetAtomicNum()==8 and a.GetFormalCharge()==-1])==3
    with pytest.raises(ValueError):deprotonate_adp(molecule('OP(=O)(O)OC'))


def parameter_files(tmp_path):
    mol=Chem.AddHs(molecule('O=C([O-])CCCC(=O)OP(=O)([O-])[O-]'),addCoords=True)
    sdf=tmp_path/'ligand.sdf';writer=Chem.SDWriter(str(sdf));writer.write(mol);writer.close()
    atoms=[dict(index=a.GetIdx(),element=a.GetSymbol(),atomic_number=a.GetAtomicNum(),formal_charge=a.GetFormalCharge(),aromatic=a.GetIsAromatic(),neighbors=[v.GetIdx() for v in a.GetNeighbors()]) for a in mol.GetAtoms()]
    component=dict(sdf=str(sdf),resname='ACP',charge=Chem.GetFormalCharge(mol),atoms=atoms,reactive_atoms=reactive_atoms(mol))
    xyz=mol.GetConformer().GetPositions();rows=[]
    for a in mol.GetAtoms():
        z=a.GetAtomicNum();typ={1:'h1',6:'c3',8:'os',15:'p5'}[z]
        if z==6 and any(b.GetBondTypeAsDouble()==2 for b in a.GetBonds()):typ='c'
        if z==8 and len(a.GetNeighbors())==1:typ='o'
        rows.append(f'{a.GetIdx()+1} {a.GetSymbol()}{a.GetIdx()+1} '+' '.join(f'{v:.4f}' for v in xyz[a.GetIdx()])+f' {typ} 1 ACP {a.GetFormalCharge()}')
    bonds=[f'{b.GetIdx()+1} {b.GetBeginAtomIdx()+1} {b.GetEndAtomIdx()+1} {int(b.GetBondTypeAsDouble())}' for b in mol.GetBonds()]
    def write(bonds_override=None,rows_override=None):
        path=tmp_path/'ligand.mol2';path.write_text('@<TRIPOS>ATOM\n'+'\n'.join(rows_override or rows)+'\n@<TRIPOS>BOND\n'+'\n'.join(bonds_override or bonds)+'\n');return path
    return mol,component,rows,bonds,write


def test_parameterization_resonance_does_not_hide_carbonyl_or_charge_changes(tmp_path):
    mol,component,rows,bonds,write=parameter_files(tmp_path)
    assert verify_parameterization(component,write())['charge_sum']==-3
    changed=bonds.copy()
    # Delocalized terminal COO/P=O notation is allowed only in the declared
    # anionic groups and with retained GAFF c/p5/o types.
    for b in mol.GetBonds():
        if b.GetBondTypeAsDouble()!=2:continue
        if b.GetBeginAtomIdx()==component['reactive_atoms']['carbonyl'] or b.GetEndAtomIdx()==component['reactive_atoms']['carbonyl']:continue
        changed[b.GetIdx()]=' '.join(changed[b.GetIdx()].split()[:3]+['1'])
    assert verify_parameterization(component,write(changed))['resonance_representation_adjustments']
    carbonyl=mol.GetBondBetweenAtoms(component['reactive_atoms']['carbonyl'],component['reactive_atoms']['carbonyl_oxygen']).GetIdx()
    changed[carbonyl]=' '.join(changed[carbonyl].split()[:3]+['1'])
    with pytest.raises(ValueError,match='Bond order|double bond'):verify_parameterization(component,write(changed))
    corrupted=rows.copy();parts=corrupted[0].split();parts[-1]=str(float(parts[-1])+1);corrupted[0]=' '.join(parts)
    with pytest.raises(ValueError,match='charge'):verify_parameterization(component,write(rows_override=corrupted))
    bad=bonds.copy();bad[-1]=' '.join(bad[-1].split()[:3]+['ar'])
    with pytest.raises(ValueError,match='Aromatic'):verify_parameterization(component,write(bad))
    nonfinite=rows.copy();parts=nonfinite[0].split();parts[-1]='nan';nonfinite[0]=' '.join(parts)
    with pytest.raises(ValueError,match='charge'):verify_parameterization(component,write(rows_override=nonfinite))
    parts=rows[0].split();parts[2]='nan';nonfinite[0]=' '.join(parts)
    with pytest.raises(ValueError,match='Nonfinite'):verify_parameterization(component,write(rows_override=nonfinite))
