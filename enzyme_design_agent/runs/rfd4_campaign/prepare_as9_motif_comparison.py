"""Author matched M0/M1 inputs from intact WT cofold coordinates.

No GPU work. Preserve historical inputs. Explicit reference microstates, no
invented waters/metal bonds, no coordinate optimization or residue exclusions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


CORE = {
    165: ['CA', 'CB', 'OG1'], 166: ['CA', 'CB', 'OG'],
    167: ['N', 'CA', 'C', 'O'], 168: ['CA', 'CB', 'OG1'],
    169: ['CA', 'CB', 'OG1'], 170: ['CA'], 171: ['CA'],
    172: ['CA'], 173: ['CA'], 301: ['CG', 'CD', 'OE1', 'NE2'],
    392: ['CB', 'CG', 'OD1', 'OD2'], 498: ['CD', 'CE', 'NZ'],
}
ATP_SMILES = 'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O'
ADI_SMILES = 'O=C([O-])CCCCC(=O)[O-]'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    a = p.parse_args()
    project, root = a.project.resolve(), a.outdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'preparation.json').exists():
        raise FileExistsError('Prepared campaign exists; preserve it')
    old = project/'enzyme_design_agent/runs/rfd4_campaigns/as9_motifonly100_20261030'
    archived = old.parent/'archive/as9_activation100_20261030'
    wt = project/'enzyme_design_agent/campaigns/diad_AS9_20261026/review_20261027/cofold_out/WT_AS9/boltz_results_WT_AS9/predictions/WT_AS9'
    shutil.copytree(old/'ccd', root/'ccd', dirs_exist_ok=True)
    os.environ['CCD_MIRROR_PATH'] = str(root/'ccd')
    for name in ('inputs', 'sources', 'figures'):
        (root/name).mkdir(exist_ok=True)
    import numpy as np
    from biotite.structure.io import pdbx
    from rdkit import Chem
    import rfproteina.conditions
    from atomworks.io import parse
    from atomworks.io.tools.rdkit import atom_array_to_rdkit
    from atomworks.io.utils.atom_array_plus import as_atom_array_plus
    from atomworks.io.utils.io_utils import to_cif_file
    from atomworks.ml.conditions import C_CRD, C_SEQ, C_IDX
    from atomworks.ml.utils.expandable_segment import insert_expandable_segment
    from atomworks.enums import ChainType
    from rfproteina.conditions import C_CRDGRP, C_INTCRD, C_IDXGRP, C_HB, C_RAS
    from rfproteina.conditions_reporting import sync_index_groups, run_pipeline_validation, check_index_group_molecule_consistency, check_index_group_res_id_gaps
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.utils.parse_presets import ANNOTATED_CIF

    def load(path):
        return as_atom_array_plus(parse(str(path), config=ANNOTATED_CIF)['assemblies']['1'][0])

    def keys(arr):
        return [(str(c), int(r), str(n)) for c,r,n in zip(arr.chain_id, arr.res_id, arr.atom_name)]

    def smiles(arr):
        mol = atom_array_to_rdkit(arr, attempt_fixing_corrupted_molecules=False)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'C':
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def ligand_graph(arr):
        ks = keys(arr)
        return sorted((min(ks[i],ks[j]), max(ks[i],ks[j]), int(order))
                      for i,j,order in arr.bonds.as_array())

    checks = (run_pipeline_validation, check_index_group_molecule_consistency, check_index_group_res_id_gaps)
    records = []
    for pose in (15, 6):
        source = wt/f'WT_AS9_model_{pose}.pdb'
        coords = {}
        for line in source.read_text().splitlines():
            if line.startswith(('ATOM  ', 'HETATM')):
                k = (line[21].strip(), int(line[22:26]), line[12:16].strip())
                if k in coords:
                    raise ValueError('Duplicate PDB atom')
                coords[k] = [float(line[j:j+8]) for j in (30,38,46)]
        # Borrow complete chemical-component definitions, never protein geometry.
        template = archived/'inputs'/f'pose{pose}.cif'
        cif = pdbx.CIFFile.read(template)
        b = cif.block
        site = b['atom_site']
        ks = list(zip(site['auth_asym_id'].as_array(str), site['auth_seq_id'].as_array(int), site['auth_atom_id'].as_array(str)))
        if len(set(ks)) != len(ks) or set(ks) != set(coords):
            raise ValueError('WT/template atom inventories differ')
        xyz = np.array([coords[k] for k in ks])
        for i, axis in enumerate(('x','y','z')):
            site['Cartn_'+axis] = pdbx.CIFColumn(xyz[:,i])
        for col in list(site):
            if col.startswith(('condition_', 'mask_')):
                del site[col]
        # Coordinate-valued annotations also live in separate CIF categories.
        # The full source intermediate is intentionally unconditioned.
        for category in list(b):
            if category.startswith(('condition_', 'mask_')):
                del b[category]
        # Derive all singly bonded terminal phosphate oxygens from the graph.
        bonds = b['chem_comp_bond']
        cid = bonds['comp_id'].as_array(str)
        n1, n2 = bonds['atom_id_1'].as_array(str), bonds['atom_id_2'].as_array(str)
        order = bonds['value_order'].as_array(str)
        atomcat = b['chem_comp_atom']
        ac = atomcat['comp_id'].as_array(str)
        an = atomcat['atom_id'].as_array(str)
        element = atomcat['type_symbol'].as_array(str)
        # Author custom ligands as complete heavy-atom graphs with implicit H.
        # This removes the old neutral phosphate OH hydrogens from the CCD too.
        keep_atoms = ~(np.isin(ac,['ATP','N66AC']) & (element=='H'))
        b['chem_comp_atom'] = pdbx.CIFCategory({col:atomcat[col].as_array(str)[keep_atoms] for col in atomcat})
        atomcat = b['chem_comp_atom']
        ac = atomcat['comp_id'].as_array(str)
        an = atomcat['atom_id'].as_array(str)
        element = atomcat['type_symbol'].as_array(str)
        elements = {n:e for n,e in zip(an[ac=='ATP'], element[ac=='ATP'])}
        # Keep bonds only between retained component atoms. RDKit determines
        # implicit hydrogens from the charged graph.
        keep_bonds = np.ones(len(cid), dtype=bool)
        for code in ('ATP','N66AC'):
            available = set(an[ac==code])
            keep_bonds &= (cid!=code) | (np.isin(n1,list(available)) & np.isin(n2,list(available)))
        b['chem_comp_bond'] = pdbx.CIFCategory({col:bonds[col].as_array(str)[keep_bonds] for col in bonds})
        neighbors = {}
        for left,right,bo in zip(n1[cid=='ATP'],n2[cid=='ATP'],order[cid=='ATP']):
            if left not in elements or right not in elements:
                continue
            neighbors.setdefault(left,[]).append((right,bo))
            neighbors.setdefault(right,[]).append((left,bo))
        charged_atp = sorted(n for n,e in elements.items() if e=='O' and
            len(neighbors.get(n,[]))==1 and neighbors[n][0][1]=='SING' and
            elements[neighbors[n][0][0]]=='P')
        if len(charged_atp) != 4:
            raise ValueError('Expected four phosphate OH groups')
        charges = atomcat['charge'].as_array(int).copy()
        charges[ac=='ATP'] = 0
        charges[(ac=='ATP') & np.isin(an,charged_atp)] = -1
        charges[ac=='N66AC'] = 0
        charges[(ac=='N66AC') & np.isin(an,['O13','O14'])] = -1
        atomcat['charge'] = pdbx.CIFColumn(charges)
        # Component-local ideal/model coordinates must agree with the new source.
        for oldcode, chain in [('ATP','B'),('N66AC','D'),('MG','C')]:
            mask = ac==oldcode
            for i,axis in enumerate(('x','y','z')):
                col = 'pdbx_model_Cartn_'+axis+'_ideal'
                vals = atomcat[col].as_array(str).astype(object)
                for row in np.flatnonzero(mask):
                    vals[row] = str(coords[(chain,1,str(an[row]))][i])
                atomcat[col] = pdbx.CIFColumn(vals.astype(str))
        # Distinct codes prevent accidental fallback to historical neutral CCDs.
        for category in b:
            cat = b[category]
            for col in list(cat):
                vals = cat[col].as_array(str)
                if np.isin(vals,['ATP','N66AC']).any():
                    vals = vals.astype(object)
                    vals[vals=='ATP'] = 'AT4'
                    vals[vals=='N66AC'] = 'AD2'
                    cat[col] = pdbx.CIFColumn(vals.astype(str))
        formal = site['pdbx_formal_charge'].as_array(str).astype(object)
        for i,(chain,rid,name) in enumerate(ks):
            if chain=='B': formal[i] = str(-1 if name in charged_atp else 0)
            if chain=='D': formal[i] = str(-1 if name in ('O13','O14') else 0)
            if chain=='C': formal[i] = '2'
        site['pdbx_formal_charge'] = pdbx.CIFColumn(formal.astype(str))
        full = root/'sources'/f'pose{pose}_same_pose_charged.cif'
        if full.exists():
            raise FileExistsError(full)
        cif.write(full)
        for code in ('AT4','AD2'):
            cc = pdbx.CIFFile()
            block = pdbx.CIFBlock()
            cc[code] = block
            for catname, idcol in [('chem_comp','id'),('chem_comp_atom','comp_id'),('chem_comp_bond','comp_id')]:
                cat = b[catname]; mask = cat[idcol].as_array(str)==code
                block[catname] = pdbx.CIFCategory({col:cat[col].as_array(str)[mask] for col in cat})
            dest = root/'ccd'/code[0]/code/(code+'.cif')
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Coordinates in the CCD are reference conformers; pose15 is reused.
            if not dest.exists(): cc.write(dest)
        fullarr = load(full)
        expected_smiles = {'B':Chem.MolToSmiles(Chem.MolFromSmiles(ATP_SMILES)),
                           'D':Chem.MolToSmiles(Chem.MolFromSmiles(ADI_SMILES))}
        for chain,target in [('B',-4),('C',2),('D',-2)]:
            if int(fullarr.charge[fullarr.chain_id==chain].sum()) != target:
                raise ValueError('Charge mismatch')
        for chain in ('B','D'):
            if smiles(fullarr[fullarr.chain_id==chain]) != expected_smiles[chain]:
                raise ValueError('Chemical graph/stereo differs from reference: '+chain)
        for arm in ('M0','M1'):
            iid = f'{arm}_pose{pose}'
            tips = {k:list(v) for k,v in CORE.items()}
            if arm=='M1':
                tips.update({310:['N','CA','C','O'],311:['N','CA','C','O']})
            aa = fullarr[(fullarr.chain_id!='A') | np.isin(fullarr.res_id,list(tips))].copy()
            orig = aa.copy()
            aa.set_annotation('chain_type', np.where(aa.chain_id=='A',int(ChainType.POLYPEPTIDE_L),int(ChainType.NON_POLYMER)).astype(np.int8))
            for name in ('is_protein','is_polymer','is_ligand'):
                aa.set_annotation(name, aa.chain_id!='A' if name=='is_ligand' else aa.chain_id=='A')
            aa = insert_expandable_segment(aa,seg_min=516,seg_max=516,atom_idx=0,insert_before=True,new_chain=False)
            protein = aa.chain_id=='A'
            motif = protein & np.isin(aa.res_id,list(tips))
            placeholder = protein & ~motif
            lig = ~protein
            fixed = lig.copy()
            definitions = {}
            for rid,names in tips.items():
                mask = protein & (aa.res_id==rid)
                fixed |= mask & np.isin(aa.atom_name,names)
                definitions[str(rid)] = dict(res_name=str(aa.res_name[mask][0]),fixed_atoms=names,sequence_fixed=rid in CORE)
            C_CRD.set_annotation(aa,np.full((len(aa),3),np.nan))
            C_CRDGRP.set_annotation(aa,np.full(len(aa),-1),on_internal_coord_mismatch='clear')
            C_INTCRD.set_annotation(aa,np.full((len(aa),3),np.nan))
            C_CRD.set_annotation_from_ground_truth(aa,fixed)
            groups = np.full(len(aa),-1);groups[fixed]=0
            C_CRDGRP.set_annotation(aa,groups,on_internal_coord_mismatch='clear')
            C_SEQ.set_annotation_from_ground_truth(aa,lig | (protein & np.isin(aa.res_id,list(CORE))))
            C_IDX.set_annotation(aa,~motif)
            C_HB.set_annotation(aa,C_HB.default_annotation(aa))
            C_RAS.set_annotation(aa,C_RAS.default_annotation(aa))
            sync_index_groups(aa,preserve_manual_edits=False)
            groups = C_IDXGRP.annotation(aa).copy(); group = int(groups.max())+1
            blocks = [list(range(165,174)),[301],[392],[498]]
            if arm=='M1': blocks.append([310,311])
            for residues in blocks:
                groups[motif & np.isin(aa.res_id,residues)] = group;group += 1
            C_IDXGRP.set_annotation(aa,groups)
            observed_anchors = int((C_CRD.mask(aa) & protein).sum())
            observed_identities = int((C_SEQ.mask(aa) & protein & (aa.atom_name=='CA')).sum())
            if observed_anchors != (31 if arm=='M0' else 39) or observed_identities != 12:
                raise ValueError('Actual coordinate/sequence masks differ from agreed design')
            errors = [e for check in checks for e in check(aa)]
            if errors: raise ValueError(errors)
            # Count output scaffold slots after actual pipeline materialization.
            data = build_design_validation_pipeline(expandable_segment_rng=np.random.default_rng(0))({'atom_array':aa.copy()})
            materialized = data['atom_array']
            emitted = (materialized.chain_type==int(ChainType.POLYPEPTIDE_L)) & C_IDX.mask(materialized)
            # Designable UNK slots need not have atoms named CA at this stage.
            emitted_slots = set(zip(map(str,materialized.chain_id[emitted]),map(int,materialized.res_id[emitted])))
            save(root/(iid+'_materialization.json'),dict(
                indexed_protein_slots=len(emitted_slots),atom_count=len(materialized),
                chain_ids=list(map(str,np.unique(materialized.chain_id))),
                indexed_atom_names=list(map(str,np.unique(materialized.atom_name[emitted])))))
            if len(emitted_slots) != 516:
                raise ValueError(f'Expected 516 scaffold positions, found {len(emitted_slots)}')
            path = root/'inputs'/(iid+'.cif')
            if path.exists(): raise FileExistsError(path)
            to_cif_file(aa,str(path),save_standard_annotations=True,chain_disambiguation='chain_iid')
            back = load(path); idx = {k:i for i,k in enumerate(keys(back))}
            if len(idx)!=len(back) or len(back)!=len(aa) or set(idx)!=set(keys(aa)):
                raise ValueError('Atom inventory changed or duplicate keys')
            perm = np.array([idx[k] for k in keys(aa)])
            for i,k in enumerate(keys(aa)):
                if not placeholder[i] and np.linalg.norm(back.coord[perm[i]]-coords[k])>.002:
                    raise ValueError('Reloaded coordinates differ from original WT source')
            for condition in (C_CRD,C_SEQ,C_IDX,C_CRDGRP,C_INTCRD,C_IDXGRP,C_HB,C_RAS):
                x,y = condition.annotation(aa),condition.annotation(back)[perm]
                same = np.allclose(x,y,atol=.002,equal_nan=True) if x.dtype.kind in 'fc' else np.array_equal(x,y)
                if not same: raise ValueError('Condition roundtrip failed: '+str(condition))
            if not np.array_equal(aa.charge,back.charge[perm]): raise ValueError('Charge roundtrip failed')
            if ligand_graph(aa[~protein])!=ligand_graph(back[back.chain_id!='A']): raise ValueError('Ligand bond graph changed')
            errors = [e for check in checks for e in check(back)]
            if errors: raise ValueError(errors)
            for chain in ('B','D'):
                if smiles(back[back.chain_id==chain])!=expected_smiles[chain]: raise ValueError('Ligand stereo/graph roundtrip')
            viewing = path.with_name(iid+'_view.cif')
            to_cif_file(aa[~placeholder],str(viewing),save_standard_annotations=True,chain_disambiguation='chain_iid')
            if max(np.linalg.norm(aa.coord[i]-coords[k]) for i,k in enumerate(keys(aa)) if not placeholder[i]) > .002:
                raise ValueError('Source coordinates changed')
            record = dict(id=iid,arm=arm,pose=pose,input=str(path),sha256=sha(path),
                view=str(viewing),source=str(source),source_sha256=sha(source),
                chemical_template=str(template),chemical_template_sha256=sha(template),
                same_pose_coordinates=True,target_length=516,materialized_scaffold_positions=516,
                fixed_sequence_residues=12,anchored_residues=len(tips),fixed_protein_atom_count=int((fixed&protein).sum()),
                motif_fixed_atoms=definitions,index_groups=blocks,unindexed_residues=sorted(tips),
                ligand_charges={'B':-4,'C':2,'D':-2},ATP_negative_oxygen_names=charged_atp,
                ligand_smiles=expected_smiles,nonmotif_coordinate_constraints=0,
                donor_sequence_policy='310/311 identities free; reject mapped311 PRO after generation' if arm=='M1' else None,
                magnesium_hydration='unresolved and omitted; no new coordination bonds or waters',
                scientific_scope='Exploratory activation encounter; no optimization or transition-state claim',
                validation='Three pipeline/index checks before and after save, exact condition/graph/charge roundtrip, carbon stereochemistry and materialized length passed')
            save(path.with_suffix('.metadata.json'),record);records.append(record)
            print(iid,'validated',flush=True)
    save(root/'preparation.json',dict(status='validated_inputs',inputs=records,expected_total_draws=100,
        draws_per_arm_pose=25,target_length=516,pro311_policy='Reject and retain; no automatic resampling',
        chemistry_sources=['https://www.ebi.ac.uk/chebi/CHEBI:30616','https://pubchem.ncbi.nlm.nih.gov/compound/200164'],
        generation_status='not_submitted'))


if __name__ == '__main__':
    main()
