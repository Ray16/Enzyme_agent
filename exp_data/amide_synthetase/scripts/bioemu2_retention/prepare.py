"""Freeze a label-blind, matched panel test using existing cofolds and MSAs."""
import argparse
import csv
import json
from pathlib import Path
import pickle

import networkx as nx
import yaml
from biotite.structure.io import load_structure
from bioemu2 import System, map_cif_atom_order, save_frames

from core import digest, write_json, frame_metrics

BASE = Path(__file__).resolve().parents[2]


def map_start(system, pdb, mol_path):
    # These are trusted local Boltz output pickles, never remote user uploads.
    molecules = list(pickle.loads(mol_path.read_bytes()).values())
    assert len(molecules) == 1
    mol = molecules[0]
    source = nx.Graph()
    for atom in mol.GetAtoms():
        source.add_node(atom.GetIdx(), element=atom.GetSymbol(), name=atom.GetProp('name'))
    for bond in mol.GetBonds():
        source.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    target = nx.Graph()
    for atom in system.topology.atoms:
        if atom.residue.chain.chain_id == 'D':
            target.add_node(atom.index, element=atom.element.symbol, name=atom.name)
    for a, b in system.topology.bonds:
        if a.index in target and b.index in target:
            target.add_edge(a.index, b.index)
    match = nx.algorithms.isomorphism.GraphMatcher(source, target,
        node_match=lambda a, b: a['element']==b['element'])
    if not match.is_isomorphic():
        raise ValueError('Substrate bond graphs differ')
    mapping = {source.nodes[i]['name']:target.nodes[j]['name'] for i,j in match.mapping.items()}
    ref = load_structure(str(pdb))
    for chain, label in [('B','ATP'),('C','MG'),('D','UNK')]:
        ref.res_name[ref.chain_id == chain] = label
    old = ref.atom_name.copy()
    for i in range(len(ref)):
        if ref.chain_id[i]=='D':
            ref.atom_name[i] = mapping[old[i]]
    return map_cif_atom_order(system, ref), mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if (out/'manifest.json').exists():
        raise ValueError('Preparation is immutable; use the existing manifest')
    out.mkdir(parents=True, exist_ok=True)
    activities = BASE/'assay_data/panel_activity.csv'
    assay = {r['id']:r for r in csv.DictReader(activities.open())}
    tasks, files = [], []
    for n in range(1,49):
        e = f'AS{n}'
        root = BASE/'results/modeling/boltz2/step1_1mg'
        input_path = root/f'inputs/step1_{e}.yaml'
        config = yaml.safe_load(input_path.read_text())
        protein = config['sequences'][0]['protein']
        if protein['sequence'] != assay[e]['sequence']:
            raise ValueError('Assay sequence mismatch: '+e)
        ligands = [r['ligand'] for r in config['sequences'][1:]]
        if [(x['id'], x.get('ccd'), x.get('smiles')) for x in ligands] != [
                ('B','ATP',None), ('C','MG',None), ('D',None,'[O-]C(=O)CCCCC(=O)[O-]')]:
            raise ValueError('Unmatched molecular setup: '+e)
        dest = out/'inputs'/e
        dest.mkdir(parents=True, exist_ok=True)
        spec = {'entities':[{'entity_type':'protein','asym_ids':['A'],'sequence':protein['sequence']},
                {'entity_type':'ligand','asym_ids':['B'],'ccd_code':'ATP'},
                {'entity_type':'ligand','asym_ids':['C'],'ccd_code':'MG'},
                {'entity_type':'ligand','asym_ids':['D'],'smiles':ligands[-1]['smiles']}],
                'bonds':[], 'asym_id_to_author_id':{}}
        system_path = dest/'system.yaml'
        write_json(system_path, spec)
        system = System.from_yaml(system_path)
        msa_path = Path(protein['msa'])
        with msa_path.open() as f:
            rows = list(csv.DictReader(f))[:128]
        if rows[0]['sequence'] != protein['sequence']:
            raise ValueError('MSA query mismatch: '+e)
        a3m = dest/'msa.a3m'
        a3m.write_text(''.join(f'>row_{i}\n{r["sequence"]}\n' for i,r in enumerate(rows)))
        for path in [input_path, msa_path, system_path, a3m]:
            files.append({'path':str(path),'sha256':digest(path)})
        bundle = root/f'results/{e}/boltz_results_step1_{e}'
        mol = bundle/f'processed/mols/step1_{e}.pkl'
        files.append({'path':str(mol),'sha256':digest(mol)})
        for pose in [0,25]:
            source = bundle/f'predictions/step1_{e}/step1_{e}_model_{pose}.pdb'
            x0, mapping = map_start(system, source, mol)
            initial = dest/f'pose_{pose}'
            save_frames(x0, initial)
            baseline = frame_metrics(x0,x0)
            write_json(initial/'mapping.json', {'source':str(source),'source_sha256':digest(source),
                'substrate_atom_map':mapping,'mapping_basis':'element-labeled bond graph; substrate symmetry allowed',
                'n_atoms':x0.n_atoms,'sequence_length':len(protein['sequence']),'baseline':baseline})
            for path in [source,initial/'topology.cif',initial/'samples.xtc',initial/'mapping.json']:
                files.append({'path':str(path),'sha256':digest(path)})
            for lag in [1,10]:
                task_id = f'{e}_pose{pose}_lag{lag}'
                tasks.append(dict(id=task_id,enzyme=e,pose=pose,lag_ns=lag,
                    seed=2026110300+n*100+pose*2+lag,
                    system=str(system_path),msa=str(a3m),initial=str(initial/'topology.cif'),
                    sequence_length=len(protein['sequence']),expected_frames=1))
        print('prepared',e,flush=True)
    checkpoint = Path('/nfs/lambda_stor_01/homes/rzhu/.cache/huggingface/hub/models--microsoft--bioemu2/snapshots/dd4b0c1b69984b68edcb6200a544213be4d75242/checkpoints/bioemu2-preview/checkpoint.ckpt')
    for path in [checkpoint,checkpoint.with_name('config.yaml'),activities]:
        files.append({'path':str(path),'sha256':digest(path)})
    manifest = dict(schema='bioemu2-retention-panel-v1',tasks=tasks,files=files,checkpoint=str(checkpoint),
        protocol={'primary_lag_ns':10,'secondary_lag_ns':1,'starting_pose_indices':[0,25],
            'endpoints_per_start_per_lag':1,'msa_rows_max':128,'precision':'32-true','batch_size':1,
            'sampler':'edm','contacts':None,'steering':None,'primary_feature':'native_contact_retention',
            'contact_cutoff_A':4.5,'pocket_cutoff_A':6,'primary_outcome':'continuous measured N66L1 product concentration',
            'secondary_label':'existing strong_producer annotation; not a newly fitted activity threshold',
            'selection':'fixed pose indices, no selection on yield, confidence, or attack geometry',
            'statistical_unit':'enzyme; endpoints and poses are not biological replicates',
            'limitations':['Retrospective exploratory panel; labels have been examined previously.',
                'Two starting poses and two lags per enzyme provide a small pilot, not a residence-time estimate.',
                'Historical one-Mg step-1 setup; outcome reflects complete assay and expression.',
                'Protein physicality does not validate ligand chemistry; extra clash diagnostics are descriptive.',
                'No calibrated equilibrium probabilities, rate constants, or binding free energies.']})
    write_json(out/'manifest.json',manifest)
    print('manifest',out/'manifest.json',len(tasks),'endpoints',flush=True)


if __name__=='__main__':
    main()
