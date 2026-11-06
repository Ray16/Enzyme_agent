"""Prepare a separate, source-linked pocket-localization arm from saved cofolds."""
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem

from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from .family_cofold import digest, file_hash, write_input_index
from .paired_analysis import carbonyl_atom_references
from .validation import _read_structure


def coordinate(atoms, chain, name):
    matches = [value['xyz'] for key, value in atoms.items() if key[0] == chain and key[2] == name]
    if len(matches) != 1:
        raise ValueError('Ambiguous mapped pocket anchor')
    return matches[0]


def activation_pocket(saved, radius=7.0):
    """Locate the activated end, using the saved first-step atom map and pose."""
    job = saved.get('reuse_source_job') or saved['job']
    manifest = json.loads(Path(saved['receipt']['prediction_manifest']).read_text())
    if manifest['input_sha256'] != job['input_sha256']:
        raise ValueError('Activation input identity changed')
    eligible = [pose for pose in saved['analysis']['poses'] if not pose['identity_issues']]
    if not eligible:
        raise ValueError('No identity-verified activation reference')
    pose = min(eligible, key=lambda p: (p['distance_A'], p['path']))
    path = Path(pose['path'])
    if file_hash(path) != pose['sha256']:
        raise ValueError('Activation reference changed')
    if pose['distance_A'] > 10:
        raise ValueError('Activation reference does not localize the carboxylate near the reactive phosphate')
    atoms, sequences = _read_structure(path)
    chain, name = pose['nucleophile']
    definition = manifest['ligand_atom_maps'][chain]
    molecule = Chem.MolFromSmiles(definition['smiles'])
    oxygen_index = next(int(i) for i, atom in definition['atoms'].items() if atom['name'] == name)
    oxygen = molecule.GetAtomWithIdx(oxygen_index)
    carbons = [a for a in oxygen.GetNeighbors() if a.GetAtomicNum() == 6]
    if oxygen.GetAtomicNum() != 8 or len(carbons) != 1:
        raise ValueError('Activation donor is not an unambiguous carboxyl oxygen')
    carbon = carbons[0]
    if len([a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8]) != 2:
        raise ValueError('Pocket anchor is not the reactive carboxyl carbon')
    carbon_name = definition['atoms'][str(carbon.GetIdx())]['name']
    center = coordinate(atoms, chain, carbon_name)
    near = {}
    for (protein_chain, number, atom_name), atom in atoms.items():
        if not atom['protein']:
            continue
        if not number.isdigit() or not 1 <= int(number) <= len(sequences[protein_chain]):
            raise ValueError('Pocket residue cannot be mapped to the full input sequence')
        distance = float(np.linalg.norm(atom['xyz'] - center))
        if distance <= radius:
            key = (protein_chain, int(number));near[key] = min(near.get(key, float('inf')), distance)
    contacts = sorted(key for key in sorted(near, key=lambda key: (near[key], key))[:20])
    if len(contacts) < 3:
        raise ValueError('Fewer than three mapped protein residues surround the activation site')
    return {'contacts': [list(value) for value in contacts], 'reference_pdb': str(path),
        'reference_sha256': pose['sha256'], 'reactive_carboxyl_carbon': [chain, carbon_name],
        'activation_distance_A': pose['distance_A'], 'contact_radius_A': radius,
        'basis': 'Model-derived activated-end pocket from an identity-verified first-step pose; conditional localization test, not an experimentally validated binding site.'}


def prepare(run, source, out, max_distance=10.0):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if not 4 <= max_distance <= 20:
        raise ValueError('Broad pocket distance must be between 4 and 20 angstroms')
    signature=digest({'queue':file_hash(source/'queue.json'),'max_distance':max_distance,
        'results':[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(source.glob('jobs/*/complete.json'))],
        'implementation':file_hash(__file__)})
    with state_lock(out/'pocket_preparation'):
        saved=out/'pocket_preparation.json'
        if saved.exists() and json.loads(saved.read_text()).get('signature')==signature:
            return json.loads((out/'queue.json').read_text())['items']
        jobs=_prepare(run,source,out,max_distance)
        atomic_write_json(saved,{'signature':signature,'source':str(source),'setup_llm_calls':0})
        return jobs


def _prepare(run, source, out, max_distance=10.0):
    """One shared decision rule per family; resolve each member's residue mapping."""
    run = Path(run).resolve();source = Path(source).resolve();out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True);(out / 'inputs').mkdir(exist_ok=True)
    original = {job['id']: job for job in json.loads((source / 'queue.json').read_text())['items']}
    members = defaultdict(dict)
    for job in original.values():
        members[job['accession']][job['step']] = job
    jobs = [];decisions = [];families = defaultdict(list)
    for accession, stages in members.items():
        if 1 not in stages or 2 not in stages:
            continue
        first, second = stages[1], stages[2]
        first_result = source / 'jobs' / first['id'] / 'complete.json'
        second_result = source / 'jobs' / second['id'] / 'complete.json'
        decision = {'enzyme_name': second['enzyme_name'], 'accession': accession, 'family_id': second['family_id']}
        if not first_result.exists() or not second_result.exists():
            decisions.append({**decision, 'status': 'pending', 'reason': 'Complete unrestrained references are not yet available.'});continue
        first_saved = json.loads(first_result.read_text());second_saved = json.loads(second_result.read_text())
        if second_saved['analysis']['geometry']['harmonic_distance_A'] <= max_distance:
            decisions.append({**decision, 'status': 'retain_unrestrained',
                'reason': 'No broad localization rescue selected: the existing attack distance is already within the proposed broad pocket scale. This is not an activity claim.'});continue
        try:
            pocket = activation_pocket(first_saved)
            producer_job = second_saved.get('reuse_source_job') or second_saved['job']
            manifest = json.loads(Path(second_saved['receipt']['prediction_manifest']).read_text())
            carbon, _, donors = carbonyl_atom_references(producer_job, manifest)
            # The same numbered pocket residues must exist in the second-step protein.
            poses = second_saved['analysis']['poses'];reference = poses[0]
            if file_hash(reference['path']) != reference['sha256']:
                raise ValueError('Second-step reference changed')
            atoms, sequences = _read_structure(Path(reference['path']))
            for chain, residue in pocket['contacts']:
                if (chain, str(residue), 'CA') not in atoms:
                    raise ValueError('Activation-pocket residue mapping is unavailable in step two')
            pocket_atoms = np.array([atom['xyz'] for key, atom in atoms.items()
                if atom['protein'] and [key[0], int(key[1])] in pocket['contacts']])
            acyl_distance = float(np.linalg.norm(pocket_atoms - coordinate(atoms, *carbon), axis=1).min())
            if acyl_distance > max_distance:
                raise ValueError('Activated intermediate is outside the mapped pocket; its placement requires investigation before amine-only guidance')
            if file_hash(second['input_yaml']) != second['input_sha256'] or file_hash(second['msa_path']) != second['msa_sha256']:
                raise ValueError('Source input or prepared MSA changed')
            document = yaml.safe_load(Path(second['input_yaml']).read_text())
            if document.get('constraints'):
                raise ValueError('Source already has restraints; do not silently stack guidance')
            binder = producer_job['entities'][producer_job['modeling_input']['geometry']['nucleophile']['role']]
            document['constraints'] = [{'pocket': {'binder': binder, 'contacts': pocket['contacts'],
                                                   'max_distance': max_distance, 'force': True}}]
            content = yaml.safe_dump(document, sort_keys=False)
            sha = hashlib.sha256(content.encode()).hexdigest()
            identifier = digest({'source_job': second['id'], 'input_sha256': sha, 'arm': 'pocket_localization_v1'})
            input_path = out / 'inputs' / (identifier + '.yaml')
            if input_path.exists() and file_hash(input_path) != sha:
                raise ValueError('Existing guided input changed')
            if not input_path.exists():
                input_path.write_text(content)
            guided = copy.deepcopy(second)
            guided.update(id=identifier, input_yaml=str(input_path), input_sha256=sha,
                analysis_arm='pocket_guided_localization', source_unrestrained_job=second['id'],
                pocket_protocol={'max_distance_A': max_distance, 'force': True, **pocket,
                    'step_2_intermediate_distance_to_pocket_A': acyl_distance,
                    'reactive_atom_distance_restraint': False, 'attack_angle_restraint': False})
            guided['modeling_input']['scoring_policy'] = {'restraints': 'broad_pocket_guided',
                'scope': 'Conditional localization rescue; retain separately from unrestrained results.'}
            jobs.extend([first, guided]);families[second['family_id']].append(accession)
            copy_to = out / 'jobs' / first['id'] / 'complete.json';copy_to.parent.mkdir(parents=True, exist_ok=True)
            if not copy_to.exists():
                atomic_write_json(copy_to, {**first_saved, 'reuse_source': str(first_result)})
            decisions.append({**decision, 'status': 'ready', 'guided_job': identifier, 'pocket': pocket,
                'reason': 'Distant unrestrained amine; first-step pocket is mapped and the activated intermediate is present in that region. Test broad amine localization without forcing reaction geometry.'})
        except (ValueError, KeyError, OSError, StopIteration) as exc:
            decisions.append({**decision, 'status': 'pending', 'reason': str(exc)})
    atomic_write_json(out / 'queue.json', {'items': jobs, 'pending': [d for d in decisions if d['status'] == 'pending'], 'setup_llm_calls': 0})
    atomic_write_json(out / 'pocket_decisions.json', {'discovery_run_id': run.name, 'source': str(source),
        'shared_policy': {'max_distance_A': max_distance, 'force': True, 'contact_radius_A': 7,
            'reactive_distance_or_angle_restraints': False, 'source': 'User-requested historical broad-pocket localization protocol'},
        'families': dict(families), 'decisions': decisions, 'setup_llm_calls': 0})
    write_input_index(jobs, out)
    return jobs


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True);parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True);parser.add_argument('--max-distance', type=float, default=10.)
    args = parser.parse_args();jobs = prepare(args.run, args.source, args.out, args.max_distance)
    print(json.dumps({'prepared_ensembles': len(jobs)//2, 'setup_llm_calls': 0}))


if __name__ == '__main__':
    main()
