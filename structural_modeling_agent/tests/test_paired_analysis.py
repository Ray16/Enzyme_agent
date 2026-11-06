"""The carbonyl angle uses the carbonyl oxygen, never the leaving oxygen."""
import copy
import math

import pytest
from rdkit import Chem

from structural_modeling_agent.paired_analysis import carbonyl_atom_references, paired_rows


@pytest.fixture
def mapped_system():
    activated = 'CC(=O)OP(=O)([O-])[O-]'
    amine = 'CCN'
    maps = {chain: {'smiles': smiles, 'atoms': {str(atom.GetIdx()): {'name': f'ATOM{atom.GetIdx()}'}
            for atom in Chem.MolFromSmiles(smiles).GetAtoms()}}
        for chain, smiles in [('X', activated), ('Y', amine)]}
    geometry = {'electrophile': {'role': 'activated', 'reactive_atom_indices': [1]},
        'charge_centre': {'role': 'activated', 'reactive_atom_indices': [2]},
        'leaving_group': {'role': 'activated', 'reactive_atom_indices': [3]},
        'nucleophile': {'role': 'amine', 'reactive_atom_indices': [2]}}
    job = {'entities': {'activated': 'X', 'amine': 'Y'}, 'modeling_input': {'geometry': geometry}}
    return job, {'ligand_atom_maps': maps}


def test_atom_identity_distinguishes_double_bonded_oxygen_from_leaving_group(mapped_system):
    job, manifest = mapped_system
    assert carbonyl_atom_references(job, manifest) == (('X', 'ATOM1'), ('X', 'ATOM2'), [('Y', 'ATOM2')])
    broken = copy.deepcopy(job)
    broken['modeling_input']['geometry']['charge_centre']['reactive_atom_indices'] = [3]
    with pytest.raises(ValueError, match='carbonyl C=O'):
        carbonyl_atom_references(broken, manifest)
    broken = copy.deepcopy(job)
    broken['modeling_input']['geometry']['nucleophile']['reactive_atom_indices'] = [0]
    with pytest.raises(ValueError, match='amine nitrogen'):
        carbonyl_atom_references(broken, manifest)


def test_unactivated_other_carboxylate_cannot_be_the_acyl_center(mapped_system):
    job, manifest = mapped_system
    manifest['ligand_atom_maps']['X']['smiles'] = 'CC(=O)O'
    with pytest.raises(ValueError, match='phosphate-activated'):
        carbonyl_atom_references(job, manifest)


def test_known_carbonyl_trajectory_is_measured_against_carbonyl_oxygen(mapped_system, tmp_path, monkeypatch):
    import hashlib
    import json
    import numpy as np
    from structural_modeling_agent.paired_analysis import carbonyl_geometry
    job, manifest = mapped_system
    job['input_sha256'] = 'verified-input'
    job['modeling_input']['geometry']['charge_centre']['atom_selector'] = 'activated_carbonyl_oxygen'
    manifest['input_sha256'] = job['input_sha256']
    manifest_path = tmp_path / 'manifest.json';manifest_path.write_text(json.dumps(manifest))
    prediction = tmp_path / 'pose.pdb';prediction.write_text('synthetic coordinates supplied by parser fixture')
    atoms = {('X', 1, 'ATOM1'): {'xyz': np.array([0., 0., 0.])},
        ('X', 1, 'ATOM2'): {'xyz': np.array([1., 0., 0.])},
        ('X', 1, 'ATOM3'): {'xyz': np.array([-1., 0., 0.])},
        ('Y', 1, 'ATOM2'): {'xyz': 3 * np.array([math.cos(math.radians(107)), math.sin(math.radians(107)), 0.])}}
    monkeypatch.setattr('structural_modeling_agent.validation._read_structure', lambda path: (atoms, {}))
    saved = {'job': job, 'receipt': {'prediction_manifest': str(manifest_path)}, 'analysis': {'poses': [
        {'path': str(prediction), 'sha256': hashlib.sha256(prediction.read_bytes()).hexdigest(),
         'distance_A': 3., 'nucleophile': ['Y', 'ATOM2']}]}}
    result = carbonyl_geometry(saved, tmp_path / 'analysis')
    assert result['weighted_carbonyl_angle_deg'] == pytest.approx(107.)
    assert result['weighted_absolute_deviation_deg'] == pytest.approx(0., abs=1e-10)


def test_joint_order_uses_weaker_step_distance_not_one_favorable_step(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr('structural_modeling_agent.paired_analysis.carbonyl_geometry', lambda *args: None)
    saved = tmp_path / 'saved.json';saved.write_text(json.dumps({}))
    summaries = []
    for accession, distances in [('balanced', (4., 5.)), ('one_close', (2., 20.))]:
        for step, distance in enumerate(distances, 1):
            summaries.append({'accession': accession, 'step': step, 'harmonic_distance_A': distance,
                'weighted_attack_angle_deg': 160., 'identity_pass_poses': 50, 'result': str(saved)})
    rows = paired_rows(summaries, {'balanced': 'Enzyme B', 'one_close': 'Enzyme A'}, tmp_path)
    assert [row['enzyme_name'] for row in rows] == ['Enzyme B', 'Enzyme A']
