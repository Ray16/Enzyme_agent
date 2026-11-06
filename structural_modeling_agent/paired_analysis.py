"""Compare completed reaction steps together, without fitting an activity score."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

from enzyme_discovery_agent.state_store import atomic_write_json


def carbonyl_atom_references(job, manifest):
    """Verify the mapped C=O and distinguish it from the leaving-group oxygen."""
    from rdkit import Chem
    geometry = job['modeling_input']['geometry'];entities = job['entities']
    carbon = geometry['electrophile'];oxygen = geometry['charge_centre'];leaving = geometry['leaving_group']
    if not (carbon['role'] == oxygen['role'] == leaving['role']):
        raise ValueError('Carbonyl and leaving-group references must describe the same activated ligand')
    chain = entities[carbon['role']];definition = manifest['ligand_atom_maps'][chain]
    molecule = Chem.MolFromSmiles(definition['smiles'])
    c, o, leaving_o = [part['reactive_atom_indices'][0] for part in (carbon, oxygen, leaving)]
    bond = molecule.GetBondBetweenAtoms(c, o)
    if (molecule.GetAtomWithIdx(c).GetAtomicNum() != 6 or molecule.GetAtomWithIdx(o).GetAtomicNum() != 8
            or bond is None or bond.GetBondType() != Chem.BondType.DOUBLE or o == leaving_o):
        raise ValueError('Expected the activated carbonyl C=O, not the leaving-group oxygen')
    bridge = molecule.GetBondBetweenAtoms(c, leaving_o)
    leaving_atom = molecule.GetAtomWithIdx(leaving_o)
    if (bridge is None or bridge.GetBondType() != Chem.BondType.SINGLE or leaving_atom.GetAtomicNum() != 8
            or not any(neighbor.GetAtomicNum() == 15 for neighbor in leaving_atom.GetNeighbors())):
        raise ValueError('Expected a phosphate-activated acyl carbon, not the other carboxylate')
    donor = geometry['nucleophile'];donor_chain = entities[donor['role']]
    donor_definition = manifest['ligand_atom_maps'][donor_chain]
    donor_molecule = Chem.MolFromSmiles(donor_definition['smiles'])
    donor_refs = []
    for index in donor['reactive_atom_indices']:
        if donor_molecule.GetAtomWithIdx(index).GetAtomicNum() != 7:
            raise ValueError('Step-two nucleophile must be the declared amine nitrogen')
        donor_refs.append((donor_chain, donor_definition['atoms'][str(index)]['name']))
    return (chain, definition['atoms'][str(c)]['name']), (chain, definition['atoms'][str(o)]['name']), donor_refs


def carbonyl_geometry(saved, directory):
    """Supplement the frozen leaving-group angle with the actual carbonyl angle."""
    from .validation import _read_structure
    from .proxies.attack_angle import weighted_attack_angle
    import numpy as np
    job = saved.get('reuse_source_job') or saved['job']
    geometry = job['modeling_input']['geometry']
    reference = geometry.get('charge_centre', {})
    if not str(reference.get('atom_selector', '')).endswith('carbonyl_oxygen'):
        return None
    manifest = json.loads(Path(saved['receipt']['prediction_manifest']).read_text())
    if manifest['input_sha256'] != job['input_sha256']:
        raise ValueError('Prediction manifest does not match the declared molecular input')
    carbon_ref, oxygen_ref, donor_refs = carbonyl_atom_references(job, manifest)
    key = hashlib.sha256(json.dumps({'geometry': geometry, 'poses': saved['analysis']['poses'],
        'manifest': manifest['ligand_atom_maps'], 'implementation': 2}, sort_keys=True).encode()).hexdigest()
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
    cache = directory / 'carbonyl_geometry.json'
    if cache.exists():
        value = json.loads(cache.read_text())
        if value.get('signature') == key:
            return value
    distances = [];angles = []
    for pose in saved['analysis']['poses']:
        path = Path(pose['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != pose['sha256']:
            raise ValueError('Saved prediction changed before carbonyl analysis')
        atoms, _ = _read_structure(path)
        def coordinate(chain, name):
            matches = [v['xyz'] for k, v in atoms.items() if k[0] == chain and k[2] == name]
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous carbonyl reaction atom')
            return matches[0]
        if tuple(pose['nucleophile']) not in donor_refs:
            raise ValueError('Saved attacking nitrogen does not match the declared substrate atom')
        center = coordinate(*carbon_ref)
        axis = coordinate(*oxygen_ref) - center
        donor = coordinate(*pose['nucleophile']) - center
        angle = math.degrees(math.acos(float(np.clip(np.dot(axis, donor) /
            (np.linalg.norm(axis) * np.linalg.norm(donor)), -1, 1))))
        distances.append(pose['distance_A']);angles.append(angle)
    mean = weighted_attack_angle(distances, angles)['weighted_attack_angle_deg']
    deviation = weighted_attack_angle(distances, [abs(a - 107.0) for a in angles])['weighted_attack_angle_deg']
    value = {'signature': key, 'atom_definition': 'nucleophile N - acyl C - carbonyl O',
        'carbon_ref': carbon_ref, 'carbonyl_oxygen_ref': oxygen_ref, 'amine_nitrogen_refs': donor_refs,
        'reference_angle_deg': 107.0, 'weighted_carbonyl_angle_deg': mean,
        'weighted_absolute_deviation_deg': deviation,
        'poses': [{'distance_A': d, 'carbonyl_angle_deg': a} for d, a in zip(distances, angles)],
        'scope': 'Supplementary carbonyl geometry; frozen legacy score unchanged; not demonstrated activity.',
        'reference': 'https://doi.org/10.1021/ja00796a058'}
    atomic_write_json(cache, value)
    return value


def paired_rows(summaries, labels, directory):
    members = {}
    for row in summaries:
        members.setdefault(row['accession'], {})[int(row['step'])] = row
    rows = []
    for accession, steps in members.items():
        if 1 not in steps or 2 not in steps:
            continue
        first, second = steps[1], steps[2]
        saved = json.loads(Path(second['result']).read_text())
        supplement = carbonyl_geometry(saved, Path(directory) / 'ensembles' / accession)
        row = {'enzyme_name': labels[accession], 'accession': accession,
            'step_1_harmonic_distance_A': first['harmonic_distance_A'],
            'step_1_weighted_angle_deg': first['weighted_attack_angle_deg'],
            'step_2_harmonic_distance_A': second['harmonic_distance_A'],
            'step_2_legacy_leaving_group_angle_deg': second['weighted_attack_angle_deg'],
            'step_2_weighted_carbonyl_angle_deg': supplement['weighted_carbonyl_angle_deg'] if supplement else None,
            'step_2_weighted_carbonyl_deviation_deg': supplement['weighted_absolute_deviation_deg'] if supplement else None,
            'larger_step_distance_A': max(first['harmonic_distance_A'], second['harmonic_distance_A']),
            'step_1_profile': first.get('numerical_profile'), 'step_2_profile': second.get('numerical_profile'),
            'step_2_analysis_arm': second.get('analysis_arm','unrestrained'),
            'step_1_identity_pass_poses': first['identity_pass_poses'],
            'step_2_identity_pass_poses': second['identity_pass_poses'],
            'scope': 'Closest approach across both steps; geometry comparison, not an activity rank.'}
        rows.append(row)
    return sorted(rows, key=lambda row: (row['larger_step_distance_A'], row['enzyme_name']))


def paired_figure(rows, limit=24):
    import matplotlib.pyplot as plt
    total = len(rows)
    rows = rows[:limit]
    height = max(5, 2.5 + .43 * len(rows))
    fig, axes = plt.subplots(1, 4, figsize=(17, height), sharey=True)
    fig.subplots_adjust(left=.30, right=.98, top=1-1.0/height, bottom=.9/height, wspace=.25)
    fields = ['step_1_harmonic_distance_A', 'step_1_weighted_angle_deg',
              'step_2_harmonic_distance_A', 'step_2_weighted_carbonyl_angle_deg']
    titles = ['Step 1: attack distance', 'Step 1: phosphate angle',
              'Step 2: N···C distance', 'Step 2: N–C=O angle']
    colors = ['#0072B2', '#0072B2', '#009E73', '#009E73']
    axes[0].set_yticks(range(len(rows)), ['\n'.join(textwrap.wrap(row['enzyme_name'], 40)) for row in rows], fontsize=9)
    for index, (ax, field, title, color) in enumerate(zip(axes, fields, titles, colors)):
        for y, row in enumerate(rows):
            ax.axhspan(y-.42, y+.42, color='#F2F5F7' if y%2 == 0 else 'white', zorder=0)
            value = row[field]
            if value is not None:
                ax.scatter(value, y, color=color, s=35, zorder=3)
                ax.annotate(f'{value:.1f}', (value, y), xytext=(5, 0), textcoords='offset points',
                            va='center', fontsize=8)
        if index in (0, 2):
            maximum = max(row[field] for row in rows)
            ax.set_xlim(0, max(5, maximum*1.35));ax.set_xlabel('Å · closer is better')
        else:
            target = 180 if index == 1 else 107
            ax.axvline(target, color='#777777', linestyle='--', linewidth=1)
            ax.set_xlim(0, 205);ax.set_xticks([0, 60, 107, 180] if index == 3 else [0, 60, 120, 180])
            ax.set_xlabel(f'Degrees · reference {target}°')
        ax.set_title(title, fontsize=11, color=color)
        ax.set_ylim(len(rows)-.5, -.5);ax.grid(axis='x', color='#DDE2E6', linewidth=.6)
        ax.spines[['top', 'right', 'left']].set_visible(False);ax.tick_params(axis='y', length=0, pad=8)
        if index:
            ax.tick_params(axis='y', labelleft=False)
    guided=any(row.get('step_2_analysis_arm')=='pocket_guided_localization' for row in rows)
    fig.suptitle('Both reaction steps — '+('pocket-guided localization arm' if guided else 'closest approach in the weaker step first'),
                 fontsize=15, fontweight='bold', y=1-.18/height)
    fig.text(.30, 1-.56/height, f'{len(rows)} of {total} paired candidates shown. '
        'Order uses the larger of the two distances; no fitted activity score.', fontsize=10, color='#555555')
    footer=('Step 2 uses broad pocket guidance; step 1 is reused unrestrained. Conditional geometry, not demonstrated activity.' if guided
        else 'Each ensemble contains 50 poses. Carbonyl angle is supplementary; legacy scores remain unchanged.')
    fig.text(.30, .015, footer, fontsize=9, color='#555555')
    return fig


def write_paired_analysis(summaries, labels, directory):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    rows = paired_rows(summaries, labels, directory)
    if not rows:
        return
    path = directory / 'both_steps_summary.csv';temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(temporary, path)
    for name, limit in [('both_steps_comparison', 24), ('both_steps_all_candidates', len(rows))]:
        fig = paired_figure(rows, limit)
        for extension in ['png', 'svg']:
            path = directory / (name + '.' + extension);temporary = path.with_name(name + '.tmp.' + extension)
            fig.savefig(temporary, dpi=180);os.replace(temporary, path)
        plt.close(fig)
