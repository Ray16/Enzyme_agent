"""Compare completed reaction steps together, without fitting an activity score."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

from enzyme_discovery_agent.state_store import atomic_write_json

# Mechanistic reference angles for the two geometries this module measures, not substrate-specific
# and not fitted thresholds: an in-line nucleophilic attack at the activated phosphorus (step one),
# and the Bürgi–Dunitz angle for nucleophilic addition to an activated carbonyl (step two).
PHOSPHATE_ATTACK_REFERENCE_DEG = 180.0
CARBONYL_ADDITION_REFERENCE_DEG = 107.0


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
    from ..validation import _read_structure
    from ..proxies.attack_angle import near_attack_mode, weighted_attack_angle
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
        'manifest': manifest['ligand_atom_maps'], 'implementation': 3}, sort_keys=True).encode()).hexdigest()
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
    deviation = weighted_attack_angle(distances, [abs(a - CARBONYL_ADDITION_REFERENCE_DEG) for a in angles])['weighted_attack_angle_deg']
    mode = near_attack_mode(distances, angles)
    value = {'signature': key, 'atom_definition': 'nucleophile N - acyl C - carbonyl O',
        'carbon_ref': carbon_ref, 'carbonyl_oxygen_ref': oxygen_ref, 'amine_nitrogen_refs': donor_refs,
        'reference_angle_deg': CARBONYL_ADDITION_REFERENCE_DEG, 'weighted_carbonyl_angle_deg': mean,
        'weighted_absolute_deviation_deg': deviation,
        'near_attack_threshold_A': mode['near_attack_threshold_A'],
        'n_near_attack_poses': mode['n_near_attack_poses'],
        'near_attack_fraction': mode['near_attack_fraction'],
        'near_mode_carbonyl_angle_deg': mode['near_mode_weighted_angle_deg'],
        'near_mode_harmonic_distance_A': mode['near_mode_harmonic_distance_A'],
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
            'step_1_near_attack_fraction': first.get('near_attack_fraction'),
            'step_1_near_mode_angle_deg': first.get('near_mode_weighted_angle_deg'),
            'step_1_near_mode_distance_A': first.get('near_mode_harmonic_distance_A'),
            'step_2_harmonic_distance_A': second['harmonic_distance_A'],
            'step_2_legacy_leaving_group_angle_deg': second['weighted_attack_angle_deg'],
            'step_2_weighted_carbonyl_angle_deg': supplement['weighted_carbonyl_angle_deg'] if supplement else None,
            'step_2_weighted_carbonyl_deviation_deg': supplement['weighted_absolute_deviation_deg'] if supplement else None,
            'step_2_near_attack_fraction': supplement['near_attack_fraction'] if supplement else None,
            'step_2_near_mode_carbonyl_angle_deg': supplement['near_mode_carbonyl_angle_deg'] if supplement else None,
            'step_2_near_mode_distance_A': supplement['near_mode_harmonic_distance_A'] if supplement else None,
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
            target = PHOSPHATE_ATTACK_REFERENCE_DEG if index == 1 else CARBONYL_ADDITION_REFERENCE_DEG
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


def _place_labels(ax, points, fontsize=6.6):
    """Label points close to their dots: small offset + light force relaxation to avoid overlap."""
    import numpy as np
    if not points:
        return
    x0, x1 = ax.get_xlim();y0, y1 = ax.get_ylim();xr = x1 - x0;yr = y1 - y0
    labs = []
    for p in points:
        a = np.array([(p['x'] - x0) / xr, (p['y'] - y0) / yr])
        labs.append({'p': p, 'anchor': a, 'pos': a + np.array([0.0, 0.05])})
    for _ in range(400):
        for i, L in enumerate(labs):
            force = (L['anchor'] + np.array([0.0, 0.055]) - L['pos']) * 0.1   # spring near its dot
            for j, M in enumerate(labs):
                if i != j:
                    d = L['pos'] - M['pos'];dist = float(np.hypot(*d))
                    if 1e-6 < dist < 0.075:
                        force += d / dist * (0.075 - dist) * 0.5              # repel other labels
                d = L['pos'] - M['anchor'];dist = float(np.hypot(*d))
                if 1e-6 < dist < 0.045:
                    force += d / dist * (0.045 - dist) * 0.5                  # repel all dots
            L['pos'] = np.clip(L['pos'] + force, 0.015, 0.985)
    for L in labs:
        lx = x0 + L['pos'][0] * xr;ly = y0 + L['pos'][1] * yr;p = L['p']
        far = float(np.hypot(*(L['pos'] - L['anchor']))) > 0.03
        # drop the accession line when the name already is the accession (e.g. a synthetic WT_AS9)
        same = p['label'].replace(' ', '_').upper() == str(p['accession']).upper()
        text = p['label'] if same else f"{p['label']}\n{p['accession']}"
        ax.annotate(text, (p['x'], p['y']), xytext=(lx, ly), color=p.get('color', 'black'),
                    fontsize=fontsize, ha='center', va='center', fontweight=p.get('weight', 'normal'),
                    arrowprops=dict(arrowstyle='-', color='0.6', lw=0.4) if far else None, zorder=5)


def near_attack_figure(rows, highlight=None, min_fraction=0.4, min_angle=140.0):
    """One scatter carrying all three geometry readouts, so both steps read at a glance.

    x = step-1 proximity-weighted attack angle A (the calibrated step-1 rank metric);
    y = step-1 near-attack fraction (how often the carboxylate reaches the reactive phosphate);
    colour = step-2 near-attack fraction (how often the amine reaches the carbonyl -- the only
    step-2 quantity the cofold can support; its angle is below resolution). So a strong dual-step
    enzyme is upper-right (activates: in-line AND often) AND bright (also reaches in step 2); a
    bright dot low/left reaches in step 2 but cannot activate, so it is not a real candidate.
    `highlight` accessions (e.g. a reference enzyme) are drawn as a labelled star.
    """
    import matplotlib.pyplot as plt
    highlight = set(highlight or [])

    def s1f(r): return r.get('step_1_near_attack_fraction')
    def s1a(r): return r.get('step_1_weighted_angle_deg')
    def s2f(r): return r.get('step_2_near_attack_fraction')
    present = [r for r in rows if s1a(r) is not None and s2f(r) is not None]
    reg = [r for r in present if r['accession'] not in highlight]
    hil = [r for r in present if r['accession'] in highlight]

    fig, ax = plt.subplots(figsize=(11.5, 8))
    ax.scatter([s1a(r) for r in reg], [s2f(r) for r in reg], s=95, color='#4C72B0',
               edgecolor='k', linewidth=0.5, zorder=3)
    for r in hil:
        ax.scatter([s1a(r)], [s2f(r)], marker='*', s=520, color='#D55E00',
                   edgecolor='k', linewidth=0.9, zorder=4)
    ax.axvline(min_angle, color='0.6', ls='--', lw=1, zorder=1)
    ax.set_xlabel('Step 1 — proximity-weighted attack angle A (°)   (validated; surfaces actives)', fontsize=12)
    ax.set_ylabel('Step 2 — fraction within 4 Å  (amine reaches carbonyl)', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_ylim(-0.04, max([s2f(r) for r in present]) * 1.12 + 0.02);ax.grid(alpha=0.25, zorder=0)

    # Label high-angle step-1 points + the best step-2 reach + references.
    high_angle = sorted([r for r in reg if s1a(r) >= min_angle], key=lambda r: -s1a(r))[:14]
    best_s2 = sorted(reg, key=lambda r: -(s2f(r) or 0.0))[:6]
    seen = set();to_label = []
    for r in high_angle + best_s2 + hil:
        if r['accession'] not in seen:
            seen.add(r['accession']);to_label.append(r)
    def short(name): return name.split(' · ')[0].split(' (')[0].strip()[:16]
    _place_labels(ax, [{'x': s1a(r), 'y': s2f(r), 'label': short(r['enzyme_name']), 'accession': r['accession']}
                       for r in to_label], fontsize=8.5)
    ax.set_title('Step 1 attack angle A (x) · Step 2 fraction within 4 Å (y) · ★ reference', fontsize=13, pad=10)
    fig.tight_layout()
    return fig


def step2_spread_figure(rows, step2_distances, highlight=None, min_angle=148.0, near=4.0):
    """Step-1 activation angle (x) vs the full step-2 distance DISTRIBUTION (y) per enzyme.

    Step-2 distance is a bimodal distribution over the 50 poses, not one number, so we show the
    spread instead of collapsing it: each enzyme sits at its step-1 proximity-weighted A (x) and
    its 50 step-2 amine→carbonyl distances are drawn as a vertical strip (y, Å; closer = higher),
    with its closest-approach marked. The `near` line is the attack-distance threshold -- poses
    below it reach the carbonyl. Only enzymes with step-1 A >= `min_angle` (plausible activation)
    plus `highlight` references are shown, to keep the strips legible. x is the validated metric;
    step-2 distance is descriptive (its angle is below cofold resolution).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    highlight = set(highlight or [])
    def s1a(r): return r.get('step_1_weighted_angle_deg')
    shown = [r for r in rows if r['accession'] in step2_distances and s1a(r) is not None
             and (s1a(r) >= min_angle or r['accession'] in highlight)]
    shown = sorted(shown, key=lambda r: -s1a(r))   # stable order: strongest activation first
    # Give every enzyme its own colour so overlapping strips are separable; references keep a star.
    palette = (list(cm.get_cmap('tab20').colors) + list(cm.get_cmap('tab20b').colors))
    colour = {r['accession']: palette[i % len(palette)] for i, r in enumerate(shown)}
    for acc in highlight:   # references get a fixed bold colour, never a pale palette slot
        if acc in colour: colour[acc] = '#111111'
    from matplotlib.lines import Line2D
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='0.9', lw=0.8, zorder=0)
    rng = np.random.default_rng(0)
    centre = {}
    for r in shown:
        d = np.array(step2_distances[r['accession']]);x = s1a(r)
        ref = r['accession'] in highlight
        c = colour[r['accession']]
        med = float(np.median(d));centre[r['accession']] = med
        jit = x + rng.uniform(-0.32, 0.32, size=len(d))
        ax.scatter(jit, d, s=11, alpha=0.40, color=c, edgecolor='none', zorder=2)
        # robust per-enzyme marker = MEDIAN of the 50 poses (not the lucky closest pose)
        ax.scatter([x], [med], marker='*' if ref else 'o', s=360 if ref else 90,
                   color=c, edgecolor='k', linewidth=0.9, zorder=4)
    ax.axhspan(0, near, color='#2ca02c', alpha=0.06, zorder=0)   # shade the attack-competent band
    ax.axhline(near, color='crimson', ls='--', lw=1.4, zorder=1)
    ax.text(ax.get_xlim()[1], near, f'{near:g} Å attack threshold  ', color='crimson',
            va='bottom', ha='right', fontsize=10, fontstyle='italic')
    ax.invert_yaxis()   # closer (smaller distance) at the top = better
    ax.set_xlabel('STEP 1  —  proximity-weighted activation angle  A  (°)        ← ranking axis (validated: surfaces known actives)',
                  fontsize=12.5, labelpad=8)
    ax.set_ylabel('STEP 2  —  amine→carbonyl distance, 50 poses per enzyme  (Å)        closer ↑',
                  fontsize=12.5, labelpad=8)
    ax.tick_params(labelsize=11)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    def short(name): return name.split(' · ')[0].split(' (')[0].strip()[:16]
    _place_labels(ax, [{'x': s1a(r), 'y': centre[r['accession']],
                        'label': short(r['enzyme_name']), 'accession': r['accession'],
                        'weight': 'bold' if r['accession'] in highlight else 'normal'}
                       for r in shown], fontsize=7.2)
    legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor='0.5', markeredgecolor='k',
                     markersize=9, label='enzyme median (50 poses)'),
              Line2D([0], [0], marker='o', color='none', markerfacecolor='0.5', markeredgecolor='none',
                     alpha=0.5, markersize=6, label='individual pose'),
              Line2D([0], [0], marker='*', color='none', markerfacecolor='#111111', markeredgecolor='k',
                     markersize=15, label='AS9 benchmark (engineered)'),
              Line2D([0], [0], color='crimson', ls='--', lw=1.4, label='4 Å attack threshold')]
    ax.legend(handles=legend, loc='lower left', fontsize=9.5, framealpha=0.95, edgecolor='0.8')
    ax.set_title('Candidate ranking — step-1 activation (x) vs step-2 reach distribution (y)',
                 fontsize=14, fontweight='bold', pad=12)
    fig.tight_layout()
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
    import re
    uniprot = r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}'
    highlight = [r['accession'] for r in rows if not re.fullmatch(uniprot, r['accession'])]
    fig = near_attack_figure(rows, highlight=highlight)
    for extension in ['png', 'svg']:
        path = directory / ('both_steps_near_attack.' + extension);temporary = path.with_name('both_steps_near_attack.tmp.' + extension)
        fig.savefig(temporary, dpi=180);os.replace(temporary, path)
    plt.close(fig)
