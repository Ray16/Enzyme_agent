"""Audit existing AS9 predictions without refolding or inferring protonation.

Run as a module. Requires numpy, scipy, gemmi and RDKit in the Boltz environment.
The reusable classifier is enzyme_design_agent.contact_analysis; this adapter
validates Boltz's local processed chemical graphs and preserves panel membership.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time

import gemmi
import numpy as np
from rdkit import Chem

from enzyme_design_agent.contact_analysis import (
    Atom, ContactPolicy, DONORS, TargetGroup, collapse_for_display,
    distinct_residue_count, evaluate_contacts, group_key,
)


BASE = Path(__file__).resolve().parents[1] / 'runs/rfd4_campaigns'
CAMPS = {
    'M3': BASE / 'as9_motif_m3_partialdiff_20261108/boltz_refolding_20261108',
    'M4': BASE / 'as9_motif_m4_glutarate_partialdiff_20261109/boltz_refolding_ccdatp_20261110',
}
POLICY = ContactPolicy()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_atoms(path):
    raw = Path(path).read_bytes()
    s = gemmi.read_pdb_string(raw.decode())
    if len(s) != 1:
        raise ValueError('Expected exactly one structural model per file')
    atoms = [Atom(ch.name, r.name, str(r.seqid).strip(), a.name, a.element.name,
                  (a.pos.x, a.pos.y, a.pos.z), a.b_iso)
             for ch in s[0] for r in ch for a in r if not a.element.is_hydrogen]
    if any(a.altloc not in ('\x00', ' ', 'A') for ch in s[0] for r in ch for a in r):
        raise ValueError('Unresolved alternate conformations')
    return atoms, hashlib.sha256(raw).hexdigest()


def graph_groups(pdb, atoms):
    """Match atom names/elements to the exact processed RDKit molecule.

    Local pickles are trusted pipeline outputs, not arbitrary uploaded data.
    Both resonance-related O atoms are acceptors for a monoanionic carboxylate.
    Fail closed for any unexpected ligand composition, charge, or atom map.
    """
    target = {a.name: a for a in atoms if a.chain == 'D'}
    if len(target) != len([a for a in atoms if a.chain == 'D']):
        raise ValueError('Repeated target atom names')
    ident = Path(pdb).name.rsplit('_model_', 1)[0]
    source = Path(pdb).parents[2] / 'processed/mols' / f'{ident}.pkl'
    raw = source.read_bytes()
    graphs = pickle.loads(raw)
    matches = []
    for key, mol in graphs.items():
        heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        if not all(a.HasProp('name') for a in heavy):
            continue
        names = {a.GetProp('name'): a.GetSymbol() for a in heavy}
        if names == {n: a.element for n, a in target.items()} and len(names) == len(heavy):
            matches.append((key, mol))
    if len(matches) != 1:
        raise ValueError(f'Need one exact target graph match, got {len(matches)}: {source}')
    key, mol = matches[0]
    pattern = Chem.MolFromSmarts('[CX3](=[OX1])[O-]')
    groups = []
    for ci, oi, oj in mol.GetSubstructMatches(pattern):
        carbon = mol.GetAtomWithIdx(ci).GetProp('name')
        names = tuple(sorted(mol.GetAtomWithIdx(i).GetProp('name') for i in (oi, oj)))
        groups.append(TargetGroup(carbon, 'D', target[names[0]].resi, names, -1, str(source)))
    if len(groups) != 2 or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != -2:
        raise ValueError('AS9 adapter expects two monoanionic carboxylates, total charge -2')
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
            continue
        distance = np.linalg.norm(np.subtract(target[a.GetProp('name')].xyz, target[b.GetProp('name')].xyz))
        if not .8 <= distance <= 1.9:
            raise ValueError('Target graph / pose bond-length mismatch')
    return groups, dict(source=str(source), sha256=hashlib.sha256(raw).hexdigest(), key=key,
                        smiles=Chem.MolToSmiles(mol), formal_charge=-2,
                        carbon_count=sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms()))


def role_map(atoms, groups, pa):
    amap = {(a.chain, a.name): a for a in atoms if a.chain in {'B', 'D'}}
    phosphorus = amap[('B', pa)]
    nearest = min((np.linalg.norm(np.subtract(amap[('D', name)].xyz, phosphorus.xyz)), name, g.group_id)
                  for g in groups for name in g.acceptors)
    return {g.group_id: 'attack' if g.group_id == nearest[2] else 'distal' for g in groups}


def role_key(c):
    # Role-based recurrence tolerates a symmetry-related flip of the ligand.
    # Chemical-group identity recurrence is retained separately.
    return c['chain'], c['ri'], c['res'], c['where'], c['conditional']


def count_at_thresholds(raw, polar, ionic, include_ambiguous=False):
    flagged = {group_key(c) for c in raw if c['reason'] in
               {'severe_clash', 'anomalously_short_contact', 'group_contains_anomalously_short_contact'}}
    return len({(c['chain'], c['ri']) for c in raw
                if c['atom'] in DONORS.get(c['res'], set())
                and (include_ambiguous or c['res'] not in {'HIS', 'CYS'})
                and group_key(c) not in flagged
                and c['d'] <= (ionic if c['kind'] == 'potential_ionic' else polar)})


def worker_init(cpus):
    index = mp.current_process()._identity[-1] - 1
    os.sched_setaffinity(0, {cpus[index % len(cpus)]})


def analyze_task(task):
    t = time.monotonic()
    camp, ident, metfile, cell, output = task
    outpath = Path(output) / 'designs' / f'{camp}_{ident}.json'
    # Resume only identical inputs and policy. PDB digest validation is done on
    # each fresh audit; completed source metrics explicitly bind pose SHA256s.
    meta_bytes = Path(metfile).read_bytes() if metfile else b''
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes() +
        (Path(__file__).parents[1]/'contact_analysis.py').read_bytes()).hexdigest()
    input_hash = hashlib.sha256(meta_bytes + json.dumps(cell, sort_keys=True).encode()).hexdigest()
    if outpath.exists():
        saved = json.loads(outpath.read_text())
        if (saved['input_hash'] == input_hash and saved['policy'] == POLICY.to_dict()
                and saved.get('implementation_hash') == implementation_hash):
            # Do not reuse changed coordinates even when the metrics are stale.
            if (hashlib.sha256(Path(saved['chemistry']['source']).read_bytes()).hexdigest() == saved['chemistry']['sha256']
                and all(Path(p['path']).is_file() and hashlib.sha256(Path(p['path']).read_bytes()).hexdigest() == p['sha256']
                        for p in saved['pose_manifest'])):
                return str(outpath)
    d = json.loads(meta_bytes) if metfile else None
    if d:
        poses = [p for p in d['poses'] if p.get('geometry')]
        if not poses:
            raise ValueError(f'No geometrically valid poses: {metfile}')
        near = [p for p in poses if p['geometry']['distance_A'] < 4 and p['geometry']['angle_deg'] > 150]
        rep = min(near or poses, key=lambda p: p['geometry']['distance_A'])['path']
        if cell:
            rep = cell['pdb']  # freeze the previously selected pose as well as membership
        pa = d['reaction_atom_maps']['ATP']['alpha_phosphorus']
        ds = np.array([p['geometry']['distance_A'] for p in poses])
        ang = np.array([p['geometry']['angle_deg'] for p in poses])
        geometry = dict(A=float(np.sum(ang / ds) / np.sum(1 / ds)),
                        dh=float(len(ds) / np.sum(1 / ds)), nac=float(np.mean((ds < 4) & (ang > 150))))
        expected = d.get('expected_poses') or len(poses)
    else:
        rep, pa = cell['pdb'], cell['paname']
        prefix = Path(rep).name.rsplit('_model_', 1)[0]
        poses = [dict(path=str(p), confidence={}) for p in sorted(Path(rep).parent.glob(prefix + '_model_*.pdb'))]
        expected, geometry = len(poses), None
    atoms, rep_sha = read_atoms(rep)
    groups, chemistry = graph_groups(rep, atoms)
    roles = role_map(atoms, groups, pa)
    raw = evaluate_contacts(atoms, groups)
    for c in raw:
        c['where'] = roles[c['group']]
    displayed = collapse_for_display(raw)
    for c in displayed:
        c['where'] = roles[c['group']]
    pocket = {a.resi for a in atoms if a.chain == 'A'
              and min(np.linalg.norm(np.subtract(a.xyz, b.xyz)) for b in atoms if b.chain == 'D') <= 6}
    counts, fixed_counts, exact_counts = {}, {}, {}
    target_identity = {(a.resi, a.name, a.element) for a in atoms if a.chain == 'D'}
    protein_identity = {(a.resi, a.residue, a.name, a.element) for a in atoms if a.chain == 'A'}
    local, global_plddt, manifest, errors, seen = [], [], [], [], set()
    for p in poses:
        try:
            ap, sha = read_atoms(p['path'])
            if {(a.resi, a.name, a.element) for a in ap if a.chain == 'D'} != target_identity:
                raise ValueError('Target identity differs from chemical graph')
            if {(a.resi, a.residue, a.name, a.element) for a in ap if a.chain == 'A'} != protein_identity:
                raise ValueError('Protein atom identity differs across samples')
            if p.get('sha256') and sha != p['sha256']:
                raise ValueError('Pose SHA256 differs from metrics manifest')
            if sha in seen:
                errors.append(dict(path=p['path'], reason='duplicate_coordinate_file_not_independent_sample'))
                continue
            seen.add(sha)
            pr = role_map(ap, groups, pa)
            contacts = evaluate_contacts(ap, groups)
            for c in contacts:
                c['where'] = pr[c['group']]
            eligible = [c for c in contacts if c['accepted'] or c['conditional']]
            for key in {role_key(c) for c in eligible}:
                counts[key] = counts.get(key, 0) + 1
            for key in {(group_key(c), c['conditional']) for c in eligible}:
                fixed_counts[key] = fixed_counts.get(key, 0) + 1
            for key in {(c['chain'], c['ri'], c['atom'], c['group'], c['O'], c['conditional']) for c in eligible}:
                exact_counts[key] = exact_counts.get(key, 0) + 1
            scores = [a.confidence for a in ap if a.chain == 'A' and a.name == 'CA' and a.resi in pocket]
            if scores:
                local.append(float(np.mean(scores)))
            conf = p.get('confidence') or {}
            if conf.get('complex_plddt') is not None:
                global_plddt.append(conf['complex_plddt'])
            manifest.append(dict(path=p['path'], sha256=sha))
        except Exception as e:
            errors.append(dict(path=p['path'], reason=f'{type(e).__name__}: {e}'))
    n = len(manifest)
    if not n:
        raise ValueError('No valid unique poses')
    for c in displayed:
        c['consistency_n'] = counts.get(role_key(c), 0)
        c['consistency_denominator'] = n
        c['consistency'] = c['consistency_n'] / n
        c['fixed_group_consistency'] = fixed_counts.get((group_key(c), c['conditional']), 0) / n
        for p in c['independent_pairs']:
            p['exact_atom_pair_consistency'] = exact_counts.get((p['chain'], p['ri'], p['atom'], p['group'], p['O'], p['conditional']), 0) / n
    old_candidates = [c for c in raw if c['atom'] in DONORS.get(c['res'], set()) and c['res'] != 'CYS' and c['d'] < 3.6]
    accepted = [c for c in displayed if not c['conditional']]
    old_raw_lines = sum(c['d'] <= 3.8 and c['atom'][0] in {'N', 'O'} for c in raw)
    row = dict(campaign=camp, ident=ident, metrics=metfile, input_hash=input_hash,
               implementation_hash=implementation_hash,
               policy=POLICY.to_dict(), chemistry=chemistry, panel_cell=cell, representative=rep,
               representative_sha256=rep_sha, expected_poses=expected, valid_unique_poses=n,
               pose_manifest=manifest, errors=errors, geometry=geometry,
               raw_contacts=raw, group_contacts=displayed,
               n_old_flat36=len({(c['chain'], c['ri']) for c in old_candidates}),
               n_typed_with_ambiguous=count_at_thresholds(raw, 3.5, 4.0, True),
               n_contacts=distinct_residue_count(raw),
               n_strict=count_at_thresholds(raw, 3.2, 3.6),
               n_loose=count_at_thresholds(raw, 3.8, 4.5),
               n_distal=sum(c['where'] == 'distal' for c in accepted),
               n_attack=sum(c['where'] == 'attack' for c in accepted),
               n_conditional=sum(c['conditional'] for c in displayed),
               n_short_groups=sum(any('short' in reason for reason in c['uncertainty_reasons']) for c in displayed),
               old_allpairs38_lines=old_raw_lines, new_display_lines=len(accepted),
               mean_consistency=float(np.mean([c['consistency'] for c in accepted])) if accepted else None,
               pocket_residues=sorted(pocket), pocket_radius_A=6,
               mean_pocket_plddt=float(np.mean(local)) if local else None,
               mean_global_plddt=float(np.mean(global_plddt)) if global_plddt else None,
               worker=dict(pid=os.getpid(), cpus=sorted(os.sched_getaffinity(0)),
                           threads=len(list(Path('/proc/self/task').iterdir()))),
               elapsed_seconds=time.monotonic() - t)
    write_json(outpath, row)
    return str(outpath)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpus', default='73,74,75,76,77,78,79')
    args = parser.parse_args()
    cpus = [int(c) for c in args.cpus.split(',')]
    if len(set(cpus)) != len(cpus) or not set(cpus) <= os.sched_getaffinity(0):
        raise RuntimeError('Invalid or overlapping worker affinity')
    for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError(f'{key}=1 required')
    if os.getloadavg()[0] + len(cpus) + 1 > os.cpu_count():
        raise RuntimeError('CPU capacity check failed')
    os.sched_setaffinity(0, {min(set(os.sched_getaffinity(0)) - set(cpus))})
    tasks = []
    for camp, run in CAMPS.items():
        cells = json.loads((args.spec_dir / f'{camp.lower()}_panel_full.json').read_text())
        by_path = {c['pdb']: c for c in cells}
        matched = set()
        for met in sorted((run / 'metrics').glob('*.json')):
            d = json.loads(met.read_text())
            hits = [by_path[p['path']] for p in d['poses'] if p['path'] in by_path]
            if len(hits) > 1:
                raise ValueError('More than one selected representative per design')
            cell = hits[0] if hits else None
            if cell:
                matched.add(cell['id'])
            tasks.append((camp, met.stem, str(met), cell, str(args.output)))
        for c in cells:
            if c['id'] not in matched:
                if c['id'] != 'WT_AS9':
                    raise ValueError(f'Panel design absent from metrics: {c["id"]}')
                tasks.append((camp, 'WT_AS9', None, c, str(args.output)))
    write_json(args.output / 'launch.json', dict(tasks=len(tasks), policy=POLICY.to_dict(),
               worker_cpus=cpus, coordinator_cpu=sorted(os.sched_getaffinity(0)),
               load=os.getloadavg(), cpu_count=os.cpu_count(), interpreter=os.sys.executable,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               classifier_sha256=hashlib.sha256((Path(__file__).parents[1]/'contact_analysis.py').read_bytes()).hexdigest(), gpu_use=False))
    print(f'Auditing {len(tasks)} full-campaign designs/controls on {len(cpus)} CPU workers', flush=True)
    with ProcessPoolExecutor(len(cpus), mp_context=mp.get_context('spawn'),
                             initializer=worker_init, initargs=(cpus,)) as pool:
        futures = {pool.submit(analyze_task, task): task for task in tasks}
        results = []
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 10 == 0 or i == len(tasks):
                print(f'{i}/{len(tasks)} designs audited', flush=True)
    write_json(args.output / 'result_index.json', sorted(results))


if __name__ == '__main__':
    main()
