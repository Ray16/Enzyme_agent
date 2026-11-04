"""Prepare reproducible redesign campaigns from explicit discovery evidence.

Selection, chemistry, assembly assumptions and measurement definitions belong
in the input manifest. This module never guesses them from a protein name.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from .pocket import define_pocket
from .interface import interface_positions
from .variants import build_variants, write_variants_fasta, write_variants_tsv
from ..structure import load_structure


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(spec_path, output_dir):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    if spec.get('schema_version') != 'saturation-1':
        raise ValueError('Expected saturation-1 schema')
    if not spec.get('selection_evidence') or not spec.get('metrics'):
        raise ValueError('Selection evidence and separate metric definitions are required')
    candidates = spec.get('candidates', [])
    if not isinstance(spec.get('n_samples'), int) or spec['n_samples'] < 1:
        raise ValueError('n_samples must be a positive integer')
    if not candidates:
        raise ValueError('No candidates supplied')
    out = Path(output_dir).resolve()
    if (out / 'manifest.json').exists():
        old = json.loads((out / 'manifest.json').read_text())
        if old['spec_sha256'] != fingerprint(spec_path):
            raise ValueError('Output belongs to another specification; use a new directory')
        for p, digest in old['input_sha256'].items():
            if fingerprint(p) != digest:
                raise ValueError(f'Input changed: {p}')
        return old
    validated = []
    seen = set()
    for c in candidates:
        label = c['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', label) or label in seen:
            raise ValueError('Candidate ids must be unique, safe identifiers')
        seen.add(label)
        if not c.get('selection_reason') or not c.get('assembly_status'):
            raise ValueError('Each candidate needs selection and assembly provenance')
        st = load_structure(c['structure_path'])
        chain = c['protein_chain']
        seq = st.sequence(chain)
        if seq != c['sequence']:
            raise ValueError(f'{label}: source structure and sequence disagree')
        proteins = [e['protein'] for e in c['boltz_input']['sequences'] if 'protein' in e]
        if len(proteins) != 1 or proteins[0]['sequence'] != seq or proteins[0]['id'] != chain:
            raise ValueError('This campaign builder currently requires one matching protein chain')
        pocket = define_pocket(st, cutoff=spec.get('cutoff_A', 5.0),
                               anchor=c['anchor'], components=c['components'],
                               protein_chain=chain)
        if pocket.excluded:
            raise ValueError(f'{label}: required components disconnected; review scaffold')
        if not pocket.positions:
            raise ValueError(f'{label}: empty design pocket')
        rows = pocket.to_rows()
        if 'interface' in c:
            interface = c['interface']
            rows = interface_positions(c['ensemble_paths'], chain,
                interface['fragment_a'], interface['fragment_b'],
                cutoff=spec.get('cutoff_A', 5.0),
                min_contact_poses=interface.get('min_contact_poses', 1))
        if not rows:
            raise ValueError(f'{label}: no positions satisfy the contact rule')
        variants = build_variants(seq, [(r['resnum'],r['wt_aa']) for r in rows],
                                  label=label, modes=('holo',))
        validated.append((c, pocket, rows, variants))
    out.mkdir(parents=True, exist_ok=True)
    all_variants, records, jobs, hashes = [], [], [], {}
    for c, pocket, rows, variants in validated:
        d = out / c['id']; d.mkdir(exist_ok=True)
        with (d / 'positions.tsv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            w.writeheader(); w.writerows(rows)
        write_variants_tsv(variants, d / 'variants.tsv', modes=('holo',))
        write_variants_fasta(variants, d / 'variants.fasta')
        hashes[str(Path(c['structure_path']).resolve())] = fingerprint(c['structure_path'])
        for path in c.get('ensemble_paths', []):
            hashes[str(Path(path).resolve())] = fingerprint(path)
        records.append({**c, 'n_positions': len(rows), 'n_variants': len(variants),
                        'admitted_components': pocket.admitted})
        for v in variants:
            jobs.append({'id': v.id, 'candidate': c['id'], 'mutation': v.mutation,
                         'sequence': v.seq, 'sequence_sha256': hashlib.sha256(v.seq.encode()).hexdigest()})
        all_variants.extend(variants)
    write_variants_fasta(all_variants, out / 'variants.fasta')
    manifest = {**spec, 'spec_path': str(spec_path), 'spec_sha256': fingerprint(spec_path),
                'input_sha256': hashes, 'candidates': records, 'jobs': jobs,
                'n_variants': len(jobs), 'expected_models': len(jobs)*spec['n_samples'],
                'status': 'prepared', 'output_dir': str(out)}
    if all(c.get('geometry') for c in records):
        (out / 'evaluation.json').write_text(json.dumps(
            {c['id']: c['geometry'] for c in records}, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
