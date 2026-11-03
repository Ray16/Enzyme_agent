"""Create a readable, evidence-preserving discovery handoff."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import discovery_status, load_candidate_registry
from enzyme_discovery_agent.models.handoff_contract import HANDOFF_KIND, HANDOFF_VERSION, HASH_PROFILE, digest, snapshot_payload, read_bundle_source, validate_handoff
from enzyme_discovery_agent.models.modeling_handoff import build_modeling_input
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent.state_store import atomic_write_json


def _first(data, metadata, key, default=None):
    if metadata.get(key):
        return metadata[key]
    if data.get(key):
        return data[key]
    return metadata.get(key, data.get(key, default))


def _evidence_layer(accession, data, metadata):
    mechanism = _first(data, metadata, 'mechanism_evidence')
    fields = ('protein_name', 'organism', 'gene_names', 'ec_numbers', 'catalytic_activities',
              'native_substrates', 'literature_evidence', 'interpro_family', 'interpro_family_id',
              'domain_architecture', 'structural_data', 'required_cofactors', 'cofactor_notes',
              'reaction_chemistry', 'claim_validation')
    conflicts = {key: {'evaluation': data[key], 'registry': metadata[key]}
                 for key in fields if data.get(key) is not None and metadata.get(key) is not None
                 and data[key] != metadata[key]}
    return {
        'identity': {'accession': accession, 'protein_name': _first(data, metadata, 'protein_name', ''),
                     'organism': _first(data, metadata, 'organism', ''),
                     'gene_names': _first(data, metadata, 'gene_names', []),
                     'ec_numbers': _first(data, metadata, 'ec_numbers', [])},
        'native_reactions': _first(data, metadata, 'catalytic_activities', []),
        'native_substrates': _first(data, metadata, 'native_substrates', []),
        'literature_reaction': _first(data, metadata, 'literature_evidence'),
        'family_assignment': {'interpro_family': _first(data, metadata, 'interpro_family'),
                              'interpro_family_id': _first(data, metadata, 'interpro_family_id')},
        'domain_architecture': _first(data, metadata, 'domain_architecture', []),
        'structures': _first(data, metadata, 'structural_data', {}),
        'cofactor_annotations': {'names': _first(data, metadata, 'required_cofactors', []),
                                 'notes': _first(data, metadata, 'cofactor_notes', [])},
        'reaction_chemistry': _first(data, metadata, 'reaction_chemistry', {}),
        'mechanism': {'status': mechanism.get('status', 'not_attempted') if isinstance(mechanism, dict) else 'not_attempted',
                      'evidence': mechanism},
        'claim_validation': _first(data, metadata, 'claim_validation'),
        'structural_homology': metadata.get('structural_homology'),
        'source_conflicts': conflicts,
    }


def _modeling_source(evidence, metadata):
    return {'catalytic_activities': evidence['native_reactions'],
            'required_cofactors': evidence['cofactor_annotations']['names'],
            'cofactor_notes': evidence['cofactor_annotations']['notes'],
            'mechanism_evidence': evidence['mechanism']['evidence'],
            'modeling_specification': metadata.get('modeling_specification')}


def _register_source(path, expected, root, sources):
    entry = {'sha256': expected, 'availability': 'unavailable', 'media_type': 'text/plain',
             'reason': 'source unavailable, unverified, or outside this run'}
    try:
        original = Path(path)
        file = (original if original.is_absolute() else root / original).resolve(strict=True)
        allowed = (root / 'artifacts').resolve()
        if not file.is_relative_to(allowed) or not file.is_file() or file.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('source not eligible for bundling')
        content = read_bundle_source(root, {'bundle_relative_path': file.relative_to(root).as_posix(),
                                            'sha256': expected})
        content.decode('utf-8')
        actual = hashlib.sha256(content).hexdigest()
        destination = root / 'handoff_bundle' / 'sources' / f'{actual}.txt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.parent.resolve().is_relative_to(root):
            raise ValueError('bundle directory escapes the run')
        if destination.is_symlink() or (destination.exists() and (not destination.is_file()
                or hashlib.sha256(destination.read_bytes()).hexdigest() != actual)):
            raise ValueError('bundle collision or corruption')
        if not destination.exists():
            fd, tmp = tempfile.mkstemp(dir=destination.parent, prefix='.source-')
            try:
                with os.fdopen(fd, 'wb') as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, destination)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        entry = {'id': actual, 'sha256': actual, 'availability': 'available',
                 'bundle_relative_path': destination.relative_to(root).as_posix(), 'media_type': 'text/plain'}
    except (OSError, ValueError, TypeError):
        pass
    source_id = entry.get('id') or (str(expected) if expected else
        'unverified-' + hashlib.sha256(str(path).encode()).hexdigest())
    entry.setdefault('id', source_id)
    if source_id not in sources or entry['availability'] == 'available':
        sources[source_id] = entry
    return source_id


def _bundle_evidence(value, root, sources):
    """Bundle fingerprinted run sources and mark all other path references unavailable."""
    if isinstance(value, list):
        return [_bundle_evidence(v, root, sources) for v in value]
    if not isinstance(value, dict):
        return value
    paths = {key: v for key, v in value.items() if (key == 'path' or key.endswith('_path')) and v}
    result = {k: _bundle_evidence(v, root, sources) for k, v in value.items()
              if k != 'path' and not k.endswith('_path')}
    refs = {}
    for key, path in paths.items():
        prefix = key[:-5] if key.endswith('_path') else ''
        expected = value.get(prefix + '_sha256') if prefix else None
        expected = expected or value.get('text_sha256') or value.get('sha256')
        refs[key] = _register_source(path, expected, root, sources)
    if len(refs) == 1:
        result['source_ref'] = next(iter(refs.values()))
    elif refs:
        result['source_refs'] = refs
    existing = [result.get('source_ref')] + list((result.get('source_refs') or {}).values()) if isinstance(result.get('source_refs') or {}, dict) else [result.get('source_ref')]
    for ref in existing:
        if isinstance(ref, str) and ref not in sources:
            sources[ref] = {'id': ref, 'availability': 'unavailable',
                            'reason': 'referenced source snapshot was not supplied'}
    return result


def write_structural_modeling_handoff(run_dir: Path, evaluations: list[dict], target: dict,
                                     max_candidates: int = 20, neighbor_map: dict | None = None,
                                     output_path: Path | None = None, *, discovery_status_value=None,
                                     coverage=None) -> Path:
    """Write admitted candidates, investigation leads and explicit exclusions."""
    root = Path(run_dir).resolve()
    path = Path(output_path).resolve() if output_path else root / 'structural_modeling_handoff.json'
    if path.parent != root:
        raise ValueError('handoff must be written in the run directory so bundle paths remain valid')
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('max_candidates must be a positive integer')
    state_files = {'candidate_registry': run_file(root, 'candidate_registry.json'),
                   'target_context': run_file(root, 'target_context.json'),
                   'ranking': run_file(root, 'pass1_ranking.json'),
                   'sequence_cache': run_file(root, 'embedding_store') / 'sequences.json',
                   'candidate_map': run_file(root, 'candidate_map.json'), 'manifest': root / 'manifest.json'}
    def state_hashes():
        return {name: hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
                for name, file in state_files.items()}
    original_hashes = state_hashes()
    registry = load_candidate_registry(root, strict=True).get('candidates', {})
    try:
        ranking = json.loads(run_file(root, 'pass1_ranking.json').read_text())
        if not isinstance(ranking, list):
            ranking = []
    except (OSError, ValueError):
        ranking = []
    try:
        cached_sequences = json.loads((run_file(root, 'embedding_store') / 'sequences.json').read_text())
        if not isinstance(cached_sequences, dict):
            cached_sequences = {}
    except (OSError, ValueError):
        cached_sequences = {}
    rank_by_acc = {r.get('accession'): r for r in ranking if isinstance(r, dict) and r.get('accession')}
    latest = {}
    for evaluation in evaluations or []:
        data = evaluation.get('data', evaluation) if isinstance(evaluation, dict) else None
        if isinstance(data, dict) and data.get('accession'):
            latest[data['accession']] = (evaluation, data)
    order = [a for a in rank_by_acc if a in latest] + [a for a in latest if a not in rank_by_acc]
    sources, candidates, leads, excluded, seen = {}, [], [], [], set()
    status = discovery_status_value or 'unknown'
    try:
        if status == 'unknown':
            status = json.loads((root / 'manifest.json').read_text()).get('status') or 'unknown'
    except (OSError, ValueError):
        pass
    for acc in order:
        evaluation, data = latest[acc]
        entry = registry.get(acc, {})
        meta = entry.get('metadata') or {}
        current = discovery_status(entry)
        validations = [v for v in (meta.get('claim_validation'), data.get('claim_validation')) if isinstance(v, dict)]
        contradictory = any(v.get('promotion_eligible') is not True for v in validations)
        if current == 'rejected' or data.get('is_enzyme') is False or meta.get('is_enzyme') is False:
            excluded.append({'accession': acc, 'reason': 'current rejection or explicit non-enzyme decision',
                             'discovery_status': current, 'claim_validation': validations})
            continue
        from enzyme_discovery_agent.models.target_review import eligible
        if not eligible(root,acc,meta):
            leads.append({'accession':acc,'reason':'Target relevance assessment is absent, deferred or stale.',
                          'discovery_status':current})
            continue
        identity_claims = [claim for v in validations for claim in (v.get('claims') or [])
                           if isinstance(claim, dict) and claim.get('claim_type') == 'accession_identity']
        identity_ok = bool(identity_claims) and all(c.get('verdict') == 'supported' and c.get('subject') == acc
                                                   for c in identity_claims)
        identity_conflicts = [key for key in ('organism', 'sequence')
                              if data.get(key) and meta.get(key) and data[key] != meta[key]]
        claimed_sequence = data.get('sequence') or meta.get('sequence')
        if claimed_sequence and cached_sequences.get(acc) and claimed_sequence != cached_sequences[acc]:
            identity_conflicts.append('sequence_cache')
        evaluation_status = evaluation.get('status')
        if (current not in {'evaluated', 'ranked'} or evaluation_status != 'success'
                or not validations or contradictory or not identity_ok or identity_conflicts
                or not all(v.get('promotion_eligible') is True for v in validations)):
            leads.append({'accession': acc, 'reason': ('identity evidence conflicts' if identity_conflicts else
                          'current evaluation or explicit eligibility is missing'),
                          'discovery_status': current, 'identity_conflicts': identity_conflicts,
                          'evidence': _evidence_layer(acc, data, meta)})
            continue
        if len(candidates) >= max_candidates:
            leads.append({'accession': acc, 'reason': 'eligible but outside the current candidate limit',
                          'discovery_status': current})
            continue
        evidence = _evidence_layer(acc, data, meta)
        modeling = build_modeling_input(_modeling_source(evidence, {'modeling_specification':
            meta.get('modeling_specification') or data.get('modeling_specification')}), target)
        rank = rank_by_acc.get(acc) or {}
        raw_rank = rank.get('rank')
        rank_value = raw_rank if type(raw_rank) is int and raw_rank > 0 else None
        sequence = _first(data, meta, 'sequence')
        sequence_source = 'discovery_evidence' if sequence else 'run_sequence_cache'
        sequence = sequence or cached_sequences.get(acc)
        if not isinstance(sequence, str) or not re.fullmatch(r'[A-Z]+', sequence):
            sequence = None
        seq_identity = {'status': 'supplied_unverified' if sequence else 'unavailable',
                        'sha256': hashlib.sha256(sequence.encode('ascii')).hexdigest() if sequence else None,
                        'source': sequence_source if sequence else None}
        namespace = 'UniProtKB' if any((c.get('evidence') or {}).get('source') == 'UniProtKB' for c in identity_claims) else 'unknown'
        candidates.append({'accession': acc, 'identity': {'accession': acc, 'namespace': namespace,
                           'resolution_status': 'supported_by_discovery_claim'}, 'sequence': sequence, 'sequence_identity': seq_identity,
                           'input_availability': {'msa': {'status': 'not_exported',
                               'reason': 'An existing alignment must be supplied explicitly to the modeling agent.'}},
                           'evidence': evidence, 'modeling_input': modeling,
                           'selection': {'source': 'evaluated_discovery_candidate',
                               'admission': 'admitted' if status == 'completed' else 'provisional',
                               'eligibility': True, 'policy': 'current-claim-validation-v1',
                               'reason': 'Current discovery evaluation passed recorded claim validation.',
                               'discovery_status': current, 'rank': rank_value,
                               'rank_note': 'saved rank is invalid or missing' if rank and rank_value is None else None,
                               'ranking_rationale': rank.get('rationale') or rank.get('reason'),
                               'ranking_status': 'unverified' if rank else 'unranked',
                               'ortholog_alternatives': rank.get('orthologs', []), 'esm_neighbors': []}})
        seen.add(acc)
    for promising in (neighbor_map or {}).get('promising', []):
        if not isinstance(promising, dict):
            continue
        for neighbor in promising.get('neighbors') or []:
            if not isinstance(neighbor, dict):
                continue
            acc = neighbor.get('accession')
            if acc and acc not in latest and acc not in seen:
                seen.add(acc)
                leads.append({'accession': acc, 'reason': 'neighbor without independent evaluation',
                              'neighbor_of': promising.get('accession'), 'distance': neighbor.get('distance')})
    for acc, entry in registry.items():
        if acc not in latest and acc not in seen:
            if discovery_status(entry) == 'rejected':
                excluded.append({'accession': acc, 'reason': 'current discovery rejection',
                                 'discovery_status': 'rejected'})
            else:
                leads.append({'accession': acc, 'reason': 'candidate has not been evaluated in this snapshot',
                              'discovery_status': discovery_status(entry)})
            seen.add(acc)
    created = datetime.now(timezone.utc).isoformat()
    doc = {'kind': HANDOFF_KIND, 'schema_version': HANDOFF_VERSION, 'created_at': created,
           'source_run': str(root), 'producer': {'run_id': root.name, 'discovery_status': status,
              'coverage': coverage if isinstance(coverage, dict) else {'status': 'unknown'},
              'source_artifact_sha256': original_hashes,
              'evaluations_input_sha256': digest(evaluations or []),
              'neighbor_map_input_sha256': digest(neighbor_map or {})},
           'target': target, 'max_candidates': max_candidates,
           'ranking': {'status': 'unverified' if ranking else 'unavailable',
                       'source_sha256': digest(ranking) if ranking else None,
                       'note': 'Ranks are proposals from the saved ranking; freshness is not established.'},
           'candidates': candidates, 'unresolved_leads': leads, 'excluded_candidates': excluded,
           'selection_status': 'selected' if candidates else 'no_admitted_candidates',
           'consumer': {'agent': 'structural_modeling_agent',
                        'authority': 'Discovery evidence and modeling proposals require independent review.'}}
    doc = _bundle_evidence(doc, root, sources)
    doc['evidence_sources'] = list(sources.values())
    doc['integrity'] = {'hash_profile': HASH_PROFILE, 'target_sha256': digest(doc['target']),
                        'snapshot_sha256': digest(snapshot_payload(doc))}
    validate_handoff(doc)
    if state_hashes() != original_hashes:
        raise RuntimeError('discovery state changed while constructing the handoff; retry on a stable snapshot')
    atomic_write_json(path, doc, allow_nan=False)
    _write_readable_summary(path.with_suffix('.md'), doc)
    return path


def _write_readable_summary(path, doc):
    def line(value):
        return str('unknown' if value is None or value == '' else value).replace('\n', ' ').replace('\r', ' ')
    producer = doc['producer']
    rows = ['# Discovery handoff', '',
            f"Discovery status: **{line(producer.get('discovery_status'))}**",
            f"Target: {line(doc['target'].get('query'))}",
            f"Coverage: {line((producer.get('coverage') or {}).get('status') or (producer.get('coverage') or {}).get('covered'))}",
            f"Admitted: {len(doc['candidates'])} · Investigation leads: {len(doc['unresolved_leads'])} · Excluded: {len(doc['excluded_candidates'])}",
            f"Snapshot SHA-256: `{doc['integrity']['snapshot_sha256']}`", '',
            'Evidence and modeling inputs are proposals for independent review.', '', '## Candidates', '']
    for candidate in doc['candidates']:
        evidence, selection, proposal = candidate['evidence'], candidate['selection'], candidate['modeling_input']
        identity = evidence.get('identity') or {}
        rows.extend([f"### {line(candidate['accession'])} — {line(identity.get('protein_name'))}",
                     f"- Organism: {line(identity.get('organism'))}; family: {line((evidence.get('family_assignment') or {}).get('interpro_family'))}",
                     f"- Admission: {line(selection.get('admission'))}; rank: {line(selection.get('rank'))} ({line(selection.get('ranking_status'))})",
                     f"- Reason: {line(selection.get('reason'))}",
                     f"- Sequence: {line(candidate['sequence_identity'].get('status'))}; SHA-256: {line(candidate['sequence_identity'].get('sha256'))}",
                     f"- Alignment: {line(candidate['input_availability']['msa']['status'])}",
                     f"- Discovery reaction evidence: {line('; '.join(str(x) for x in evidence.get('native_reactions') or []))}",
                     f"- Mechanism evidence status: {line((evidence.get('mechanism') or {}).get('status'))}",
                     f"- Proposal status: {line(proposal.get('status'))}",
                     f"- Unresolved: {line('; '.join(proposal.get('unresolved') or []))}",
                     f"- Warnings: {line('; '.join(proposal.get('warnings') or []))}",
                     f"- Conflicting fields: {line(', '.join((evidence.get('source_conflicts') or {}).keys()))}", ''])
    for title, entries in [('Investigation leads', doc['unresolved_leads']), ('Excluded', doc['excluded_candidates'])]:
        rows.extend([f'## {title}', ''])
        rows.extend(f"- {line(item.get('accession'))}: {line(item.get('reason'))}" for item in entries)
        if not entries:
            rows.append('- None')
        rows.append('')
    rows.extend(['## Source snapshots', ''])
    rows.extend(f"- {line(item.get('id') or item.get('sha256'))}: {line(item.get('availability'))}" for item in doc['evidence_sources'])
    if not doc['evidence_sources']:
        rows.append('- None supplied')
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.handoff-summary-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(rows) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
