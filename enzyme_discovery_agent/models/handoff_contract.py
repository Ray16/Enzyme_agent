"""Versioned, portable discovery-handoff integrity and admission contract."""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import os
import re
import stat
from pathlib import Path

HANDOFF_VERSION = '3.0'
HANDOFF_KIND = 'discovery_handoff'
HASH_PROFILE = 'python-json-sortkeys-utf8-v1'


def canonical_bytes(value):
    """Versioned deterministic Python JSON; not RFC 8785 canonicalization."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def snapshot_payload(document):
    return {key: value for key, value in document.items() if key != 'integrity'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'non-finite JSON number: {value}')


def load_document(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def read_bundle_source(bundle_root, reference):
    """Read the verified bytes through no-follow descriptors rooted in the bundle."""
    name = reference.get('bundle_relative_path')
    path = Path(name) if isinstance(name, str) else None
    if not path or path.is_absolute() or any(p in {'', '.', '..'} for p in path.parts):
        raise ValueError('invalid bundle-relative path')
    expected = reference.get('sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('source fingerprint is missing')
    descriptors = []
    try:
        fd = os.open(Path(bundle_root), os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(fd)
        for component in path.parts[:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            descriptors.append(fd)
        fd = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        descriptors.append(fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
            raise ValueError('source is not an eligible regular file')
        with os.fdopen(os.dup(fd), 'rb') as source:
            contents = source.read(16 * 1024 * 1024 + 1)
        if len(contents) > 16 * 1024 * 1024 or hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError('source unavailable or fingerprint mismatch')
        return contents
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def validate_handoff(doc):
    if not isinstance(doc, dict) or doc.get('kind') != HANDOFF_KIND or doc.get('schema_version') != HANDOFF_VERSION:
        raise ValueError('expected a discovery handoff with schema 3.0')
    allowed = {'kind', 'schema_version', 'created_at', 'source_run', 'producer', 'target',
               'max_candidates', 'ranking', 'candidates', 'unresolved_leads', 'excluded_candidates',
               'evidence_sources', 'selection_status', 'consumer', 'integrity'}
    unexpected = set(doc) - allowed
    if unexpected:
        raise ValueError(f'unsupported handoff fields: {sorted(unexpected)}')
    for key in ('producer', 'target', 'ranking', 'integrity'):
        if not isinstance(doc.get(key), dict):
            raise ValueError(f'handoff {key} must be an object')
    for key in ('candidates', 'unresolved_leads', 'excluded_candidates', 'evidence_sources'):
        if not isinstance(doc.get(key), list):
            raise ValueError(f'handoff {key} must be an array')
    try:
        created = datetime.fromisoformat(doc['created_at'])
        if created.tzinfo is None:
            raise ValueError('timezone is required')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('handoff created_at must be a timezone-aware timestamp') from exc
    producer = doc['producer']
    if not isinstance(producer.get('run_id'), str) or not producer['run_id']:
        raise ValueError('producer run identity is required')
    if producer.get('discovery_status') not in {'completed', 'incomplete', 'running', 'unknown',
                                                'failed', 'budget_exhausted', 'discovery_incomplete', 'interrupted'}:
        raise ValueError('invalid discovery status')
    if not isinstance(producer.get('coverage'), dict):
        raise ValueError('producer coverage is required')
    artifact_hashes = producer.get('source_artifact_sha256')
    required_artifacts = {'candidate_registry', 'target_context', 'ranking', 'sequence_cache', 'candidate_map', 'manifest'}
    if not isinstance(artifact_hashes, dict) or not required_artifacts <= set(artifact_hashes):
        raise ValueError('source artifact fingerprint inventory is incomplete')
    def valid_digest(value):
        return isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value))
    if any(value is not None and not valid_digest(value) for value in artifact_hashes.values()):
        raise ValueError('invalid source artifact fingerprint')
    if any(not valid_digest(producer.get(key)) for key in ('evaluations_input_sha256', 'neighbor_map_input_sha256')):
        raise ValueError('input fingerprints are required')
    if type(doc.get('max_candidates')) is not int or doc['max_candidates'] < 1 or len(doc['candidates']) > doc['max_candidates']:
        raise ValueError('invalid candidate limit')
    ranking = doc['ranking']
    if ranking.get('status') not in {'unverified', 'unavailable'} or not isinstance(ranking.get('note'), str):
        raise ValueError('ranking status and limitation are required')
    if ranking.get('status') == 'unverified' and not valid_digest(ranking.get('source_sha256')):
        raise ValueError('ranking source fingerprint is required')
    if ranking.get('status') == 'unavailable' and ranking.get('source_sha256') is not None:
        raise ValueError('unavailable ranking cannot have a source fingerprint')
    if doc['integrity'].get('hash_profile') != HASH_PROFILE:
        raise ValueError('unsupported handoff hash profile')
    if doc['integrity'].get('target_sha256') != digest(doc['target']):
        raise ValueError('target fingerprint mismatch')
    if doc['integrity'].get('snapshot_sha256') != digest(snapshot_payload(doc)):
        raise ValueError('handoff snapshot fingerprint mismatch')
    if not isinstance(doc['producer'].get('discovery_status'), str) or not doc['producer']['discovery_status']:
        raise ValueError('discovery status is required')
    if doc.get('selection_status') != ('selected' if doc['candidates'] else 'no_admitted_candidates'):
        raise ValueError('selection status contradicts admitted candidates')
    source_ids = set()
    for source in doc['evidence_sources']:
        if not isinstance(source, dict) or source.get('availability') not in {'available', 'unavailable'}:
            raise ValueError('invalid evidence source status')
        source_id = source.get('id') or source.get('sha256') or 'unverified-source'
        if source_id in source_ids:
            raise ValueError('duplicate evidence source ID')
        source_ids.add(source_id)
        if source['availability'] == 'available':
            path = Path(source.get('bundle_relative_path') or '')
            if not source.get('id') or path.is_absolute() or not path.parts or '..' in path.parts:
                raise ValueError('invalid bundled source reference')
    def check_refs(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'source_ref' and (not isinstance(child, str) or child not in source_ids):
                    raise ValueError('evidence source reference is missing')
                if key == 'source_refs':
                    if not isinstance(child, dict) or any(not isinstance(ref, str) or ref not in source_ids for ref in child.values()):
                        raise ValueError('evidence source reference is missing')
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)
    for section in ('target', 'candidates', 'unresolved_leads', 'excluded_candidates'):
        check_refs(doc[section])
    seen = set()
    for candidate in doc['candidates']:
        if not isinstance(candidate, dict):
            raise ValueError('candidate must be an object')
        acc = candidate.get('accession')
        if not isinstance(acc, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', acc) or acc in {'.', '..'} or acc in seen:
            raise ValueError('candidate IDs must be unique safe identifiers')
        seen.add(acc)
        if not all(isinstance(candidate.get(k), dict) for k in ('evidence', 'modeling_input', 'selection', 'identity')):
            raise ValueError('candidate lacks identity, evidence, modeling input or selection')
        selection = candidate['selection']
        if selection.get('admission') not in {'admitted', 'provisional'} or selection.get('eligibility') is not True:
            raise ValueError('candidate lacks an explicit eligible admission')
        expected_admission = 'admitted' if producer['discovery_status'] == 'completed' else 'provisional'
        if selection['admission'] != expected_admission:
            raise ValueError('candidate admission contradicts discovery status')
        rank = selection.get('rank')
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError('candidate rank must be a positive integer or null')
        if selection.get('ranking_status') not in {'unverified', 'unranked'} or (selection['ranking_status'] == 'unranked' and rank is not None):
            raise ValueError('invalid candidate ranking status')
        if not selection.get('policy') or not selection.get('reason'):
            raise ValueError('candidate admission lacks policy or reason')
        validation = candidate['evidence'].get('claim_validation') or {}
        identity = [claim for claim in validation.get('claims', []) if isinstance(claim, dict)
                    and claim.get('claim_type') == 'accession_identity']
        if (validation.get('promotion_eligible') is not True or not identity or
                any(c.get('subject') != acc or c.get('verdict') != 'supported' for c in identity)):
            raise ValueError('candidate lacks supported discovery identity and eligibility')
        if candidate['identity'].get('accession') != acc:
            raise ValueError('candidate identity does not match its accession')
        sequence = candidate.get('sequence')
        seq_info = candidate.get('sequence_identity') or {}
        if sequence is not None:
            if not isinstance(sequence, str) or not re.fullmatch(r'[A-Z]+', sequence):
                raise ValueError('invalid candidate sequence')
            if seq_info.get('sha256') != hashlib.sha256(sequence.encode('ascii')).hexdigest():
                raise ValueError('candidate sequence fingerprint mismatch')
    for name in ('unresolved_leads', 'excluded_candidates'):
        for item in doc[name]:
            if not isinstance(item, dict) or not isinstance(item.get('accession'), str) or not item['accession']:
                raise ValueError(f'{name} entries require an identifier')
            if item['accession'] in seen:
                raise ValueError('an identifier occurs in multiple handoff categories')
            seen.add(item['accession'])
    return doc


def adapt_legacy_handoff(doc, original_sha256):
    """Validate schema 2.0; keep missing admission evidence as unknown leads."""
    if not isinstance(doc, dict) or doc.get('schema_version') != '2.0':
        raise ValueError('unsupported discovery handoff schema; expected 2.0 or 3.0')
    if not isinstance(doc.get('target'), dict) or not isinstance(doc.get('candidates'), list):
        raise ValueError('invalid legacy discovery handoff')
    leads, seen = [], set()
    for candidate in doc['candidates']:
        if not isinstance(candidate, dict) or not all(isinstance(candidate.get(k), dict)
                                                      for k in ('evidence', 'modeling_input', 'selection')):
            raise ValueError('legacy handoff candidate lacks required layers')
        acc = candidate.get('accession')
        if not isinstance(acc, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', acc) or acc in {'.', '..'} or acc in seen:
            raise ValueError('legacy candidate IDs must be unique safe identifiers')
        seen.add(acc)
        leads.append({'accession': acc, 'reason': 'legacy handoff does not establish current admission',
                      'legacy_candidate': copy.deepcopy(candidate)})
    return {'kind': HANDOFF_KIND, 'schema_version': HANDOFF_VERSION, 'source_run': doc.get('source_run'),
            'producer': {'discovery_status': 'unknown', 'source_schema': '2.0',
                         'legacy_source_sha256': original_sha256},
            'target': copy.deepcopy(doc['target']), 'ranking': {'status': 'unknown'},
            'candidates': [], 'unresolved_leads': leads, 'excluded_candidates': [],
            'evidence_sources': [], 'selection_status': 'no_admitted_candidates',
            'legacy_original': copy.deepcopy(doc)}
