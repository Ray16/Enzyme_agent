"""Preserve reaction-specific annotation provenance separately from target fit."""
from pathlib import Path
import hashlib
import json


def record_digest(entry):
    return hashlib.sha256(json.dumps(entry,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def summarize_native_annotation(entry):
    """Describe cited evidence; reviewed status alone never implies an assay."""
    reactions = []
    for comment in entry.get('comments', []):
        if comment.get('commentType') != 'CATALYTIC ACTIVITY':
            continue
        reaction = comment.get('reaction') or {}
        evidence = reaction.get('evidences') or []
        codes = {item.get('evidenceCode') for item in evidence}
        if 'ECO:0000269' in codes:
            level = 'experimental_evidence_cited'
        elif codes & {'ECO:0000256', 'ECO:0000259'}:
            level = 'automatically_inferred'
        elif codes & {'ECO:0000250', 'ECO:0000255', 'ECO:0000305'}:
            level = 'inferred_annotation'
        else:
            level = 'evidence_not_specified'
        reactions.append({'reaction': reaction.get('name'), 'evidence_level': level,
                          'evidences': evidence})
    levels = {row['evidence_level'] for row in reactions}
    return {
        'accession': entry.get('primaryAccession'),
        'record_status': entry.get('entryType', 'unknown'),
        'protein_existence': entry.get('proteinExistence', 'unknown'),
        'native_activity_evidence': next(iter(levels)) if len(levels) == 1 else
            'mixed_reaction_evidence' if levels else 'native_reaction_unknown',
        'reactions': reactions,
        'scope': 'Evidence attached to native-reaction annotations. This does not establish target activity; inspect the cited study and exact protein identity.',
    }


def native_annotation_for_report(run_dir, accession, data):
    """Use preserved reaction evidence or a separately retrieved source snapshot."""
    for claim in (data.get('claim_validation') or {}).get('claims', []):
        if claim.get('claim_type') == 'native_reaction':
            preserved = (claim.get('evidence') or {}).get('annotation_provenance')
            if isinstance(preserved, dict) and preserved.get('accession') == accession:
                return preserved
    # Publication snapshots are reporting evidence, not fresh discovery seeds.
    root = Path(run_dir).resolve()
    path = root / 'artifacts/native_annotations' / (str(accession) + '.json')
    if path.resolve().is_relative_to(root) and path.is_file():
        try:
            saved = json.loads(path.read_text())
            entry = saved['record']
            if entry.get('primaryAccession') == accession and record_digest(entry) == saved.get('record_sha256'):
                return summarize_native_annotation(entry)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return {'native_activity_evidence': 'evidence_not_preserved', 'reactions': [],
            'scope': 'The saved annotation does not preserve reaction-specific evidence codes. Reviewed status and search hit counts do not establish experimental characterization.'}
