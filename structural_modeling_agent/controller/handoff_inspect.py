"""Inspect unadmitted discovery leads offline without starting a model."""
import json
from pathlib import Path

from enzyme_discovery_agent.models.handoff_contract import adapt_legacy_handoff, load_document, validate_handoff
from structural_modeling_agent.validation import file_digest


def lead_page(task, accession, offset=0):
    if type(offset) is not int or offset < 0:
        raise ValueError('offset must be a nonnegative integer')
    lead = next((row for row in task.get('unresolved_leads', []) if row.get('accession') == accession), None)
    if lead is None:
        raise ValueError('unknown investigation lead')
    text = json.dumps(lead, ensure_ascii=False, allow_nan=False)
    return {'accession': accession, 'offset': offset, 'characters': len(text), 'text': text[offset:offset + 10000],
            'next_offset': offset + 10000 if offset + 10000 < len(text) else None,
            'admission': 'investigation_only'}


def inspect_handoff(path, accession=None, offset=0):
    path = Path(path)
    task = load_document(path)
    if isinstance(task, dict) and task.get('schema_version') == '3.0':
        task = validate_handoff(task)
    elif isinstance(task, dict) and task.get('schema_version') == '2.0':
        task = adapt_legacy_handoff(task, file_digest(path))
    else:
        raise ValueError('unsupported discovery handoff schema')
    if accession:
        return lead_page(task, accession, offset)
    return {'schema_version': task['schema_version'], 'discovery_status': (task.get('producer') or {}).get('discovery_status'),
            'admitted_candidates': len(task['candidates']), 'investigation_leads': [
                {'accession': row.get('accession'), 'reason': row.get('reason')} for row in task['unresolved_leads']],
            'excluded_candidates': [{'accession': row.get('accession'), 'reason': row.get('reason')}
                                    for row in task['excluded_candidates']],
            'note': 'Inspection does not admit a lead or start modeling.'}
