"""Bound evidence previews with explicit omissions and intact identifiers."""

import hashlib
import json
from pathlib import Path

from enzyme_discovery_agent.state_store import atomic_write_json


def evidence_snapshot(root, value, category='context_memory'):
    """Save a complete, immutable projection at an inspectable campaign path."""
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    path = Path(root) / 'artifacts/evidence' / category / (hashlib.sha256(raw.encode()).hexdigest() + '.json')
    if not path.exists():
        atomic_write_json(path, json.loads(raw))
    return {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def preview_evidence(value, *, text_limit=None, list_limit=None):
    """Return (preview, omissions); no claim selection or scientific inference.

    Molecular strings, identities and source locators are indivisible. Callers
    supply the complete artifact reference whenever omissions are present.
    """
    omissions = []

    def visit(item, path='', key=''):
        if isinstance(item, dict):
            return {k: visit(v, path + '/' + str(k).replace('~', '~0').replace('/', '~1'), k)
                    for k, v in item.items()}
        if isinstance(item, list):
            kept = item if list_limit is None else item[:list_limit]
            if len(kept) != len(item):
                omissions.append({'path': path, 'total_items': len(item), 'omitted_items': len(item) - len(kept)})
            return [visit(v, path + '/' + str(i), key) for i, v in enumerate(kept)]
        protected = (key in {'id', 'accession', 'source', 'artifact', 'path', 'locator', 'doi', 'pmid', 'url'}
                     or any(part in key for part in ('smiles', 'sha256', '_id', '_path', '_ref')))
        if isinstance(item, str) and text_limit is not None and len(item) > text_limit and not protected:
            omissions.append({'path': path, 'total_characters': len(item), 'omitted_characters': len(item) - text_limit})
            return item[:text_limit] + ' [preview; inspect complete source]'
        return item

    return visit(value), omissions
