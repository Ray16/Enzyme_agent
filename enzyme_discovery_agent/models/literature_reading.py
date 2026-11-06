"""Durable download and extraction work; a batch size is not a campaign cap."""

from pathlib import Path
import hashlib
import json
from enzyme_discovery_agent.models.family_learning import _read
from enzyme_discovery_agent.state_store import atomic_write_json

TERMINAL_EXTRACTIONS = {"done", "irrelevant"}


def content_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def paper_signature(paper):
    # Ranking scores and transport annotations are not new source evidence.
    return content_signature({key: paper.get(key) for key in
        ('title', 'abstract', 'doi', 'pmid', 'pmcid', 'url', 'year')})


def reading_needs(run_dir):
    """Current claims to verify and uncertainties to investigate; no reading verdicts."""
    from .candidate_registry import load_candidate_registry
    from .target_review import scientific_record, digest
    from enzyme_discovery_agent.run_layout import run_file
    root = Path(run_dir)
    target = _read(run_file(root, 'target_context.json'), {})
    needs = []
    for accession, entry in sorted(load_candidate_registry(root, strict=True)['candidates'].items()):
        meta = entry.get('metadata') or {}
        assessment = meta.get('target_assessment') or {}
        origin = meta.get('literature_lead') or {}
        if not (assessment or meta.get('is_literature_lead')):
            continue
        needs.append({'accession': accession, 'protein_name': meta.get('protein_name'),
            'identity_status': meta.get('identity_resolution_status'),
            'source_paper': origin.get('paper'),
            'decision': assessment.get('decision'), 'claims_to_verify': assessment.get('strengths', []),
            'mechanism_and_step': assessment.get('mechanism_and_step'),
            'target_roles': assessment.get('target_roles', []),
            'uncertainties': assessment.get('unknowns', []), 'constraints': assessment.get('constraints', []),
            'revisit_when': assessment.get('revisit_when'),
            'evidence_version': digest(scientific_record(meta))})
    return {'target': {**{key: target.get(key) for key in ('query', 'substrate_smiles', 'target_profile')},
                       'target_reasoning': target.get('target_reasoning') or _read(run_file(root, 'target_reasoning.json'), {})},
            'candidate_needs': needs}


def needs_signature(run_dir):
    return content_signature(reading_needs(run_dir))


def deferred_papers(run_dir, model=None):
    priority = _read(Path(run_dir)/'agent_state/literature_paper_priority.json', {})
    if priority.get('needs_signature') != needs_signature(run_dir):
        return set()
    if model is not None and priority.get('model') != model:
        return set()
    return set(priority.get('deferred', []))


def reading_state(run_dir):
    return _read(Path(run_dir) / "agent_state" / "literature_reading.json", {"downloads": {}, "extractions": {}})


def save_reading_state(run_dir, state):
    atomic_write_json(Path(run_dir) / "agent_state" / "literature_reading.json", state)


def pending_reading(run_dir, papers):
    from enzyme_discovery_agent.tools.fulltext import _fulltext_key
    from enzyme_discovery_agent.tools.literature_extraction import _is_non_article
    root = Path(run_dir)
    state = reading_state(root)
    acquired = _read(root / "artifacts" / "fulltext" / "manifest.json", [])
    done = {m.get("key") for m in acquired}
    pending = []
    deferred = deferred_papers(root)
    for paper in papers:
        key = _fulltext_key(paper)
        if key and key not in done and paper_signature(paper) not in deferred and not _is_non_article(paper.get("title", "")):
            if state["downloads"].get(key, {}).get("attempts", 0) < 2:
                pending.append(key)
    for row in acquired:
        key = row.get("key")
        extraction = state["extractions"].get(key, {})
        if row.get("text_file") and extraction.get("status") not in TERMINAL_EXTRACTIONS and extraction.get("attempts", 0) < 2:
            pending.append(key)
    return sorted(set(pending))
