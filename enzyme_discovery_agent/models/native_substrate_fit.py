"""Build offline substrate comparisons from one source-resolved native reaction.

Reporting never performs fresh compound lookups or pools independent activities.
Unresolved structures remain an evidence gap for the discovery workflow.
"""
import json
from pathlib import Path


def target_substrate_smiles(root):
    from enzyme_discovery_agent.run_layout import run_file
    try:
        context = json.loads(run_file(Path(root), 'target_context.json').read_text())
    except (OSError, ValueError):
        return ''
    return str(context.get('substrate_smiles') or '')


def add_property_comparisons(root, evaluations):
    """Fill missing comparisons using distinct, recorded reactant occurrences."""
    target = target_substrate_smiles(root)
    if not target:
        return evaluations
    from enzyme_discovery_agent.tools.family_reactions import _substrate_facts
    from enzyme_discovery_agent.substrate_accommodation import accommodation_facts
    targets = [part for part in target.split('.') if part]
    for evaluation in evaluations:
        data = evaluation.get('data', evaluation)
        if not isinstance(data, dict):
            continue
        similarity = data.get('substrate_similarity') or {}
        if similarity.get('property_comparisons'):
            continue
        scope = data.get('native_substrate_reaction_scope') or {}
        if scope:
            substrates = scope.get('native_substrates') or []
        elif len(data.get('catalytic_activities') or []) == 1:
            substrates = data.get('native_substrates') or []
        else:
            continue
        # Missing structures cannot be silently discarded and scored as a full
        # native reactant set. Occurrences, including repeated species, stay distinct.
        if not substrates or any(not isinstance(s, dict) or not s.get('smiles') for s in substrates):
            continue
        reaction = scope.get('reaction') or (data.get('catalytic_activities') or [''])[0]
        if not all(s.get('occurrence_id') for s in substrates):
            from enzyme_discovery_agent.tools.native_reactions import substrate_occurrences
            substrates = substrate_occurrences(reaction, substrates)
        natives = [s['smiles'] for s in substrates]
        facts = _substrate_facts(natives, targets)
        if not facts:
            continue
        data['substrate_similarity'] = {
            **similarity, 'property_comparisons': facts['property_comparisons'],
            'accommodation': similarity.get('accommodation') or accommodation_facts(targets, natives),
            'property_comparison_basis': {
                'source': 'source-resolved native reactant occurrences',
                'reaction': reaction,
                'target_smiles': target, 'native_smiles': natives,
                'scope': 'Computed analogue resemblance; target activity remains unverified'}}
    return evaluations
