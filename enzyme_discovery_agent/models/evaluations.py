"""Current candidate eligibility shared by reports and structural handoffs."""

from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status, load_candidate_registry,
)


def registered_hypothesis(root,accession,metadata):
    """A source-cited hypothesis can be assessed without a native reaction annotation."""
    import json
    priority=metadata.get('priority_assessment') or {}
    if metadata.get('is_literature_lead') or priority.get('decision') not in {'investigate','prioritize'}:return False
    if not priority.get('strengths') or not priority.get('evidence_refs'):return False
    path=(Path(root)/(metadata.get('priority_artifact') or '')).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():return False
    try:
        if json.loads(path.read_text())!=priority:return False
    except (OSError,ValueError):return False
    from .target_review import identity_verified
    return identity_verified(root,accession,metadata)


def shortlist_eligible(data):
    return (data.get('priority_assessment') or {}).get('decision')!='exclude'


def priority_concerns(data):
    """Known conflicts cannot be compensated by favorable unrelated metrics."""
    concerns=list(data.get('critical_warnings') or [])
    assessment=data.get('priority_assessment') or {}
    concerns.extend(item.get('finding','') for item in assessment.get('weaknesses') or []
                    if isinstance(item,dict) and item.get('basis')=='observed_mismatch' and item.get('severity')=='major')
    accommodation=(data.get('substrate_similarity') or {}).get('accommodation') or {}
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
    roles=role_transfer_summary(data)
    if not str(roles.get('record_status') or '').startswith('stale'):
        for mapping in roles.get('mappings') or []:
            if mapping.get('role') not in {None,'unresolved'}:
                concerns.append('The fitted native participant has a different transfer role; whole-molecule similarity cannot establish its replacement.')
    from enzyme_discovery_agent.models.reaction_step import step_summary
    mapped=step_summary(data).get('local_bond_changes_match') is True
    for pair in accommodation.get('per_target') or []:
        if not pair.get('native_smiles'):
            continue  # An unresolved structure is not a demonstrated bad partner.
        if (pair.get('target_carbons') or 0)>=2 and (pair.get('native_is_cofactor_scale') is True or pair.get('native_scaffold_comparable') is False):
            concerns.append('A required organic input maps only to a structurally incomparable native partner.')
        if (not mapped and pair.get('target_functional_groups') and pair.get('functional_group_similarity') is not None
                and pair.get('shares_functional_groups') is False):
            concerns.append('A resolved native counterpart lacks the target reactive-group context; role transfer needs investigation.')
    return list(dict.fromkeys(concerns))


def prioritized_hypothesis(data):
    """A top-tier starting point needs a recorded mechanistic rationale."""
    if priority_concerns(data):
        return False
    assessment=data.get('priority_assessment') or {}
    if assessment.get('decision')=='prioritize':
        return True
    from enzyme_discovery_agent.models.reaction_step import step_summary
    return step_summary(data).get('local_bond_changes_match') is True


def current_evaluations(run_dir: Path, evaluations: list[dict]) -> list[dict]:
    """Keep the latest evaluation per accession, respecting current decisions.

    A registry's current rejection or failed claim validation takes precedence
    over historical artifacts. Artifact-only legacy runs remain readable.
    """
    registry = load_candidate_registry(Path(run_dir), strict=True).get("candidates", {})
    latest = {}
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        data = evaluation.get("data", evaluation)
        if not isinstance(data, dict):
            continue
        accession = data.get("accession")
        if accession:
            latest[accession] = evaluation
    retained = []
    for accession, evaluation in latest.items():
        data = evaluation.get("data", evaluation)
        entry = registry.get(accession, {})
        metadata = entry.get("metadata", {})
        if metadata.get('priority_assessment'):
            data={**data,'priority_assessment':metadata['priority_assessment']}
            evaluation={**evaluation,'data':data} if 'data' in evaluation else data
        if not shortlist_eligible(data):
            continue
        if metadata.get('reaction_step'):
            data={**data,'reaction_step':metadata['reaction_step']}
            evaluation={**evaluation,'data':data} if 'data' in evaluation else data
        if metadata.get('reaction_roles') and metadata.get('substrate_similarity'):
            data={**data,'reaction_roles':metadata['reaction_roles'],
                  'substrate_similarity':metadata['substrate_similarity']}
            evaluation={**evaluation,'data':data} if 'data' in evaluation else data
        if current_candidate_status(entry) == "rejected":
            continue
        if data.get("is_enzyme") is False or metadata.get("is_enzyme") is False:
            continue
        validations = (metadata.get("claim_validation"), data.get("claim_validation"))
        if any(v and v.get("promotion_eligible") is not True for v in validations) and not registered_hypothesis(run_dir,accession,metadata):
            continue
        retained.append(evaluation)
    return retained
