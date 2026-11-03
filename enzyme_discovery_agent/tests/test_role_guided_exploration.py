"""Participant interpretations must change executed work, not just report prose."""
import copy
import json

import pytest

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.models.expansion import expansion_assessments, require_evaluated_seed, select_expansion_seeds
from enzyme_discovery_agent.models.family_learning import refresh_family_learning
from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.output.report_generator import _annotate_reasoned_tiers, _candidate_evidence_payload, _deterministic_mechanistic_ranking
from enzyme_discovery_agent.tools.reaction_roles import record_reaction_roles


def prepare(root):
    for accession, fit in [('SYNTH_A', .99), ('SYNTH_B', .4)]:
        data = {'accession':accession, 'protein_name':accession, 'gene_names':[accession],
                'interpro_family_id':'SYNTH_SHARED_FOLD',
                'catalytic_activities':['Synthetic participants form the declared product.'],
                'function_comments':['The synthetic participant supplies a reaction component through an intermediate.'],
                'required_cofactors':['synthetic cofactor'],
                'native_substrates':[{'name':'synthetic participant','smiles':'SYNTH_NATIVE'}],
                'substrate_fit':fit, 'claim_validation':{'promotion_eligible':True},
                'substrate_similarity':{'accommodation':{'per_target':[
                    {'target_smiles':'SYNTH_TARGET','native_smiles':'SYNTH_NATIVE'}]},
                    'property_comparisons':[{'mw_similarity':fit}]}}
        update_candidate_registry(root, accession, 'evaluated', data)
    (root/'pass1_ranking.json').write_text(json.dumps([
        {'accession':'SYNTH_A','rank':1}, {'accession':'SYNTH_B','rank':2}]))


def assign(root, kind='precursor', unit='atom', site='tunneled_intermediate'):
    return record_reaction_roles(str(root), 'SYNTH_A', [{
        'native_smiles':'SYNTH_NATIVE', 'role':kind, 'transferred_unit':unit, 'site_relation':site,
        'interpretation':'Synthetic interpretation to test work allocation; target activity remains unknown.',
        'source':{'locator':'function_comments/0',
                  'quote':'The synthetic participant supplies a reaction component through an intermediate.'}}])


def registry(root):
    return load_candidate_registry(root)['candidates']


@pytest.mark.parametrize('kind,unit,site', [
    ('precursor','atom','tunneled_intermediate'), ('cofactor','electrons','same_site'),
    ('carrier','fragment','carrier_tethered'), ('unresolved','unresolved','unresolved')])
def test_recorded_roles_redirect_expansion_and_open_focused_review(tmp_path, kind, unit, site):
    prepare(tmp_path)
    assert select_expansion_seeds(registry(tmp_path), 1, tmp_path)==['SYNTH_A']
    old_signal=decisions._signal(decisions.observations(tmp_path)[0])
    assert assign(tmp_path, kind, unit, site)['status']=='success'

    # Same complete saved ranking and legacy fit; new evidence changes the action.
    assert select_expansion_seeds(registry(tmp_path), 1, tmp_path, method='foldseek')==['SYNTH_B']
    policy=refresh_family_learning(tmp_path)
    assert policy['candidates']['SYNTH_A']['state']=='unresolved'
    assert policy['candidates']['SYNTH_A']['fit'] is None
    assert policy['families']['ipr:SYNTH_SHARED_FOLD']['state']=='supported'
    action=expansion_assessments(tmp_path, registry(tmp_path), policy, 'foldseek')['SYNTH_A']
    assert action['decision']=='gather_evidence'
    assert 'reacting site' in action['reason']
    assert require_evaluated_seed(tmp_path, 'SYNTH_A', method='foldseek')['status']=='skipped'
    assert decisions.review_due(tmp_path)  # A role finding need not wait for a large batch.
    assert decisions._signal(decisions.observations(tmp_path)[0])!=old_signal
    packet=decisions.decision_context(tmp_path)
    challenges=[row for row in packet['observations'] if row['facts'].get('accession')=='SYNTH_A']
    assert challenges[0]['facts']['participant_roles']['mappings'][0]['role']==kind

    # Reports receive the same uncertainty; an old scalar cannot restore it.
    data=registry(tmp_path)['SYNTH_A']['metadata']['evaluation_data']
    assert _candidate_evidence_payload(data)['participant_roles']['review_required']
    ranked=_deterministic_mechanistic_ranking([
        {'data':data}, {'data':registry(tmp_path)['SYNTH_B']['metadata']}], {},
        fallback_substrate_fit_by_acc={'SYNTH_A':.99})
    if kind!='unresolved':
        assert ranked[0]['accession']=='SYNTH_B'
    else:
        assert next(row for row in ranked if row['accession']=='SYNTH_A')['tier']==2
    assert 'unresolved' in next(row for row in ranked if row['accession']=='SYNTH_A')['rationale']
    assert registry(tmp_path)['SYNTH_A']['statuses'][-1]['status']=='evaluated'

    # Final tier labelling has its own historical survey-fit fallback.
    data.update(is_swissprot=True, literature={'characterization_papers':10})
    row=[{'accession':'SYNTH_A','rank':1}]
    _annotate_reasoned_tiers(row, {'SYNTH_A':{'data':data}}, {'SYNTH_A'}, set(), {'SYNTH_A':.99})
    assert row[0]['tier']!=1 and row[0]['substrate_fit'] is None


def test_recovery_keeps_role_questions_in_the_literature_frontier(tmp_path, monkeypatch):
    from enzyme_discovery_agent.models.recovery import next_recovery_action
    prepare(tmp_path)
    assign(tmp_path)
    monkeypatch.setattr('enzyme_discovery_agent.models.literature_strategy.target_routes', lambda root:[])
    action=next_recovery_action(tmp_path, {'search_pubmed'})
    assert action['lane']=='participant_role'
    assert action['accession']=='SYNTH_A'
    assert 'SYNTH_A' in action['args']['query']
    assert 'reacting site' in action['reason']


def test_role_constraints_survive_disabled_adaptive_feedback(tmp_path, monkeypatch):
    from enzyme_discovery_agent import config
    prepare(tmp_path)
    assign(tmp_path)
    monkeypatch.setattr(config, 'ENABLE_EXPANSION_FEEDBACK', False)
    assert select_expansion_seeds(registry(tmp_path), 1, tmp_path)==['SYNTH_B']
    assert select_expansion_seeds(registry(tmp_path), 1)==['SYNTH_B']
    assert require_evaluated_seed(tmp_path, 'SYNTH_A')['status']=='skipped'


def test_source_locator_contract_survives_model_schema_compaction():
    from enzyme_discovery_agent.context import compact_tool_definitions
    from enzyme_discovery_agent.tool_registry import get_tool_info
    schema=compact_tool_definitions([get_tool_info('record_reaction_roles')['schema']])[0]
    locator=schema['function']['parameters']['properties']['roles']['items']['properties']['source']['properties']['locator']
    assert locator['pattern']==r'^(function_comments|catalytic_activities)/[0-9]+$'
    assert 'function_comments/0' in locator['description']


def test_object_source_contract_reaches_the_model_with_its_required_fields():
    from enzyme_discovery_agent.context import compact_tool_definitions
    from enzyme_discovery_agent.tool_registry import get_tool_info
    schema=compact_tool_definitions([get_tool_info('record_reaction_step')['schema']])[0]
    source=schema['function']['parameters']['properties']['source']
    assert source['required']==['locator','quote']
    assert source['properties']['locator']['pattern']==r'^catalytic_activities/[0-9]+$'
    assert source['properties']['quote']['minLength']==20


def test_changed_evidence_requires_review_and_fresh_role_assertions_can_reopen_work(tmp_path):
    prepare(tmp_path)
    assign(tmp_path, 'direct_reactant', 'whole_molecule', 'same_site')
    assert select_expansion_seeds(registry(tmp_path), 1, tmp_path)==['SYNTH_A']
    current=copy.deepcopy(registry(tmp_path)['SYNTH_A']['metadata']['evaluation_data'])
    current['required_cofactors']=['different retrieved cofactor']
    update_candidate_registry(tmp_path, 'SYNTH_A', None, {
        'evaluation_data':current, 'required_cofactors':current['required_cofactors']})
    summary=role_transfer_summary(current)
    assert summary['record_status'].startswith('stale')
    assert summary['review_required']
    assert select_expansion_seeds(registry(tmp_path), 1, tmp_path)==['SYNTH_B']
    assert assign(tmp_path, 'direct_reactant', 'whole_molecule', 'same_site')['status']=='success'
    assert select_expansion_seeds(registry(tmp_path), 1, tmp_path)==['SYNTH_A']


def test_role_revision_with_same_annotation_changes_signal_and_restores_comparison(tmp_path):
    prepare(tmp_path)
    assign(tmp_path)
    before=decisions._signal(decisions.observations(tmp_path)[0])
    assign(tmp_path, 'direct_reactant', 'whole_molecule', 'same_site')
    assert decisions._signal(decisions.observations(tmp_path)[0])!=before
    assert select_expansion_seeds(registry(tmp_path), 1, tmp_path)==['SYNTH_A']
