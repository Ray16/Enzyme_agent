"""Synthetic regressions for chemistry and complete assessment observations."""
import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from enzyme_discovery_agent.chemistry_logic import extract_target_bond_type
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry


@pytest.mark.parametrize('text,bonds', [
    ('ester formation; assess whether extension is possible', ['ester']),
    ('thioester hydrolysis', ['thioester']),
    ('phosphodiester cleavage', ['phosphodiester']),
    ('polyester hydrolysis', ['ester']),
    ('ether cleavage', ['ether']),
])
def test_bond_labels_are_tokens_not_substrings(text, bonds):
    assert extract_target_bond_type(text)['bond_types'] == bonds


def test_evaluation_uses_saved_direction_despite_conflicting_instruction_words(tmp_path):
    from enzyme_discovery_agent.tools import reasoning
    (tmp_path/'target_reasoning.json').write_text(json.dumps({
        'reaction_direction':'formation', 'bond_types':['ester'],
        'substrate_topology':'free_small_molecule'}))
    entry={'primaryAccession':'P12345', 'entryType':'UniProtKB reviewed (Swiss-Prot)',
        'sequence':{'length':200}, 'organism':{'scientificName':'Synthetic organism'},
        'proteinDescription':{'recommendedName':{'fullName':{'value':'Synthetic catalyst'}}}}
    update_candidate_registry(tmp_path, 'P12345', 'screened')
    with ExitStack() as stack:
        for name,value in {
            '_fetch_uniprot_entry':entry,
            '_count_pubmed_characterization':{'characterization_papers':0,'engineering_papers':0},
            '_check_pdb_availability':{'pdb_ids':[]}, '_check_bacterial_expression':{},
            '_fallback_substrate_lookup':[]}.items():
            stack.enter_context(patch.object(reasoning,name,return_value=value))
        stack.enter_context(patch('enzyme_discovery_agent.tools.database.get_protein_domains',return_value={'data':[]}))
        select=stack.enter_context(patch('enzyme_discovery_agent.tools.native_reactions.select_reaction_group', return_value=None))
        compare=stack.enter_context(patch.object(reasoning,'check_bond_type_mismatch',return_value=[]))
        result=reasoning.evaluate_candidate(tmp_path,'P12345',
            target_reaction_type='ester formation; distinguish it from hydrolysis; assess whether extension occurs')
    assert result['status']=='success'
    assert select.call_args.args[1:3] == ('formation',['ester'])
    assert compare.call_args.args[1] == {'direction':'formation','bond_types':['ester']}


def test_assessment_context_keeps_every_sampled_record(tmp_path):
    from enzyme_discovery_agent.models import research_decisions as decisions
    from enzyme_discovery_agent.tests.test_review_protocol_recovery import pending
    prior,outcome=pending(tmp_path)
    from enzyme_discovery_agent.tests.test_research_decision_loop import plan
    identifier=decisions.start_decision(tmp_path,
        plan(tmp_path,revises=prior,evidence_refs=[outcome]), {'inspect_synthetic_evidence'})
    sample=[]
    for i in range(12):
        acc=f'SYNTH_{i}'
        update_candidate_registry(tmp_path,acc,'evidence_pending', {
            'evaluation_data':{'protein_name':f'Synthetic catalyst {i}',
                               'catalytic_activities':[f'Synthetic reaction {i}']}})
        sample.append({'accession':acc,'status':'evidence_pending'})
    decisions.finish_decision(tmp_path, identifier, {'status':'success','data':{
        'actual_sample':sample,'attempted':12,'evidence_pending':12}})
    records=decisions.decision_context(tmp_path)['previous_test']['assessment']['records']
    assert len(records)==12
    assert records[-1]['native_reactions']==['Synthetic reaction 11']


def test_complementary_reaction_roles_outrank_whole_substrate_overlap():
    from enzyme_discovery_agent.tests.test_reactive_end_transfer import data, context, arguments
    from enzyme_discovery_agent.models.reaction_step import validate_step
    from enzyme_discovery_agent.output.report_generator import _deterministic_mechanistic_ranking
    candidate=data()
    candidate['reaction_step']=validate_step(candidate,context(),**arguments())
    candidate['substrate_similarity']={'property_comparisons':[{
        'functional_group_similarity':.1,'mw_similarity':.6,'volume_similarity':.6}]}
    similar={'accession':'SYNTH_SIMILAR','substrate_similarity':{'property_comparisons':[{
        'functional_group_similarity':1.,'mw_similarity':.99,'volume_similarity':.99}]},
        'mechanism_evidence':{'status':'ok','facts':{'confidence':'high'},'paper':{'pmcid':'SYNTH_PAPER'}}}
    ranking=_deterministic_mechanistic_ranking([{'data':similar},{'data':candidate}],{1:8,2:5,3:2})
    assert ranking[0]['accession']==candidate['accession']
    assert 'mechanism=experimental' in ranking[1]['rationale']


def test_reporting_never_pools_independent_native_reactions(tmp_path):
    from enzyme_discovery_agent.models.native_substrate_fit import add_property_comparisons
    (tmp_path/'target_context.json').write_text(json.dumps({'substrate_smiles':'CCO.CCN'}))
    candidate={'catalytic_activities':['first independent reaction','second independent reaction'],
        'native_substrates':[{'smiles':'CCO'},{'smiles':'CCN'}]}
    add_property_comparisons(tmp_path,[candidate])
    assert 'substrate_similarity' not in candidate


def test_reporting_cannot_reuse_one_native_occurrence_for_two_target_partners(tmp_path):
    from enzyme_discovery_agent.models.native_substrate_fit import add_property_comparisons
    (tmp_path/'target_context.json').write_text(json.dumps({'substrate_smiles':'CCO.CCO'}))
    candidate={'catalytic_activities':['one substrate = one product'],
        'native_substrates':[{'smiles':'CCO'}]}
    add_property_comparisons(tmp_path,[candidate])
    accommodation=candidate['substrate_similarity']['accommodation']
    assert len(accommodation['unaccommodated_targets'])==1


def test_reporting_preserves_source_supported_repeated_occurrences(tmp_path):
    from enzyme_discovery_agent.models.native_substrate_fit import add_property_comparisons
    (tmp_path/'target_context.json').write_text(json.dumps({'substrate_smiles':'CCO.CCO'}))
    candidate={'catalytic_activities':['2 synthetic substrate = synthetic product'],
        'native_substrates':[{'name':'synthetic substrate','smiles':'CCO'}]}
    add_property_comparisons(tmp_path,[candidate])
    accommodation=candidate['substrate_similarity']['accommodation']
    assert not accommodation['unaccommodated_targets']
    assert len(accommodation['per_target'])==2


def test_batched_search_evidence_reaches_the_next_model_decision(tmp_path):
    from enzyme_discovery_agent.models import research_decisions as decisions
    from enzyme_discovery_agent.tests.test_research_decision_loop import plan
    from enzyme_discovery_agent.context import build_model_context
    identifier=decisions.start_decision(tmp_path,plan(tmp_path),{'inspect_synthetic_evidence'})
    results=[{'tool_name':f'synthetic_source_{i}','arguments':{},'status':'success',
              'data':[{'title':f'SOURCE_{i}_TARGET_EVIDENCE','abstract':f'Retrieved observation from source {i}.'}]}
             for i in range(4)]
    decisions.finish_decision(tmp_path,identifier,{'status':'success','data':results})
    messages=[{'role':'system','content':'Reason from retrieved evidence.'},
              {'role':'user','content':'Find candidates for the requested transformation.'}]
    context=json.dumps(build_model_context(messages,tmp_path))
    for i in range(4):
        assert f'SOURCE_{i}_TARGET_EVIDENCE' in context
        assert f'Retrieved observation from source {i}.' in context


def test_literature_enrichment_cannot_reuse_a_native_occurrence():
    from enzyme_discovery_agent.models.candidate_provenance import _accommodation_facts
    assert _accommodation_facts('CCO.CCO',['CCO'])==[]
    assert len(_accommodation_facts('CCO.CCO',['CCO','CCO']))==2


def test_semantic_direction_override_updates_prose_claims_too():
    from enzyme_discovery_agent.models.target_reasoning import derive_target_reasoning
    from enzyme_discovery_agent.tests.test_direction_topology_fixes import _StubLLM
    llm=_StubLLM(json.dumps({'reaction_direction':'formation','bond_types':['ester'],
        'substrate_topology':'free_small_molecule','uncertainty':'low'}))
    contract=derive_target_reasoning('Ester formation; distinguish this from hydrolysis.',llm_client=llm)
    assert contract['reaction_direction']=='formation'
    assert not any(c['claim']=='The required reaction direction is cleavage.' for c in contract['claims'])
    assert any(c['claim']=='The required reaction direction is formation.' for c in contract['claims'])
    assert contract['extraction_disagreements'][0]['keyword_extraction']=='cleavage'


def test_initial_search_map_is_bounded_and_saved_in_verified_trace(tmp_path):
    from types import SimpleNamespace
    from enzyme_discovery_agent.gates.mechanism_family_coverage import MechanismFamilyGate
    from enzyme_discovery_agent.export_reasoning_trace import export_trace
    from enzyme_discovery_agent.tools.mechanism_families import unexplored_promising_families
    family={'name':'Synthetic distinct fold','descriptors':['synthetic architecture'],
        'literature_terms':['synthetic transformation'],'required_catalytic_capability':'A source-grounded reacting role.'}
    paper={'title':'Synthetic primary evidence','abstract':'Source describes the declared transformation.'}
    gate=MechanismFamilyGate(SimpleNamespace(active_label='synthetic'))
    with patch('enzyme_discovery_agent.tools.mechanism_families.derive_mechanism_families',return_value=[family]) as derive:
        result=gate.execute(SimpleNamespace(run_dir=tmp_path,accumulated_papers=[paper]))
    assert result.state_updates['mechanism_families_done']
    assert derive.call_args.kwargs['max_attempts']==2
    assert derive.call_args.kwargs['reflect'] is False
    assert unexplored_promising_families([family],[])==[family]
    trace=export_trace(tmp_path)
    event=trace['events'][0]
    assert event['type']=='mechanism_hypotheses_recorded'
    assert event['capture_mode']=='live'
    assert event['record']['retrieved_papers']==[paper]
    assert trace['hash_chain_valid'] and trace['source_files_intact']
    assert 'Synthetic distinct fold' in (tmp_path/'reasoning/trace.md').read_text()
    from enzyme_discovery_agent.models.research_decisions import _root, review_due
    assert json.loads((_root(tmp_path)/'state.json').read_text())['initial_sources_reviewed']
    assert not review_due(tmp_path)
    (tmp_path/event['record']['source']['path']).write_text('{}')
    assert not export_trace(tmp_path)['source_files_intact']
