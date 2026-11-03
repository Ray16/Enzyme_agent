"""The primary panel preserves catalytic roles and distinct evidence scopes."""
import json

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.tools.discovery_proposals import record_proposals, write_primary_portfolio


def setup(root):
    (root/'artifacts').mkdir()
    (root/'artifacts/source.json').write_text(json.dumps({'text':'Synthetic catalyst transfers between both supplied partners.'}))
    (root/'target_context.json').write_text(json.dumps({'substrate_smiles':'CCO.O=C=O'}))
    update_candidate_registry(root,'SYNTH_A','discovered',{'protein_name':'Synthetic catalyst',
        'claim_validation':{'claims':[{'claim_type':kind,'subject':'SYNTH_A','verdict':'supported'}
            for kind in ('accession_identity','record_active')]}})


def proposal(**changes):
    return {'accessions':['SYNTH_A'],'tier':'primary','family':'Synthetic family',
        'hypothesis':'Both supplied components can be placed at the retrieved catalytic machinery.',
        'stage':'requested_transformation','support_basis':'catalytic_architecture',
        'target_roles':[{'component_index':i,'role':'Synthetic reacting role',
            'basis':'Retrieved synthetic comparator','recognition_change':'Acceptance remains unresolved'} for i in [1,2]],
        'evidence':[{'path':'artifacts/source.json','quote':'Synthetic catalyst transfers between both supplied partners.',
                     'supports':'Synthetic catalytic-transfer hypothesis'}],
        'structural_question':'Can the two partners occupy complementary catalytic positions?',
        'unknowns':['Native and target substrate acceptance remain unresolved'], 'variant_rationale':'',**changes}


def test_unannotated_sequence_can_be_primary_with_source_linked_architecture(tmp_path):
    setup(tmp_path);result=record_proposals(tmp_path,[proposal()])
    row,=result['data']['candidates']
    assert row['enzyme']=='Synthetic catalyst (SYNTH_A)'
    assert row['target_activity']=='unestablished'
    assert not (tmp_path/'artifacts/target_reviews').exists()


def test_family_only_and_later_stage_cannot_fill_primary_slots(tmp_path):
    setup(tmp_path)
    for changes in [{'support_basis':'family_only'},{'stage':'later_extension'},
                    {'target_roles':proposal()['target_roles'][:1]}]:
        result=record_proposals(tmp_path,[proposal(**changes)])
        assert result['status']=='error' and result['data']['selected']==0


def test_exploratory_hypothesis_is_retained_outside_primary_panel(tmp_path):
    setup(tmp_path);result=record_proposals(tmp_path,[proposal(tier='exploratory',support_basis='family_only')])
    assert result['data']['selected']==0 and result['data']['other_proposals']==1
    assert 'SYNTH_A' in (tmp_path/'exploratory_candidates.csv').read_text()


def test_changed_evidence_invalidates_primary_admission(tmp_path):
    setup(tmp_path);assert record_proposals(tmp_path,[proposal()])['data']['selected']==1
    (tmp_path/'artifacts/source.json').write_text('Changed source')
    assert write_primary_portfolio(tmp_path)['selected']==0


def test_generated_proposal_cannot_cite_itself_as_independent_support(tmp_path):
    setup(tmp_path);folder=tmp_path/'artifacts/discovery_quality_proposals';folder.mkdir()
    (folder/'self.json').write_text('Self-generated hypothesis')
    p=proposal(evidence=[{'path':'artifacts/discovery_quality_proposals/self.json',
                         'quote':'Self-generated hypothesis','supports':'Itself'}])
    assert record_proposals(tmp_path,[p])['status']=='error'


def test_primary_only_modeling_never_falls_back_to_family_leads(tmp_path):
    from enzyme_discovery_agent.models.modeling_portfolio import select_modeling_portfolio
    setup(tmp_path);(tmp_path/'agent_state').mkdir(exist_ok=True)
    (tmp_path/'agent_state/discovery_quality_policy.json').write_text(json.dumps({'primary_only':True}))
    assert select_modeling_portfolio(tmp_path,max_candidates=48)==[]
    record_proposals(tmp_path,[proposal()])
    assert select_modeling_portfolio(tmp_path,max_candidates=48)[0]['accession']=='SYNTH_A'


def test_interpretation_schema_discloses_the_runtime_length_limit():
    from enzyme_discovery_agent.tools import research_decisions
    from enzyme_discovery_agent.tool_registry import get_tool_info
    schema=get_tool_info('interpret_research_outcome')['schema']['function']['parameters']['properties']
    assert schema['interpretation']['minLength']==20
    assert schema['interpretation']['maxLength']==research_decisions.BRIEF_FIELD_LIMIT
