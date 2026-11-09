import json
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.output.discovery_portfolio import write_discovery_portfolio


def candidate(root, accession, name, family, *, verified=True):
    update_candidate_registry(root,accession,'discovered',{
        'protein_name':name,'screening_family_id':family,
        'screening_annotation':{'catalytic_activities':['Synthetic reaction'], 'organism':'Synthetic organism'},
        'claim_validation':{'claims':[{'claim_type':kind,'subject':accession,'verdict':'supported'}
            for kind in ('accession_identity','record_active')] if verified else []}})


def setup(root):
    (root/'family_landscape.json').write_text(json.dumps([
        {'fold_id':fold,'fold_name':'Synthetic fold '+fold,
         'derived_family_hypotheses':[{'name':'Source-derived synthetic hypothesis '+fold,
             'required_catalytic_capability':'Transfer at the requested reaction center'}]}
        for fold in ('FOLD_A','FOLD_B')]))


def test_unreviewed_discoveries_can_reach_structural_investigation_without_target_review(tmp_path):
    setup(tmp_path);candidate(tmp_path,'SYNTH_A','Synthetic catalyst','FOLD_A')
    result=write_discovery_portfolio(tmp_path)
    row,=result['candidates']
    assert row['target_assessment']=='not_assessed'
    assert row['target_activity']=='unestablished'
    assert row['modeling_setup_status']=='requires_step_partner_and_cofactor_validation'
    source=json.loads((tmp_path/row['proposal_source']).read_text())
    assert source['source_record']['catalytic_activities']==['Synthetic reaction']
    assert not (tmp_path/'artifacts/target_reviews').exists()


def test_family_breadth_and_name_groups_precede_more_related_records(tmp_path):
    setup(tmp_path)
    candidate(tmp_path,'A_FIRST','Same synthetic enzyme','FOLD_A')
    candidate(tmp_path,'A_COPY','Same synthetic enzyme','FOLD_A')
    candidate(tmp_path,'A_OTHER','Different synthetic enzyme','FOLD_A')
    candidate(tmp_path,'Z_NEW','Other-family synthetic enzyme','FOLD_B')
    result=write_discovery_portfolio(tmp_path,limit=2)
    assert {r['family_id'] for r in result['candidates']}=={'FOLD_A','FOLD_B'}
    assert result['available_annotation_groups']==3 and result['alternatives']==1


def test_missing_identity_and_unsupported_families_do_not_become_proposals(tmp_path):
    setup(tmp_path)
    candidate(tmp_path,'SYNTH_UNKNOWN','Unknown identity','FOLD_A',verified=False)
    candidate(tmp_path,'SYNTH_NO_BASIS','No hypothesis','FOLD_OTHER')
    assert write_discovery_portfolio(tmp_path)['selected']==0


def test_conflicting_related_assessments_are_visible_not_silently_reconciled(tmp_path):
    setup(tmp_path)
    for acc,decision in [('SYNTH_A','prioritize'),('SYNTH_B','investigate')]:
        candidate(tmp_path,acc,'Shared source annotation','FOLD_A')
        update_candidate_registry(tmp_path,acc,None,{'target_assessment':{'decision':decision}})
    result=write_discovery_portfolio(tmp_path)
    assert result['selected']==1
    assert result['candidates'][0]['group_priority_status']=='conflicting_assessments_need_comparison'


def test_discovery_proposals_reach_modeling_selection_without_activity_review(tmp_path, monkeypatch):
    from enzyme_discovery_agent.models.modeling_portfolio import select_modeling_portfolio
    from enzyme_discovery_agent import config
    # This test exercises the no-review discovery->modeling path, selected when
    # AUTO_TARGET_REVIEW is off. The default now couples it on in direct mode, so
    # pin it off here to test the path this test is about.
    monkeypatch.setattr(config, 'AUTO_TARGET_REVIEW', False)
    setup(tmp_path);candidate(tmp_path,'SYNTH_A','Synthetic catalyst','FOLD_A')
    rows=select_modeling_portfolio(tmp_path,max_candidates=1)
    assert len(rows)==1 and rows[0]['priority']=='explore'
    assert rows[0]['modeling_setup_status']=='requires_step_partner_and_cofactor_validation'


def test_small_modeling_batch_does_not_shrink_the_discovery_deliverable(tmp_path, monkeypatch):
    from enzyme_discovery_agent.models.modeling_portfolio import select_modeling_portfolio
    from enzyme_discovery_agent import config
    monkeypatch.setattr(config, 'AUTO_TARGET_REVIEW', False)
    setup(tmp_path)
    candidate(tmp_path,'SYNTH_A','First synthetic catalyst','FOLD_A')
    candidate(tmp_path,'SYNTH_B','Other synthetic catalyst','FOLD_B')
    assert len(select_modeling_portfolio(tmp_path,max_candidates=1))==1
    assert json.loads((tmp_path/'discovery_modeling_handoff.json').read_text())['selected']==2


def test_relaxed_lookup_cannot_substitute_an_unrelated_family():
    from enzyme_discovery_agent.gates.family_relevance import resolve_family_folds
    def search(term,max_results=8):
        return {'used_query':'enzyme','data':[
            {'accession':'IPR_SYNTH','name':'Unrelated scavenger','type':'family'}]}
    assert resolve_family_folds({'name':'Zorvik family','descriptors':['Zorvik enzyme']},
                                {'IPR_SYNTH':5},search)==[]


def test_a_proposal_carries_the_candidate_specific_reasoning_already_recorded(tmp_path):
    """Without this the artifact cannot say why THIS enzyme was proposed."""
    setup(tmp_path); candidate(tmp_path,'SYNTH_A','Synthetic catalyst','FOLD_A')
    assessment={'decision':'investigate','evidence_level':'mechanistic_transfer',
                'reason':'Native donor resembles the requested one at the reacting end.',
                'mechanism_and_step':'Activates the carboxylate before amine attack.',
                'strengths':['Reported tolerance of a longer donor'],
                'unknowns':['Acceptance of the requested acceptor is untested'],
                'constraints':[{'finding':'Acceptor is longer','basis':'missing_evidence','severity':'minor'}],
                'target_roles':[{'participant':'T1','role':'donor','basis':'transfer'}],
                'revisit_when':'A kinetic measurement on the requested acceptor appears.',
                'source_refs':['R1'],'created_at':'2026-11-08T00:00:00+00:00'}
    update_candidate_registry(tmp_path,'SYNTH_A',None,{'target_assessment':assessment,
        'target_assessment_artifact':'artifacts/target_reviews/synthetic.json'})
    row,=write_discovery_portfolio(tmp_path)['candidates']
    recorded=json.loads((tmp_path/row['proposal_source']).read_text())['recorded_assessment']
    assert recorded['decision']=='investigate'
    assert recorded['reason']==assessment['reason']
    assert recorded['mechanism_and_step']==assessment['mechanism_and_step']
    assert recorded['strengths']==assessment['strengths']
    assert recorded['unknowns']==assessment['unknowns']
    assert recorded['revisit_when']==assessment['revisit_when']
    # The full record stays inspectable rather than being summarized away.
    assert recorded['artifact']=='artifacts/target_reviews/synthetic.json'


def test_an_unassessed_proposal_says_so_instead_of_looking_reasoned(tmp_path):
    setup(tmp_path); candidate(tmp_path,'SYNTH_A','Synthetic catalyst','FOLD_A')
    row,=write_discovery_portfolio(tmp_path)['candidates']
    recorded=json.loads((tmp_path/row['proposal_source']).read_text())['recorded_assessment']
    assert recorded['decision']=='not_assessed'
    assert 'no candidate-specific reasoning was recorded' in recorded['scope']
    assert 'artifact' not in recorded
