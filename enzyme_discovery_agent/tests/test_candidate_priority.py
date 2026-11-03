"""Priority changes must be holistic, reversible, and visible in rolling output."""
import csv
import json
from types import SimpleNamespace
from unittest.mock import Mock

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry,update_candidate_registry,discovery_status
from enzyme_discovery_agent.models.research_decisions import decision_context
from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates
from enzyme_discovery_agent.output.report_generator import _deterministic_mechanistic_ranking,_priority_fronts
from enzyme_discovery_agent.tool_registry import get_tool_info
import enzyme_discovery_agent.tools.research_decisions


def prepare(root):
    for accession in ('SYNTH_A','SYNTH_B'):
        update_candidate_registry(root,accession,'evaluated',{'accession':accession,
            'protein_name':accession,'catalytic_activities':['Synthetic inputs form the declared product.'],
            'claim_validation':{'promotion_eligible':True}})
    agent=EnzymeAgent.__new__(EnzymeAgent)
    agent.run_dir,agent.llm=root,SimpleNamespace(active_label='synthetic:caller')
    agent.tools=[get_tool_info('set_candidate_focus')['schema']]
    agent._decision_action_tools={'set_candidate_focus'}
    agent._require_research_decision=True
    agent._search_history,agent._search_queries,agent._search_log=[],[],[]
    agent.log,agent._log_event=Mock(),Mock()
    return agent


def proposal(root,decision,*,missing=False,previous=None):
    packet=decision_context(root)
    source=next(row['id'] for row in packet['observations'] if row['facts'].get('accession')=='SYNTH_A')
    refs=[source]
    if previous:refs.append(previous['outcome_evidence'])
    return {'hypothesis':'This native role may be a poor starting point for the target.',
        'alternative':'A source-grounded alternative mapping may support further investigation.',
        'scope':'search_strategy','prediction':'The saved priority will guide the next investigation.',
        'contrasting_prediction':'The candidate will remain an unresolved alternative.',
        'why_this_test':'Compare all known strengths, weaknesses and unknowns instead of one score.',
        'revision_trigger':'Reopen if new evidence supports the missing reacting role.',
        'evidence_refs':refs,'tool_name':'set_candidate_focus','revises':previous['research_decision'] if previous else '',
        'belief_update':'suspend','arguments':{'accession':'SYNTH_A','decision':decision,
            'reason':'The observed native reacting role differs; preserve the candidate for a future revision.',
            'strengths':['The native reaction is source-resolved.'],
            'weaknesses':[{'finding':'Synthetic role mismatch requires a different catalytic step.',
                           'basis':'missing_evidence' if missing else 'observed_mismatch','severity':'major'}]
                          if decision in {'exclude','deprioritize'} else [],
            'unknowns':['Target acceptance has not been established.'],
            'revisit_when':'New source evidence establishes a role-compatible reaction step.'}}


def test_candidate_can_leave_and_reenter_the_shortlist_without_losing_history(tmp_path):
    agent=prepare(tmp_path)
    removed=agent._execute_single_tool('plan_research_step',proposal(tmp_path,'exclude'),1)
    assert removed['status']=='success'
    rows=list(csv.DictReader((tmp_path/'candidate_assessment_queue.csv').open()))
    assert [row['accession'] for row in rows]==['SYNTH_B']
    registry=load_candidate_registry(tmp_path)['candidates']
    assert discovery_status(registry['SYNTH_A'])=='evaluated'
    assert registry['SYNTH_A']['metadata']['priority_assessment']['decision']=='exclude'
    reopened=agent._execute_single_tool('plan_research_step',proposal(tmp_path,'investigate',previous=removed),2)
    assert reopened['status']=='success'
    assert {row['accession'] for row in csv.DictReader((tmp_path/'candidate_assessment_queue.csv').open())}=={'SYNTH_A','SYNTH_B'}
    assert not list(csv.DictReader((tmp_path/'ranked_candidates.csv').open()))
    assert len(list((tmp_path/'artifacts/shortlist_decisions/SYNTH_A').glob('*.json')))==2


def test_missing_evidence_cannot_be_used_as_an_exclusion(tmp_path):
    agent=prepare(tmp_path)
    result=agent._execute_single_tool('plan_research_step',proposal(tmp_path,'exclude',missing=True),1)
    assert result['status']=='error'
    assert not load_candidate_registry(tmp_path)['candidates']['SYNTH_A']['metadata'].get('priority_assessment')


def test_missing_candidate_citation_returns_its_current_record_without_assigning_priority(tmp_path):
    agent=prepare(tmp_path)
    args=proposal(tmp_path,'investigate')
    args['evidence_refs']=[next(row['id'] for row in decision_context(tmp_path)['observations']
                                if row['facts'].get('accession')=='SYNTH_B')]
    result=agent._execute_single_tool('plan_research_step',args,1)
    assert result['status']=='error'
    references=result['repair_context']['candidate_evidence']
    assert references and all(row['facts']['accession']=='SYNTH_A' for row in references)
    assert not load_candidate_registry(tmp_path)['candidates']['SYNTH_A']['metadata'].get('priority_assessment')


def test_direct_tool_guard_rejects_unoffered_calls_but_preserves_planned_actions(tmp_path):
    agent=prepare(tmp_path);agent._offered_direct_tools={'plan_research_step'}
    denied=agent._execute_model_tool('get_decision_context',{},1)
    assert denied['status']=='error' and not denied['decision_executed']
    accepted=agent._execute_model_tool('plan_research_step',proposal(tmp_path,'investigate'),2)
    assert accepted['status']=='success'
    assert not agent._executing_model_selection


def test_a_major_known_conflict_beats_high_scores_and_publication_counts():
    bad={'accession':'SYNTH_FAMOUS','literature':{'characterization_papers':10000},
         'substrate_similarity':{'property_comparisons':[{'mw_similarity':.99,'volume_similarity':.99}],
             'accommodation':{'per_target':[{'native_smiles':'C','target_smiles':'CCCCCC',
                 'target_carbons':6,'native_is_cofactor_scale':True}]}},'is_swissprot':True}
    unknown={'accession':'SYNTH_UNKNOWN'}
    reasonable={'accession':'SYNTH_ANALOGUE','substrate_similarity':{'property_comparisons':[{'mw_similarity':.5,'volume_similarity':.5}]}}
    ranked=_deterministic_mechanistic_ranking([{'data':bad},{'data':unknown},{'data':reasonable}],{})
    assert ranked[-1]['accession']=='SYNTH_FAMOUS' and ranked[-1]['tier']==3
    assert next(row for row in ranked if row['accession']=='SYNTH_UNKNOWN')['tier']==2
    assert all(row['tier']!=1 for row in ranked)


def test_tradeoffs_and_missing_dimensions_are_not_averaged_into_a_false_winner():
    rows=[{'accession':'A','axes':{'size':.99,'role_context':.1}},
          {'accession':'B','axes':{'size':.5,'role_context':.5}},
          {'accession':'C','axes':{}}]
    _priority_fronts(rows)
    assert [row['front'] for row in rows]==[0,0,0]


def test_rolling_export_adds_new_evaluations_instead_of_reusing_an_old_ranking(tmp_path):
    prepare(tmp_path)
    refresh_ranked_candidates(tmp_path)
    update_candidate_registry(tmp_path,'SYNTH_NEW','evaluated',{'accession':'SYNTH_NEW','protein_name':'New synthetic lead'})
    refresh_ranked_candidates(tmp_path)
    assert {row['accession'] for row in csv.DictReader((tmp_path/'candidate_assessment_queue.csv').open())}=={'SYNTH_A','SYNTH_B','SYNTH_NEW'}


def test_recorded_scientific_priority_changes_the_next_expansion_seed(tmp_path):
    from enzyme_discovery_agent.models.expansion import select_expansion_seeds
    agent=prepare(tmp_path)
    for accession,value in [('SYNTH_A',.2),('SYNTH_B',.99)]:
        update_candidate_registry(tmp_path,accession,None,{'substrate_similarity':{
            'property_comparisons':[{'mw_similarity':value}]}})
    refresh_ranked_candidates(tmp_path)
    assert select_expansion_seeds(load_candidate_registry(tmp_path)['candidates'],1,tmp_path)==['SYNTH_B']
    result=agent._execute_single_tool('plan_research_step',proposal(tmp_path,'prioritize'),1)
    assert result['status']=='success'
    assert select_expansion_seeds(load_candidate_registry(tmp_path)['candidates'],1,tmp_path)==['SYNTH_A']


def test_duplicate_leaders_do_not_hide_an_unresolved_relevant_family(tmp_path):
    from enzyme_discovery_agent.models.expansion import select_expansion_seeds
    # All are evaluated, eligible leads. The second family has an unresolved
    # native interpretation; it has no observed conflict with the target.
    for accession, family, fit in [('SYNTH_A','FOLD_A',.9),
                                   ('SYNTH_B','FOLD_A',.8),
                                   ('SYNTH_C','FOLD_B',.7)]:
        update_candidate_registry(tmp_path,accession,'evaluated',{
            'accession':accession,'interpro_family_id':family,
            'substrate_similarity':{'property_comparisons':[{'mw_similarity':fit}]}})
    selected=select_expansion_seeds(load_candidate_registry(tmp_path)['candidates'],
                                    2,tmp_path,method='foldseek')
    assert selected==['SYNTH_A','SYNTH_C']


def test_display_surfaces_families_among_incomparable_leads_without_rescuing_conflicts():
    data=[{'accession':'A1','interpro_family_id':'FOLD_A'},
          {'accession':'A2','interpro_family_id':'FOLD_A'},
          {'accession':'B1','interpro_family_id':'FOLD_B'},
          {'accession':'C1','interpro_family_id':'FOLD_C',
           'critical_warnings':['Source-resolved reaction conflicts with the target.']}]
    ranked=_deterministic_mechanistic_ranking([{'data':row} for row in data],{})
    assert [row['accession'] for row in ranked]==['A1','B1','A2','C1']


def test_family_breadth_preserves_observed_dominance():
    data=[{'accession':'A1','interpro_family_id':'FOLD_A'},
          {'accession':'A2','interpro_family_id':'FOLD_A'},
          {'accession':'B1','interpro_family_id':'FOLD_B'}]
    for row,value in zip(data,(.9,.8,.2)):
        row['substrate_similarity']={'property_comparisons':[{'target_smiles':'CCO','mw_similarity':value}]}
    ranked=_deterministic_mechanistic_ranking([{'data':row} for row in data],{})
    assert [row['accession'] for row in ranked]==['A1','A2','B1']
