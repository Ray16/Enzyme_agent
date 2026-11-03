"""Mode-dependent score semantics and unresolved structural identity evidence."""
from enzyme_discovery_agent.tools.structural_homolog_search import _parse_foldseek_results,_tm_score


def test_eval_is_not_a_tm_score_without_explicit_tmalign_mode():
    assert _tm_score({'eval':1e-30},'3di')==(None,None)
    assert _tm_score({'eval':.9},None)==(None,None)
    assert _tm_score({'eval':.9},'tmalign')==(.9,'tmalign_mean_query_target_tm_via_eval')


def test_explicit_tm_field_precedes_eval_and_invalid_scores_stay_unknown():
    assert _tm_score({'alntmscore':.8,'eval':1e-30},'3di')==(.8,'alntmscore')
    assert _tm_score({'alntmscore':'nan','eval':.9},'tmalign')==(None,None)
    assert _tm_score({'eval':2},'tmalign')==(None,None)


def test_unmapped_structure_hits_are_retained_without_inventing_sequence_identity():
    payload={'results':[{'db':'synthetic_structures','alignments':[[{
        'theader':'SYNTHETIC_STRUCTURE_CHAIN','eval':.8}]]}]}
    unresolved=[]
    assert _parse_foldseek_results(payload,'SYNTHETIC_SEED',search_mode='tmalign',unresolved=unresolved)==[]
    assert unresolved[0]['tm_score']==.8
    assert 'accession' not in unresolved[0]
    assert unresolved[0]['identity_status']=='requires_sequence_identity_mapping_before_candidate_registration'
