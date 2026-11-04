"""Both required partners constrain discovery and evaluation together."""
from types import SimpleNamespace

import pytest

from enzyme_discovery_agent.substrate_fit import accommodation_score, comparison_coverage
from enzyme_discovery_agent.chemistry_logic import check_bond_type_mismatch, canonical_bond_types


def pair(target,native,fit):
    return {'target_smiles':target,'native_smiles':native,'mw_similarity':fit,
            'volume_similarity':fit,'chain_length_similarity':fit,
            'functional_group_similarity':1.0,'tpsa_similarity':1.0,'logP_similarity':1.0}


def test_weaker_partner_is_decisive_without_averaging():
    data={'target_smiles':['FIRST','SECOND'],'n_target_molecules':2,
          'property_comparisons':[pair('FIRST','NATIVE_A',.95),pair('SECOND','NATIVE_B',.12)]}
    assert comparison_coverage(data)['complete'] is True
    assert accommodation_score(data)==pytest.approx(.12)


def test_unresolved_partner_is_unknown_even_when_the_other_pair_is_good():
    data={'target_smiles':['FIRST','SECOND'],'n_target_molecules':2,
          'property_comparisons':[pair('FIRST','NATIVE_A',.95)],
          'accommodation':{'unaccommodated_targets':['SECOND']}}
    assert comparison_coverage(data)['missing_target_smiles']==['SECOND']
    assert accommodation_score(data) is None


def test_a_named_partner_without_valid_property_values_is_still_unresolved():
    data={'target_smiles':['FIRST','SECOND'],'n_target_molecules':2,
          'property_comparisons':[pair('FIRST','NATIVE_A',.95),
              {'target_smiles':'SECOND','native_smiles':'NATIVE_B','mw_similarity':float('nan')}]}
    assert comparison_coverage(data)['complete'] is False
    assert accommodation_score(data) is None


def test_repeated_target_occurrences_each_need_a_comparison():
    data={'target_smiles':['REPEATED','REPEATED'],'n_target_molecules':2,
          'property_comparisons':[pair('REPEATED','NATIVE_A',.9)]}
    assert comparison_coverage(data)['complete'] is False
    data['property_comparisons'].append(pair('REPEATED','NATIVE_B',.3))
    assert comparison_coverage(data)['complete'] is True
    assert accommodation_score(data)==pytest.approx(.3)


def test_balanced_candidate_ranks_above_one_with_a_strong_and_a_weak_partner():
    from enzyme_discovery_agent.output.report_generator import _deterministic_mechanistic_ranking
    def candidate(accession,fits):
        return {'data':{'accession':accession,'protein_name':'Synthetic catalyst',
            'substrate_similarity':{'target_smiles':['FIRST','SECOND'],'n_target_molecules':2,
                'property_comparisons':[pair(t,n,fit) for t,n,fit in
                    zip(['FIRST','SECOND'],['NATIVE_A','NATIVE_B'],fits)]}}}
    ranking=_deterministic_mechanistic_ranking([
        candidate('SYNTH_A_UNBALANCED',[.95,.1]),candidate('SYNTH_Z_BALANCED',[.6,.6])],{})
    assert ranking[0]['accession']=='SYNTH_Z_BALANCED'


def test_family_expansion_profile_uses_the_same_weakest_partner():
    from enzyme_discovery_agent.tools.family_reactions import _holistic_fit, summarise_family_profile
    data={'target_smiles':['FIRST','SECOND'],'n_target_molecules':2,
          'dimensions':{'mw_similarity':.9},
          'property_comparisons':[pair('FIRST','NATIVE_A',.95),pair('SECOND','NATIVE_B',.12)]}
    assert _holistic_fit(data)==pytest.approx(.12)
    assert summarise_family_profile([{'substrate':data}])['holistic_fit']['max']==.12
    data['property_comparisons'].pop()
    assert _holistic_fit(data) is None


def test_unknown_joint_fit_is_incomparable_without_creating_a_dominance_cycle():
    from enzyme_discovery_agent.output.report_generator import _priority_fronts
    rows=[{'accession':'SYNTH_A','bottleneck_fit':.6,'axes':{'size':.1}},
          {'accession':'SYNTH_B','bottleneck_fit':.4,'axes':{'size':.9}},
          {'accession':'SYNTH_C','bottleneck_fit':None,'axes':{'size':.5}}]
    _priority_fronts(rows)
    assert rows[0]['front']<rows[1]['front']
    assert rows[2]['front']==rows[0]['front']


def test_descriptive_bond_labels_do_not_create_a_false_critical_warning():
    target={'direction':'formation','bond_types':['ester C–O bond formed between the supplied partners']}
    assert check_bond_type_mismatch({'reaction_direction':'formation','bond_types':['ester']},target)==[]
    assert check_bond_type_mismatch({'bond_types':['glycosidic']},target)
    assert canonical_bond_types(['C–O'])=={'C-O'}


def test_ontology_aliases_exclude_a_product_without_dropping_unknown_participants():
    from enzyme_discovery_agent.tools.family_reactions import _reactant_side_chebi_ids, _norm_chem
    record={'comments':[{'commentType':'CATALYTIC ACTIVITY','reaction':{
        'name':'source acid + source alcohol = source product',
        'reactionCrossReferences':[{'database':'ChEBI','id':name} for name in ['CHEBI:A','CHEBI:B','CHEBI:C']]}}]}
    aliases={'CHEBI:A':['preferred acid(1-)','source acid'],
             'CHEBI:B':['preferred alcohol','source alcohol'],
             'CHEBI:C':['preferred product(1+)','source product']}
    assert _reactant_side_chebi_ids(record,aliases.get)==['CHEBI:A','CHEBI:B']
    assert _norm_chem('N(2)-substituted')!=_norm_chem('N(6)-substituted')


def test_ontology_aliases_preserve_recorded_stoichiometry(monkeypatch):
    from enzyme_discovery_agent import chemistry_logic as chemistry
    from enzyme_discovery_agent.tools.native_reactions import substrate_occurrences
    monkeypatch.setattr(chemistry,'_chebi_alias_cache',{'CHEBI:SYNTH':['source monomer']})
    result=substrate_occurrences('2 source monomer = source product',[
        {'name':'preferred monomer','smiles':'CCO','chebi_id':'CHEBI:SYNTH'}])
    assert len(result)==2
    assert all(row['stoichiometry']==2 and row['stoichiometry_status']=='source_recorded' for row in result)


def test_discovery_and_accommodation_choose_the_same_maximin_property_assignment(monkeypatch):
    from enzyme_discovery_agent import chemistry_logic as chemistry
    from enzyme_discovery_agent.tools import family_reactions, substrate
    from enzyme_discovery_agent.substrate_accommodation import accommodation_facts
    targets=['CCCO','CCCOC'];natives=['CCCCO','CCCCC']
    values={(targets[0],natives[0]):.95,(targets[0],natives[1]):.3,
            (targets[1],natives[0]):.8,(targets[1],natives[1]):.1}
    def compare(t,n,*args):
        return {**pair(t,n,values[t,n]),'functional_groups_target':['hydroxyl'],
                'functional_groups_native':['hydroxyl'],'mw_target':100,'mw_native':100}
    monkeypatch.setattr(chemistry,'compare_substrate_properties',compare)
    monkeypatch.setattr(family_reactions,'compare_substrate_properties',compare)
    monkeypatch.setattr(substrate,'compute_substrate_properties',lambda *a:SimpleNamespace(
        is_aromatic=False,longest_aliphatic_chain=4,molecular_weight=100))
    facts=family_reactions._substrate_facts(natives,targets)
    accommodation=accommodation_facts(targets,natives)
    expected=[(targets[0],natives[1]),(targets[1],natives[0])]
    assert [(p['target_smiles'],p['native_smiles']) for p in facts['property_comparisons']]==expected
    assert [(p['target_smiles'],p['native_smiles']) for p in accommodation['per_target']]==expected
    assert accommodation_score({**facts,'accommodation':accommodation})==pytest.approx(.3)
