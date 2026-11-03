"""A protein's different activities cannot contribute fictional joint substrates."""
from enzyme_discovery_agent.tools.native_reactions import reaction_groups,select_reaction_group


def test_each_activity_resolves_its_own_participants_and_retains_provenance():
    comments=[{'commentType':'CATALYTIC ACTIVITY','reaction':{
        'name':'synthetic first transformation','ecNumber':'6.3.2.99',
        'reactionCrossReferences':[{'database':'Rhea','id':'SYNTH_R1'}]}},
        {'commentType':'CATALYTIC ACTIVITY','reaction':{
            'name':'synthetic second transformation','ecNumber':'2.1.1.99',
            'reactionCrossReferences':[{'database':'Rhea','id':'SYNTH_R2'}]}}]
    def participants(entry):
        assert len(entry['comments'])==1
        return ['SYNTH_PARTNER_1'] if entry['comments'][0] is comments[0] else ['SYNTH_PARTNER_2']
    groups=reaction_groups({'comments':comments},participants)
    assert len(groups)==2
    assert groups[0]['chebi_ids']==['SYNTH_PARTNER_1']
    assert groups[1]['chebi_ids']==['SYNTH_PARTNER_2']
    assert groups[0]['rhea_ids']==['SYNTH_R1']


def test_unrelated_activity_cannot_supply_a_better_fitting_missing_partner():
    relevant={'reaction_index':0,'reaction_direction':'formation','bond_types':['amide'],
              'native_substrates':['SYNTH_REACTANT_A','SYNTH_REACTANT_B']}
    unrelated={'reaction_index':1,'reaction_direction':'transfer','bond_types':['C-C'],
               'native_substrates':['SYNTH_OTHER_PARTNER']}
    chosen=select_reaction_group([relevant,unrelated],'formation',{'amide'},
                                lambda group:.2 if group is relevant else 1.0)
    assert chosen is relevant
    assert 'SYNTH_OTHER_PARTNER' not in chosen['native_substrates']


def test_similar_reactions_still_compete_by_fit_without_combining_partners():
    groups=[{'reaction_index':i,'reaction_direction':'formation','bond_types':['C-C'],
             'native_substrates':[f'SYNTH_PARTNER_{i}'],'fit':fit}
            for i,fit in enumerate([.2,.8])]
    chosen=select_reaction_group(groups,'formation',{'C-C'},lambda group:group['fit'])
    assert chosen is groups[1] and chosen['native_substrates']==['SYNTH_PARTNER_1']


def test_unknown_activity_does_not_borrow_global_ec_from_another_domain():
    entry={'ec_numbers':['6.3.2.99'],'comments':[{'commentType':'CATALYTIC ACTIVITY',
            'reaction':{'name':'Synthetic unresolved transformation'}}]}
    result=reaction_groups(entry,lambda one:[])
    assert result[0]['ec_number'] is None
    assert result[0]['bond_types']==[]


def test_reactive_center_is_scoped_while_cross_activity_conflict_is_retained():
    from enzyme_discovery_agent.tools.native_reactions import scoped_activation
    first='substrate_A + ATP = product_A + AMP + diphosphate'
    second='substrate_B + ATP = product_B + ADP + phosphate'
    selected={'reaction_index':0,'reaction':first,'ec_number':'6.3.1.99'}
    result=scoped_activation(selected,[first,second])
    assert result['activities']==[first]
    assert result['activation']['reactive_phosphate']=='Palpha'
    assert result['all_activity_observations']['status']=='conflicting'
    assert result['reaction_scope']['reaction_index']==0
