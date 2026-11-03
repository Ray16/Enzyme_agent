"""Different group distributions can preserve a two-occurrence reaction step."""
import copy
import json

import pytest

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry,update_candidate_registry
from enzyme_discovery_agent.models.reaction_step import validate_step,step_summary
from enzyme_discovery_agent.models.research_decisions import observations,_signal
from enzyme_discovery_agent.substrate_accommodation import accommodation_facts
from enzyme_discovery_agent.substrate_fit import accommodation_score
from enzyme_discovery_agent.tools.family_reactions import _substrate_facts,select_relevant_members
from enzyme_discovery_agent.tools.native_reactions import substrate_occurrences
from enzyme_discovery_agent.tools.reaction_roles import get_reaction_step_context,record_reaction_step

OMEGA='NCCCC(=O)O'
TARGETS=['O=C(O)CCCC(=O)O','NCCCCN']
NATIVE=('[NH2:1][CH2:2][CH2:3][CH2:4][C:5](=[O:6])[OH:7].'
        '[NH2:8][CH2:9][CH2:10][CH2:11][C:12](=[O:13])[OH:14]>>'
        '[NH2:1][CH2:2][CH2:3][CH2:4][C:5](=[O:6])[NH:8]'
        '[CH2:9][CH2:10][CH2:11][C:12](=[O:13])[OH:14].[OH2:7]')
TARGET=('[O:1]=[C:2]([OH:3])[CH2:4][CH2:5][CH2:6][C:7](=[O:8])[OH:9].'
        '[NH2:10][CH2:11][CH2:12][CH2:13][CH2:14][NH2:15]>>'
        '[O:1]=[C:2]([NH:10][CH2:11][CH2:12][CH2:13][CH2:14][NH2:15])'
        '[CH2:4][CH2:5][CH2:6][C:7](=[O:8])[OH:9].[OH2:3]')
REACTION='2 synthetic substrate = synthetic coupled product + water'


def data():
    return {'accession':'SYNTH_STEP', 'catalytic_activities':[REACTION],
        'native_substrates':[{'name':'synthetic substrate','smiles':OMEGA}]}


def context():
    return {'query':'Form a bond between complementary reacting ends and assess further extension.',
            'substrate_smiles':'.'.join(TARGETS)}


def arguments():
    return dict(native_reaction=NATIVE,target_reaction=TARGET,
        source={'locator':'catalytic_activities/0','quote':REACTION},
        interpretation='Two native substrate occurrences fill separate acyl-donor and attacking-nitrogen roles. '
            'The target has the same local bond edits despite a different distribution of terminal groups.',
        activation_assessment='Net bond changes omit activation intermediates; enzyme acceptance and the activation pathway remain unestablished.',
        continuation_assessment='The target product retains one acid end and one amino end. Their presence does not prove '
            'further enzyme turnover, correct final product distribution or processivity.')


@pytest.mark.parametrize('reaction',[REACTION,'synthetic substrate + synthetic substrate = synthetic coupled product + water'])
def test_two_copies_of_one_species_are_distinct_source_supported_occurrences(reaction):
    inputs=substrate_occurrences(reaction,data()['native_substrates'])
    assert len(inputs)==2
    assert len({item['occurrence_id'] for item in inputs})==2
    natives=[item['smiles'] for item in inputs]
    accommodation=accommodation_facts(TARGETS,natives)
    assert accommodation['unaccommodated_targets']==[]
    substrate=_substrate_facts(natives,TARGETS)
    assert len(substrate['property_comparisons'])==2
    assert accommodation_score({**substrate,'accommodation':accommodation})>0
    chosen=select_relevant_members([{'accession':'SYNTH_STEP','direction_vs_target':'same',
        'substrate':substrate,'accommodation':accommodation}])
    assert chosen[0]['accession']=='SYNTH_STEP'


def test_complementary_ends_preserve_the_local_step_without_matching_whole_group_counts():
    record=validate_step(data(),context(),**arguments())
    assert record['local_bond_changes_match']
    assert len(record['native_bond_changes'])==len(record['target_bond_changes'])==2
    assert len(record['target_products'])==2  # Leaving water is retained, not lost.
    assert 'repeated extension remain scientific hypotheses' in record['validation_scope']
    changed={**context(),'substrate_smiles':'CCO'}
    assert step_summary({**data(),'reaction_step':record},changed)['status']=='stale'


def test_functionality_alone_does_not_invent_a_second_native_occurrence():
    source=data()
    source['catalytic_activities']=['synthetic substrate = synthetic rearranged product']
    args=arguments()
    args['source']['quote']=source['catalytic_activities'][0]
    with pytest.raises(ValueError,match='native occurrences'):
        validate_step(source,context(),**args)


def test_a_leaving_fragment_cannot_disappear_from_the_proposed_reaction():
    args=arguments()
    args['target_reaction']=TARGET.rsplit('.[OH2:3]',1)[0]
    with pytest.raises(ValueError,match='conserve mapped atom'):
        validate_step(data(),context(),**args)


def test_an_intramolecular_cycle_does_not_establish_two_partner_ligation():
    source=data()
    source['catalytic_activities']=['synthetic substrate = synthetic cyclic product + water']
    args=arguments()
    args['source']['quote']=source['catalytic_activities'][0]
    args['native_reaction']=('[NH2:1][CH2:2][CH2:3][CH2:4][C:5](=[O:6])[OH:7]>>'
                             '[NH:1]1[CH2:2][CH2:3][CH2:4][C:5]1=[O:6].[OH2:7]')
    record=validate_step(source,context(),**args)
    assert not record['local_bond_changes_match']


def test_tool_records_a_step_and_exposes_it_to_exploration(tmp_path):
    (tmp_path/'target_context.json').write_text(json.dumps(context()))
    update_candidate_registry(tmp_path,'SYNTH_STEP','evaluated',data())
    packet=get_reaction_step_context(str(tmp_path),'SYNTH_STEP')
    assert packet['status']=='success'
    assert len(packet['data']['native_occurrences'])==2
    before=_signal(observations(tmp_path)[0])
    result=record_reaction_step(str(tmp_path),'SYNTH_STEP',**arguments())
    assert result['status']=='success'
    assert _signal(observations(tmp_path)[0])!=before
    saved=load_candidate_registry(tmp_path)['candidates']['SYNTH_STEP']['metadata']
    assert saved['reaction_step']['local_bond_changes_match']
    assert get_reaction_step_context(str(tmp_path),'SYNTH_STEP')['data']['previous_step']['status']=='source_checked_hypothesis'


def test_step_and_its_limits_are_recalled_after_history_is_discarded(tmp_path):
    from enzyme_discovery_agent.models.research_memory import recall_memory
    from enzyme_discovery_agent.context import build_model_context
    (tmp_path/'target_context.json').write_text(json.dumps(context()))
    update_candidate_registry(tmp_path,'SYNTH_STEP','evaluated',data())
    assert record_reaction_step(str(tmp_path),'SYNTH_STEP',**arguments())['status']=='success'
    # A new reader has no conversation history or in-memory tool output.
    recalled=recall_memory(type(tmp_path)(str(tmp_path)),'reasoning',query='reacting terminal continuation')
    assert recalled['total']==1
    step=recalled['records'][0]['participant_roles']['reaction_step']
    assert step['local_bond_changes_match']
    assert 'does not prove' in step['continuation_assessment']
    assert (tmp_path/recalled['records'][0]['source']).is_file()
    messages=[{'role':'system','content':'Use retrieved evidence and preserve uncertainty.'},
              {'role':'user','content':context()['query']}]
    messages += [{'role':'assistant','content':'Obsolete narrative '*1000} for _ in range(30)]
    compacted=build_model_context(messages,tmp_path)
    assert 'SYNTH_STEP' in json.dumps(compacted)
    assert 'continuation_assessment' in json.dumps(compacted)
    assert len(json.dumps(compacted))<16000
    from enzyme_discovery_agent.export_reasoning_trace import export_trace
    exported=export_trace(tmp_path)
    assert exported['hash_chain_valid'] and exported['source_files_intact']
    assert exported['events'][0]['type']=='reaction_step_recorded'
    assert exported['events'][0]['capture_mode']=='live'
    assert 'continuation' in (tmp_path/'reasoning/trace.md').read_text().lower()


def test_a_direct_fragment_transfer_can_be_supported_by_the_explicit_local_step():
    from enzyme_discovery_agent.models.reaction_roles import validate_roles,apply_roles
    source=data()
    source['reaction_step']=validate_step(source,context(),**arguments())
    roles=validate_roles(source,[{'native_smiles':OMEGA,'role':'direct_reactant',
        'transferred_unit':'fragment','site_relation':'same_site',
        'interpretation':'The reacting scaffold is retained while the leaving group is lost, as shown by the mapped step.',
        'source':{'locator':'catalytic_activities/0','quote':REACTION}}])
    native=[OMEGA,OMEGA]
    result=apply_roles(source,roles,accommodation_facts(TARGETS,native))
    assert result.get('role_transfer_status')!='unresolved'
    changed=copy.deepcopy(source)
    changed['reaction_step']['local_bond_changes_match']=False
    assert apply_roles(changed,roles,accommodation_facts(TARGETS,native))['role_transfer_status']=='unresolved'


def test_a_recorded_partial_step_uses_its_own_native_participants():
    source=data()
    cycle='synthetic substrate = synthetic cyclic product + water'
    source['catalytic_activities']=[cycle,REACTION]
    source['native_substrate_reaction_scope']={'reaction':cycle}
    source['native_reaction_groups']=[{'reaction':cycle,'native_substrates':source['native_substrates']},
        {'reaction':REACTION,'native_substrates':substrate_occurrences(REACTION,source['native_substrates'])}]
    args=arguments();args['source']['locator']='catalytic_activities/1'
    record=validate_step(source,context(),**args)
    assert record['local_bond_changes_match']
    assert step_summary({**source,'reaction_step':record},context())['status']=='source_checked_hypothesis'
    changed=copy.deepcopy(source)
    changed['native_reaction_groups'][1]['native_substrates'][0]['smiles']='NCC(=O)O'
    assert step_summary({**changed,'reaction_step':record},context())['status']=='stale'
