"""Reaction discovery must retain native chemistry before sequence mapping."""
from unittest.mock import patch

import pytest

from enzyme_discovery_agent.tools import kbase as database
from enzyme_discovery_agent.context import ARTIFACT_CATEGORIES


def response(compounds=None,**extra):
    return {'response':{'numFound':3,'docs':[{'id':'rxn99991',
        'definition':'Synthetic first partner + second partner = coupled product',
        'ec_numbers':['2.1.1.900'],'compound_ids':compounds or ['cpd99991','cpd99992','cpd99993'],
        'aliases':['Rhea: 99990; 99991','MetaCyc: SYNTHETIC-REACTION'],**extra}]}}


def test_full_ec_is_exact_and_retains_all_participants_and_mapping_identifiers():
    with patch.object(database,'_solr_query',return_value=response()) as query:
        result=database.search_modelseed_reactions('2.1.1.900')
    assert query.call_args.args[1]=='ec_numbers:2.1.1.900'
    record=result['data'][0]
    assert record['compound_ids']==['cpd99991','cpd99992','cpd99993']
    assert record['cross_references']=={'Rhea':['99990','99991'],'MetaCyc':['SYNTHETIC-REACTION']}
    assert record['complete_ec_numbers']==['2.1.1.900']
    assert result['total_count']==3 and result['next_offset']==1


@pytest.mark.parametrize('prefix',['2.1','2.1.','2.1.*','2.1.-'])
def test_partial_ec_searches_descendants(prefix):
    with patch.object(database,'_solr_query',return_value=response()) as query:
        database.search_modelseed_reactions(prefix)
    assert query.call_args.args[1]=='ec_numbers:2.1.*'


def test_chemical_search_does_not_require_an_enzyme_name_or_ec():
    with patch.object(database,'_solr_query',return_value=response()) as query:
        result=database.search_modelseed_reactions(keyword='synthetic partner')
    assert result['count']==1
    assert 'ec_numbers:' not in query.call_args.args[1]
    query_text=query.call_args.args[1]
    assert 'definition:/.*' in query_text and '[sS][yY]' in query_text and '[pP][aA]' in query_text
    assert ' AND ' in query_text


def test_compound_query_is_a_participation_filter_not_a_claim_of_reacting_role():
    with patch.object(database,'_solr_query',return_value=response()) as query:
        result=database.search_modelseed_reactions(compound_ids=['cpd99991','cpd99992'])
    assert query.call_args.args[1]=='compound_ids:cpd99991 AND compound_ids:cpd99992'
    assert 'target remains a hypothesis' in result['data'][0]['scope']


def test_legacy_compound_lists_and_missing_ec_remain_recoverable():
    with patch.object(database,'_solr_query',return_value=response(['cpd99991; cpd99992','cpd99993'],ec_numbers=None)):
        result=database.search_modelseed_reactions(reaction_id='rxn99991')
    assert result['data'][0]['compound_ids']==['cpd99991','cpd99992','cpd99993']
    assert result['data'][0]['ec_numbers']==[]


def test_incomplete_or_nonstandard_ec_annotations_are_retained_without_becoming_exact_mapping_ids():
    annotations=['2.1.1.900','2.1.1.900-','2.1.-.-']
    with patch.object(database,'_solr_query',return_value=response(ec_numbers=annotations)):
        record=database.search_modelseed_reactions(reaction_id='rxn99991')['data'][0]
    assert record['ec_numbers']==annotations
    assert record['complete_ec_numbers']==['2.1.1.900']


def test_pagination_exposes_unsearched_results_without_claiming_exhaustion():
    with patch.object(database,'_solr_query',return_value=response()) as query:
        result=database.search_modelseed_reactions('2',max_results=1,offset=2)
    assert query.call_args.kwargs['start']==2
    assert result['count']==1 and result['total_count']==3 and result['next_offset'] is None


@pytest.mark.parametrize('arguments',[{}, {'ec_class':'2 OR *:*'}, {'compound_ids':['bad:id']},
    {'reaction_id':'bad:id'}, {'keyword':'synthetic','offset':-1}, {'keyword':'synthetic','max_results':0}])
def test_invalid_or_unbounded_queries_make_no_remote_request(arguments):
    with patch.object(database,'_solr_query') as query:
        assert database.search_modelseed_reactions(**arguments)['status']=='error'
    query.assert_not_called()


def test_native_reaction_and_compound_results_have_durable_artifact_categories():
    assert ARTIFACT_CATEGORIES['search_modelseed_reactions']=='database'
    assert ARTIFACT_CATEGORIES['get_modelseed_compound']=='database'


@pytest.mark.parametrize('identifier',['','cpd*','x OR *:*'])
def test_compound_lookup_requires_an_exact_retrieved_identifier(identifier):
    with patch.object(database,'_solr_query') as query:
        assert database.get_modelseed_compound(identifier)['status']=='error'
    query.assert_not_called()
