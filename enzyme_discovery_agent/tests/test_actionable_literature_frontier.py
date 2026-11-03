"""Primary title evidence routes pending identities without assuming their biology."""
import json

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.models.research_decisions import decision_context
from enzyme_discovery_agent.tools.identity_resolution import source_associations
from enzyme_discovery_agent.output.research_portfolio import write_research_portfolio


def test_encoded_cluster_source_is_distinguished_from_expression_host():
    text='The enzymes ZyxQ and QrsT, encoded in the Synthetic species biosynthetic gene cluster, when expressed in Different host, form the product.'
    source={'text':text,'path':'synthetic-source','sha256':'synthetic-hash','source_id':'synthetic:paper'}
    result=source_associations('ZyxQ',[source])
    assert [row['organism'] for row in result]==['Synthetic species']
    assert not source_associations('UnrelatedGene',[source])


def test_pending_paper_lead_with_retrieved_primary_title_is_actionable(tmp_path):
    update_candidate_registry(tmp_path,'literature:synthetic','evidence_pending',{
        'is_literature_lead':True,'literature_name':'ZyxQ','identity_resolution_status':'pending'})
    folder=tmp_path/'artifacts/literature';folder.mkdir(parents=True)
    (folder/'synthetic.json').write_text(json.dumps({'result':{'data':[{
        'title':'Characterization of ZyxQ in a synthetic transformation','pmid':'SYNTH_PRIMARY',
        'abstract':'ZyxQ from Synthetic species performs the source-described reaction.'}]}}))
    packet=decision_context(tmp_path)
    lead=packet['recent_unassessed_leads'][0]
    assert lead['name']=='ZyxQ' and lead['primary_title_matches'][0]['pmid']=='SYNTH_PRIMARY'
    assert lead['identity_status']=='pending'


def test_unannotated_source_cited_hypothesis_stays_visible_without_activity_admission(tmp_path):
    update_candidate_registry(tmp_path,'SYNTHETIC','evidence_pending',{'protein_name':'Synthetic catalyst',
        'claim_validation':{'promotion_eligible':False},'priority_assessment':{
            'decision':'investigate','reason':'Source supports a conditional catalytic hypothesis.',
            'strengths':['Retrieved machinery evidence'],'unknowns':['Native substrate scope'],
            'evidence_refs':['synthetic:source'],'revisit_when':'A discriminating test returns.'}})
    result=write_research_portfolio(tmp_path)
    assert result['source_cited_hypotheses']==1
    text=(tmp_path/'research_portfolio.csv').read_text()
    assert 'SYNTHETIC' in text and 'False' in text and 'unestablished' in text
