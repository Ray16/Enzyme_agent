"""Relevance, primary records and publisher links must survive initial retrieval."""
import json
from urllib.parse import urlsplit,parse_qs

from enzyme_discovery_agent.tools import literature as literature


def configure(monkeypatch,payload):
    requests=[]
    def request(req,**kwargs):
        requests.append(parse_qs(urlsplit(req.full_url).query))
        return json.dumps(payload).encode()
    monkeypatch.setattr(literature,'request_with_retry',request)
    monkeypatch.setattr(literature,'_semantic_rerank',lambda **kwargs:kwargs['papers'][:kwargs['max_results']])
    monkeypatch.setattr(literature.config,'OPENALEX_API_KEY','')
    return requests


def test_relevant_retrieval_is_the_default_and_complementary_orders_are_explicit(monkeypatch):
    requests=configure(monkeypatch,{'results':[{'title':'Synthetic low-citation primary study','publication_year':2026,
        'doi':'https://doi.org/10.test/new','cited_by_count':0}]})
    assert literature.search_openalex('synthetic transformation')['retrieval_order']=='relevance'
    assert 'sort' not in requests[-1]
    literature.search_openalex('synthetic transformation',sort_order='recent')
    assert requests[-1]['sort']==['publication_year:desc']
    literature.search_openalex('synthetic transformation',sort_by_citations=True)
    assert requests[-1]['sort']==['cited_by_count:desc']


def test_review_letters_do_not_displace_primary_papers_or_erase_their_links(monkeypatch):
    relation={'is-review-of':[{'id-type':'doi','id':'10.test/primary'}]}
    link={'URL':'https://publisher.example/source.pdf','content-type':'application/pdf'}
    requests=configure(monkeypatch,{'message':{'items':[
        {'DOI':'10.test/review','title':['Review of synthetic study'],'type':'peer-review','relation':relation},
        {'DOI':'10.test/component','title':['Synthetic supplement'],'type':'component'},
        {'DOI':'10.test/primary','title':['Synthetic primary study'],'type':'journal-article','link':[link]}]}})
    result=literature.search_crossref('synthetic primary study',max_results=1)
    assert result['data'][0]['doi']=='10.test/primary'
    assert result['data'][0]['fulltext_links']==[link]
    assert len(result['related_material'])==2
    assert result['related_material'][0]['relations']==relation
    assert {'relation','link'}<=set(requests[-1]['select'][0].split(','))
    all_records=literature.search_crossref('synthetic primary study',max_results=3,include_related_material=True)
    assert len(all_records['data'])==3
