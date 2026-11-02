import json
from unittest.mock import Mock

from enzyme_discovery_agent.models.literature_leads import register_literature_lead
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry,update_candidate_registry
from enzyme_discovery_agent.models.lead_followup import reconcile_directions,service_leads
from enzyme_discovery_agent.models.evaluations import registered_hypothesis,current_evaluations


def setup(root):
    (root/'target_reasoning.json').write_text(json.dumps({'reaction_direction':'formation','bond_types':['ester']}))
    source=root/'source.txt';source.write_text('A cascade uses hydrolysis of a polymer with SynBreak and subsequent synthesis by SynJoin.')
    return source


def test_direction_filter_attributes_cascade_roles_before_identity(tmp_path):
    root=tmp_path;source=setup(root)
    bad=register_literature_lead(root,{'name':'SynBreak','relationship':'cited'},{'doi':'10.test/cascade'},source)
    good=register_literature_lead(root,{'name':'SynJoin','mechanism':{'direction':'formation'}},{'doi':'10.test/cascade'},source)
    records=load_candidate_registry(root,strict=True)['candidates']
    assert records[bad]['metadata']['target_relevance_status']=='excluded_opposite_direction'
    assert records[bad]['statuses'][-1]['status']=='rejected'
    assert records[good]['statuses'][-1]['status']=='evidence_pending'


def test_new_leads_get_followup_without_a_model_request_and_are_not_repeated(tmp_path,monkeypatch):
    source=setup(tmp_path)
    lead=register_literature_lead(tmp_path,{'name':'SynJoin'},{'doi':'10.test/source'},source)
    query=Mock(return_value={'status':'success','data':[]})
    resolve=Mock(return_value={'status':'partial','data':{'status':'pending'}})
    monkeypatch.setattr('enzyme_discovery_agent.tools.literature.search_pubmed',query)
    monkeypatch.setattr('enzyme_discovery_agent.tools.identity_resolution.resolve_literature_lead',resolve)
    assert lead in service_leads(tmp_path)['attempted']
    assert not service_leads(tmp_path)['attempted']
    assert query.call_count==2
    assert query.call_args.args[0]=='SynJoin'
    assert (tmp_path/'candidate_coverage.csv').exists()


def test_native_annotation_absence_does_not_hide_a_verified_cited_hypothesis(tmp_path):
    priority={'decision':'investigate','strengths':['Synthetic source supports a testable role'],'evidence_refs':['synthetic-source']}
    artifact=tmp_path/'priority.json';artifact.write_text(json.dumps(priority))
    metadata={'priority_assessment':priority,'priority_artifact':artifact.name,
        'claim_validation':{'promotion_eligible':False,'blocking_claims':['native_reaction'],'claims':[
            {'claim_type':kind,'subject':'SYNTHETIC1','verdict':'supported'} for kind in ['accession_identity','record_active']]}}
    update_candidate_registry(tmp_path,'SYNTHETIC1','evidence_pending',metadata)
    assert registered_hypothesis(tmp_path,'SYNTHETIC1',metadata)
    assert current_evaluations(tmp_path,[{'data':{'accession':'SYNTHETIC1',**metadata}}])


def _pending_mocks(monkeypatch):
    query=Mock(return_value={'status':'success','data':[]})
    resolve=Mock(return_value={'status':'partial','data':{
        'status':'pending','reason':'Retrieve a source linking this name to a taxon or deposited identifier'}})
    monkeypatch.setattr('enzyme_discovery_agent.tools.literature.search_pubmed',query)
    monkeypatch.setattr('enzyme_discovery_agent.tools.identity_resolution.resolve_literature_lead',resolve)
    return query,resolve


def _cool_down(root):
    """Expire the inter-attempt cooldown without patching the global clock."""
    path=root/'agent_state/lead_followup.json'
    ledger=json.loads(path.read_text())
    for row in ledger.values():row['retry_after']=0
    path.write_text(json.dumps(ledger))
    return ledger


def test_a_lead_that_needs_more_evidence_is_retried_not_abandoned(tmp_path,monkeypatch):
    """'identity_pending' asks for a taxon; treating it as terminal dropped the lead for good."""
    source=setup(tmp_path)
    lead=register_literature_lead(tmp_path,{'name':'SynJoin'},{'doi':'10.test/source'},source)
    _pending_mocks(monkeypatch)
    assert lead in service_leads(tmp_path)['attempted']
    assert not service_leads(tmp_path)['attempted']      # cooldown still suppresses a hot retry
    entry=next(iter(_cool_down(tmp_path).values()))
    assert entry['stage']=='identity_pending'
    assert 'taxon' in entry['needs']                      # what it waits for is recorded
    assert lead in service_leads(tmp_path)['attempted']   # and it does come back


def test_retries_stop_at_the_bounded_attempt_budget(tmp_path,monkeypatch):
    from enzyme_discovery_agent.models.lead_followup import LEAD_ATTEMPTS
    source=setup(tmp_path)
    register_literature_lead(tmp_path,{'name':'SynJoin'},{'doi':'10.test/source'},source)
    _pending_mocks(monkeypatch)
    for _ in range(LEAD_ATTEMPTS+2):
        service_leads(tmp_path);_cool_down(tmp_path)
    entry=next(iter(json.loads((tmp_path/'agent_state/lead_followup.json').read_text()).values()))
    assert entry['attempts']==LEAD_ATTEMPTS
    assert not service_leads(tmp_path)['attempted']


def test_followup_history_survives_the_registry_rekeying_a_lead_to_its_name(tmp_path,monkeypatch):
    """The registry re-keys a lead once its name is known; history must not be orphaned."""
    source=setup(tmp_path)
    lead=register_literature_lead(tmp_path,{'name':'SynJoin'},{'doi':'10.test/source'},source)
    _pending_mocks(monkeypatch)
    service_leads(tmp_path)
    carried=load_candidate_registry(tmp_path,strict=True)['candidates'][lead]['metadata']
    update_candidate_registry(tmp_path,'SynJoin','evidence_pending',dict(carried))
    _cool_down(tmp_path)
    service_leads(tmp_path,lead_ids=['SynJoin'])   # service it under its new canonical key
    ledger=json.loads((tmp_path/'agent_state/lead_followup.json').read_text())
    # One lead, one ledger row. Keying by the registry key produced a second, orphaned row and
    # made drained work look new. (A genuinely changed evidence basis may still refresh the
    # attempt budget -- that is new evidence, not lost history.)
    assert len(ledger)==1,'re-keying must not orphan the follow-up history'
    row=next(iter(ledger.values()))
    assert row['registry_key']=='SynJoin' and row['literature_name']=='SynJoin'


def test_a_backlog_is_drained_faster_than_a_fixed_trickle(tmp_path,monkeypatch):
    from enzyme_discovery_agent.models.lead_followup import MAX_LEADS_PER_CALL
    source=setup(tmp_path)
    for i in range(16):
        register_literature_lead(tmp_path,{'name':f'SynJoin{i}'},{'doi':f'10.test/s{i}'},source)
    _pending_mocks(monkeypatch)
    attempted=service_leads(tmp_path)['attempted']
    assert len(attempted)>2,'a 16-lead backlog must not be serviced two at a time'
    assert len(attempted)<=MAX_LEADS_PER_CALL


def test_an_unresolved_lead_is_reported_as_unresolved_even_when_relevance_is_unreviewed(tmp_path):
    import csv
    from enzyme_discovery_agent.output.research_portfolio import write_research_portfolio
    source=setup(tmp_path)
    register_literature_lead(tmp_path,{'name':'SynJoin'},{'doi':'10.test/source'},source)
    update_candidate_registry(tmp_path,'SynJoin',None,{'target_relevance_status':'needs_relevance_review'})
    write_research_portfolio(tmp_path)
    rows=list(csv.DictReader((tmp_path/'unresolved_literature_leads.csv').open()))
    assert rows, 'the identity gap must not be hidden by the relevance label'
    assert rows[0]['relevance_status']=='needs_relevance_review'


def test_a_taxon_already_known_elsewhere_is_used_to_resolve_the_lead(tmp_path,monkeypatch):
    """Both facts were already in the registry; nothing joined them, so retries re-asked."""
    source=setup(tmp_path)
    lead=register_literature_lead(tmp_path,{'name':'SynJoin'},{'doi':'10.test/source'},source)
    # A separate entry for the same enzyme already records where it comes from.
    update_candidate_registry(tmp_path,'SynJoin','evidence_pending',
                              {'protein_name':'SynJoin','organism':'Synthetica exempla'})
    queries=[]
    def search(query,**kwargs):
        queries.append(query)
        return {'status':'success','data':[]}
    monkeypatch.setattr('enzyme_discovery_agent.tools.literature.search_pubmed',search)
    monkeypatch.setattr('enzyme_discovery_agent.tools.identity_resolution.resolve_literature_lead',
                        lambda *a,**k:{'status':'partial','data':{'status':'pending','reason':'x'}})
    service_leads(tmp_path,lead_ids=[lead])
    assert queries and 'Synthetica exempla' in queries[0], queries
    carried=load_candidate_registry(tmp_path,strict=True)['candidates'][lead]['metadata']
    assert carried['organism']=='Synthetica exempla'
    assert 'narrows identity lookup' in carried['organism_basis']


def test_disagreeing_taxa_are_left_unresolved_rather_than_guessed(tmp_path,monkeypatch):
    from enzyme_discovery_agent.models.lead_followup import sibling_organism
    source=setup(tmp_path)
    lead=register_literature_lead(tmp_path,{'name':'SynJoin'},{'doi':'10.test/source'},source)
    update_candidate_registry(tmp_path,'SynJoin','evidence_pending',
                              {'protein_name':'SynJoin','organism':'Synthetica exempla'})
    update_candidate_registry(tmp_path,'SynJoin_b','evidence_pending',
                              {'literature_name':'SynJoin','organism':'Alia exempla'})
    registry=load_candidate_registry(tmp_path,strict=True)['candidates']
    meta=registry[lead]['metadata']
    assert sibling_organism(registry,lead,meta) is None


def test_a_taxon_arriving_later_reopens_the_lead_with_a_fresh_budget(tmp_path,monkeypatch):
    """The thing resolution waits for usually arrives after the first attempt."""
    source=setup(tmp_path)
    lead=register_literature_lead(tmp_path,{'name':'SynJoin'},{'doi':'10.test/source'},source)
    _pending_mocks(monkeypatch)
    service_leads(tmp_path,lead_ids=[lead])
    first=json.loads((tmp_path/'agent_state/lead_followup.json').read_text())
    assert next(iter(first.values()))['stage']=='identity_pending'
    # Still inside the cooldown, so nothing would normally be retried...
    assert not service_leads(tmp_path,lead_ids=[lead])['attempted']
    # ...but learning the organism is new evidence, not a repeat of the same question.
    update_candidate_registry(tmp_path,lead,None,{'organism':'Synthetica exempla'})
    assert lead in service_leads(tmp_path,lead_ids=[lead])['attempted']
    ledger=json.loads((tmp_path/'agent_state/lead_followup.json').read_text())
    assert len(ledger)==1, 'still one lead, not a duplicate row'
    assert next(iter(ledger.values()))['attempts']==1, 'new evidence restores the attempt budget'
