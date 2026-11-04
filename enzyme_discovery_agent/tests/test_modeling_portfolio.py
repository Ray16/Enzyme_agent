"""Unknown native annotation protects exploration without inventing activity evidence."""
import json
import pytest

from enzyme_discovery_agent.models import modeling_portfolio as portfolio


@pytest.fixture(autouse=True)
def target_review_precedes_scheduling(monkeypatch):
    # These cases exercise scheduling; the admission boundary has its own tests.
    monkeypatch.setattr('enzyme_discovery_agent.models.target_review.eligible',lambda *a,**k:True)


def entry(tmp_path, identifier, *, annotated=False, family='family-a', decision='investigate'):
    priority = {'decision': decision, 'reason': 'Source-supported synthetic catalytic hypothesis',
                'strengths': ['A source supports a testable interaction'],
                'evidence_refs': ['synthetic-source'], 'revisit_when': 'A discriminating measurement is available'}
    path = tmp_path / (identifier.replace(':', '_') + '.json')
    path.write_text(json.dumps(priority))
    return {'statuses': [{'status': 'evidence_pending', 'timestamp': '2026-01-01'}],
            'metadata': {'priority_assessment': priority, 'priority_artifact': path.name,
                         'interpro_family_id': family,
                         'native_substrates': ['synthetic participant'] if annotated else []}}


def test_unannotated_source_scoped_candidate_is_eligible_without_native_rank(tmp_path, monkeypatch):
    records = {'SYNTH_A': entry(tmp_path, 'SYNTH_A', annotated=True),
               'source:SYNTH_B': entry(tmp_path, 'source:SYNTH_B', family=['family-b'])}
    monkeypatch.setattr(portfolio, 'load_candidate_registry', lambda *a, **k: {'candidates': records})
    rows = portfolio.select_modeling_portfolio(tmp_path, max_candidates=1, protected_unannotated_slots=1)
    assert [row['accession'] for row in rows] == ['source:SYNTH_B']
    assert rows[0]['native_substrates_resolved'] is False
    assert not (tmp_path / 'ranked_candidates.csv').exists()


def test_missing_annotation_alone_does_not_justify_modeling(tmp_path, monkeypatch):
    records = {'SYNTH_A': entry(tmp_path, 'SYNTH_A', annotated=True),
               'SYNTH_B': entry(tmp_path, 'SYNTH_B'), 'SYNTH_C': entry(tmp_path, 'SYNTH_C')}
    records['SYNTH_B']['metadata']['priority_assessment']['evidence_refs'] = []
    records['SYNTH_C']['metadata']['is_literature_lead'] = True
    monkeypatch.setattr(portfolio, 'load_candidate_registry', lambda *a, **k: {'candidates': records})
    assert [row['accession'] for row in portfolio.select_modeling_portfolio(
        tmp_path, max_candidates=2, protected_unannotated_slots=1)] == ['SYNTH_A']


def test_candidates_are_taken_from_each_new_registry_snapshot(tmp_path, monkeypatch):
    records = {'SYNTH_A': entry(tmp_path, 'SYNTH_A')}
    monkeypatch.setattr(portfolio, 'load_candidate_registry', lambda *a, **k: {'candidates': records})
    assert len(portfolio.select_modeling_portfolio(tmp_path, max_candidates=2)) == 1
    records['SYNTH_B'] = entry(tmp_path, 'SYNTH_B', family='family-b')
    assert len(portfolio.select_modeling_portfolio(tmp_path, max_candidates=2)) == 2


def test_one_damaged_priority_artifact_does_not_block_other_ready_candidates(tmp_path, monkeypatch):
    records = {'SYNTH_A': entry(tmp_path, 'SYNTH_A'), 'SYNTH_B': entry(tmp_path, 'SYNTH_B')}
    (tmp_path / records['SYNTH_A']['metadata']['priority_artifact']).write_text('{incomplete')
    monkeypatch.setattr(portfolio, 'load_candidate_registry', lambda *a, **k: {'candidates': records})
    assert [row['accession'] for row in portfolio.select_modeling_portfolio(
        tmp_path, max_candidates=2)] == ['SYNTH_B']
    report = json.loads((tmp_path / 'modeling_portfolio.json').read_text())
    assert report['skipped_artifacts'][0]['accession'] == 'SYNTH_A'
    assert records['SYNTH_A']['statuses'][-1]['status'] == 'evidence_pending'


def test_ranked_policy_tracks_expansion_without_a_primary_or_annotation_gate(tmp_path,monkeypatch):
    policy=tmp_path/'agent_state/discovery_quality_policy.json';policy.parent.mkdir()
    policy.write_text(json.dumps({'primary_only':True,'modeling_selection':'ranked'}))
    records={'SYNTH_A':entry(tmp_path,'SYNTH_A'),'SYNTH_B':entry(tmp_path,'SYNTH_B'),
             'SYNTH_UNRANKED':entry(tmp_path,'SYNTH_UNRANKED')}
    monkeypatch.setattr(portfolio,'load_candidate_registry',lambda *a,**k:{'candidates':records})
    ranking=tmp_path/'ranked_candidates.csv'
    ranking.write_text('rank,accession,protein_name\n1,SYNTH_A,First synthetic enzyme\n')
    assert [r['accession'] for r in portfolio.select_modeling_portfolio(tmp_path,max_candidates=None)]==['SYNTH_A']
    ranking.write_text(ranking.read_text()+'2,SYNTH_B,Second synthetic enzyme\n')
    rows=portfolio.select_modeling_portfolio(tmp_path,max_candidates=None)
    assert [r['accession'] for r in rows]==['SYNTH_A','SYNTH_B']
    assert rows[1]['enzyme_name']=='Second synthetic enzyme'
    assert rows[1]['native_substrates_resolved'] is False
    assert all(r['modeling_setup_status']=='requires_independent_source_and_target_setup_review' for r in rows)
