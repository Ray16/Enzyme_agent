"""Gate-5 comparison rule is frozen and discriminates improvement from regression."""
import csv
from pathlib import Path

from enzyme_discovery_agent.eval.discovery_benefit import compare


def _run(root, ranked_rows):
    root.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    (root/'logs'/'events.jsonl').write_text('')
    (root/'agent_state').mkdir(exist_ok=True)
    (root/'agent_state'/'token_usage.json').write_text('{"calls":[],"total_tokens":100000}')
    fields=['accession','target_assessment_decision','evidence_level','catalytic_activities',
            'mechanism_family','interpro_family','protein_name','family_relevance']
    with open(root/'ranked_candidates.csv','w') as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        for r in ranked_rows: w.writerow({k:r.get(k,'') for k in fields})
    return root


def test_improvement_passes_and_regression_fails(tmp_path):
    # Baseline: two annotated supported candidates, no unannotated reach.
    base=_run(tmp_path/'base',[
        {'accession':'P1','target_assessment_decision':'investigate','evidence_level':'medium',
         'catalytic_activities':'ATP + x = y'},
        {'accession':'P2','target_assessment_decision':'investigate','evidence_level':'high',
         'catalytic_activities':'ATP + a = b'}])
    # Treatment: same two plus an unannotated candidate reached by expansion.
    treat=_run(tmp_path/'treat',[
        {'accession':'P1','target_assessment_decision':'investigate','evidence_level':'medium',
         'catalytic_activities':'ATP + x = y'},
        {'accession':'P2','target_assessment_decision':'investigate','evidence_level':'high',
         'catalytic_activities':'ATP + a = b'},
        {'accession':'P3','target_assessment_decision':'investigate','evidence_level':'medium',
         'catalytic_activities':''}])  # unannotated reach
    out=compare(base,treat,'amide_synthetase')
    assert out['criteria']['improves_breadth_recall_or_reach'] is True
    assert out['criteria']['no_more_unsupported_promotions'] is True
    assert out['treatment']['unannotated_reach']==1 and out['baseline']['unannotated_reach']==0

    # A treatment that adds an UNSUPPORTED prioritized candidate must fail that criterion.
    bad=_run(tmp_path/'bad',[
        {'accession':'P1','target_assessment_decision':'investigate','evidence_level':'medium',
         'catalytic_activities':'ATP + x = y'},
        {'accession':'P2','target_assessment_decision':'investigate','evidence_level':'high',
         'catalytic_activities':'ATP + a = b'},
        {'accession':'P9','target_assessment_decision':'prioritize','evidence_level':'speculative',
         'catalytic_activities':''}])
    out_bad=compare(base,bad,'amide_synthetase')
    assert out_bad['criteria']['no_more_unsupported_promotions'] is False
    assert out_bad['pair_passes_all_criteria'] is False


def test_self_comparison_cannot_show_improvement(tmp_path):
    run=_run(tmp_path/'r',[
        {'accession':'P1','target_assessment_decision':'investigate','evidence_level':'high',
         'catalytic_activities':'ATP + x = y'}])
    out=compare(run,run,'amide_synthetase')
    # Equal on every "not reduced" axis, but a run never improves over itself.
    assert out['criteria']['primary_recall_not_reduced'] is True
    assert out['criteria']['improves_breadth_recall_or_reach'] is False
    assert out['pair_passes_all_criteria'] is False
