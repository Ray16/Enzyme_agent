import json
from enzyme_discovery_agent.models.native_annotation import (
    native_annotation_for_report, record_digest, summarize_native_annotation,
)


def entry(code=None, reviewed=True):
    evidence=[{'evidenceCode':code,'source':'PubMed' if code=='ECO:0000269' else 'SyntheticRule','id':'synthetic-source'}] if code else []
    return {'primaryAccession':'SYNTH1','entryType':'reviewed' if reviewed else 'unreviewed',
            'proteinExistence':'1: Evidence at protein level',
            'comments':[{'commentType':'CATALYTIC ACTIVITY','reaction':{
                'name':'Synthetic reactants = synthetic product','evidences':evidence}}]}


def test_reviewed_record_and_protein_existence_do_not_establish_functional_experiment():
    assert summarize_native_annotation(entry())['native_activity_evidence']=='evidence_not_specified'
    assert summarize_native_annotation(entry('ECO:0000250'))['native_activity_evidence']=='inferred_annotation'


def test_reaction_evidence_is_separate_from_review_status():
    assert summarize_native_annotation(entry('ECO:0000256'))['native_activity_evidence']=='automatically_inferred'
    assert summarize_native_annotation(entry('ECO:0000269',False))['native_activity_evidence']=='experimental_evidence_cited'


def test_different_reactions_do_not_inherit_one_experimental_label():
    record=entry('ECO:0000269')
    record['comments']+=entry('ECO:0000256')['comments']
    result=summarize_native_annotation(record)
    assert result['native_activity_evidence']=='mixed_reaction_evidence'
    assert [r['evidence_level'] for r in result['reactions']]==['experimental_evidence_cited','automatically_inferred']


def test_publication_source_snapshot_requires_identity_and_unchanged_record(tmp_path):
    folder=tmp_path/'artifacts/native_annotations';folder.mkdir(parents=True)
    record=entry('ECO:0000256')
    path=folder/'SYNTH1.json';saved={'record':record,'record_sha256':record_digest(record)}
    path.write_text(json.dumps(saved))
    assert native_annotation_for_report(tmp_path,'SYNTH1',{})['native_activity_evidence']=='automatically_inferred'
    saved['record']['comments']=entry('ECO:0000269')['comments'];path.write_text(json.dumps(saved))
    assert native_annotation_for_report(tmp_path,'SYNTH1',{})['native_activity_evidence']=='evidence_not_preserved'
