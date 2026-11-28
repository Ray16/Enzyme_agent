"""Comparison plots remain readable as completed homolog ensembles accumulate."""
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from structural_modeling_agent.family.analysis import _short_name,comparison_figure


def test_display_prefers_enzyme_symbols_without_inventing_family_members():
    assert _short_name('Long enzyme description',['xyzB'])=='XyzB'
    assert _short_name('Unreviewed enzyme annotation',literature='Named enzyme')=='Named enzyme'
    assert _short_name('Mixed enzyme family protein')=='Unassigned enzyme'
    assert _short_name('Descriptive protein XyzB, unrelated family annotation')=='XyzB'


def test_long_labels_have_dedicated_rows_and_cannot_collapse_metric_axes():
    rows=[{'accession':'synthetic_'+str(i),'weighted_attack_angle_deg':170-i,
           'harmonic_distance_A':3+i,'numerical_profile':'exact'} for i in range(35)]
    labels={r['accession']:'Named enzyme '+str(i)+' · organism with a long descriptive name'
            for i,r in enumerate(rows)}
    original=copy.deepcopy(rows)
    fig=comparison_figure(rows,labels,1)
    fig.canvas.draw()
    axes=fig.axes
    assert all(axis.get_position().width>.2 for axis in axes)
    tick_labels=[value.get_text() for value in axes[0].get_yticklabels()]
    assert len(tick_labels)==35 and all('Named enzyme' in text for text in tick_labels)
    assert all('synthetic_' not in text for text in tick_labels)
    assert rows==original
    assert axes[1].get_xscale()=='log'
    plt.close(fig)


def test_family_report_exports_to_schema3_and_imports_into_discovery(tmp_path):
    import json
    from enzyme_discovery_agent.import_structural_evidence import import_structural_evidence
    from enzyme_discovery_agent.models.candidate_registry import (
        load_candidate_registry,
        update_candidate_registry,
    )
    from structural_modeling_agent.evidence import modeling_evidence
    from structural_modeling_agent.family.analysis import export_family_report

    run_dir = tmp_path / "discovery_run"
    run_dir.mkdir()
    update_candidate_registry(run_dir, "SYNTH1", "evaluated", metadata={"protein_name": "Demo ligase"})

    fam_dir = run_dir / "family_cofold"
    result_file = fam_dir / "jobs" / "SYNTH1_step1" / "complete.json"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(json.dumps({"ok": True}))

    report_path = export_family_report(
        fam_dir,
        [
            {
                "accession": "SYNTH1",
                "enzyme_name": "DemoLig",
                "family_id": "FAM01",
                "step": 1,
                "analysis_arm": "unrestrained",
                "weighted_attack_angle_deg": 112.5,
                "harmonic_distance_A": 3.4,
                "n_poses": 50,
                "near_attack_fraction": 0.42,
                "holo_seating_status": "seated",
                "identity_pass_poses": 50,
                "numerical_profile": "exact",
                "result": str(result_file),
            }
        ],
    )
    evidence = modeling_evidence(report_path)
    assert evidence["schema_version"] == "3.0"
    assert evidence["kind"] == "modeling_validation"
    assert evidence["candidates"][0]["execution_status"] == "complete"
    assert evidence["candidates"][0]["geometry_metrics"]["step_1_unrestrained"]["weighted_attack_angle_deg"] == 112.5

    review_path = import_structural_evidence(run_dir, report_path)
    assert review_path.name == "structural_modeling_review.json"
    reg = load_candidate_registry(run_dir)["candidates"]["SYNTH1"]
    assert reg["metadata"]["modeling_review_status"] == "ready_for_modeling_review"

    # Tampering with complete.json must flag incomplete status on re-verification.
    result_file.write_text(json.dumps({"ok": False}))
    tampered = modeling_evidence(report_path)
    assert tampered["candidates"][0]["execution_status"] == "incomplete"
    assert tampered["candidates"][0]["review_errors"]
