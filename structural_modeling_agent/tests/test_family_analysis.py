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


def test_promote_family_cofold_and_design_validation_roundtrip(tmp_path):
    import json
    from structural_modeling_agent.barrier.parameters import DEFAULT_GAFF2_POLARIZABILITY_OVERRIDES
    from structural_modeling_agent.barrier.prep_cofold import asset_dir
    from structural_modeling_agent.controller.tool_registry import get_registry
    from structural_modeling_agent.family.analysis import export_family_report
    from structural_modeling_agent.family.promote import (
        build_design_validation_task,
        promote_family_cofold,
    )

    # 1. Durable default cofactor assets exist in package tree.
    assets = asset_dir()
    assert (assets / "ATP.lib").is_file()
    assert (assets / "ATP.frcmod").is_file()

    # 2. Both bridge tools are registered in ToolRegistry under followups domain.
    reg = get_registry()
    assert "promote_family_cofold" in reg.names_for_domain("followups")
    assert "build_design_validation_task" in reg.names_for_domain("followups")

    # 3. Promote family-cofold -> MD manifest + strategy-open Design handoff.
    fam_dir = tmp_path / "family_cofold"
    pose_pdb = fam_dir / "jobs" / "SYNTH1_step1" / "model_0.pdb"
    pose_pdb.parent.mkdir(parents=True)
    pose_pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    result_file = fam_dir / "jobs" / "SYNTH1_step1" / "complete.json"
    result_file.write_text(json.dumps({
        "analysis": {
            "poses": [
                {"path": str(pose_pdb), "distance_A": 3.25, "angle_deg": 162.0, "identity_issues": []},
            ]
        }
    }))
    export_family_report(
        fam_dir,
        [
            {
                "accession": "SYNTH1",
                "enzyme_name": "DemoLig",
                "family_id": "FAM01",
                "step": 1,
                "analysis_arm": "unrestrained",
                "weighted_attack_angle_deg": 145.0,
                "harmonic_distance_A": 3.4,
                "n_poses": 50,
                "near_attack_fraction": 0.6,
                "holo_seating_status": "seated",
                "identity_pass_poses": 50,
                "numerical_profile": "exact",
                "result": str(result_file),
            }
        ],
    )
    promo = promote_family_cofold(fam_dir, top_k=2, step=1)
    assert promo["n_candidates"] == 1
    md_doc = json.loads(open(promo["md_promotion_path"]).read())
    md_setup = md_doc["candidates"][0]["md_setup"]
    assert md_setup["polarizability_overrides"] == DEFAULT_GAFF2_POLARIZABILITY_OVERRIDES
    assert md_setup["protonation_audit"] is True
    assert md_setup["prep_python"] != md_setup["md_python"]

    design_doc = json.loads(open(promo["design_handoff_path"]).read())
    assert design_doc["kind"] == "enzyme_design_request"
    assert design_doc["requested_tool"] is None

    # 4. Close Design -> Structural Modeling validation loop.
    design_run = tmp_path / "design_run"
    (design_run / "artifacts" / "design").mkdir(parents=True)
    (design_run / "manifest.json").write_text(json.dumps({
        "agent": "enzyme_design_agent",
        "status": "completed",
        "strategy": "redesign",
        "designs": 1,
    }))
    (design_run / "design_output_handoff.json").write_text(json.dumps({
        "kind": "design_output_handoff",
        "agent": "enzyme_design_agent",
        "strategy": "redesign",
    }))
    (design_run / "artifacts" / "design" / "build_saturation_library_001.json").write_text(json.dumps({
        "result": {
            "status": "success",
            "data": {"variants": [{"id": "SYNTH1_E463Q", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"}]},
        }
    }))
    val = build_design_validation_task(design_run, target={"name": "amide_ligation"})
    assert val["n_candidates"] == 1
    assert val["candidate_ids"] == ["SYNTH1_E463Q"]
    task_doc = json.loads(open(val["task_path"]).read())
    assert task_doc["kind"] == "design_validation_task"
    assert task_doc["candidates"][0]["evidence"]["origin"] == "enzyme_design_agent"
