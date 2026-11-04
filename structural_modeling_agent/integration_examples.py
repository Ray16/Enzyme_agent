"""Connect the two existing experimental studies to saved discovery candidates.

This is downstream campaign configuration, never discovery prompt context.
Original artifacts remain the evidence sources; new outputs carry their hashes.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .integration import ROOT, export_bundle, read_rows, apply_bundle, save_application, write_json
from .proxy_policy import resolve_mechanism, proxy_mechanism_for_family
from .proxies import parse_pose
from .proxies.workflow import evaluate_plan


def campaign_configs() -> dict[str, dict]:
    synthetase = "exp_data/amide_synthetase/structural_proxies/tables/"
    hydrolase = "exp_data/nylon_hydrolase/structural_proxies/attempts/cofold_yield_20261007/"
    return {
        "amide_activation": {
            "bundle_id": "amide_activation_inverse_distance_attack_angle_v1",
            "selected_artifact": "structural_modeling_agent/knowledge/attack_angle_metric.json",
            "calibration_geometry": synthetase + "continuous_pose_geometry.csv",
            "excluded_ids": ["SfaB"],
            "definition": {"kind": "proximity_weighted_attack_angle", "required_poses": 50,
                           "score_units": "degrees",
                           "supported_mechanisms": ["acyl-adenylate", "acyl-phosphate"],
                           "phosphate_by_mechanism": {"acyl-adenylate": "PA", "acyl-phosphate": "PG"},
                           "atom_selection": "nearest substrate carboxylate oxygen to the reaction-specific ATP phosphorus; corresponding O--P--bridging-O angle",
                           "aggregation": "sum(angle_deg / distance_A) / sum(1 / distance_A)"},
            "scope": {"step_index": 1, "mechanisms": ["acyl-adenylate"], "families": ["ANL/AMP-binding"]},
            "assay_endpoint": "Mean product concentration (ug/mL), not a kinetic rate constant",
            "validation_status": "exploratory_retrospective_fixed_metric",
            "limitations": ["The current metric has no fitted parameters; historical Gaussian-score calibration does not validate it.",
                            "Not a general activity ranker; report harmonic distance and setup validity alongside angle.",
                            "Other enzyme families and P-gamma chemistry are unvalidated mechanistic transfers.",
                            "Score omits the amine-dependent step and does not validate metal placement or assemblies."],
            "sources": {
                "assay_data": ["exp_data/amide_synthetase/assay_data/panel_activity.csv"],
                "validation": [synthetase + "synthetase_fact_check.json", synthetase + "broader_recovery_results.json"],
                "calibration_features": [synthetase + "continuous_pose_geometry.csv"],
                "extractor_code": ["structural_modeling_agent/proxies/attack_angle.py"]}},
        "ntn_hydrolysis": {
            "bundle_id": "ntn_ensemble_yield_20261007_v1",
            "selected_artifact": hydrolase + "frozen_candidate.json",
            "definition": {"kind": "standardized_linear", "required_poses": 10,
                           "feature_definition_id": "ntn_two_site_internal_register_chunk1_v1",
                           "score_units": "predicted_log10_L2_ugml",
                           "assembly_quality": "reference interface Jaccard >= 0.5; one-to-one processed-chain and ligand assignment",
                           "aggregation": "first chunk models 0..9; internal non-cap amides; equal weight to two sites",
                           "sampling": "first10"},
            "scope": {"step_index": 1, "mechanisms": ["ntn_hydrolysis"], "families": ["NylC-like"],
                      "assembly": "mature_two_unit_interface"},
            "assay_endpoint": "Lysate L2 accumulation on powder at 65 C, log10 ug/mL",
            "validation_status": "exploratory_internal_cross_validation",
            "limitations": ["Coefficients frozen from the original 43-enzyme calibration cohort; no refit on discovery candidates.",
                            "Repeated cross-validation is internal validation, not an independent prospective test.",
                            "The model is specific to processed NylC-like two-site assemblies and this assay endpoint.",
                            "Lysate product accumulation combines abundance, maturation, access and chemistry."],
            "sources": {
                "assay_data": ["exp_data/nylon_hydrolase/data/activity/amide_hydrolase_biorep_activity.csv"],
                "calibration_features": [hydrolase + "first10_joined.csv"],
                "validation": [hydrolase + "prediction_metrics.json", hydrolase + "increment_over_length.json", hydrolase + "protocol.json"],
                "extractor_code": ["structural_modeling_agent/proxies/ntn_ensemble.py"]}},
    }


def synthetase_index() -> list[dict]:
    audit = json.loads((ROOT / "exp_data/amide_synthetase/structural_proxies/tables/synthetase_fact_check.json").read_text())
    source = next(p for p in audit["sources_sha256"] if p.endswith("/ranked_candidates.csv"))
    revised = next(p for p in audit["sources_sha256"] if p.endswith("/ranked_candidates_rescored.csv"))
    original = {r["accession"]: r for r in read_rows(ROOT / source)}
    candidates = []
    for r in read_rows(ROOT / revised):
        acc = r["accession"]
        row = original[acc]
        geometry = audit["discovery"]["raw_geometry"].get(acc, {})
        resolved, basis = resolve_mechanism(row.get("interpro_family"), row.get("catalytic_activities"))
        fallback = proxy_mechanism_for_family(row.get("interpro_family"))
        phosphate = geometry.get("phosphate") or (resolved.phosphate if resolved else None)
        mechanism = resolved.mechanism if resolved else ({"PA": "acyl-adenylate", "PG": "acyl-phosphate"}.get(phosphate))
        if resolved is not None and geometry and resolved.phosphate != geometry["phosphate"]:
            mechanism = None
            basis = "conflict_between_current_annotation_and_archived_coordinate"
        candidates.append({"accession": acc, "discovery_rank": int(r["new_rank"]),
                           "mechanism": mechanism, "mechanism_basis": basis if resolved else "archived_coordinate_audit",
                           "family": fallback.family_group if fallback else row.get("interpro_family"),
                           "step_index": 1, "phosphate": phosphate,
                           "prediction_dir": geometry.get("directory"),
                           "discovery_source": source, "ranking_source": revised})
    return candidates


def hydrolase_index() -> list[dict]:
    ranked = ROOT / "enzyme_discovery_agent/runs/imported_hydrolase_discovery/ranked_candidates.csv"
    ranking = {r["accession"]: r for r in read_rows(ranked)}
    setup = json.loads((ROOT / "exp_data/nylon_hydrolase/cofolding/discovery_cofold/build_manifest.json").read_text())
    rows = []
    for c in setup:
        acc = c["accession"]
        r = ranking.get(acc, {})
        directory = f"exp_data/nylon_hydrolase/cofolding/discovery_cofold/results/{acc}/boltz_results_{acc}/predictions/{acc}"
        rows.append({"accession": acc, "discovery_rank": int(r["rank"]) if r.get("rank") else None,
                     "step_index": 1, "mechanism": "ntn_hydrolysis" if c["form"] == "ntn" else r.get("mechanism_family"),
                     "family": r.get("interpro_family"), "prediction_dir": directory,
                     "assembly": "mature_single_unit" if c["form"] == "ntn" else c["form"],
                     "ranking_source": str(ranked.relative_to(ROOT))})
    return rows


def append_confidence_baselines(report: dict, candidates: list[dict]):
    index = {c["accession"]: c for c in candidates}
    for row in report["candidates"]:
        candidate = index[row["accession"]]
        directory = candidate.get("prediction_dir")
        if not directory:
            continue
        paths = sorted((ROOT / directory).glob("*_model_*.pdb"))
        if not paths:
            continue
        # These baselines remain computable even when a calibrated mechanism-
        # specific model is inapplicable to the candidate's assembly or family.
        plan = {"schema_version": "1.0", "mechanism": candidate.get("mechanism") or "unresolved",
                "reaction_step": "declared discovery model; confidence-only assessment",
                "assay_endpoint": report["bundle"]["calibration"]["assay_endpoint"], "context": {},
                "hypotheses": [{"proxy_id": p, "rationale": "Prediction-confidence baseline",
                                 "expected_direction": None, "evidence_ids": ["prediction:confidence_sidecar"]}
                                for p in ("complex_plddt", "ligand_iptm")]}
        try:
            poses = [parse_pose(str(p)) for p in paths]
            row["confidence_baselines"] = evaluate_plan(poses, [None]*len(poses), plan)["measurements"]
        except (OSError, ValueError) as exc:
            row["confidence_baseline_error"] = str(exc)


def run_examples(output: Path):
    configs = campaign_configs()
    write_json(output / "campaign_configurations.json", configs)
    summaries = []
    for name, config in configs.items():
        folder = output / name
        bundle = export_bundle(config)
        candidates = synthetase_index() if name == "amide_activation" else hydrolase_index()
        write_json(folder / "bundle.json", bundle)
        write_json(folder / "candidate_index.json", {"schema_version": "1.0", "candidates": candidates})
        report = apply_bundle(bundle, candidates)
        append_confidence_baselines(report, candidates)
        save_application(folder, report)
        summaries.append({"campaign": name, "n_candidates": len(candidates),
                          "statuses": dict(Counter(r["status"] for r in report["candidates"])),
                          "transfer_statuses": dict(Counter(r["transfer_status"] for r in report["candidates"])),
                          "report": str(folder / "candidate_proxy_evidence.json")})
        if name == "ntn_hydrolysis":
            # Reuse existing additional-member descriptors to verify that the
            # frozen model reproduces the saved application without using labels.
            source = "exp_data/nylon_hydrolase/structural_proxies/attempts/catalytic_stages_20261007/new_members/predictions.csv"
            members = []
            for r in read_rows(ROOT / source):
                members.append({"accession": r["id"], "calibration_id": r["id"],
                                "mechanism": "ntn_hydrolysis", "family": "NylC-like", "step_index": 1,
                                "assembly": "mature_two_unit_interface", "n_poses": int(r["n_poses"]),
                                "feature_definition_id": bundle["definition"]["feature_definition_id"],
                                "feature_source": source,
                                "features": {k: float(r[k]) for k in bundle["definition"]["features"] + ["interface_jaccard_min"]}})
            verification = apply_bundle(bundle, members)
            save_application(folder / "additional_panel_members", verification)
            expected = {r["id"]: float(r["prediction"]) for r in read_rows(ROOT / source)}
            for r in verification["candidates"]:
                if r["status"] != "scored" or abs(r["score"]-expected[r["accession"]]) > 1e-10:
                    raise ValueError("frozen-model replay differs from saved additional-member application")
    write_json(output / "summary.json", summaries)
    print(json.dumps(summaries, indent=2))
