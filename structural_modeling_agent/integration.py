"""Bridge experimental calibration artifacts to discovery candidate evidence.

Exports frozen definitions with source fingerprints; application never refits on
candidate labels or changes discovery ranks. Raw assays and structures stay in
their existing directories. Registry import is an explicit command.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from enzyme_discovery_agent.state_store import atomic_write_json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    path = relocated_source(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocated_source(path: Path) -> Path:
    """Resolve a preserved evidence path through the explicit case-study move map.

    Existing files take precedence. This does not rewrite frozen source hashes.
    """
    if path.exists():
        return path
    try:
        old = str(path.relative_to(ROOT))
    except ValueError:
        return path
    manifest = ROOT / "exp_data/REFACTOR_PATHS_20261009.json"
    if manifest.exists():
        rows = json.loads(manifest.read_text())["moves"]
        for row in rows:
            if row["old"] == old:
                return ROOT / row["new"]
        for row in sorted(rows, key=lambda row: len(row["old"]), reverse=True):
            if row.get("kind") == "directory" and old.startswith(row["old"].rstrip("/") + "/"):
                return ROOT / row["new"] / old[len(row["old"].rstrip("/")) + 1:]
    return path


def definition_digest(definition: dict) -> str:
    return hashlib.sha256(json.dumps(definition, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data, allow_nan=False)


def export_bundle(config: dict, root: Path = ROOT) -> dict:
    """Freeze a previously selected score or fitted model, not a new selection."""
    source = root / config["selected_artifact"]
    selected = json.loads(source.read_text())
    definition = {**config["definition"]}
    if definition["kind"] == "proximity_weighted_attack_angle":
        from .proxies.attack_angle import METRIC_ID
        if selected.get("metric_id") != METRIC_ID or selected.get("kind") != definition["kind"]:
            raise ValueError("current metric identity differs from the frozen definition")
        definition.update({key: selected[key] for key in ("metric_id", "aggregation", "companion", "fitted_parameters")})
        geometry = read_rows(root / config["calibration_geometry"])
        training_ids = sorted({r["id"] for r in geometry if r["group"] == "calibration" and r["id"] not in config.get("excluded_ids", [])})
    elif definition["kind"] == "geometry_ensemble":
        params = selected["selected_parameters"]
        if params["angle_function"] != "gaussian":
            raise ValueError("this extractor implements only the frozen Gaussian angle definition")
        definition["parameters"] = {
            "distance_midpoint_A": params["midpoint_A"],
            "distance_softness_A": params["softness_A"],
            "angular_width_deg": config["angular_width_deg"],
            "regularization": params["regularization"],
        }
        geometry = read_rows(root / config["calibration_geometry"])
        training_ids = sorted({r["id"] for r in geometry
                               if r["group"] == "calibration" and r["id"] not in config.get("excluded_ids", [])})
    elif definition["kind"] == "standardized_linear":
        for key in ("features", "means", "scales", "coefficients", "intercept"):
            definition[key] = selected[key]
        training_ids = selected["training_ids"]
    else:
        raise ValueError("unsupported frozen score definition")
    sources = [{"path": str(Path(path)), "sha256": digest(root / path), "role": role}
               for role, paths in config["sources"].items() for path in paths]
    # Selection artifact is always included, even if a caller omits it in sources.
    if config["selected_artifact"] not in {s["path"] for s in sources}:
        sources.append({"path": config["selected_artifact"], "sha256": digest(source), "role": "selected_artifact"})
    bundle = {"schema_version": "1.0", "bundle_id": config["bundle_id"],
              "created_at": datetime.now(timezone.utc).isoformat(),
              "definition": definition, "definition_sha256": definition_digest(definition),
              "calibration": {"training_ids": training_ids,
                              "assay_endpoint": config["assay_endpoint"],
                              "scope": config["scope"], "validation_status": config["validation_status"],
                              "limitations": config["limitations"]},
              "sources": sources}
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict):
    if bundle.get("schema_version") != "1.0":
        raise ValueError("unsupported calibration-bundle schema")
    definition = bundle["definition"]
    if definition_digest(definition) != bundle.get("definition_sha256"):
        raise ValueError("frozen definition fingerprint does not match its contents")
    if "approach_consistency" in definition.get("features", []) and (definition.get("descriptor_versions") or {}).get("approach_consistency") != "2.0":
        raise ValueError("approach_consistency changed to definition 2.0; review/recalibrate the frozen model before reuse")
    if not isinstance(definition.get("required_poses"), int) or definition["required_poses"] < 1:
        raise ValueError("frozen definition requires an explicit positive pose count")
    if definition["kind"] == "standardized_linear":
        n = len(definition["features"])
        if n == 0 or len(set(definition["features"])) != n:
            raise ValueError("frozen feature names must be nonempty and unique")
        for key in ("means", "scales", "coefficients"):
            if len(definition[key]) != n or not np.isfinite(definition[key]).all():
                raise ValueError(f"invalid frozen model {key}")
        if not np.isfinite(definition["intercept"]) or np.any(np.asarray(definition["scales"]) <= 0):
            raise ValueError("invalid frozen intercept or scales")
    elif definition["kind"] == "proximity_weighted_attack_angle":
        from .proxies.attack_angle import METRIC_ID
        if definition.get("metric_id") != METRIC_ID or definition.get("score_units") != "degrees" or definition.get("fitted_parameters") != []:
            raise ValueError("invalid current angle metric definition")
    elif definition["kind"] == "geometry_ensemble":
        values = definition["parameters"]
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in values.values()):
            raise ValueError("geometry score parameters must be finite and positive")
    else:
        raise ValueError("unknown frozen model kind")


def scope_status(bundle: dict, candidate: dict) -> tuple[str, bool, list[str]]:
    scope = bundle["calibration"]["scope"]
    definition = bundle["definition"]
    reasons = []
    if candidate.get("step_index") != scope["step_index"]:
        return "incompatible_step", False, ["candidate reaction step differs from calibration"]
    mechanism = candidate.get("mechanism")
    if not mechanism:
        return "unresolved_mechanism", False, ["candidate mechanism must be declared"]
    allowed = definition.get("supported_mechanisms", scope["mechanisms"])
    if mechanism not in allowed:
        return "incompatible_mechanism", False, ["frozen extraction is not defined for this mechanism"]
    if scope.get("assembly") and candidate.get("assembly") != scope["assembly"]:
        return "incompatible_assembly", False, ["candidate molecular assembly differs from the frozen feature definition"]
    if mechanism not in scope["mechanisms"]:
        return "mechanism_transfer", True, ["descriptive geometry only; activity transfer is unvalidated"]
    family = candidate.get("family")
    if family not in scope.get("families", []):
        reasons.append("candidate family is outside or unresolved relative to the calibration panel")
        if definition["kind"] == "standardized_linear":
            return "family_transfer_unavailable", False, reasons
        return "family_transfer", True, reasons
    return "within_calibration_scope", True, reasons


def model_paths(candidate: dict, root: Path) -> list[Path]:
    if "prediction_files" in candidate:
        return [root / p for p in candidate["prediction_files"]]
    if candidate.get("prediction_dir"):
        return sorted((root / candidate["prediction_dir"]).glob("*_model_*.pdb"))
    return []


def frozen_geometry_score(distances, angles, definition: dict) -> dict:
    d, a = np.asarray(distances, float), np.asarray(angles, float)
    if d.shape != a.shape or d.shape != (definition["required_poses"],):
        raise ValueError("geometry does not have the frozen pose count")
    if not np.isfinite(d).all() or not np.isfinite(a).all() or (d < 0).any() or ((a < 0) | (a > 180)).any():
        raise ValueError("invalid distance or angle")
    p = definition["parameters"]
    logit = (p["distance_midpoint_A"]-d)/p["distance_softness_A"]
    # Stable logistic, matching the original extractor without overflow.
    w = np.exp(-np.logaddexp(0., -logit))
    q = np.exp(-.5*((180-a)/p["angular_width_deg"])**2)
    total, square = float(w.sum()), float((w*w).sum())
    return {"score": float((w*q).sum()/(p["regularization"]+total)),
            "distance_support_sum": total,
            "effective_weight_count": total*total/square if square else 0.,
            "minimum_distance_A": float(d.min()), "mean_angle_deg": float(a.mean())}


def frozen_linear_score(features: dict, definition: dict) -> dict:
    x = np.array([float(features[k]) for k in definition["features"]])
    if not np.isfinite(x).all():
        raise ValueError("model features must be finite; missing values are not imputed")
    z = (x-np.asarray(definition["means"]))/np.asarray(definition["scales"])
    terms = z*np.asarray(definition["coefficients"])
    return {"score": float(terms.sum()+definition["intercept"]),
            "features": {k: float(v) for k, v in zip(definition["features"], x)},
            "standardized_features": {k: float(v) for k, v in zip(definition["features"], z)},
            "model_terms": {k: float(v) for k, v in zip(definition["features"], terms, strict=True)}}


def extract_linear_features(candidate: dict, bundle: dict, paths: list[Path], root: Path) -> dict:
    definition = bundle["definition"]
    if candidate.get("features") is not None:
        if candidate.get("feature_definition_id") != definition["feature_definition_id"]:
            raise ValueError("supplied feature table uses an unrecognized descriptor definition")
        if candidate.get("n_poses") != definition["required_poses"]:
            raise ValueError("supplied feature table uses the wrong ensemble size")
        if float(candidate["features"].get("interface_jaccard_min", -1)) < .5:
            raise ValueError("supplied feature table fails the frozen assembly-quality gate")
        return candidate["features"]
    if not candidate.get("template"):
        raise ValueError("frozen assembly extraction requires a candidate-specific reference template")
    from structural_modeling_agent.proxies import ntn_ensemble as provider
    template = provider.ca_cif(root / candidate["template"])
    alpha_len = len(template["A"])
    reference = [{**template[a], **{r+alpha_len: v for r, v in template[b].items()}}
                 for a, b in (("A", "B"), ("C", "D"))]
    if len(paths) != definition["required_poses"]:
        raise ValueError("prediction files do not have the frozen ensemble size")
    poses = [provider.score_pose(p, reference) for p in paths]
    if sorted(p["model"] for p in poses) != list(range(definition["required_poses"])):
        raise ValueError("frozen extraction requires models 0 through the declared final model")
    return provider.aggregate(candidate["accession"], poses)


def apply_bundle(bundle: dict, candidates: list[dict], root: Path = ROOT) -> dict:
    validate_bundle(bundle)
    results = []
    for candidate in candidates:
        accession = candidate.get("accession")
        if not accession:
            raise ValueError("each candidate requires an accession")
        scope, eligible, reasons = scope_status(bundle, candidate)
        item = {"accession": accession, "discovery_rank": candidate.get("discovery_rank"),
                "mechanism": candidate.get("mechanism"), "family": candidate.get("family"),
                "bundle_id": bundle["bundle_id"], "definition_sha256": bundle["definition_sha256"],
                "transfer_status": scope, "score": None,
                "score_units": bundle["definition"]["score_units"], "reasons": reasons,
                "calibration_status": bundle["calibration"]["validation_status"],
                "calibration_id_is_training_member": candidate.get("calibration_id", accession)
                    in bundle["calibration"]["training_ids"],
                "training_membership_note": "ID comparison only; aliases require identity review.",
                "definition_integrity": "verified",
                "input_provenance": "unverified",
                "calibration_source_integrity": "unverified",
                "source_files": []}
        # A source list establishes integrity only when every listed file exists
        # and matches. It cannot verify source attribution or calibration quality.
        try:
            sources = bundle.get("sources") or []
            if sources and all(digest(root / source["path"]) == source["sha256"] for source in sources):
                item["calibration_source_integrity"] = "verified"
        except (OSError, KeyError, TypeError):
            pass
        if not eligible:
            item["status"] = "not_applicable"
            results.append(item)
            continue
        try:
            for source in bundle["sources"]:
                if source["role"] == "extractor_code" and digest(root / source["path"]) != source["sha256"]:
                    raise ValueError("extractor code differs from the frozen bundle; export a reviewed new version")
            paths = model_paths(candidate, root)
            item["n_input_poses"] = len(paths) if paths else candidate.get("n_poses", 0)
            if len({str(p.resolve()) for p in paths}) != len(paths):
                raise ValueError("duplicate prediction files would inflate the ensemble")
            if paths and len({digest(p) for p in paths}) != len(paths):
                raise ValueError("byte-identical predictions cannot establish the frozen ensemble coverage")
            definition = bundle["definition"]
            if paths and len(paths) != definition["required_poses"]:
                raise ValueError("actual prediction files do not have the frozen ensemble size")
            item["sampling_independence"] = "not_established_by_file_hashes"
            item["producer_sample_identity"] = "unverified"
            producer_path = candidate.get("prediction_manifest")
            if producer_path or definition.get("require_producer_sample_ids"):
                if not producer_path or not paths:
                    raise ValueError("this bundle requires a producer manifest and actual prediction files")
                producer_path = root / producer_path
                producer = json.loads(producer_path.read_text())
                entries = producer.get("predictions") or []
                ids = [e.get("sample_id") for e in entries]
                producer_files = {(str((Path(e["path"]) if Path(e["path"]).is_absolute() else producer_path.parent / e["path"]).resolve()), e["sha256"]) for e in entries}
                actual_files = {(str(p.resolve()), digest(p)) for p in paths}
                if (not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids) or
                        len(entries) != len(producer_files) or producer_files != actual_files):
                    raise ValueError("producer sample identities or prediction file bindings are missing, duplicated or inconsistent")
                if definition.get("require_producer_sample_ids") and not candidate.get("input_sha256"):
                    raise ValueError("producer sample identity requires the expected input SHA-256")
                if candidate.get("input_sha256") and producer.get("input_sha256") != candidate["input_sha256"]:
                    raise ValueError("producer input identity differs from the candidate")
                item["producer_sample_identity"] = "manifest_content_verified_not_independence_proof"
                item["source_files"].append({"path": str(producer_path), "sha256": digest(producer_path)})
            if definition["kind"] in {"geometry_ensemble", "proximity_weighted_attack_angle"}:
                if len(paths) != definition["required_poses"]:
                    raise ValueError("prediction files do not have the frozen ensemble size")
                phosphate = candidate.get("phosphate")
                expected = definition["phosphate_by_mechanism"][candidate["mechanism"]]
                if phosphate != expected:
                    raise ValueError("reactive phosphate disagrees with the declared mechanism")
                if definition["kind"] == "proximity_weighted_attack_angle":
                    from .proxies.attack_angle import geometry_from_pdb, weighted_attack_angle
                    values = np.array([geometry_from_pdb(p, phosphate) for p in paths])
                    metrics = weighted_attack_angle(values[:, 0], values[:, 1])
                    item.update(metrics, score=metrics["weighted_attack_angle_deg"], metric_id=definition["metric_id"])
                else:
                    # A historical bundle keeps its original extractor as well
                    # as its parameters. An installed wheel without that study
                    # cannot silently substitute a newer extraction definition.
                    from exp_data.amide_synthetase.code.synthetase_geometry import geometry
                    values = np.array([geometry(p, phosphate) for p in paths])
                    item.update(frozen_geometry_score(values[:, 0], values[:, 1], definition))
            else:
                features = extract_linear_features(candidate, bundle, paths, root)
                item.update(frozen_linear_score(features, definition))
            item["source_files"].extend({"path": str(p), "sha256": digest(p)} for p in paths)
            if candidate.get("feature_source"):
                source = root / candidate["feature_source"]
                source_bytes = source.read_bytes()
                source_hash = hashlib.sha256(source_bytes).hexdigest()
                item["source_files"].append({"path": str(source), "sha256": source_hash})
                if candidate.get("feature_source_sha256"):
                    if source_hash != candidate["feature_source_sha256"]:
                        raise ValueError("feature source fingerprint differs from the declared fingerprint")
                    source_data = json.loads(source_bytes)
                    if not isinstance(source_data, dict) or candidate.get("features") is None:
                        raise ValueError("verified feature source must be an object accompanying supplied features")
                    for field in ("accession", "feature_definition_id", "n_poses", "features"):
                        if source_data.get(field) != candidate.get(field):
                            raise ValueError(f"feature source {field} differs from the supplied candidate")
                    item["input_provenance"] = "source_content_verified"
                    item["reasons"].append("source content matches supplied values; origin and extraction remain unverified")
                else:
                    item["reasons"].append("feature source was fingerprinted; supplied values were not verified against it")
            if candidate.get("features") is not None:
                if item["input_provenance"] == "unverified":
                    item["reasons"].append("supplied features have not been independently verified against their source")
            elif paths:
                item["input_provenance"] = "file_integrity_recorded"
            item["status"] = "scored"
        except (OSError, ValueError, KeyError, AssertionError, ImportError) as exc:
            item.update(status="unavailable", score=None)
            item["reasons"].append(str(exc))
        results.append(item)
    return {"schema_version": "2.0", "bundle": bundle, "candidates": results,
            "note": "Frozen experimental calibration applied without refitting. Discovery ranks are retained. Scores remain exploratory; transfer status and assay scope accompany every result."}


def save_application(output: Path, report: dict):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "candidate_proxy_evidence.json", report)
    fields = ["accession", "discovery_rank", "mechanism", "family", "bundle_id", "status",
              "transfer_status", "score", "score_units", "harmonic_distance_A", "metric_id", "n_input_poses", "calibration_status", "reasons"]
    with (output / "candidate_proxy_scores.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in report["candidates"]:
            writer.writerow({k: "; ".join(r[k]) if k == "reasons" else r.get(k) for k in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("config", type=Path)
    export.add_argument("--output", type=Path, required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("bundle", type=Path)
    apply.add_argument("candidates", type=Path)
    apply.add_argument("--output-dir", type=Path, required=True)
    examples = sub.add_parser("examples")
    examples.add_argument("--output-dir", type=Path, default=ROOT / "structural_modeling_agent/integrated_campaigns")
    args = parser.parse_args()
    if args.command == "export":
        write_json(args.output, export_bundle(json.loads(args.config.read_text())))
        print(args.output)
    elif args.command == "apply":
        manifest = json.loads(args.candidates.read_text())
        candidates = manifest if isinstance(manifest, list) else manifest["candidates"]
        save_application(args.output_dir, apply_bundle(json.loads(args.bundle.read_text()), candidates))
        print(args.output_dir / "candidate_proxy_evidence.json")
    else:
        from structural_modeling_agent.integration_examples import run_examples
        run_examples(args.output_dir)


if __name__ == "__main__":
    main()
