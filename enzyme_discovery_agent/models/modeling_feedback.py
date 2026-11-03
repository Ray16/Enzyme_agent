"""Project saved structural observations without granting activity authority."""

import math

_BOOKKEEPING = {"timestamp", "ts", "updated_at", "derived_at", "checked_at", "checked_utc",
                "created_at", "path", "source_report", "reason", "rationale", "note"}


def _facts(value):
    if isinstance(value, dict):
        return {k: _facts(v) for k, v in value.items() if k not in _BOOKKEEPING}
    if isinstance(value, list):
        return [_facts(v) for v in value]
    return value


def modeling_feedback_basis(metadata):
    """Hash observations, not rewritten report prose or report-wrapper hashes."""
    basis = {}
    report = metadata.get("modeling_validation_evidence")
    if isinstance(report, dict):
        basis["validation"] = _facts({k: report.get(k) for k in (
            "revision", "status", "review_errors", "setup_validation", "pose_validation",
            "geometry_metrics", "proxy_evaluation")})
    proxy = metadata.get("structural_proxy_evidence")
    if isinstance(proxy, dict):
        basis["proxy"] = _facts({k: proxy.get(k) for k in (
            "status", "bundle_id", "definition_sha256", "score", "score_units", "transfer_status",
            "calibration_status", "n_input_poses", "harmonic_distance_A", "metric_id",
            "features", "measurements", "source_files")})
    return basis


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def modeling_feedback_summary(accession, metadata, source):
    """Bounded review cues; saved validation is not independently rerun here."""
    report = metadata.get("modeling_validation_evidence")
    if isinstance(report, dict):
        metrics = report.get("geometry_metrics") or {}
        sites = []
        for site, values in sorted(metrics.items()):
            if not isinstance(values, dict):
                continue
            sites.append({"site": str(site)[:100], **{key: _number(values.get(key)) for key in (
                "weighted_attack_angle_deg", "harmonic_distance_A", "n_poses")},
                "complete_observed_ensemble": values.get("complete_observed_ensemble")})
        return {"accession": accession, "kind": "modeling_validation",
                "reported_status": report.get("status", "unknown"), "revision": report.get("revision"),
                "setup_status": (report.get("setup_validation") or {}).get("status", "unknown"),
                "pose_status": (report.get("pose_validation") or {}).get("status", "unknown"),
                "site_metrics": sites[:4], "omitted_sites": max(0, len(sites)-4),
                "review_errors": [str(x)[:180] for x in report.get("review_errors", [])[:4]],
                "source": source,
                "interpretation": "Review setup, distance, angle and sampling jointly. Saved checks are not revalidated by this view; target activity is unestablished."}
    proxy = metadata.get("structural_proxy_evidence")
    if isinstance(proxy, dict):
        return {"accession": accession, "kind": "frozen_proxy_application",
                "reported_status": proxy.get("status", "unknown"), "score": _number(proxy.get("score")),
                "score_units": proxy.get("score_units"), "transfer_status": proxy.get("transfer_status"),
                "calibration_status": proxy.get("calibration_status"),
                "harmonic_distance_A": _number(proxy.get("harmonic_distance_A")), "source": source,
                "interpretation": "Keep original units and transfer scope; a proxy application does not establish target activity."}
    return None
