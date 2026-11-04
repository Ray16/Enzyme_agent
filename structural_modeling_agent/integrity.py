"""Integrity checks for reported artifacts; these do not establish scientific validity."""
import json
from pathlib import Path

from .validation import file_digest, fingerprint


def proxy_integrity(plan, evaluation, artifact, poses=None):
    if evaluation is None:
        return {"status": "not_evaluated", "errors": []}
    errors = []
    try:
        if plan is None or evaluation.get("plan") != plan or evaluation.get("plan_sha256") != fingerprint(plan):
            errors.append("proxy plan differs from the evaluated plan")
        if not artifact or fingerprint(json.loads(Path(artifact).read_text())) != fingerprint(evaluation):
            errors.append("proxy evaluation artifact is unavailable or changed")
        sources = evaluation.get("source_files") or []
        if not sources:
            errors.append("proxy prediction inputs are missing")
        if poses is not None:
            evaluated = {(str(Path(x["path"]).resolve()), x["sha256"]) for x in sources}
            inspected = {(str(Path(x["path"]).resolve()), x["sha256"]) for x in poses}
            if len(evaluated) != len(sources) or evaluated != inspected:
                errors.append("proxy inputs differ from the inspected prediction ensemble")
        for entry in sources + (evaluation.get("confidence_sources") or []):
            if file_digest(entry["path"]) != entry["sha256"]:
                errors.append("proxy input changed: " + entry["path"])
        expected_sidecars = {str(Path(x["path"]).with_name("confidence_" + Path(x["path"]).stem + ".json").resolve())
                             for x in sources}
        recorded_sidecars = {str(Path(x["path"]).resolve()) for x in evaluation.get("confidence_sources") or []}
        existing_sidecars = {p for p in expected_sidecars if Path(p).exists()}
        if existing_sidecars != recorded_sidecars:
            errors.append("proxy confidence sidecar set changed")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append("proxy supporting artifact unavailable or invalid: " + str(exc))
    return {"status": "verified" if not errors else "unverified", "errors": errors}
