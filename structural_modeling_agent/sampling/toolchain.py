"""Read-only inventory of explicitly scoped multiscale tool validation.

No accelerator initialization, download, submission or scientific setup defaults
occur here. Revalidate evidence bytes before reporting a recorded check current.
"""
from pathlib import Path
import json

from ..validation import file_digest as digest


def inspect_toolchain(manifest_path=None):
    path = Path(manifest_path) if manifest_path else Path(__file__).with_name("toolchain_manifest.json")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "processivity-toolchain-1.0":
        raise ValueError("Unsupported toolchain inventory")
    results = []
    for item in manifest["checks"]:
        errors = []
        for evidence in item["evidence"]:
            try:
                evidence_path = Path(evidence["path"])
                if not evidence_path.is_absolute():
                    evidence_path = path.parent / evidence_path
                if digest(evidence_path) != evidence["sha256"]:
                    errors.append(f"Changed evidence: {evidence['path']}")
            except OSError:
                errors.append(f"Unavailable evidence: {evidence['path']}")
        results.append({**item, "evidence_current": not errors, "errors": errors})
    environments = []
    for env in manifest["environments"]:
        entrypoint = ("interpreter_available", "python") if "python" in env else ("executable_available", "executable")
        environments.append({**env, entrypoint[0]: Path(env[entrypoint[1]]).is_file()})
    configuration_errors = []
    for evidence in manifest.get("preparation_configuration_evidence", []):
        try:
            location = Path(evidence["path"])
            location = location if location.is_absolute() else path.parent / location
            if digest(location) != evidence["sha256"]:
                configuration_errors.append(f"Changed preparation configuration evidence: {location}")
        except OSError:
            configuration_errors.append(f"Unavailable preparation configuration evidence: {location}")
    profiles = [{**profile, "interpreter_available": Path(profile["python"]).is_file()}
                for profile in manifest.get("preparation_operator_profiles", [])]
    return {**manifest, "checks": results, "environments": environments,
            "preparation_operator_profiles": profiles, "configuration_errors": configuration_errors,

            "recorded_evidence_current": all(c["evidence_current"] for c in results) and not configuration_errors,
            "live_hardware_tested_now": False, "execution_authorized": False,
            "chemistry_validated": False, "processivity_demonstrated": False,
            "interpretation": "Recorded software/numerical scope only. Agent-selected chemical setup, reference accuracy and convergence remain separate gates."}
