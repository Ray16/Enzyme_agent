"""Validated, mechanism-agnostic target profiles for evidence-aware discovery."""

from __future__ import annotations

import json
from pathlib import Path

_STRUCTURAL_EVIDENCE_CHECKS = frozenset({
    "catalytic_center", "pocket_engagement", "cofold_confidence", "ligand_burial",
})


def _normalize_discovery_coverage(value: object) -> dict:
    """Normalize a target-supplied discovery coverage contract.

    This is deliberately a retrieval/evaluation stop condition, not a ranking
    formula.  The target owner can describe the kinds of evidence that must be
    represented before candidates are compared, without baking a family name
    or a target-specific exception into the agent.
    """
    default = {
        "minimum_screened_candidates": 4,
        "minimum_distinct_families": 4,
        "evidence_buckets": [],
    }
    if value is None:
        return default
    if not isinstance(value, dict):
        raise ValueError("discovery_coverage must be an object.")
    result = dict(default)
    result.update(value)
    for field in ("minimum_screened_candidates", "minimum_distinct_families"):
        number = result[field]
        if not isinstance(number, int) or number < 1:
            raise ValueError(f"discovery_coverage.{field} must be a positive integer.")
    buckets = result["evidence_buckets"]
    if not isinstance(buckets, list):
        raise ValueError("discovery_coverage.evidence_buckets must be a list.")
    normalized_buckets = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            raise ValueError("Each discovery coverage evidence bucket must be an object.")
        name = bucket.get("name")
        terms = bucket.get("match_any")
        minimum = bucket.get("minimum_candidates", 1)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each evidence bucket needs a non-empty name.")
        if (not isinstance(terms, list) or not terms
                or not all(isinstance(term, str) and term.strip() for term in terms)):
            raise ValueError("Each evidence bucket needs a non-empty match_any list.")
        if not isinstance(minimum, int) or minimum < 1:
            raise ValueError("evidence bucket minimum_candidates must be a positive integer.")
        normalized_buckets.append({
            "name": name.strip(),
            "match_any": [term.strip() for term in terms],
            "minimum_candidates": minimum,
        })
    result["evidence_buckets"] = normalized_buckets
    return result


def normalize_target_profile(profile: dict | None) -> dict:
    """Validate and normalize a portable target profile without target-specific rules."""
    if profile is None:
        return {
            "schema_version": "1.0",
            "required_evidence": sorted(_STRUCTURAL_EVIDENCE_CHECKS),
            "review_requirements": [],
            "discovery_coverage": _normalize_discovery_coverage(None),
        }
    if not isinstance(profile, dict):
        raise ValueError("Target profile must be a JSON object.")
    if profile.get("schema_version", "1.0") != "1.0":
        raise ValueError("Unsupported target-profile schema; expected version 1.0.")
    result = dict(profile)
    result["schema_version"] = "1.0"
    for field in ("reaction", "substrate", "enzyme_requirements"):
        value = result.setdefault(field, {})
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be an object.")
    required_evidence = result.setdefault("required_evidence", sorted(_STRUCTURAL_EVIDENCE_CHECKS))
    if not isinstance(required_evidence, list) or not all(isinstance(item, str) and item.strip() for item in required_evidence):
        raise ValueError("required_evidence must be a list of non-empty strings.")
    unknown = set(required_evidence) - _STRUCTURAL_EVIDENCE_CHECKS
    if unknown:
        raise ValueError(
            "required_evidence may contain only automatically assessed structural checks: "
            f"{sorted(_STRUCTURAL_EVIDENCE_CHECKS)}. Put other questions in review_requirements."
        )
    review_requirements = result.setdefault("review_requirements", [])
    if not isinstance(review_requirements, list) or not all(isinstance(item, str) and item.strip() for item in review_requirements):
        raise ValueError("review_requirements must be a list of non-empty strings.")
    result["discovery_coverage"] = _normalize_discovery_coverage(
        result.get("discovery_coverage")
    )
    return result


def load_target_profile(path: str | Path) -> dict:
    """Load a JSON target profile suitable for the discovery CLI."""
    return normalize_target_profile(json.loads(Path(path).read_text()))
