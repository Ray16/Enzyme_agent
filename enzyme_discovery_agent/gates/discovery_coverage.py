"""Coverage guidance that broadens discovery without blocking rolling evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from enzyme_discovery_agent.gates import AgentState, GateResult
from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status,
    discovery_status,
    family_label,
    load_candidate_registry,
)
from enzyme_discovery_agent.models.exploration_log import append_exploration_log
from enzyme_discovery_agent.models.target_profile import normalize_target_profile
from enzyme_discovery_agent.run_layout import run_file


def _coverage_contract(run_dir: Path) -> dict:
    try:
        context = json.loads((run_file(run_dir, 'target_context.json')).read_text())
    except (OSError, json.JSONDecodeError):
        context = {}
    return normalize_target_profile(context.get("target_profile")).get("discovery_coverage", {})


def assess_discovery_coverage(run_dir: Path) -> dict:
    """Assess evidence representation without assigning a candidate score."""
    contract = _coverage_contract(run_dir)
    candidates = load_candidate_registry(run_dir).get("candidates", {})
    screened = {
        accession: entry for accession, entry in candidates.items()
        if discovery_status(entry) in {"screened", "evaluated", "ranked"}
    }
    families = {family_label(entry.get("metadata", {}), accession)
                for accession, entry in screened.items()}
    canonical_families = {family for family in families if family.startswith("ipr:")}
    missing = []
    if len(screened) < contract["minimum_screened_candidates"]:
        missing.append(
            f"{contract['minimum_screened_candidates'] - len(screened)} more screened candidates"
        )
    if len(families) < contract["minimum_distinct_families"]:
        missing.append(
            f"{contract['minimum_distinct_families'] - len(families)} more distinct families"
        )

    bucket_status = []
    for bucket in contract.get("evidence_buckets", []):
        terms = [term.casefold() for term in bucket["match_any"]]
        matches = []
        for accession, entry in screened.items():
            meta = entry.get("metadata", {})
            searchable = " ".join(str(meta.get(field, "")) for field in (
                "protein_name", "function", "catalytic_activities", "gene_names",
                "ec_numbers", "structural_homology",
            )).casefold()
            if any(term in searchable for term in terms):
                matches.append(accession)
        item = {"name": bucket["name"], "matches": sorted(matches),
                "minimum_candidates": bucket["minimum_candidates"]}
        bucket_status.append(item)
        if len(matches) < bucket["minimum_candidates"]:
            missing.append(
                f"{bucket['name']} ({len(matches)}/{bucket['minimum_candidates']} evidence-backed candidates)"
            )
    return {
        "covered": not missing,
        "enforcement": "advisory",
        "screened_candidates": len(screened),
        "distinct_families": len(families),
        "canonical_families": len(canonical_families),
        "family_labels": sorted(families),
        "buckets": bucket_status,
        "missing": missing,
        "contract": contract,
    }


class DiscoveryCoverageGate:
    """Report target-defined evidence coverage alongside rolling evaluation."""

    def should_fire(self, state: AgentState, has_eval_calls: bool, messages: list[dict]) -> bool:
        return state.candidate_count > 0 and not state.eval_gate_fired

    def execute(self, state: AgentState, messages: list[dict]) -> GateResult:
        coverage = assess_discovery_coverage(state.run_dir)
        if coverage["covered"]:
            return GateResult(intercept=False, state_updates={
                "discovery_coverage_done": True,
                "discovery_coverage_incomplete": False,
            })

        # Redirect only once.  Subsequent turns must be allowed to execute
        # screening/retrieval calls that close the recorded gaps. Coverage is
        # advisory: it does not disable rolling evaluation.
        if state.discovery_coverage_redirected:
            return GateResult(intercept=False, state_updates={
                "discovery_coverage_incomplete": False,
            })

        missing = "\n".join(f"- {item}" for item in coverage["missing"])
        buckets = coverage["contract"].get("evidence_buckets", [])
        bucket_guidance = ""
        if buckets:
            bucket_guidance = (
                "\nTarget evidence categories still need representation. For each category, seek "
                "primary-literature or curated catalytic evidence whose annotation contains one of "
                "its target-profile descriptors:\n"
                + "\n".join(
                    f"- {bucket['name']}: {', '.join(bucket['match_any'])}"
                    for bucket in buckets
                )
            )
        append_exploration_log(
            state.run_dir,
            entry_type="COVERAGE",
            title="Discovery coverage incomplete; continue evaluation and broaden discovery",
            details={"Missing": "; ".join(coverage["missing"]),
                     "Screened candidates": coverage["screened_candidates"],
                     "Distinct families": coverage["distinct_families"]},
        )
        return GateResult(
            intercept=True,
            new_messages=[{
                "role": "user",
                "content": (
                    "DISCOVERY COVERAGE CHECKPOINT: continue evaluating the available candidates. "
                    "The current screened set needs broader evidence for a defensible comparison. "
                    "Search targeted primary literature and resolve concrete accessions for the missing "
                    "evidence types, then screen only evidence-backed accessions. EC class may be used "
                    "as a compatibility filter, but not as the source of a broad candidate expansion.\n"
                    f"\nCoverage gaps:\n{missing}{bucket_guidance}\n\n"
                    "Do not screen generic database result lists. Add a candidate only when a paper or "
                    "curated reaction annotation supplies a substrate/mechanism rationale."
                ),
            }],
            log_message="Discovery coverage incomplete; rolling evaluation remains available",
            state_updates={
                "discovery_coverage_incomplete": False,
                "discovery_coverage_redirected": True,
            },
        )
