"""Gate 5: does candidate expansion actually improve discovery? Frozen before runs.

CHEMICAL_REASONING.md gate 5 requires a budget-matched baseline/ablation comparison
with scoring frozen BEFORE outputs are inspected. This module is that frozen scoring,
committed before any ablation run exists, so the success rule cannot be shaped to the
result -- the same discipline that made the factoring comparison trustworthy.

The ablation is the expansion change shipped this session (commit a1de65ac):
  baseline  = ENZYME_EXPANSION_EVAL_INTERVAL unset or expansion effectively off
  treatment = expansion reachable (the shipped default)
Both run at the SAME token budget on the SAME benchmark query. This module does NOT
run the campaigns (that is real compute, launched separately and explicitly); it
scores two finished run directories against the never-seen oracle in eval/benchmarks.

Pre-registered success rule (frozen)
------------------------------------
Expansion is beneficial on a benchmark only if, at matched budget, the treatment run:
  1. does not reduce primary recall (judged specific accessions/members), and
  2. does not reduce family-coverage fraction at the candidate stage, and
  3. does not increase unsupported promotions, and
  4. improves at least one of: family coverage, primary recall, or unannotated
     reach (candidates carried that keyword retrieval alone would not surface),
  5. without worsening tokens-per-supported-hypothesis by more than 10%.
A single paired run cannot establish this; multiple budget-matched trials per
benchmark are required before any claim, and unjudged candidates are never counted
as false positives. This module reports the per-criterion verdict; it does not
declare overall success from one pair.
"""

from pathlib import Path

from enzyme_discovery_agent.eval.benchmarks import BENCHMARKS
from enzyme_discovery_agent.eval.metrics import score_run

TOKENS_REGRESSION_TOLERANCE = 0.10


def _primary_recall(score):
    rec = score.get("recall") or {}
    primary = [name for name in rec if rec[name].get("hit") is not None]
    found = sum(len(rec[name]["found"]) for name in primary)
    total = sum(rec[name]["n"] for name in primary)
    return (found / total) if total else 0.0


def _candidate_family_fraction(score):
    cov = ((score.get("family_coverage") or {}).get("candidates") or {})
    return cov.get("fraction") or 0.0


def _unsupported_promotions(run_dir):
    """Prioritised candidates resting on speculative evidence -- should stay zero."""
    import csv
    path = Path(run_dir) / "ranked_candidates.csv"
    if not path.exists():
        return None
    bad = 0
    for row in csv.DictReader(open(path)):
        decision = (row.get("target_assessment_decision") or "").strip()
        level = (row.get("evidence_level") or "").strip().lower()
        if decision == "prioritize" and level in {"", "none", "speculative", "low"}:
            bad += 1
    return bad


def _supported_count(run_dir):
    import csv
    path = Path(run_dir) / "ranked_candidates.csv"
    if not path.exists():
        return 0
    return sum(1 for row in csv.DictReader(open(path))
               if (row.get("target_assessment_decision") or "").strip() in {"prioritize", "investigate"})


def _tokens_per_supported(run_dir, score):
    tokens = score.get("tokens")
    supported = _supported_count(run_dir)
    if not tokens or not supported:
        return None
    return tokens / supported


def _unannotated_reach(run_dir):
    """Ranked candidates with no native catalytic annotation -- the reach keyword
    retrieval cannot provide, which expansion is meant to open up."""
    import csv
    path = Path(run_dir) / "ranked_candidates.csv"
    if not path.exists():
        return 0
    return sum(1 for row in csv.DictReader(open(path))
               if not (row.get("catalytic_activities") or "").strip())


def compare(baseline_dir, treatment_dir, benchmark_key):
    """Score a budget-matched baseline/treatment pair against the frozen rule.

    Returns per-criterion verdicts. Does NOT declare overall success: one pair
    cannot, and the module says so.
    """
    if benchmark_key not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark_key!r}")
    benchmark = BENCHMARKS[benchmark_key]
    base, treat = score_run(baseline_dir, benchmark), score_run(treatment_dir, benchmark)
    base_tps, treat_tps = _tokens_per_supported(baseline_dir, base), _tokens_per_supported(treatment_dir, treat)
    criteria = {
        "primary_recall_not_reduced": _primary_recall(treat) >= _primary_recall(base),
        "family_coverage_not_reduced":
            _candidate_family_fraction(treat) >= _candidate_family_fraction(base),
        "no_more_unsupported_promotions":
            (_unsupported_promotions(treatment_dir) or 0) <= (_unsupported_promotions(baseline_dir) or 0),
        "improves_breadth_recall_or_reach": (
            _candidate_family_fraction(treat) > _candidate_family_fraction(base)
            or _primary_recall(treat) > _primary_recall(base)
            or _unannotated_reach(treatment_dir) > _unannotated_reach(baseline_dir)),
        "tokens_per_supported_not_worse": (
            base_tps is None or treat_tps is None
            or treat_tps <= base_tps * (1 + TOKENS_REGRESSION_TOLERANCE)),
    }
    return {
        "benchmark": benchmark_key,
        "baseline": {"dir": str(baseline_dir), "primary_recall": _primary_recall(base),
                     "candidate_family_fraction": _candidate_family_fraction(base),
                     "unsupported_promotions": _unsupported_promotions(baseline_dir),
                     "unannotated_reach": _unannotated_reach(baseline_dir),
                     "tokens": base.get("tokens"), "tokens_per_supported": base_tps},
        "treatment": {"dir": str(treatment_dir), "primary_recall": _primary_recall(treat),
                      "candidate_family_fraction": _candidate_family_fraction(treat),
                      "unsupported_promotions": _unsupported_promotions(treatment_dir),
                      "unannotated_reach": _unannotated_reach(treatment_dir),
                      "tokens": treat.get("tokens"), "tokens_per_supported": treat_tps},
        "criteria": criteria,
        "pair_passes_all_criteria": all(criteria.values()),
        "scope": ("One budget-matched pair. Multiple trials per benchmark are required before "
                  "any discovery-benefit claim; unjudged candidates are not false positives."),
    }
