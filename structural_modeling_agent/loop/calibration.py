#!/usr/bin/env python3
"""Calibration ledger: keep the structural proxy honest against experiment.

The structural proxy is a weak surrogate (rho~=0.09 raw on our panel; restraints invert
it; calibrated on ~3 positives). It must NEVER enter the reward uncalibrated. This module
records every (proxy score, assay result) pair and, per family (mechanism -- and so
calibration -- transfers by family), computes:

  * ``correlation`` -- running Spearman/Pearson of proxy vs assay, the surrogate's
    trustworthiness; also the Goodhart-drift detector (proxy climbing while assays flat).
  * ``calibrated_proxy`` -- a proxy score shrunk toward a prior by how much (and how
    trustworthily) we have actually observed, returned WITH an uncertainty.

Abstained candidates (fail-closed reward) are recorded but excluded from correlation so
they cannot poison the surrogate. Stdlib-only: Spearman is Pearson on ranks.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enzyme_discovery_agent.state_store import atomic_write_json

# With no/weak per-family evidence, shrink the calibrated estimate toward this neutral
# prior and report high uncertainty -- we do not trust the raw proxy on its own.
PRIOR_MEAN = 0.5
# Pseudo-count controlling how fast trust grows with observations.
SHRINK_K = 5.0


@dataclass
class CalibrationRecord:
    candidate_id: str
    proxy_score: float | None
    assay_value: float | None
    family: str | None = None
    proxy_version: str | None = None
    abstained: bool = False
    conditions: dict = field(default_factory=dict)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _ranks(values: list[float]) -> list[float]:
    """Average (fractional) ranks, so ties do not bias Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


class CalibrationLedger:
    def __init__(self, records: list[CalibrationRecord] | None = None):
        self.records: list[CalibrationRecord] = list(records or [])

    def add(self, record: CalibrationRecord) -> None:
        self.records.append(record)

    def _pairs(self, family: str | None, *, proxy_version=None, conditions=None) -> tuple[list[float], list[float]]:
        # Omitted scope selects legacy unversioned/unspecified records only.
        # A candidate contributes once; repeated measurements are not new enzymes.
        pairs = {}
        for r in self.records:
            if r.abstained or r.proxy_score is None or r.assay_value is None:
                continue
            if family is not None and r.family != family:
                continue
            if r.proxy_version != proxy_version or r.conditions != (conditions or {}):
                continue
            if not all(math.isfinite(v) for v in (r.proxy_score, r.assay_value)):
                raise ValueError("calibration observations must be finite")
            pairs[(r.family, r.candidate_id)] = (r.proxy_score, r.assay_value)
        return ([p[0] for p in pairs.values()], [p[1] for p in pairs.values()])

    def correlation(self, family: str | None = None, *, proxy_version=None, conditions=None) -> dict:
        xs, ys = self._pairs(family, proxy_version=proxy_version, conditions=conditions)
        return {
            "family": family, "proxy_version": proxy_version, "conditions": conditions or {}, "n": len(xs),
            "spearman": _spearman(xs, ys), "pearson": _pearson(xs, ys),
        }

    def surrogate_uncertainty(self, family: str | None = None, *, proxy_version=None, conditions=None) -> float:
        """Heuristic lack-of-trust score in [0, 1], not a predictive uncertainty interval."""
        corr = self.correlation(family, proxy_version=proxy_version, conditions=conditions)
        n, rho = corr["n"], corr["spearman"]
        # The declared score direction is frozen. Negative association cannot
        # validate it; reversing or fitting a score requires a separate study.
        trust = max(0.0, rho) if rho is not None else 0.0
        if family is None:
            trust = 0.0  # pooled families cannot establish candidate-level transfer
        # Shrink by both amount of data and how correlated it is.
        confidence = (n / (n + SHRINK_K)) * trust
        return round(1.0 - confidence, 4)

    def calibrated_proxy(self, proxy_score: float, family: str | None = None, *, proxy_version=None, conditions=None) -> dict:
        """Shrink a raw proxy score toward the prior using heuristic trust.

        Returns the calibrated estimate (usable as the reward's proxy factor) and its
        uncertainty. With little/uncorrelated data the estimate collapses to the prior.
        Neither value is a validated prediction or a statistical uncertainty interval.
        """
        if not math.isfinite(proxy_score) or not 0 <= proxy_score <= 1:
            raise ValueError("normalized proxy score must be finite and in [0, 1]")
        scope = {"proxy_version": proxy_version, "conditions": conditions}
        u = self.surrogate_uncertainty(family, **scope)
        confidence = 1.0 - u
        estimate = confidence * proxy_score + u * PRIOR_MEAN
        return {"estimate": round(estimate, 4), "uncertainty": u,
                "uncertainty_kind": "heuristic_lack_of_trust_not_prediction_interval",
                "validation_status": "not_established_by_this_ledger",
                "status": "supported_direction" if confidence > 0 else "untrusted",
                "confidence": round(confidence, 4), **self.correlation(family, **scope)}

    # --- persistence -------------------------------------------------------------
    def to_json(self, path: str) -> None:
        atomic_write_json(path, [asdict(r) for r in self.records], allow_nan=False)

    @classmethod
    def from_json(cls, path: str) -> "CalibrationLedger":
        with open(path) as fh:
            return cls([CalibrationRecord(**r) for r in json.load(fh)])
