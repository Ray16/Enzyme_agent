#!/usr/bin/env python3
"""Acquisition: choose which candidates to promote to the next (expensive) fidelity.

The loop is multi-fidelity active search (see ``docs/discovery_loop_design.md``): the
acquisition function scores each candidate by a UCB-style trade-off of exploitation
(the calibrated multi-objective reward) and exploration (surrogate uncertainty +
novelty), then selects a batch under the polite/expensive-fidelity budget.

    acquisition = reward + beta * (w_u * uncertainty + w_n * novelty)

Fail-closed candidates (abstained / vetoed reward) are NEVER selected -- they carry no
reward and must not be promoted to cofold or assay. Novelty is supplied per candidate
(e.g. 1 - max structural similarity to already-assayed hits); ``novelty_from_similarity``
is a helper for the common case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class Candidate:
    id: str
    reward: float | None            # from reward.combine_reward(); None if abstained/vetoed-out
    uncertainty: float = 0.0        # surrogate uncertainty for this candidate's family [0,1]
    novelty: float = 0.0            # distance from what we already know [0,1]
    abstained: bool = False
    meta: dict = field(default_factory=dict)
    eligible: bool = True


def novelty_from_similarity(similarities: list[float]) -> float:
    """Novelty = 1 - (max similarity to any known candidate). Empty -> maximally novel."""
    if not similarities:
        return 1.0
    return max(0.0, 1.0 - max(similarities))


def acquisition_score(cand: Candidate, *, beta: float = 0.5,
                      w_uncertainty: float = 0.5, w_novelty: float = 0.5) -> float | None:
    """UCB-style acquisition; ``None`` for candidates that must not be selected."""
    if cand.abstained or cand.reward is None or not cand.eligible or cand.reward == 0:
        return None
    for name, value in (("reward", cand.reward), ("uncertainty", cand.uncertainty), ("novelty", cand.novelty)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if any(not math.isfinite(v) or v < 0 for v in (beta, w_uncertainty, w_novelty)):
        raise ValueError("acquisition weights must be finite and nonnegative")
    explore = w_uncertainty * cand.uncertainty + w_novelty * cand.novelty
    return cand.reward + beta * explore


def rank_candidates(candidates: list[Candidate], **kw) -> list[tuple[Candidate, float]]:
    """Return selectable candidates with their acquisition scores, best first."""
    scored = []
    for c in candidates:
        s = acquisition_score(c, **kw)
        if s is not None:
            scored.append((c, s))
    scored.sort(key=lambda cs: -cs[1])
    return scored


def select_batch(candidates: list[Candidate], k: int, **kw) -> list[Candidate]:
    """Select the top-k candidates to promote to the next fidelity (excludes abstained)."""
    if type(k) is not int or k < 0:
        raise ValueError("batch size must be a nonnegative integer")
    if len({c.id for c in candidates}) != len(candidates):
        raise ValueError("candidate identifiers must be unique")
    return [c for c, _ in rank_candidates(candidates, **kw)[:k]]
