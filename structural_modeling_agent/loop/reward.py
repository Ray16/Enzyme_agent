#!/usr/bin/env python3
"""Multi-objective, evidence-grounded, fail-closed reward.

    reward = mechanistic_plausibility x calibrated_activity_proxy x developability

Each factor is in [0, 1] and can VETO by being ``None`` (abstained) or 0 (vetoed):

  * mechanistic_plausibility -- from the modeling_reasoning evaluator; ``None`` when the
    reaction coordinate is unknown or the fail-closed step-consistency gate blocks.
  * calibrated_activity_proxy -- the structural proxy AFTER calibration against
    experiment (``calibration.py``); a raw proxy score must never enter the reward
    uncalibrated.
  * developability -- from the developability layer; ``0.0`` on a curated veto (e.g.
    a documented expression liability), ``None`` when it abstains.

Fail-closed rule: if ANY factor is ``None``, the candidate is *abstained* -- it gets no
reward AND contributes no calibration data (it must not poison the surrogate). A ``0``
factor yields reward ``0`` with ``eligible=False`` but is NOT abstained. This is the
Goodhart guardrail: a candidate cannot ride a high proxy past an unknown mechanism or a
documented liability.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Reward:
    reward: float | None
    mechanism: float | None
    proxy: float | None
    developability: float | None
    abstained: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return not self.abstained and all(v is not None and v > 0 for v in
                                         (self.mechanism, self.proxy, self.developability))

    def to_dict(self) -> dict:
        return {
            "reward": self.reward, "abstained": self.abstained, "eligible": self.eligible,
            "mechanism": self.mechanism, "proxy": self.proxy,
            "developability": self.developability, "reasons": self.reasons,
        }


def _check(name: str, value: float | None, reasons: list[str]) -> bool:
    if value is None:
        reasons.append(f"{name}: abstained (None) -> fail-closed")
        return False
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} factor must be in [0,1], got {value}")
    if value == 0.0:
        reasons.append(f"{name}: vetoed (0.0)")
    return True


def combine_reward(*, mechanism: float | None, proxy: float | None,
                   developability: float | None) -> Reward:
    """Combine the three factors into a fail-closed multi-objective reward."""
    reasons: list[str] = []
    ok = all([
        _check("mechanism", mechanism, reasons),
        _check("proxy", proxy, reasons),
        _check("developability", developability, reasons),
    ])
    if not ok:
        return Reward(reward=None, mechanism=mechanism, proxy=proxy,
                      developability=developability, abstained=True, reasons=reasons)
    value = mechanism * proxy * developability
    return Reward(reward=value, mechanism=mechanism, proxy=proxy,
                  developability=developability, abstained=False, reasons=reasons)
