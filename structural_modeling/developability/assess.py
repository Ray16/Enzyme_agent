#!/usr/bin/env python3
"""Developability assessment: fuse curated liability evidence with a real predictor.

This is the developability factor of the closed-loop reward (see
``docs/discovery_loop_design.md``). It is **evidence-first and fail-closed**:

  * A curated, literature-grounded liability (``evidence.py``) is authoritative. A
    ``veto``-severity liability forces a veto regardless of the predictor -- a
    documented liability must never be overridden by a model score. Every curated
    record carries a verifiable citation; no memory-based claims.
  * The sequence predictor (``predictors.py``, default NetSolP) supplies a real,
    published solubility estimate, used to *flag* likely-poor candidates when no curated
    record settles the question.
  * With neither a usable prediction nor a curated record, the assessment **abstains**
    (no developability score, no reward) rather than fabricating a number.

A poor verdict (curated veto or predicted below the floor) sets an
``exploration_trigger``: the loop should look for a soluble homolog of the same family,
because the mechanism transfers (see ``family_assign.py`` + the reaction coordinates).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

try:  # allow both "python assess.py" and package import
    from .evidence import LiabilityKnowledge, LiabilityRecord
    from .predictors import SolubilityPredictor, compute_descriptors, get_predictor
except ImportError:  # pragma: no cover - script mode
    from evidence import LiabilityKnowledge, LiabilityRecord
    from predictors import SolubilityPredictor, compute_descriptors, get_predictor


@dataclass
class DevelopabilityAssessment:
    verdict: str                      # "veto" | "flag" | "ok" | "abstain"
    score: float | None              # developability factor in [0, 1], or None if abstained
    reasons: list[str] = field(default_factory=list)
    liabilities: list[dict] = field(default_factory=list)
    prediction: dict | None = None
    descriptors: dict = field(default_factory=dict)
    exploration_trigger: dict | None = None

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "score": self.score, "reasons": self.reasons,
            "liabilities": self.liabilities, "prediction": self.prediction,
            "descriptors": self.descriptors,
            "exploration_trigger": self.exploration_trigger,
        }


def _trigger(reason: str, hint: str | None) -> dict:
    return {"triggered": True, "reason": reason,
            "hint": hint or "seek a soluble homolog of the same family (mechanism transfers)"}


def assess_developability(
    *,
    sequence: str | None = None,
    uniprot: str | None = None,
    family_ids: tuple[str, ...] | list[str] = (),
    knowledge: LiabilityKnowledge | None = None,
    predictor: SolubilityPredictor | None = None,
) -> DevelopabilityAssessment:
    """Assess one candidate's developability, evidence-first and fail-closed."""
    knowledge = knowledge or LiabilityKnowledge.load()
    predictor = predictor or NetSolPPredictor()

    records: list[LiabilityRecord] = knowledge.lookup(uniprot=uniprot, family_ids=family_ids)
    liabilities = [r.to_dict() for r in records]
    descriptors = compute_descriptors(sequence) if sequence else {}

    # 1) Curated evidence is authoritative.
    veto = next((r for r in records if r.severity == "veto"), None)
    if veto is not None:
        return DevelopabilityAssessment(
            verdict="veto", score=0.0,
            reasons=[f"curated liability ({veto.liability}): {veto.claim}"],
            liabilities=liabilities, descriptors=descriptors,
            exploration_trigger=_trigger(
                f"documented {veto.liability} liability", veto.exploration_hint),
        )

    flag_rec = next((r for r in records if r.severity == "flag"), None)

    # 2) Real predictor (only informative when we have a sequence).
    prediction = predictor.predict(sequence).to_dict() if sequence else None

    # 3) Fail closed: no usable prediction -> abstain (never fabricate a score).
    if prediction is None or prediction.get("score") is None:
        note = (prediction or {}).get("note")
        if flag_rec is not None:
            return DevelopabilityAssessment(
                verdict="flag", score=None,
                reasons=[f"curated liability ({flag_rec.liability}): {flag_rec.claim}"],
                liabilities=liabilities, prediction=prediction, descriptors=descriptors,
                exploration_trigger=_trigger(
                    f"documented {flag_rec.liability} liability", flag_rec.exploration_hint))
        return DevelopabilityAssessment(
            verdict="abstain", score=None,
            reasons=[f"no usable prediction ({note or 'no predictor'}) and no curated "
                     "liability -- abstain (no reward)"],
            liabilities=liabilities, prediction=prediction, descriptors=descriptors)

    score = prediction["score"]
    reasons = [f"predicted solubility={score} ({prediction['method']})"]

    if flag_rec is not None:
        reasons.append(f"curated liability ({flag_rec.liability}): {flag_rec.claim}")
        return DevelopabilityAssessment(
            verdict="flag", score=score, reasons=reasons, liabilities=liabilities,
            prediction=prediction, descriptors=descriptors,
            exploration_trigger=_trigger(
                f"documented {flag_rec.liability} liability", flag_rec.exploration_hint))

    if score < SOLUBILITY_FLOOR:
        reasons.append(f"below solubility floor {SOLUBILITY_FLOOR}")
        return DevelopabilityAssessment(
            verdict="flag", score=score, reasons=reasons, liabilities=liabilities,
            prediction=prediction, descriptors=descriptors,
            exploration_trigger=_trigger("predicted low solubility", None))

    return DevelopabilityAssessment(
        verdict="ok", score=score, reasons=reasons, liabilities=liabilities,
        prediction=prediction, descriptors=descriptors)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", help="candidate amino-acid sequence")
    ap.add_argument("--uniprot", help="UniProt accession (for curated-evidence lookup)")
    ap.add_argument("--family", action="append", default=[], help="InterPro family id (repeatable)")
    args = ap.parse_args(argv)

    a = assess_developability(sequence=args.sequence, uniprot=args.uniprot,
                              family_ids=tuple(args.family))
    print(json.dumps(a.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
