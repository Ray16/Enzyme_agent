#!/usr/bin/env python3
"""Load and match the vendored, PDF-extracted cofolding guidance.

The knowledge is a snapshot of the reasoning module's exact-grounded candidate
rules (see ``knowledge/PROVENANCE.md``). Each rule is keyed by ``decision_point``
and guarded by ``applies_when`` feature flags. This module joins those rules onto
a candidate's features so a resolved handoff can cite the PDF evidence
(``paper_id`` + ``claim_id`` + verbatim quote) that justified each modeling
decision.

Rules are candidate guidance, never verified defaults. Matching a rule attaches
provenance; it does not by itself decide anything.
"""
from __future__ import annotations

import json
import os
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KNOWLEDGE = os.path.join(HERE, "knowledge", "cofolding_rules.json")


class LiteratureKnowledge:
    """A queryable view over vendored candidate rules."""

    def __init__(self, document: dict[str, Any], *, source: str | None = None):
        self.source = source
        self.schema_version = document.get("schema_version")
        self.corpus = document.get("corpus", {})
        self.caveat = document.get("caveat")
        self.rules: list[dict[str, Any]] = list(document.get("candidate_rules") or [])

    @classmethod
    def load(cls, path: str = DEFAULT_KNOWLEDGE) -> "LiteratureKnowledge":
        with open(path) as handle:
            return cls(json.load(handle), source=os.path.abspath(path))

    @staticmethod
    def _guard_satisfied(applies_when: dict[str, Any], features: dict[str, Any]) -> bool:
        """A rule applies when every guard key is present and equal in features.

        An empty guard applies unconditionally. A guard key absent from features
        is treated as unsatisfied (fail closed) so we never over-claim relevance.
        """
        for key, expected in (applies_when or {}).items():
            if features.get(key) != expected:
                return False
        return True

    def rules_for(
        self, decision_point: str, features: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Applicable rules for one decision point, most-supported first."""
        matched = [
            rule
            for rule in self.rules
            if rule.get("decision_point") == decision_point
            and self._guard_satisfied(rule.get("applies_when") or {}, features)
        ]
        matched.sort(key=lambda r: _independent_papers(r), reverse=True)
        return matched

    def basis_for(
        self, decision_point: str, features: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Compact, provenance-preserving citation blocks for a decision point."""
        blocks = []
        for rule in self.rules_for(decision_point, features):
            papers = sorted({e.get("paper_id") for e in rule.get("evidence") or [] if e.get("paper_id")})
            blocks.append(
                {
                    "decision_point": decision_point,
                    "guidance": rule.get("guidance"),
                    "applies_when": rule.get("applies_when"),
                    "support": (
                        "cross-paper-candidate"
                        if len(papers) >= 2
                        else "single-paper-candidate"
                    ),
                    "papers": papers,
                    "evidence": [
                        {
                            "paper_id": e.get("paper_id"),
                            "claim_id": e.get("claim_id"),
                            "quote": e.get("quote"),
                        }
                        for e in (rule.get("evidence") or [])
                    ],
                    "knowledge_source": self.source,
                }
            )
        return blocks


def _independent_papers(rule: dict[str, Any]) -> int:
    return len({e.get("paper_id") for e in rule.get("evidence") or [] if e.get("paper_id")})
