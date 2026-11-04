#!/usr/bin/env python3
"""Curated, literature-grounded developability liabilities.

Authoritative when present: a cited liability with the matching construct/host scope overrides
the sequence-based predictor surrogate. Records are keyed by UniProt accession and/or
InterPro family; family-keyed records TRANSFER to unannotated homologs of that family,
exactly like the reaction coordinates. Mirrors the ``mechanism_evidence`` pattern --
every record carries a citation quote.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIABILITIES = os.path.join(_HERE, "knowledge", "liabilities.json")

SEVERITY_RANK = {"veto": 3, "flag": 2, "note": 1}


@dataclass
class LiabilityRecord:
    id: str
    liability: str
    severity: str
    claim: str
    uniprot: str | None = None
    families: tuple[str, ...] = ()
    exploration_hint: str | None = None
    evidence: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "liability": self.liability, "severity": self.severity,
            "claim": self.claim, "uniprot": self.uniprot,
            "families": list(self.families), "exploration_hint": self.exploration_hint,
            "evidence": list(self.evidence),
        }


class LiabilityKnowledge:
    def __init__(self, records: list[LiabilityRecord]):
        self.records = records

    @classmethod
    def load(cls, path: str = DEFAULT_LIABILITIES) -> "LiabilityKnowledge":
        with open(path) as fh:
            doc = json.load(fh)
        recs = []
        for r in doc.get("records") or []:
            recs.append(LiabilityRecord(
                id=r["id"], liability=r["liability"], severity=r["severity"],
                claim=r["claim"], uniprot=r.get("uniprot"),
                families=tuple(r.get("families") or ()),
                exploration_hint=r.get("exploration_hint"),
                evidence=tuple(r.get("evidence") or ()),
            ))
        return cls(recs)

    def lookup(self, *, uniprot: str | None = None,
               family_ids: tuple[str, ...] | list[str] = ()) -> list[LiabilityRecord]:
        """Return liabilities matching this candidate, most severe first.

        A record matches by exact UniProt accession OR by shared InterPro family (so a
        family-keyed liability transfers to homologs, including unannotated Atlas hits).
        """
        fam = set(family_ids or ())
        hits = []
        for rec in self.records:
            if uniprot and rec.uniprot and rec.uniprot == uniprot:
                hits.append(rec)
            elif fam and set(rec.families) & fam:
                hits.append(rec)
        return sorted(hits, key=lambda r: -SEVERITY_RANK.get(r.severity, 0))
