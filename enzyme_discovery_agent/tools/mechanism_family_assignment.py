"""Assign each candidate to one of the run's *mechanism families*.

The run derives a small set of mechanism families up front (see
``tools/mechanism_families.py`` -> ``mechanism_families.json``): the distinct
enzyme fold/superfamily/family classes that could catalyse the target reaction,
each with model-written ``descriptors`` (the annotation/domain words that
identify its members) and a ``promise`` rating.  That derivation is a *prior*;
nothing downstream ever tied the actual evaluated candidates back to it.  This
module closes that loop, deterministically and offline:

  * :func:`assign_mechanism_family` labels one candidate with the best-matching
    family by distinctive-token overlap between the candidate's annotations
    (InterPro family name, protein name, EC, gene) and the family's descriptors.
  * :func:`summarize_assignments` rolls the labelled pool up into a per-family
    table (level, promise, count, best rank) for the exploration summary.

Firewall note: this code names NO specific enzyme family.  Every family label,
descriptor and promise comes from the run's own LLM-derived
``mechanism_families.json`` at runtime; the matching reuses the generic
distinctive-token machinery in ``mechanism_families`` (reaction-shape stopwords,
no answer vocabulary).  It therefore works identically for any target reaction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from enzyme_discovery_agent.tools.mechanism_families import (
    _GENERIC_FAMILY_TOKENS,
    _family_search_tokens,
)

# Structural-rank words the model uses in a family NAME to say how broad the
# class is.  Generic taxonomy vocabulary (already in the shared stopword set),
# not any enzyme or answer name.  Ordered broad -> narrow; first hit wins.
_RANK_WORDS = ("clan", "fold", "superfamily", "family")

_PROMISE_RANK = {"high": 3, "medium": 2, "low": 1}


def load_mechanism_families(run_dir: Path) -> list[dict]:
    """Return the run's derived mechanism families (empty list if none)."""
    try:
        data = json.loads((Path(run_dir) / "mechanism_families.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [f for f in data if isinstance(f, dict)]
    return []


def mechanism_family_level(name: str) -> str | None:
    """Structural rank implied by a family's own name (fold/superfamily/family).

    The model already encodes this in the label it writes (e.g. "... *fold* ..."
    vs "... *superfamily* ..."); we read it back rather than guessing.  Returns
    ``None`` when the name states no rank."""
    low = str(name or "").lower()
    for word in _RANK_WORDS:
        if re.search(rf"\b{word}\b", low):
            return word
    return None


def _candidate_tokens(meta: dict) -> set[str]:
    """Lowercase annotation tokens for a candidate, from the fields most likely
    to carry a family/domain signature.  We deliberately do NOT stopword-filter
    the candidate side; the discrimination comes from the family's *distinctive*
    tokens (which are already generic-filtered)."""
    parts: list[str] = []
    for key in ("interpro_family", "protein_name", "function"):
        val = meta.get(key)
        if isinstance(val, str):
            parts.append(val)
    for key in ("ec_numbers", "gene_names", "catalytic_activities"):
        val = meta.get(key)
        if isinstance(val, (list, tuple)):
            parts.extend(str(v) for v in val)
        elif isinstance(val, str):
            parts.append(val)
    text = " ".join(parts).lower()
    return set(re.findall(r"[a-z][a-z0-9]{2,}", text))


def assign_mechanism_family(meta: dict, families: list[dict]) -> dict | None:
    """Best-matching mechanism family for one candidate, or ``None``.

    Scores each family by how many of its *distinctive* tokens (name +
    descriptors, minus generic reaction vocabulary) appear in the candidate's
    annotation tokens.  Requires at least one shared distinctive token; ties
    break toward the higher-promise family, then the earlier-derived one (stable
    and reproducible).  Returns ``{"name", "level", "promise", "score",
    "matched_tokens"}``.
    """
    if not families:
        return None
    cand = _candidate_tokens(meta)
    if not cand:
        return None

    best = None
    best_key: tuple = ()
    for idx, fam in enumerate(families):
        fam_tokens = _family_search_tokens(fam)
        matched = sorted(fam_tokens & cand)
        score = len(matched)
        if score == 0:
            continue
        promise = str(fam.get("promise", "")).lower()
        # Higher score, then higher promise, then earlier index (negated so a
        # smaller index sorts as "better" under max()).
        key = (score, _PROMISE_RANK.get(promise, 0), -idx)
        if best is None or key > best_key:
            best_key = key
            best = {
                "name": str(fam.get("name", "")),
                "level": mechanism_family_level(fam.get("name", "")),
                "promise": promise or None,
                "score": score,
                "matched_tokens": matched,
            }
    return best


def summarize_assignments(entries: list[dict], families: list[dict]) -> list[dict]:
    """Roll assigned candidates up into a per-family table for the summary.

    ``entries`` are candidate dicts each carrying a ``mechanism_family`` key
    (name string or ``None``) and, when known, a numeric ``rank``.  The rollup
    preserves the derived-family order and appends an ``Unassigned`` bucket for
    candidates no family claimed.  Each row is machine-readable and stable across
    runs with the same pool.
    """
    by_name: dict[str, dict] = {}
    for fam in families or []:
        name = str(fam.get("name", ""))
        if not name:
            continue
        by_name[name] = {
            "name": name,
            "level": mechanism_family_level(name),
            "promise": str(fam.get("promise", "")).lower() or None,
            # Pass the model's own activation description straight through; we do
            # not parse or name intermediates in code.
            "activation_strategy": fam.get("proposed_activation_strategy") or None,
            "n_candidates": 0,
            "best_rank": None,
            "accessions": [],
        }
    unassigned = {
        "name": "Unassigned", "level": None, "promise": None,
        "activation_strategy": None, "n_candidates": 0,
        "best_rank": None, "accessions": [],
    }

    for e in entries or []:
        fam_name = e.get("mechanism_family")
        bucket = by_name.get(fam_name) if fam_name else None
        if bucket is None:
            bucket = unassigned
        bucket["n_candidates"] += 1
        acc = e.get("accession")
        if acc:
            bucket["accessions"].append(acc)
        rank = e.get("rank")
        if isinstance(rank, int) and rank > 0:
            if bucket["best_rank"] is None or rank < bucket["best_rank"]:
                bucket["best_rank"] = rank

    rows = [b for b in by_name.values() if b["n_candidates"] > 0]
    if unassigned["n_candidates"] > 0:
        rows.append(unassigned)
    return rows
