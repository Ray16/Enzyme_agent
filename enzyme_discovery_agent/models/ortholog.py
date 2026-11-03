"""Ortholog grouping key — the SAME enzyme across organisms.

Shared by the pre-evaluation selection (so the evaluation budget buys DISTINCT
enzymes, not many copies of one) and the final ranking collapse (so orthologs
never dominate the table).  One source of truth keeps the two consistent.
"""
from __future__ import annotations

import re
import hashlib


def _ec_subclass(ec: str) -> str:
    """Normalise an EC number to its sub-subclass (first three fields).

    A candidate is often annotated with only a partial EC (e.g. ``A.B.C.-``)
    while its ortholog carries the fully resolved number (``A.B.C.n``).  Keying
    on the raw string splits the same enzyme into two groups, so we compare at
    the sub-subclass level, which is stable across that annotation gap while
    still separating genuinely different chemistries (``A.B.C`` vs ``A.B.D``).
    """
    parts = [p for p in (ec or "").strip().split(".") if p and p != "-"]
    return ".".join(parts[:3])


def ortholog_key(data: dict) -> str | None:
    """Return an exact-sequence group only when sequence evidence is present.

    Shared labels and coarse classifications are tentative associations and do
    not establish interchangeable sequences or justify suppressing candidates.
    """
    sequence = data.get("sequence")
    if isinstance(sequence, str) and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence.upper()):
        return "sequence-sha256:" + hashlib.sha256(sequence.upper().encode()).hexdigest()
    return None


def tentative_label_key(data: dict) -> str | None:
    """Provide a name-based hint for exploration, never identity-based pruning."""
    genes = data.get("gene_names") or []
    gene = (genes[0] if genes else "").lower().strip()
    # Strip only a *separated* copy/paralog suffix (geneX-1, geneX_2, geneX II),
    # never a digit fused to the name itself (geneA2 stays distinct from geneA).
    gene = re.sub(r"[\s_\-]+(?:[ivx]+|\d+)$", "", gene)
    ec = _ec_subclass((data.get("ec_numbers") or [""])[0])
    if gene and ec:
        return f"{gene}|{ec}"
    if gene:
        return f"gene:{gene}"
    name = re.sub(r"\b\d+\b", "", (data.get("protein_name") or "").lower()).strip()
    name = re.sub(r"\s+", " ", name)
    return f"name:{name}" if name else None
