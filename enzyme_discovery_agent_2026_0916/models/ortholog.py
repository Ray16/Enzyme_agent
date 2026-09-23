"""Ortholog grouping key — the SAME enzyme across organisms.

Shared by the pre-evaluation selection (so the evaluation budget buys DISTINCT
enzymes, not many copies of one) and the final ranking collapse (so orthologs
never dominate the table).  One source of truth keeps the two consistent.
"""
from __future__ import annotations

import re


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
    """Group true orthologs — the SAME enzyme in different organisms — together.

    Uses the normalised gene symbol plus the primary EC sub-subclass.  Copy
    suffixes that only distinguish duplicates of one enzyme (``geneX-1``,
    ``geneX_2``, ``geneX II``) are stripped so they collapse, but a digit fused
    directly to the stem is kept, because it names a DISTINCT gene (``geneA2`` is
    a different enzyme from ``geneA`` and must not merge).  Genuinely different
    enzymes that merely share a gene letter or an EC class still stay apart
    (different gene stem or different EC sub-subclass → different key).  Only
    i/v/x are treated as roman suffixes so a gene stem legitimately ending in a
    consonant such as 'c'/'d'/'m' is left intact.
    """
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
