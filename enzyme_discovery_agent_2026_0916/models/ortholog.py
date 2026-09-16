"""Ortholog grouping key — the SAME enzyme across organisms.

Shared by the pre-evaluation selection (so the evaluation budget buys DISTINCT
enzymes, not many copies of one) and the final ranking collapse (so orthologs
never dominate the table).  One source of truth keeps the two consistent.
"""
from __future__ import annotations

import re


def ortholog_key(data: dict) -> str | None:
    """Group true orthologs — the SAME enzyme in different organisms — together.

    Uses the normalised gene symbol plus the primary EC number.  Suffix noise that
    only distinguishes copies of one enzyme (e.g. ``geneX``, ``geneX-1``,
    ``geneX-II``, ``geneX_2``) is stripped so they collapse, while genuinely
    different enzymes that merely share a gene letter or an EC do NOT (different
    gene stem or different EC → different key).  Only i/v/x are treated as roman
    suffixes so a gene legitimately ending in 'c'/'d'/'m' (murC, iucC) is intact.
    """
    genes = data.get("gene_names") or []
    gene = (genes[0] if genes else "").lower().strip()
    gene = re.sub(r"[\s_\-]*(?:[ivx]+|\d+)$", "", gene)  # drop paralog/copy suffix
    ec = (data.get("ec_numbers") or [""])[0]
    if gene and ec:
        return f"{gene}|{ec}"
    if gene:
        return f"gene:{gene}"
    name = re.sub(r"\b\d+\b", "", (data.get("protein_name") or "").lower()).strip()
    name = re.sub(r"\s+", " ", name)
    return f"name:{name}" if name else None
