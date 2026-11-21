"""Ortholog grouping key — the SAME enzyme across organisms.

Shared by the pre-evaluation selection (so the evaluation budget buys DISTINCT
enzymes, not many copies of one) and the final ranking collapse (so orthologs
never dominate the table).  One source of truth keeps the two consistent.
"""
from __future__ import annotations

import re
import hashlib
from difflib import SequenceMatcher


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


def sequence_identity(seq_a: str | None, seq_b: str | None) -> float | None:
    """Estimate pairwise amino-acid sequence identity in [0, 1] without external tools.

    Returns ``None`` when either sequence is absent or non-canonical. Uses a
    length-ratio and k-mer prefilter so distant sequences fail fast, then
    computes exact ``SequenceMatcher`` matching-character fraction over the
    longer sequence.
    """
    if not isinstance(seq_a, str) or not isinstance(seq_b, str):
        return None
    a = seq_a.strip().upper()
    b = seq_b.strip().upper()
    if not a or not b or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", a) or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", b):
        return None
    if a == b:
        return 1.0
    longer = max(len(a), len(b))
    shorter = min(len(a), len(b))
    if shorter / longer < 0.5:
        return shorter / longer * 0.5
    # Exact matching residues / longer length (conservative upper-bounded identity)
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return round(matched / longer, 4)


def filter_highly_similar_orthologs(
    rows: list[dict],
    *,
    identity_threshold: float = 0.85,
    seq_getter=None,
    family_getter=None,
) -> tuple[list[dict], list[dict]]:
    """Collapse only highly similar sequences (>= identity_threshold) within each family.

    Retains all distinct families and all divergent/less-similar orthologs
    (< identity_threshold), as well as any entries whose sequences are missing.
    Returns ``(kept_rows, collapsed_records)``.
    """
    get_seq = seq_getter or (lambda r: (r.get("data") or r).get("sequence"))
    get_fam = family_getter or (lambda r: r.get("family") or (r.get("data") or r).get("interpro_family_id") or "")

    kept: list[dict] = []
    collapsed: list[dict] = []
    # Track kept sequences per family so comparisons stay local and fast
    kept_by_family: dict[str, list[tuple[str, str]]] = {}

    for row in rows:
        seq = get_seq(row)
        fam = str(get_fam(row) or "")
        acc = row.get("accession") or ""
        if not isinstance(seq, str) or not seq.strip():
            kept.append(row)
            continue
        match_acc = None
        match_id = None
        for prev_acc, prev_seq in kept_by_family.get(fam, []):
            ident = sequence_identity(seq, prev_seq)
            if ident is not None and ident >= identity_threshold:
                match_acc = prev_acc
                match_id = ident
                break
        if match_acc is not None:
            collapsed.append({
                "accession": acc,
                "protein_name": row.get("enzyme_name") or row.get("protein_name") or (row.get("data") or {}).get("protein_name"),
                "representative_accession": match_acc,
                "sequence_identity": match_id,
                "family": fam,
                "reason": f"near_identical_ortholog (>={identity_threshold:.0%} identity to {match_acc})",
            })
            continue
        kept.append(row)
        kept_by_family.setdefault(fam, []).append((acc, seq))

    return kept, collapsed
