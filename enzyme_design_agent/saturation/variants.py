"""Build a saturation-mutagenesis variant set from a pocket.

Saturation means *every* non-wild-type amino acid at *every* selected position,
including positions a mechanism argument says are essential.  Those are kept
rather than masked out, because a scan whose geometric score has no measured
negatives needs internal negative controls, and a catalytic knockout folded at
the same sample count as everything else is the cleanest one available.  Masking
them would mean the data never exists.

Residues one might want excluded are therefore an ANNOTATION -- see
``annotate``/``Variant.annotation`` -- that downstream analysis can filter on,
not an exclusion applied here.

Nothing in this module knows a ligand or an enzyme.  ``modes`` is an arbitrary
caller-supplied tuple of condition labels (ligand compositions, buffer states,
whatever the campaign varies); dedup is tracked per ``(variant, mode)`` so a
variant already measured under one condition is not refolded under it while
still being built for the others.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

AA20 = "ACDEFGHIKLMNPQRSTVWY"

WT_MUTATION = "WT"


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    kind: str                            # "wt" | "single"
    mutation: str                        # "S190T", or "WT"
    resnum: int | None
    wt_aa: str
    mut_aa: str
    seq: str
    annotation: str = ""
    #: mode -> whether this (variant, mode) still needs to be run
    needs: dict[str, bool] = field(default_factory=dict)

    @property
    def needed_modes(self) -> tuple[str, ...]:
        return tuple(m for m, need in self.needs.items() if need)


def mutate(seq: str, resnum: int, wt_aa: str, mut_aa: str) -> str:
    """Apply one substitution by 1-based residue number, verifying the wild type.

    The wild-type check is the whole point: an off-by-one or a renumbered
    structure produces a perfectly valid-looking variant set that measures the
    wrong positions, and nothing downstream can detect it.
    """
    i = resnum - 1
    if not 0 <= i < len(seq):
        raise ValueError(f"residue {resnum} is outside a sequence of length {len(seq)}")
    if seq[i] != wt_aa:
        raise ValueError(
            f"residue {resnum} is {seq[i]!r} in the sequence but the position table says "
            f"{wt_aa!r} -- the structure and the sequence disagree on numbering"
        )
    if mut_aa == wt_aa:
        raise ValueError(f"{wt_aa}{resnum}{mut_aa} is not a substitution")
    return seq[:i] + mut_aa + seq[i + 1:]


def build_variants(
    seq: str,
    positions,
    label: str = "",
    alphabet: str = AA20,
    modes: tuple[str, ...] = (),
    existing: set[tuple[str, str]] | None = None,
    dedup: bool = False,
    annotate=None,
    include_wt: bool = True,
) -> list[Variant]:
    """Every single substitution at every position.

    Args:
        seq: wild-type sequence, 1-based (``seq[n-1]`` is residue ``n``).
        positions: iterable of pocket ``Position`` objects, or of
            ``(resnum, wt_aa)`` pairs.
        label: prefix for the stable variant id, e.g. an enzyme or arm name.
            Ids are ``<label>_<WT><resnum><MUT>``, or ``<WT><resnum><MUT>`` bare.
        alphabet: substitutions to try. Default is all 20; the wild type at each
            position is skipped, giving 19 per position.
        modes: condition labels every variant is built for.
        existing: ``(variant_id, mode)`` pairs already measured.
        dedup: when True, mark those pairs as not needed. When False (the
            default) everything is rebuilt, so one campaign produces one
            internally comparable dataset -- mixing runs is unsafe when the
            effect sizes are near the run-to-run spread.
        annotate: optional ``callable(resnum, wt_aa) -> str`` recording why a
            position is special. Annotation only; it never excludes.
        include_wt: emit the unmutated reference as its own variant.
    """
    existing = existing or set()
    pairs: list[tuple[int, str]] = []
    for p in positions:
        if hasattr(p, "resnum"):
            pairs.append((p.resnum, p.wt_aa))
        else:
            rn, wt = p
            pairs.append((int(rn), str(wt)))

    def _needs(vid: str) -> dict[str, bool]:
        return {m: not (dedup and (vid, m) in existing) for m in modes}

    out: list[Variant] = []
    if include_wt:
        wid = f"{label}_{WT_MUTATION}" if label else WT_MUTATION
        out.append(Variant(id=wid, label=label, kind="wt", mutation=WT_MUTATION,
                           resnum=None, wt_aa="", mut_aa="", seq=seq,
                           annotation="", needs=_needs(wid)))

    for resnum, wt_aa in pairs:
        ann = annotate(resnum, wt_aa) if annotate else ""
        for mut_aa in alphabet:
            if mut_aa == wt_aa:
                continue
            mutation = f"{wt_aa}{resnum}{mut_aa}"
            vid = f"{label}_{mutation}" if label else mutation
            out.append(Variant(id=vid, label=label, kind="single", mutation=mutation,
                               resnum=resnum, wt_aa=wt_aa, mut_aa=mut_aa,
                               seq=mutate(seq, resnum, wt_aa, mut_aa),
                               annotation=ann, needs=_needs(vid)))

    _check_unique(out)
    return out


def _check_unique(variants: list[Variant]) -> None:
    """Ids and sequences must both be unique.

    A duplicate sequence means two different ids fold to the same thing, which
    would quietly average two labels into one measurement.
    """
    ids = [v.id for v in variants]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate variant ids: {dupes[:5]}")
    seqs: dict[str, str] = {}
    for v in variants:
        if v.seq in seqs:
            raise ValueError(
                f"variants {seqs[v.seq]} and {v.id} have identical sequences"
            )
        seqs[v.seq] = v.id


VARIANT_FIELDS = (
    "id", "label", "kind", "mutation", "resnum", "wt_aa", "mut_aa",
    "annotation", "n_mut", "seq",
)


def write_variants_tsv(variants: list[Variant], path: str | Path,
                       modes: tuple[str, ...] = ()) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(VARIANT_FIELDS) + [f"need_{m}" for m in modes]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for v in variants:
            row = {
                "id": v.id, "label": v.label, "kind": v.kind, "mutation": v.mutation,
                "resnum": v.resnum if v.resnum is not None else "",
                "wt_aa": v.wt_aa, "mut_aa": v.mut_aa, "annotation": v.annotation,
                "n_mut": 0 if v.kind == "wt" else 1, "seq": v.seq,
            }
            for m in modes:
                row[f"need_{m}"] = int(bool(v.needs.get(m, True)))
            w.writerow(row)
    return p


def write_variants_fasta(variants: list[Variant], path: str | Path,
                         needed_only: bool = True) -> Path:
    """FASTA of the sequences to fold. One record per variant, never per mode."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for v in variants:
        if needed_only and v.needs and not v.needed_modes:
            continue
        lines.append(f">{v.id}")
        lines.extend(v.seq[i:i + 60] for i in range(0, len(v.seq), 60))
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    return p


def job_counts(variants: list[Variant], modes: tuple[str, ...],
               n_samples: int = 1) -> dict:
    """What this variant set costs to run, per mode and in total."""
    per_mode = {m: sum(1 for v in variants if v.needs.get(m, True)) for m in modes}
    jobs = sum(per_mode.values())
    return {
        "n_variants": len(variants),
        "n_positions": len({v.resnum for v in variants if v.resnum is not None}),
        "jobs_per_mode": per_mode,
        "n_jobs": jobs,
        "n_samples": n_samples,
        "n_models": jobs * n_samples,
    }
