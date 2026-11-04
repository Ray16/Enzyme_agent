"""Collect a saturation campaign's predicted structures into a scored table.

**This module owns bookkeeping, not the metric.**  What a variant's structures
*mean* is a question the metric answers, and the metric evolves as a campaign
runs -- a confidence number that looked discriminating at n=3 can turn out to be
protein-dominated and near-flat at n=50, and a geometry proxy can be refuted and
replaced.  So the science metric is supplied by the structural modeling
pipeline (``structural_modeling_agent.proxies``), which carries the catalogue, the
ensemble machinery and the validation that decides whether a proxy earned
belief.  Swapping or adding a metric must not require touching this file.

What IS stable, and lives here:

* discovering which models exist, and recovering ``(variant, mode, sample)``
  from the filename rather than the directory -- cofolding tools disagree on
  layout, and the same campaign often has two of them;
* refusing to double-count a model that appears under two layouts;
* long format as the only stored truth, one row per model, keyed
  ``(id, mode, sample)``. Wide and per-variant tables are derived from it;
* carrying ``n_models`` and ``complete`` everywhere, so a half-finished variant
  can never be silently averaged as though it were done.

Raw per-model confidence numbers are read through too, but as *provenance from
the folding tool*, not as the metric: they are what the predictor reported, and
they are recorded so a metric can later be checked against them as a confound.
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

#: ``confidence_<variant_id>_model_<k>.json`` -- the one naming convention that
#: survives across the layouts (stock ``boltz_results_<id>/predictions/<id>/``
#: and the accelerated kit's ``by_seed/<id>/s<seed>/``).
CONFIDENCE_RE = re.compile(r"^confidence_(?P<id>.+)_model_(?P<sample>\d+)\.json$")


@dataclass(frozen=True)
class ModelRef:
    """One predicted model on disk."""

    variant_id: str
    mode: str
    sample: int
    confidence_path: str
    structure_path: str | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.variant_id, self.mode, self.sample)


@dataclass
class Campaign:
    root: str
    models: list[ModelRef] = field(default_factory=list)
    modes: tuple[str, ...] = ()
    duplicates: int = 0
    unknown_ids: tuple[str, ...] = ()

    def groups(self) -> dict[tuple[str, str], list[ModelRef]]:
        """(variant_id, mode) -> its models, sample-ordered."""
        out: dict[tuple[str, str], list[ModelRef]] = {}
        for m in self.models:
            out.setdefault((m.variant_id, m.mode), []).append(m)
        for v in out.values():
            v.sort(key=lambda m: m.sample)
        return out


def _structure_beside(confidence_path: Path) -> str | None:
    """The model file matching a confidence JSON, if the predictor wrote one."""
    stem = confidence_path.name[len("confidence_"):-len(".json")]
    for ext in (".pdb", ".cif", ".mmcif"):
        cand = confidence_path.parent / f"{stem}{ext}"
        if cand.exists():
            return str(cand)
    return None


def discover(
    root: str | Path,
    modes: tuple[str, ...] = (),
    known_ids: set[str] | None = None,
) -> Campaign:
    """Walk a results tree and index every model found.

    Args:
        root: directory to walk. Symlinked subtrees ARE followed -- campaigns
            routinely stage results by symlink, and ``Path.rglob`` would skip
            them, silently reporting a partial campaign as complete.
        modes: condition labels. A model's mode is whichever of these appears as
            a path segment under ``root``; with no match, or no modes given, the
            mode is ``""`` (a single-condition campaign).
        known_ids: variant ids from the build step. Ids outside this set are
            reported separately rather than merged -- an unrecognised id means
            the tree holds something this campaign did not ask for.
    """
    root_p = Path(root).expanduser()
    seen: set[tuple[str, str, int]] = set()
    models: list[ModelRef] = []
    unknown: set[str] = set()
    dupes = 0

    for dirpath, _dirnames, filenames in os.walk(root_p, followlinks=True):
        rel_parts = Path(dirpath).relative_to(root_p).parts if Path(dirpath) != root_p else ()
        mode = next((m for m in modes if m in rel_parts), "")
        for fn in filenames:
            m = CONFIDENCE_RE.match(fn)
            if not m:
                continue
            vid, sample = m.group("id"), int(m.group("sample"))
            if known_ids is not None and vid not in known_ids:
                unknown.add(vid)
                continue
            key = (vid, mode, sample)
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            cpath = Path(dirpath) / fn
            models.append(ModelRef(variant_id=vid, mode=mode, sample=sample,
                                   confidence_path=str(cpath),
                                   structure_path=_structure_beside(cpath)))

    models.sort(key=lambda r: (r.variant_id, r.mode, r.sample))
    found_modes = tuple(sorted({r.mode for r in models})) if models else ()
    return Campaign(root=str(root_p), models=models,
                    modes=modes or found_modes, duplicates=dupes,
                    unknown_ids=tuple(sorted(unknown)))


# --------------------------------------------------------------------------
# Provenance from the folding tool (NOT the metric)
# --------------------------------------------------------------------------
def read_confidence(ref: ModelRef, chain_pairs: dict[str, tuple[str, str]] | None = None,
                    scalars: tuple[str, ...] = ("iptm", "ligand_iptm", "complex_plddt",
                                                "complex_iplddt", "confidence_score")) -> dict:
    """Numbers the predictor itself reported for one model.

    ``chain_pairs`` maps an output column to a pair of chain indices in
    ``pair_chains_iptm``; the value recorded is the MINIMUM of the two
    directions.  Min, not mean: the matrix is strongly asymmetric for a small
    ligand, and the protein-dominated direction is near-flat, so a mean is
    mostly the uninformative half.  The pairs are an argument because which
    chain is the substrate is campaign knowledge, not something this file can
    know -- there is no default.
    """
    try:
        data = json.loads(Path(ref.confidence_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    row: dict = {k: data.get(k) for k in scalars}
    pc = data.get("pair_chains_iptm") or {}
    for name, (a, b) in (chain_pairs or {}).items():
        vals = [pc[x][y] for x, y in ((a, b), (b, a))
                if isinstance(pc.get(x), dict) and pc[x].get(y) is not None]
        row[name] = round(min(vals), 6) if vals else None
        if len(vals) == 2:
            row[f"{name}_asym"] = round(abs(vals[0] - vals[1]), 6)
    return row


# --------------------------------------------------------------------------
# The metric: delegated to the structural modeling pipeline
# --------------------------------------------------------------------------
def score_groups(
    campaign: Campaign,
    resolver,
    proxies=None,
    min_poses: int = 1,
) -> dict[tuple[str, str], dict]:
    """Score each (variant, mode) ensemble with ``structural_modeling_agent.proxies``.

    The resolver -- which atom attacks, which is attacked, what leaves -- is
    reaction knowledge, so it is written per system by the caller and passed in,
    exactly as the proxies package requires.  Validate it with
    ``check_resolver`` before trusting anything this returns; that is the
    pipeline's rule and this function does not override it.

    Returns ``(variant_id, mode) -> {proxy_id: score}``, plus ``n_poses``.
    Groups whose structures are missing are reported with an ``error`` rather
    than dropped, so a gap in the campaign stays visible.
    """
    try:
        from structural_modeling_agent.proxies import build_ensemble, evaluate, parse_pose
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "scoring needs the structural modeling pipeline "
            "(structural_modeling_agent.proxies), which supplies the metric catalogue "
            f"and its validation: {exc}"
        ) from exc

    out: dict[tuple[str, str], dict] = {}
    for key, refs in campaign.groups().items():
        paths = [r.structure_path for r in refs if r.structure_path]
        if len(paths) < min_poses:
            out[key] = {"n_poses": len(paths),
                        "error": f"only {len(paths)} structure(s) on disk; "
                                 f"need {min_poses}"}
            continue
        poses = [parse_pose(p) for p in paths]
        sites = [resolver(p) for p in poses]
        usable = [(p, s) for p, s in zip(poses, sites) if s is not None]
        if len(usable) < min_poses:
            out[key] = {"n_poses": len(poses), "n_resolved": len(usable),
                        "error": "resolver returned no reactive site for these poses"}
            continue
        ens = build_ensemble([p for p, _ in usable], [s for _, s in usable])
        scores = evaluate(ens, proxies) if proxies is not None else evaluate(ens)
        out[key] = {"n_poses": len(poses), "n_resolved": len(usable), **scores}
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def long_rows(campaign: Campaign,
              chain_pairs: dict[str, tuple[str, str]] | None = None,
              meta: dict[str, dict] | None = None) -> list[dict]:
    """One row per model. The only stored truth; everything else derives from it."""
    rows = []
    for r in campaign.models:
        row = {"id": r.variant_id, "mode": r.mode, "sample": r.sample}
        row.update((meta or {}).get(r.variant_id, {}))
        row.update(read_confidence(r, chain_pairs))
        rows.append(row)
    return rows


def aggregate(rows: list[dict], value_key: str, expect: int | None = None,
              group_keys: tuple[str, ...] = ("id", "mode"),
              carry: tuple[str, ...] = ()) -> list[dict]:
    """Per-group summary of one metric column.

    ``value_key`` is passed in rather than fixed: the column worth aggregating
    is whichever metric the campaign currently believes, and that changes.
    Every row carries ``n_models``, and ``complete`` whenever ``expect`` is
    given, so a partial group is never mistaken for a finished one.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in group_keys), []).append(r)

    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        vals = [r[value_key] for r in rs
                if isinstance(r.get(value_key), (int, float))]
        row = dict(zip(group_keys, key))
        for c in carry:
            row[c] = rs[0].get(c, "")
        row["n_models"] = len(vals)
        row[f"{value_key}_mean"] = round(statistics.fmean(vals), 6) if vals else ""
        row[f"{value_key}_median"] = round(statistics.median(vals), 6) if vals else ""
        row[f"{value_key}_sd"] = round(statistics.stdev(vals), 6) if len(vals) > 1 else ""
        row[f"{value_key}_min"] = round(min(vals), 6) if vals else ""
        row[f"{value_key}_max"] = round(max(vals), 6) if vals else ""
        if expect is not None:
            row["complete"] = int(len(vals) >= expect)
        out.append(row)
    return out


def write_tsv(rows: list[dict], path: str | Path,
              fields: list[str] | None = None) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return p
