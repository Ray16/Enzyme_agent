"""Publication-quality ESM-embedding map of a discovery run's explored candidates.

Embeds every explored candidate of a run with an ESM2 model (on a reserved GPU,
via the fair-esm worker), projects them to 2-D with UMAP, colors by InterPro
family, and highlights the top-ranked candidates with numbered markers keyed to a
clean side table -- so nothing overlaps.

Answer-agnostic: it plots the run's OWN pool and its OWN ranking. Any reference
accessions to star are passed on the command line only (never hardcoded here --
this file is scanned by the leakage firewall).

Usage:
    python figures/plot_esm_embedding_map.py <run_dir> \
        [--title "Amide synthetases (nylon-6,6)"] [--top 10] \
        [--model facebook/esm2_t33_650M_UR50D] [--out figures/xxx.png] \
        [--refs A,B] [--no-embed]

Runs under the palm python (umap/matplotlib/sklearn); the embedding step shells
out to the fair-esm GPU env via gpu_reserve (config.CANDIDATE_MAP_EMBED_PYTHON).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.models import embedding_store as es
from enzyme_discovery_agent.models.candidate_registry import (
    family_display, load_candidate_registry,
)
from enzyme_discovery_agent.tools.embedding_map import _run_embed_subprocess


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #
def batch_fetch_sequences(accessions: list[str], chunk: int = 200) -> dict[str, str]:
    """Fetch sequences for many accessions with few requests (UniProt FASTA stream)."""
    out: dict[str, str] = {}
    for i in range(0, len(accessions), chunk):
        block = accessions[i:i + chunk]
        q = " OR ".join(f"accession:{a}" for a in block)
        url = ("https://rest.uniprot.org/uniprotkb/stream?format=fasta&query="
               + urllib.parse.quote(q))
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                fasta = resp.read().decode()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"  batch fetch failed ({exc}); continuing\n")
            continue
        acc, seq = None, []
        for line in fasta.splitlines():
            if line.startswith(">"):
                if acc and seq:
                    out[acc] = "".join(seq)
                # header: sp|ACC|NAME ... or tr|ACC|NAME
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
        if acc and seq:
            out[acc] = "".join(seq)
    return out


def embed_all_candidates(run_dir: Path, model: str) -> str:
    """Ensure every registry candidate with a sequence is in the embedding store."""
    registry = load_candidate_registry(run_dir).get("candidates", {})
    pending = es.pending_accessions(run_dir, list(registry.keys()))
    if not pending:
        return "all candidates already embedded"
    cache = es.load_sequences(run_dir)
    need = [a for a in pending if a not in cache]
    if need:
        print(f"  fetching {len(need)} sequences from UniProt (batched)...")
        fetched = batch_fetch_sequences(need)
        es.cache_sequences(run_dir, fetched)
        cache.update(fetched)
    items = [
        {"accession": a, "seq_sha1": es.seq_sha1(cache[a]), "sequence": cache[a]}
        for a in pending if cache.get(a)
    ]
    if not items:
        return "no sequences resolved"
    print(f"  embedding {len(items)} sequences on GPU ({model})...")
    rows, note = _run_embed_subprocess(run_dir, items, model)
    if not rows:
        return f"embedding skipped: {note}"
    added = es.append_vectors(run_dir, model, rows)
    return f"embedded {added} new candidate(s) ({note})"


# --------------------------------------------------------------------------- #
# Ranking / promise
# --------------------------------------------------------------------------- #
def top_candidates(run_dir: Path, registry: dict, n: int) -> list[str]:
    """Top-ranked accessions: pass1_ranking.json order if present, else the
    evaluated set ordered by substrate_fit."""
    p1 = run_dir / "pass1_ranking.json"
    if p1.exists():
        try:
            ranking = json.loads(p1.read_text())
            accs = [e.get("accession") for e in ranking if e.get("accession")]
            if accs:
                return accs[:n]
        except (OSError, json.JSONDecodeError):
            pass
    scored = []
    for acc, info in registry.items():
        statuses = [s.get("status") for s in info.get("statuses", [])]
        if "evaluated" not in statuses or (statuses and statuses[-1] == "rejected"):
            continue
        try:
            fit = float(info.get("metadata", {}).get("substrate_fit") or 0.0)
        except (TypeError, ValueError):
            fit = 0.0
        scored.append((acc, fit))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [a for a, _ in scored[:n]]


def _short_name(meta: dict, acc: str, maxlen: int = 26) -> str:
    name = (meta.get("protein_name") or "").strip()
    if not name and meta.get("gene_names"):
        name = str(meta["gene_names"][0])
    name = name or acc
    return name if len(name) <= maxlen else name[: maxlen - 1] + "…"


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def render(run_dir: Path, out_path: Path, title: str, n_top: int,
           refs: set[str]) -> Path | None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    store = es.load_store(run_dir)
    if store.matrix is None or len(store) < 3:
        print("  not enough embedded candidates to plot")
        return None
    registry = load_candidate_registry(run_dir).get("candidates", {})

    # UMAP projection.
    import umap
    n_neighbors = max(2, min(15, len(store) - 1))
    xy = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.15, metric="cosine",
                   random_state=42).fit_transform(store.matrix)
    xy = np.asarray(xy)

    # Family per point; color the most common families, rest -> "Other".
    fams = [family_display(registry.get(a, {}).get("metadata", {}), a) for a in store.accessions]
    from collections import Counter
    counts = Counter(fams)
    top_fams = [f for f, _ in counts.most_common(12)]
    palette = plt.get_cmap("tab20")
    fam_color = {f: palette(i % 20) for i, f in enumerate(top_fams)}
    OTHER = "#c9ccd1"

    def color_of(f):
        return fam_color.get(f, OTHER)

    tops = [a for a in top_candidates(run_dir, registry, n_top) if a in store.acc_to_row]
    top_set = set(tops)

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1.15], height_ratios=[6.0, 1.35],
                          hspace=0.16, wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1]); ax_leg.axis("off")
    ax_tab = fig.add_subplot(gs[1, :]); ax_tab.axis("off")

    # Background points (non-top, non-ref).
    bx, by, bc = [], [], []
    for i, acc in enumerate(store.accessions):
        if acc in top_set or acc in refs:
            continue
        bx.append(xy[i, 0]); by.append(xy[i, 1]); bc.append(color_of(fams[i]))
    ax.scatter(bx, by, s=20, c=bc, alpha=0.55, linewidths=0, zorder=1)

    # Reference enzymes (starred), if supplied out-of-band.
    for i, acc in enumerate(store.accessions):
        if acc in refs:
            ax.scatter(xy[i, 0], xy[i, 1], s=300, marker="*", facecolor="white",
                       edgecolors="black", linewidths=1.4, zorder=4)

    # Top candidates: bold family-colored markers with a numbered halo label.
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    for rank, acc in enumerate(tops, start=1):
        i = store.acc_to_row[acc]
        ax.scatter(xy[i, 0], xy[i, 1], s=150, c=[color_of(fams[i])],
                   edgecolors="black", linewidths=1.5, zorder=5)
        ax.text(xy[i, 0], xy[i, 1], str(rank), ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="black",
                path_effects=halo, zorder=6)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=11); ax.set_ylabel("UMAP 2", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12, loc="left")
    ax.text(0.0, 1.015, f"{len(store)} explored candidates - ESM2 embedding",
            transform=ax.transAxes, fontsize=10, color="#555555")

    # Family legend (right column).
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          markerfacecolor=fam_color[f], markeredgecolor="none",
                          label=(f if len(f) <= 30 else f[:29] + "…"))
               for f in top_fams]
    if len(top_fams) < len(counts):
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                                  markerfacecolor=OTHER, markeredgecolor="none",
                                  label=f"Other ({len(counts) - len(top_fams)} families)"))
    leg = ax_leg.legend(handles=handles, title="InterPro family", loc="upper left",
                        bbox_to_anchor=(-0.05, 1.0), fontsize=8.2, title_fontsize=9.5,
                        frameon=False, handletextpad=0.5, labelspacing=0.5)
    leg._legend_box.align = "left"
    marker_note = "Numbered = top-ranked candidate"
    if refs:
        marker_note += "   ★ = reference"
    ax_leg.text(-0.05, 0.02, marker_note, transform=ax_leg.transAxes,
                fontsize=8.2, color="#333333", va="bottom")

    # Top-candidate table (full width, two balanced columns), no overlap.
    ax_tab.text(0.0, 1.0, "Top-ranked candidates", fontsize=10.5,
                fontweight="bold", va="top", transform=ax_tab.transAxes)
    per_col = (len(tops) + 1) // 2 if tops else 1
    for rank, acc in enumerate(tops, start=1):
        meta = registry.get(acc, {}).get("metadata", {})
        entry = f"{rank:>2}.  {_short_name(meta, acc)}  ({acc})"
        col = 0 if rank <= per_col else 1
        row = (rank - 1) if col == 0 else (rank - 1 - per_col)
        x0 = 0.0 if col == 0 else 0.52
        ax_tab.text(x0, 0.80 - row * 0.155, entry, fontsize=8.6, va="top",
                    family="monospace", transform=ax_tab.transAxes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path} (+ .pdf)")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--title", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--model", default=config.CANDIDATE_MAP_ESM_MODEL)
    ap.add_argument("--out", default="")
    ap.add_argument("--refs", default="")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.stderr.write(f"run dir not found: {run_dir}\n")
        return 2
    title = args.title or run_dir.name
    out = Path(args.out) if args.out else (Path(__file__).parent / f"esm_map_{run_dir.name[:40]}.png")
    refs = {a.strip() for a in args.refs.replace(",", " ").split() if a.strip()}

    if not args.no_embed:
        print(embed_all_candidates(run_dir, args.model))
    result = render(run_dir, out, title, args.top, refs)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
