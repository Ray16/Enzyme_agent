"""Visualize the candidate ESM-embedding map.

Projects the run's persisted protein embeddings to 2-D (UMAP primary + PCA
cross-check), colored by InterPro family, with the promising candidates and their
ESM neighbors highlighted and (optionally) reference enzymes starred.

Answer-agnostic by construction: it plots the run's OWN discovered pool and its
OWN scores. Reference-enzyme accessions to star are supplied out-of-band (CLI arg,
``ENZYME_AGENT_MAP_REFERENCE_ACCS`` env, or ``run_dir/reference_enzymes.json``) --
never hardcoded, so no benchmark answer lives in package source (CLAUDE.md).

Best-effort: any failure (no store, umap/matplotlib absent, degenerate data) is
logged and returns None; it never breaks finalization. Also runnable standalone on
a finished run directory::

    python -m enzyme_discovery_agent.output.candidate_map_plot <run_dir> [--refs A,B,C]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")


def _resolve_reference_accessions(run_dir: Path, reference_accessions) -> set[str]:
    """CLI/arg > env > optional run_dir/reference_enzymes.json > empty."""
    if reference_accessions:
        return {a.strip() for a in reference_accessions if a and a.strip()}
    from enzyme_discovery_agent import config
    env = (config.CANDIDATE_MAP_REFERENCE_ACCESSIONS or "").strip()
    if env:
        return {a.strip() for a in env.replace(",", " ").split() if a.strip()}
    path = run_dir / "reference_enzymes.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            accs = data if isinstance(data, list) else data.get("accessions", [])
            return {str(a).strip() for a in accs if str(a).strip()}
        except (OSError, json.JSONDecodeError):
            pass
    return set()


def _project(matrix):
    """Return (umap_xy, pca_xy); either may be None if its library is missing."""
    import numpy as np

    umap_xy = None
    pca_xy = None
    n = matrix.shape[0]
    try:
        import umap  # noqa: F401
        n_neighbors = max(2, min(15, n - 1))
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine",
                            random_state=42)
        umap_xy = reducer.fit_transform(matrix)
    except Exception as exc:  # noqa: BLE001
        logger.info("Candidate-map UMAP unavailable (%s)", exc)
    try:
        from sklearn.decomposition import PCA
        pca_xy = PCA(n_components=2, random_state=42).fit_transform(matrix)
    except Exception as exc:  # noqa: BLE001
        logger.info("Candidate-map PCA unavailable (%s)", exc)
    return umap_xy, pca_xy


def render_candidate_map(run_dir, log_fn=None, reference_accessions=None):
    """Render ``candidate_map.png`` from the persisted embedding store. Returns the
    path, or None if it could not be produced."""
    run_dir = Path(run_dir)
    try:
        from enzyme_discovery_agent.models.embedding_store import load_store
        from enzyme_discovery_agent.models.candidate_registry import (
            family_display, load_candidate_registry,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("Candidate-map plot skipped (imports: %s)", exc)
        return None

    store = load_store(run_dir)
    if store.matrix is None or len(store) < 3:
        return None

    registry = load_candidate_registry(run_dir).get("candidates", {})

    # Sets to highlight, from the persisted map artifact (if present).
    promising: set[str] = set()
    neighbors: set[str] = set()
    try:
        cmap_data = json.loads((run_file(run_dir, 'candidate_map.json')).read_text())
        for p in cmap_data.get("promising", []):
            promising.add(p.get("accession"))
            for nbr in p.get("neighbors", []):
                neighbors.add(nbr.get("accession"))
    except (OSError, json.JSONDecodeError):
        pass
    references = _resolve_reference_accessions(run_dir, reference_accessions)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        logger.info("Candidate-map plot skipped (matplotlib: %s)", exc)
        return None

    umap_xy, pca_xy = _project(store.matrix)
    panels = [(xy, title) for xy, title in
              ((umap_xy, "UMAP (cosine)"), (pca_xy, "PCA cross-check")) if xy is not None]
    if not panels:
        return None

    # Color by InterPro family (categorical).
    fams = [family_display(registry.get(acc, {}).get("metadata", {}), acc)
            for acc in store.accessions]
    uniq = sorted(set(fams))
    palette = plt.get_cmap("tab20")
    fam_color = {f: palette(i % 20) for i, f in enumerate(uniq)}

    try:
        fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 7.0))
        if len(panels) == 1:
            axes = [axes]
        for ax, (xy, title) in zip(axes, panels):
            xy = np.asarray(xy)
            for f in uniq:
                idx = [i for i, ff in enumerate(fams) if ff == f]
                ax.scatter(xy[idx, 0], xy[idx, 1], s=26, color=fam_color[f],
                           alpha=0.65, edgecolors="none",
                           label=(f[:36] if len(f) > 36 else f))
            # Highlight neighbors (ring), promising (bold ring), references (star).
            for i, acc in enumerate(store.accessions):
                if acc in references:
                    ax.scatter(xy[i, 0], xy[i, 1], s=240, marker="*",
                               color="black", zorder=5)
                elif acc in promising:
                    ax.scatter(xy[i, 0], xy[i, 1], s=120, facecolors="none",
                               edgecolors="crimson", linewidths=2.0, zorder=4)
                elif acc in neighbors:
                    ax.scatter(xy[i, 0], xy[i, 1], s=90, facecolors="none",
                               edgecolors="#1f77b4", linewidths=1.4, zorder=3)
            ax.set_title(title, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        # One shared legend for families; annotate the highlight semantics.
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = 2 if len(handles) > 8 else 1
        note = "crimson ring = promising · blue ring = ESM neighbor"
        if references:
            note += " · ★ = reference"
        fig.legend(handles, labels, title=f"InterPro family    ({note})",
                   loc="lower center", ncol=min(4, max(1, len(uniq))),
                   fontsize=8, title_fontsize=9, frameon=True)
        fig.suptitle("Candidate map in ESM embedding space", fontsize=15, y=0.99)
        fig.tight_layout(rect=[0, 0.14, 1, 0.96])
        out = run_file(run_dir, "candidate_map.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if log_fn:
            log_fn(f"  Candidate map: projected {len(store)} embedded candidate(s) -> {out.name}")
        return out
    except Exception as exc:  # noqa: BLE001 - never break finalization on a plot
        logger.warning("Candidate-map plot failed: %s", exc)
        return None


def _parse_refs(argv: list[str]) -> list[str]:
    for i, a in enumerate(argv):
        if a == "--refs" and i + 1 < len(argv):
            return [x for x in argv[i + 1].replace(",", " ").split() if x]
    return []


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: python -m enzyme_discovery_agent.output.candidate_map_plot "
            "<run_dir> [--refs A,B,C]\n"
        )
        raise SystemExit(2)
    logging.basicConfig(level=logging.INFO)
    result = render_candidate_map(
        sys.argv[1], log_fn=print, reference_accessions=_parse_refs(sys.argv[2:])
    )
    print(f"wrote {result}" if result else "no candidate map produced")
