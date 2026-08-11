"""Native-substrate-fit visualization for the surveyed family landscape.

Turns the grounded family survey into a picture an experimentalist can read at a
glance: for the top-fitting fold families, how closely their reviewed members'
NATIVE substrates resemble the target across every physicochemical axis (a radar),
and what native SIZE RANGE each family's pocket is built for (a bar, with the
target marked).  Both are even-handed — every axis is shown, none is privileged —
and answer-agnostic: everything is measured from the run's OWN discovered folds
against its OWN target.

Best-effort: any failure (no landscape, matplotlib absent, degenerate data) is
logged and skipped; it never breaks finalization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("enzyme_agent")

# Similarity axes shown on the radar, in display order, with short labels.  Keys
# match tools.family_reactions._HOLISTIC_SIM_AXES / the survey's axis_medians.
_AXES: list[tuple[str, str]] = [
    ("functional_group_similarity", "Functional\ngroups"),
    ("mw_similarity", "Molecular\nweight"),
    ("chain_length_similarity", "Main-chain\nlength"),
    ("volume_similarity", "Volume"),
    ("tpsa_similarity", "Polarity\n(TPSA)"),
    ("logp_similarity", "Hydrophob.\n(logP)"),
]


def _target_chain_length(run_dir: Path) -> int | None:
    """Main-chain length of the target substrate(s), for the size-range bar's
    reference marker.  None when it cannot be computed."""
    try:
        ctx = json.loads((run_dir / "target_context.json").read_text())
        smi = ctx.get("substrate_smiles")
        if not smi:
            return None
        from enzyme_discovery_agent.tools.substrate import compute_substrate_properties
        # A dotted SMILES is several substrates; take the longest main chain.
        return max(
            compute_substrate_properties(s).longest_aliphatic_chain
            for s in str(smi).split(".") if s
        )
    except Exception:  # noqa: BLE001
        return None


def _fold_short(fold_id: str, source: str | None) -> str:
    src = (source or "").split(":")[0]
    return f"{fold_id}" + (f"\n({src})" if src and src != "discovered" else "")


def render_family_fit_plot(
    run_dir: str | Path, top_n: int = 5, log_fn=None,
) -> Path | None:
    """Render the native-substrate-fit radar + size-range bar for the top-fitting
    families to ``family_substrate_fit.png``.  Returns the path, or None if it
    could not be produced (no landscape, no plottable folds, matplotlib absent)."""
    run_dir = Path(run_dir)
    landscape_path = run_dir / "family_landscape.json"
    if not landscape_path.exists():
        return None
    try:
        folds = json.loads(landscape_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(folds, list):
        return None

    # Keep only folds that carry the multi-axis fit vector, ranked by their
    # even-weight holistic fit (max), best first.
    plottable = [
        f for f in folds
        if isinstance(f, dict) and (f.get("axis_medians") or {})
    ]

    def _holistic_max(f: dict) -> float:
        return ((f.get("holistic_fit") or {}).get("max")) or -1.0

    plottable.sort(key=_holistic_max, reverse=True)
    plottable = plottable[:top_n]
    if not plottable:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001 - viz is optional
        logger.info("Family-fit plot skipped (matplotlib unavailable: %s)", exc)
        return None

    try:
        labels = [lbl for _, lbl in _AXES]
        n_ax = len(_AXES)
        angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        fig = plt.figure(figsize=(13, 5.6))
        ax_r = fig.add_subplot(1, 2, 1, polar=True)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(1, len(plottable) - 1)) for i in range(len(plottable))]

        for f, color in zip(plottable, colors):
            med = f.get("axis_medians") or {}
            vals = [med.get(key) if isinstance(med.get(key), (int, float)) else 0.0
                    for key, _ in _AXES]
            vals += vals[:1]
            label = f.get("fold_id", "?")
            if f.get("members_spanning_aromatic_and_aliphatic"):
                label += " ✦"  # broad pocket: spans aromatic + aliphatic natives
            ax_r.plot(angles, vals, color=color, linewidth=1.8, label=label)
            ax_r.fill(angles, vals, color=color, alpha=0.12)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(labels, fontsize=8)
        ax_r.set_ylim(0, 1)
        ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_r.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax_r.set_title("Native-substrate fit vs target\n(median per axis, 1 = identical)",
                       fontsize=10, pad=18)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=7,
                    title="Fold family", title_fontsize=8)
        ax_r.annotate("✦ pocket spans aromatic + aliphatic natives",
                      xy=(0.5, -0.12), xycoords="axes fraction",
                      ha="center", fontsize=7, style="italic")

        # Right panel: native main-chain length RANGE per family (pocket size has
        # direction — favours larger vs smaller), with the target marked.
        ax_b = fig.add_subplot(1, 2, 2)
        ys = list(range(len(plottable)))[::-1]
        for f, y, color in zip(plottable, ys, colors):
            rng = f.get("native_chain_length_range")
            if not rng:
                continue
            lo, hi = rng[0], rng[1]
            ax_b.plot([lo, hi], [y, y], color=color, linewidth=6, solid_capstyle="round",
                      alpha=0.7)
            ax_b.plot([lo, hi], [y, y], "|", color=color, markersize=10)
        ax_b.set_yticks(ys)
        ax_b.set_yticklabels([f.get("fold_id", "?") for f in plottable], fontsize=8)
        tcl = _target_chain_length(run_dir)
        if tcl is not None:
            ax_b.axvline(tcl, color="crimson", linestyle="--", linewidth=1.4,
                         label=f"target main-chain ≈ {tcl}")
            ax_b.legend(loc="lower right", fontsize=8)
        ax_b.set_xlabel("Native substrate main-chain length (carbons)", fontsize=9)
        ax_b.set_title("Pocket size range each family is built for", fontsize=10, pad=12)
        ax_b.margins(y=0.15)
        ax_b.grid(axis="x", alpha=0.3)

        fig.suptitle("Family native-substrate fit — top candidates for the target",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        out = run_dir / "family_substrate_fit.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        if log_fn:
            log_fn(f"  Family-fit plot: rendered {len(plottable)} top families -> {out.name}")
        return out
    except Exception as exc:  # noqa: BLE001 - never break finalization on a plot
        logger.warning("Family-fit plot failed: %s", exc)
        return None
