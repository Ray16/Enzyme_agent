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
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")

# Similarity axes shown on the radar, in display order, with short labels.  Keys
# match tools.family_reactions._HOLISTIC_SIM_AXES / the survey's axis_medians.
_AXES: list[tuple[str, str]] = [
    ("functional_group_similarity", "Functional\ngroups"),
    ("mw_similarity", "Molecular\nweight"),
    ("chain_length_similarity", "Main-chain\nlength"),
    ("volume_similarity", "Volume"),
    ("tpsa_similarity", "Polarity\n(TPSA)"),
    ("logP_similarity", "Hydrophob.\n(logP)"),
]


def _target_chain_length(run_dir: Path) -> int | None:
    """Main-chain length of the target substrate(s), for the size-range bar's
    reference marker.  None when it cannot be computed."""
    try:
        ctx = json.loads((run_file(run_dir, 'target_context.json')).read_text())
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


def _fold_label(f: dict, max_len: int = 34) -> str:
    """Human-readable family label — the InterPro NAME, never the bare id.  Uses
    the landscape's stored name, else resolves the id, else falls back to the id."""
    name = f.get("fold_name")
    if not name:
        try:
            from enzyme_discovery_agent.tools.family_reactions import interpro_name
            name = interpro_name(f.get("fold_id", ""))
        except Exception:  # noqa: BLE001
            name = None
    name = (name or f.get("fold_id") or "?").strip()
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def render_family_fit_plot(
    run_dir: str | Path, top_n: int = 6, log_fn=None,
) -> Path | None:
    """Render the native-substrate-fit radar + size-range bar for the top-fitting
    families to ``family_substrate_fit.png``.  Returns the path, or None if it
    could not be produced (no landscape, no plottable folds, matplotlib absent)."""
    run_dir = Path(run_dir)
    landscape_path = run_file(run_dir, 'family_landscape.json')
    if not landscape_path.exists():
        return None
    try:
        folds = json.loads(landscape_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(folds, list):
        return None

    # Keep only folds carrying the multi-axis fit vector.  Collapse near-duplicate
    # InterPro entries of the SAME fold (a domain and its "N-terminal" /
    # "conserved site" sub-entries share a base name before the first comma) into
    # ONE representative — otherwise two entries of one family crowd out other
    # distinct families — keeping the best-fitting representative.  Then rank by
    # even-weight holistic fit (max), best first.
    def _holistic_max(f: dict) -> float:
        return ((f.get("holistic_fit") or {}).get("max")) or -1.0

    def _base_key(f: dict) -> str:
        return _fold_label(f, max_len=10_000).split(",")[0].strip().lower()

    best_by_base: dict[str, dict] = {}
    n_total = 0
    for f in folds:
        if not (isinstance(f, dict) and (f.get("axis_medians") or {})):
            continue
        n_total += 1
        k = _base_key(f)
        if k not in best_by_base or _holistic_max(f) > _holistic_max(best_by_base[k]):
            best_by_base[k] = f
    n_families = len(best_by_base)
    plottable = sorted(best_by_base.values(), key=_holistic_max, reverse=True)[:top_n]
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

        fig = plt.figure(figsize=(14, 7.6))
        ax_r = fig.add_subplot(1, 2, 1, polar=True)
        # Categorical palette so distinct families read as distinct colours (a
        # sequential map made adjacent families hard to tell apart, and its pale end
        # was low-contrast on white).
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(plottable))]

        for f, color in zip(plottable, colors):
            med = f.get("axis_medians") or {}
            vals = [med.get(key) if isinstance(med.get(key), (int, float)) else 0.0
                    for key, _ in _AXES]
            vals += vals[:1]
            label = _fold_label(f, max_len=48)
            if f.get("members_spanning_aromatic_and_aliphatic"):
                label += " ✦"  # broad pocket: spans aromatic + aliphatic natives
            ax_r.plot(angles, vals, color=color, linewidth=2.2, label=label)
            ax_r.fill(angles, vals, color=color, alpha=0.13)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(labels, fontsize=10)
        ax_r.set_ylim(0, 1)
        ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_r.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color="#555555")
        ax_r.tick_params(pad=6)
        ax_r.set_title("Native-substrate fit vs target\n(median per axis, 1 = identical)",
                       fontsize=12, pad=26)

        # Right panel: native main-chain length RANGE per family (pocket size has
        # direction — favours larger vs smaller), with the target marked.
        ax_b = fig.add_subplot(1, 2, 2)
        ys = list(range(len(plottable)))[::-1]
        for f, y, color in zip(plottable, ys, colors):
            rng = f.get("native_chain_length_range")
            if not rng:
                continue
            lo, hi = rng[0], rng[1]
            ax_b.plot([lo, hi], [y, y], color=color, linewidth=7, solid_capstyle="round",
                      alpha=0.8)
            ax_b.plot([lo, hi], [y, y], "|", color=color, markersize=11)
        ax_b.set_yticks(ys)
        ax_b.set_yticklabels([_fold_label(f, max_len=34) for f in plottable], fontsize=9)
        tcl = _target_chain_length(run_dir)
        if tcl is not None:
            ax_b.axvline(tcl, color="crimson", linestyle="--", linewidth=1.6,
                         label=f"target main-chain ≈ {tcl}")
            ax_b.legend(loc="lower right", fontsize=9)
        ax_b.set_xlabel("Native substrate main-chain length (carbons)", fontsize=11)
        ax_b.tick_params(axis="x", labelsize=9)
        ax_b.set_title("Pocket size range each family is built for", fontsize=12, pad=12)
        ax_b.margins(y=0.15)
        ax_b.grid(axis="x", alpha=0.3)

        n_more = max(0, n_families - len(plottable))
        supt = "Family native-substrate fit — top candidates for the target"
        if n_more:
            supt += f"   (showing {len(plottable)} of {n_families} families with a resolved fit)"
        fig.suptitle(supt, fontsize=15, y=0.995)
        # ONE figure-level legend BELOW both panels (was an in-axes box overlapping
        # the radar title + axis labels).  Full family names, one legend for the run.
        handles, leg_labels = ax_r.get_legend_handles_labels()
        note = "✦ pocket spans aromatic + aliphatic natives"
        ncol = 2 if len(handles) > 3 else 1
        leg = fig.legend(handles, leg_labels, title=f"Fold family    ({note})",
                         loc="lower center", bbox_to_anchor=(0.5, 0.0),
                         ncol=ncol, fontsize=10, title_fontsize=11, frameon=True,
                         borderaxespad=0.4)
        leg._legend_box.align = "left"
        # Leave room at the bottom for the legend; keep panels clear of it.
        fig.tight_layout(rect=[0, 0.15 if ncol == 1 else 0.20, 1, 0.95])
        out = run_dir / "family_substrate_fit.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if log_fn:
            log_fn(f"  Family-fit plot: rendered {len(plottable)} top families -> {out.name}")
        return out
    except Exception as exc:  # noqa: BLE001 - never break finalization on a plot
        logger.warning("Family-fit plot failed: %s", exc)
        return None
