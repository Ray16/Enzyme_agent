"""Reaction energy-diagram plot from a scan barrier JSON (the `scan_curve` written by
scan.py). One PNG per enzyme: relative electronic energy vs the forming-bond reaction
coordinate d(nuc-elec), with reactant / TS / product marked and DeltaE-dagger annotated.
Generalizable: reads only the JSON, so it renders for any enzyme the pipeline ran."""
from __future__ import annotations
import json
from pathlib import Path


def energy_diagram(barrier_json: str | Path, out_png: str | Path | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(Path(barrier_json).read_text())
    curve = d["scan_curve"]
    # prefer the monotone two-leg coordinate xi = d(elec-leav) - d(nuc-elec); fall back to d(nuc-elec)
    use_xi = all("xi" in c for c in curve)
    xs = [c["xi"] for c in curve] if use_xi else [c["d_nuc_elec"] for c in curve]
    ys = [c["dE_kcal"] for c in curve]
    iR, iT, iP = d["i_reactant"], d["i_ts"], d["i_product"]
    tag = d["enzyme_id"]
    out_png = str(out_png) if out_png else str(Path(barrier_json).with_name(f"energy_diagram_{tag}.png"))

    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=200)
    ax.plot(xs, ys, "-", color="#444444", lw=1.6, zorder=1)
    ax.scatter(xs, ys, s=22, color="#888888", zorder=2)
    # for an adenylation step the "product" of the SCANNED step is the acyl-adenylate
    # INTERMEDIATE of the full two-step reaction (final product = amide, after step 2),
    # so label it accordingly; otherwise it is the step product.
    prod_lab = "intermediate\n(acyl-AMP)" if str(d["reaction_id"]).startswith("adenylation") else "product"
    for idx, lab, col in ((iR, "reactant", "#2b7bba"), (iT, "TS", "#d1495b"), (iP, prod_lab, "#2a9d5c")):
        ax.scatter([xs[idx]], [ys[idx]], s=90, color=col, zorder=3, edgecolor="k", linewidth=0.6)
        ax.annotate(lab, (xs[idx], ys[idx]), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, color=col, fontweight="bold")
    # barrier + reaction-energy annotations
    dEa, dEr = d["dE_act_kcal"], d["dE_rxn_kcal"]
    ax.annotate("", xy=(xs[iT], ys[iT]), xytext=(xs[iT], ys[iR]),
                arrowprops=dict(arrowstyle="<->", color="#d1495b", lw=1.2))
    ax.text(xs[iT] - 0.06, (ys[iR] + ys[iT]) / 2, f"$\\Delta E^\\ddagger$ = {dEa:.1f} kcal/mol",
            rotation=90, va="center", ha="right", fontsize=10, color="#d1495b")
    gate = d["mechanism_gate"]["ok"]
    is_aden = str(d["reaction_id"]).startswith("adenylation")
    e_lab = "$\\Delta E_\\mathrm{to\\,int.}$" if is_aden else "$\\Delta E_\\mathrm{rxn}$"
    sub = (f"{d['reaction_id']}  |  {d['n_atoms']} atoms ({d['n_free']} free), q={d['charge']}  |  "
           f"{e_lab}={dEr:+.1f}  |  mechanism gate: {'PASS' if gate else 'FAIL'}")
    fig.suptitle(f"{tag} — adenylation barrier (UMA {d['model']})", fontsize=12, y=0.99)
    ax.set_title(sub, fontsize=7.8, color="#555", pad=8)
    if "ddE_vs_wt_kcal" in d:
        ax.text(0.98, 0.04, f"$\\Delta\\Delta E^\\ddagger$ vs WT = {d['ddE_vs_wt_kcal']:+.1f} kcal/mol",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#333")
    if use_xi:
        ax.set_xlabel("reaction coordinate  ξ = d(Pα–O3A) − d(O–Pα)  [Å]", fontsize=10)
    else:
        ax.set_xlabel("reaction coordinate  d(nucleophile–electrophile)  [Å]", fontsize=10)
        ax.invert_xaxis()   # bond forms as d decreases -> reactant on the left visually
    ax.set_ylabel("relative electronic energy  [kcal/mol]", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import sys
    print(energy_diagram(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
