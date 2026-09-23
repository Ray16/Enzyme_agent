# Nylon-hydrolase feature surfacing: structure does not predict within-fold hydrolytic rate

**Date:** 2026-09-23 · **Verdict:** For the 96-enzyme Ntn-hydrolase panel (Peptidase S58 / NylC
family) assayed on crystalline nylon powder, **no catalytic-geometry proxy predicts activity.**
The only correct-direction signal is model confidence — and the *best* of it is **active-site /
autocleavage confidence** (active-site-shell pLDDT, catalytic-Thr pLDDT, β-chain pTM, A|B-junction
ipTM), which is mechanistically motivated: these enzymes require AUTOCLEAVAGE to expose the
catalytic N-terminal Thr, so confidence in the active-site / mature-junction region reports whether
the model reconstructs the correctly self-processed, catalytically-poised state. This is **distinct
from global foldability** (shell-pLDDT correlates only ~0.5 with complex-pLDDT and retains a partial
ρ ≈ +0.14 for L1 after controlling for it) — *not* merely a foldability confound. It remains a
**weak** predictor, though (raw ρ ≈ 0.12–0.18), and it is a confidence measure, not catalytic
geometry: the honest headline is that **no structural catalytic-geometry proxy predicts within-fold
hydrolytic rate; the best (weak) signal is active-site/autocleavage model confidence.**

## Question
Repeating the amide-synthetase feature-surfacing analysis for the hydrolases: does any structural
proxy computed on the Boltz-2 50-sample cofold ensemble rank the most experimentally-active nylon
hydrolases? Proxies chosen with the `kg-memory-system` agent (391 rec/95 papers), adapted to
Ntn / Ser-nucleophile amide hydrolysis.

## Method (matched to the synthetase study)
- Cofold: mature enzyme split at the catalytic Thr (chain B res 1 = free N-terminal Oγ nucleophile),
  + nylon-6,6 tetrad ligand, 50 diffusion samples. 96 enzymes × 50 = 4,800 models.
- Proxies (`analysis/hydrolase/proxy_battery.py`): near-attack joint gate (Oγ→scissile carbonyl,
  Bürgi–Dunitz), TI-template RMSD, oxyanion-hole H-bond geometry, nucleophile activation
  (α-amine⋯Oγ), reactive-center RMSF, oxyanion ESP; + confidence nulls (pLDDT, ipTM).
- Metric: surfacing AUROC at nested cutoffs (top-5/10/20 + > control) + mean, **directions fixed
  by mechanism** (`analysis/hydrolase/feature_surfacing.py`).
- Activity: L2 = PA66 dimer release (primary, 38/96 active), L1 = PA66 monomer (15/96).

## Result
| feature (fixed dir) | L2 mean / top-5 | L1 mean / top-5 |
|---|---|---|
| **complex pLDDT** (null) | **0.66 / 0.72** | 0.57 / 0.57 |
| **active-site pLDDT** (null) | 0.51 / 0.46 | **0.64 / 0.81** |
| reactive-center RMSF | 0.63 / 0.73 | 0.51 / 0.42 |
| near-attack joint gate | 0.58 / 0.69 | 0.39 / 0.23 |
| TI-template RMSD | 0.52 / 0.62 | 0.30 / 0.11 |
| oxyanion-hole / ESP / base | ≤0.57, several *backwards* | ≤0.54, mostly backwards |

- **The best signal is active-site / autocleavage confidence — distinct from global foldability.**
  Autocleavage-junction ipTM (A|B) tops L2 (top-5 0.81); catalytic-Thr pLDDT and β-chain pTM top L1
  (top-5 0.81 / 0.77). These are mechanistically loaded for autoprocessing Ntn enzymes and are only
  moderately correlated with global complex-pLDDT (~0.5 for shell/Thr pLDDT, 0.26 for β-chain pTM);
  active-site-shell pLDDT retains a partial ρ of +0.14 (L1) / +0.08 (L2) after controlling for
  complex-pLDDT — so there is an active-site-specific component, not just global foldability. Caveat:
  still weak (raw ρ ≈ 0.12–0.18), and the crude-lysate assay means part of any pLDDT signal reflects
  expressibility/detectability. The best catalytic-GEOMETRY proxy (near-attack joint gate, L2 top-5
  0.69) is matched by these confidence terms and collapses for L1.
- **The RMSF "signal" is a confidence confound** — `preorg:og1_rmsf` correlates ρ = −0.63 with
  complex pLDDT; active enzymes are simply modeled more rigidly/confidently.
- **Oxyanion-hole, ESP, and base-distance proxies are near-random or backwards** (`*`). The
  earlier apparent "L1 is predictable at AUROC 0.9" was a **sign-flip artifact** of picking
  direction from a near-zero correlation on 15 positives; under fixed mechanistic direction those
  values are ~0.1–0.4.
- **Everyone binds; near-attack is rare for everyone.** Median best Oγ···C ≈ 2.5 Å in both actives
  and inactives; 47/96 enzymes have zero productive joint-NAC poses. The catalytic geometry is
  essentially invariant across this single fold.

## Why (the informative contrast with synthetases)
Synthetases are fold- and substrate-**diverse**, so how well the reactive (in-line) geometry forms
varies with the enzyme — and Boltz-2 near-attack angle surfaces the top-3 catalysts at AUROC 0.94.
The hydrolase panel is a **single conserved fold**; its measured activity on **crystalline nylon
powder** is gated by polymer access / adsorption / thermostability, which a soluble-oligomer cofold
does not capture. Static structural proxies therefore succeed for the synthetases and fail here.
This directly supports the manuscript's premise that crystallinity and H-bonding physically
restrict enzyme access to the polymer backbone.

## Caveats
- Only the Michaelis-like cofold state is modeled; KG-recommended deacylation-state preorganization
  and a DFT-refined TI template (v2) are not accessible without a second modeled state / QM.
- L1 has only 15 positives — top-k AUROCs there are high-variance (hence the fixed-direction guard).
- Structures are predicted cofolds; PROPKA on the buried N-terminal α-amine was omitted as unreliable.

## Files
- Proxies: `analysis/hydrolase_proxies.csv` · rankings: `analysis/feature_surfacing_ranking_{L2,L1,total}.csv`
- Figure: `figures/hydrolase_feature_surfacing.png`
- Code: `analysis/hydrolase/proxy_battery.py`, `feature_surfacing.py`, `figures/make_hydrolase_feature_surfacing_figure.py`
- Proxy shortlist + lit anchors: `analysis/hydrolase/STRUCTURAL_PROXIES.md`
