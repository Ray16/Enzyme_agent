# Cross-model near-attack: Boltz-2 is the signal-carrier (amide-synthetase step-1)

**Date:** 2026-09-23 · **Verdict: use Boltz-2.** Of the AF3-class cofolders tested, only Boltz-2
surfaces the most-active synthetases by static near-attack geometry.

## Question
Does any newer cofolder (Protenix-v2, AF3, ESMFold2) capture the step-1 near-attack signal that
predicts activity better than Boltz-2 — i.e. does the "distance<4 Å gate → in-line angle" proxy
rank the real top catalysts?

## Method (matched across models)
- **50 diffusion samples / complex** for every model (Boltz-2 was already 50; Protenix/AF3 re-folded
  at 50 — a 5-sample first pass was invalid, under-sampled). ESMFold2 excluded (see caveats).
- Cofold input = protein + ATP + 1 Mg + adipate (corrected 1-Mg stoichiometry).
- Proxy (validated): nucleophile = adipate proximal carboxyl O; electrophile = ATP **Pα**;
  gate d(O···Pα) < 4 Å **then** rank by in-line angle ∠(O,Pα,O3A) > 150°. `na_frac` = coarser
  fraction passing both gates.
- Ground truth: measured panel activity (`assay_data/panel_activity.csv`). Top-3 actives =
  **AS9 (9.28), AS17 (1.91), AS23 (1.45)**; binary label = activity > SfaB (reference).

## Result (50-sample)
| model | top-3 AUROC (d<4Å→angle) | top-3 ranks | AS9 rank | binary AUROC | ρ |
|---|---|---|---|---|---|
| **Boltz-2** | **0.94** | **3 \| 5 \| 6** | 6/46 | 0.61 | 0.17 |
| Protenix-v2 | 0.49 | 22 \| 23 \| 29 | 29/47 | 0.63 | 0.05 |
| AF3 | 0.40 | 27 \| 46 \| 47 | 27/44 | 0.51 | 0.06 |

**Only Boltz-2 puts the three most active enzymes in the top 6** (AUROC-top3 0.94, reproducing the
original tuned-proxy value now at matched 50 samples). Protenix/AF3 are near-random on the angle
proxy. (Protenix's `na_frac` AUROC 0.74 is moderate but still ranks the top-3 at 4|14|23.)

## It is angle fidelity, not ligand placement
All three models dock adipate into the pocket equally well — **48/48 targets within 6 Å of ATP Pα,
0 displaced** (median min-d: Boltz 3.33, Protenix 3.21, AF3 3.55 Å). They agree on *where* adipate
binds; only Boltz-2's *in-line attack angles* track activity. So the Boltz-specific advantage is
geometric fidelity of the productive trajectory, not pocket occupancy.

## SfaB (the activity reference) — off-substrate, correctly poor
SfaB's native substrate is 5-chlorovalerate + cadaverine (a **C5 mono-acid**), not adipate (C6
di-acid). In the poses it binds adipate **bent/compact** (distal carboxyl ~7.4–9 Å from Pα vs ~10 Å
extended in AS9) against hydrophobic residues (Val227/Leu322), so the reacting carboxylate is tilted
~30–60° off-axis (angle 119–128° in AF3/Protenix). This is the geometry correctly reflecting SfaB's
poor adipate turnover — consistent with the known SfaB-on-diacid cofold false-negative.

## Caveats
- **Binary AUROC is weak for all models (~0.5–0.63)** — the informative metric is **top-3 surfacing**,
  where Boltz-2 is uniquely strong.
- **ESMFold2 excluded:** its speedup kit fails at 50 diffusion samples in `full_msa`/`fast`
  (`rc=1`, 0 CIFs, `rollout_fallback_num_diffusion_samples=1`); not resolved, campaign concluded first.
- Synthetase panel only; hydrolase cross-model comparison not run.
- Structures are predicted cofolds; the SfaB mechanistic read is a structure-grounded hypothesis.

## Files
- Per-target: `analysis/crossmodel_nearattack.csv` · summary: `analysis/crossmodel_predictivity_summary.csv`
- Figure: `figures/synthetase_crossmodel_nearattack_50sample.png`
- Reusable: `analysis/synthetase/crossmodel_driver.py`, `analysis/synthetase/crossmodel_nearattack.py`,
  `figures/make_crossmodel_figure.py`
- 50-sample structures: `exp_data/amide_synthetase/{boltz2,protenix,af3}/step1_1mg/` (5-sample versions archived).
