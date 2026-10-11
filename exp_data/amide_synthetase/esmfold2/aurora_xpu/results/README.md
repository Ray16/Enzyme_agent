# ESMFold2 amide-synthetase panel, stock numerics on Aurora XPU (2026-10-10)

49 targets x **50 diffusion samples = 2450 poses**, protein + ATP + Mg + adipate
dianion, MSA-conditioned. This replaces the archived `--mode fast` run as the
quality reference, and it is the first ESMFold2 panel here at the 50-sample
house standard (the archived one was 5, which is why it fed no pose-level
analysis; Boltz-2 ran 50 on this same panel).

Raw output: `/lus/flare/projects/FRAME-IDP/rzhu/esmfold2_xpu/out_panel/`
(880 MB, `<target>/sample_XX.cif` + `sample_XX.json` + `ranking.tsv`).

## Configuration — uniform across all 2450 samples, verified not assumed

```
used_msa True | loops 20 | steps 200 | msa_max_depth 1024 | float32 (+bf16 autocast)
checkpoint biohub/ESMFold2 @ 1ebf0e34, ESMC-6B @ 45b0fa5d  (kit stock/PINS.json, sha256-checked)
49 targets x exactly 50 cif + 50 json, 0 undersized files
```

## Quality

| | min | median | max |
|---|---|---|---|
| mean pLDDT | 79.1 | 88.8 | 91.3 |
| ipTM | 0.750 | 0.880 | 0.928 |
| ps_ipTM (protein↔**ATP**) | 0.466 | 0.648 | 0.703 |

## Agreement with the archived fast-mode CUDA run

Per-target medians, n=49 (`panel_stock_vs_fast.csv`):

| | Spearman ρ | median Δ (stock − fast) | range |
|---|---|---|---|
| ipTM | **0.888** (p=1.7e-17) | +0.0013 | −0.012 … +0.042 |
| pTM | **0.933** (p=1.9e-22) | +0.0013 | −0.008 … +0.028 |

Read that against the fast run's **own** within-target ipTM spread, median
**0.007**: the cross-backend difference is several times smaller than the
model's own seed-to-seed noise. Structurally, on AS9, Cα RMSD between the two
runs was 1.29 Å against a within-fast spread of 0.90–2.07 Å — inside the noise.

So the two backends agree, and nothing in the panel's conclusions should change
because of the backend. What *does* change is sample depth: 50 per target
instead of 5.

## Cost

17.2 GPU-hours total, 25.0 s median per sample, ~30 min wall on 6 Aurora nodes
(72 tiles). Against the +354,751 node-hour allocation this is free; the Polaris
CUDA route was unavailable at −1,088 node hours.

## Caveats

- These are **stock** numerics (SDPA, no flash-attn / transformer_engine), which
  the kit calls `off` — at or above its `exact` mode, so this satisfies the
  house "highest-quality configuration" standard. It is *not* literally the
  kit's `--mode exact`, which remains unrun anywhere.
- The two 2025 assay batches still do not reproduce against each other
  (ρ=0.204, p=0.16). Better structures do not fix a ground-truth problem, and
  no proxy conclusion should be revisited on the strength of this run alone.
- ⚠ **`ps_ipTM` in this run is protein↔ATP, not protein↔adipate.** The input
  chain order is protein, ATP, MG, adipate, and the reducer inherited from the
  Polaris driver reads pair (0,1), which was that project's ligand of interest
  but is ATP here. The protein↔adipate value is **not recoverable** from the
  saved JSON -- only the single reduced number was stored. The driver now
  records the full `pair_chains_iptm` matrix instead, so a re-run would carry
  every interface, but this panel does not have it.
  This does not affect the planned downstream work: the pose-level geometry
  (donor-Pα distance, attack angle, ligand displacement) is measured from the
  CIF coordinates, not from ipTM.

## Which cofolder surfaces the measured actives? (Boltz-2, not this one)

`panel_nac_esmfold2_vs_boltz2.csv` scores these 2450 poses and the matched
2450-pose Boltz-2 1-Mg panel (`boltz2/step1_1mg`) through one element-based
carboxylate/P-alpha selector (`nac_comparison/`). Ranking the 49 targets by the
proximity-weighted angle `A`:

| positives | Boltz-2 AUROC | ESMFold2 AUROC |
|---|---|---|
| AS9, AS17, AS23 | **0.942** (ranks 4/1/9) | 0.688 (21/24/4) |
| **AS9, AS17 only** (see below) | **0.979** (ranks 4 and 1) | 0.553 (21 and 24) |
| AS23 only | 0.833 | 0.938 |

Paired bootstrap on the 3-positive label: dAUROC −0.254, 95% CI
[−0.355, −0.159]. Against an 8-test selection-corrected null (4 metrics x 2
models) Boltz-2's `A` survives (p=0.014); ESMFold2's does not (p=0.503).

**AS23 is not a reproducible active** — 2.89 in the 1/24 batch vs 0.012 in the
01/09 batch (238x, i.e. the BL21 control level), against 1.4x for AS9 and 2.4x
for AS17. Dropping it is what splits the two models cleanly: ESMFold2's entire
apparent signal was AS23, and it sits at chance on the two solid actives.

Two caveats on reading this as a Boltz-2 endorsement. It is **specifically `A`**
that carries Boltz-2 — its `D_h` (0.457) and `n_gated` (0.489) are at chance, so
denser docking is not what helps; ESMFold2 in fact docks more (it puts adipate
at P-alpha in 33/50 SfaB poses where Boltz-2 manages 1/50, on an enzyme that is
experimentally near-dead). And this is a **comparative** result: `A` was
selected on this same panel historically, only 2 trustworthy positives remain,
and the known fold-confound on `A` is untested here.

On the *continuous* question — rank correlation against yield rather than
retrieval of the actives — neither model wins: ESMFold2 rho=+0.262 (p=0.069),
Boltz-2 +0.226 (p=0.118), paired difference p=0.84.
