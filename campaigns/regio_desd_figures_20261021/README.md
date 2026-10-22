# Does the proximity-weighted attack angle `A` separate active from inactive enzymes?

A test of our in-silico metric against **every quantitative figure** of
`paper/Regioselective biosynthesis of oligoamides as precursors for sequence-controlled
co-polyamides.pdf` (bioRxiv 2025.11.03.685623).

Read in this order: **`PREREGISTRATION.md`** (the test, fixed before folding) →
**`MECHANISM.md`** (why the setup is what it is) → `analysis/`.

## Status (2026-10-21)

| Track | State |
|---|---|
| Step-1 cofolds (58) | 54 done; remainder auto-retrying on V100s |
| Step-2 cofolds (90) | 19 done; running |
| Fig 2 labels | pre-existing, independently re-digitized, verdict ACCEPT |
| Fig 4 / 5 labels | digitized this session (60 rows each); Fig 6 has no quantitative panel |
| **Positive control** | **PASSES** — 0.23 Å best ligand placement, 50/50 poses on-site (see caveat) |
| Driver | `fleet/autodrive.sh` detached: watchdog + full replot every 10 min to completion |

### The positive control, and its limit

DesD + native HSC vs crystal **7TGM**: backbone Cα RMSD **0.38–0.51 Å**, ligand centroid
displacement **best 0.23 / median 0.69 Å**, contact Jaccard 0.933, **50/50** poses within 3 Å.

This proves the **pipeline is correctly configured** — chains, atom selection, protonation,
cofactor placement, pocket identity. It does **not** prove predictive ability: 7TGM was released
2022-07-06 and is very likely inside Boltz-2's training set, so this is plausibly **recall**. The
campaign's actual inputs (DesD+adipate, DdaG+MA, AcsA+AMA) are certainly not in the PDB — genuine
predictions, but with **no crystal to check them against**. That is the honest ceiling here.

## Layout

```
PREREGISTRATION.md     the hypotheses, tests, exact p-values and power ceiling, fixed up front
MECHANISM.md           Palpha vs Pgamma, Mg count, the protonation rule, step-1 vs step-2 mapping
sequences/             the 6 enzymes + provenance (how each identity was established)
msa/                   batched ColabFold MSAs (one job, 56 s) + index.tsv
structures/            7TGM (DesD+HSC+AMS), 2W02 (AcsD+ATP+Mg) reference crystals
cofolds/step1|step2/   inputs/ (YAML), results/ (ensembles), job_map.csv
fig2_diads/ fig4_triads/ fig5_tetrads/ fig7_regioselectivity/    one figure per experimental panel
analysis/              step1_scores.csv, control_7tgm_vs_predicted.csv, diagnostics
fleet/                 claim-based resumable Boltz workers (items_*.txt, claims/, logs/)
archive/               superseded files (never deleted)
```

## Reproduce

```bash
conda activate boltz-2
python3 substrates_oligomer.py    # verify every ligand's formula + charge
python3 build_inputs.py           # 58 step-1 YAMLs
python3 validate_inputs.py        # GATE: inputs. must print ALL CHECKS PASSED
bash   fleet/launch_all.sh 16     # fan across free GPUs via gpu_reserve
bash   roll.sh                    # GATE: structures -> score with A -> replot panels
python3 analyze_control.py        # the positive control, once it is folded
```

## What is load-bearing here

**The metric is frozen.** `A` is imported from
`structural_modeling_agent/proxies/attack_angle.py` (`inverse_distance_attack_angle_v1`) and never
reimplemented, refitted, or threshold-tuned. `n_gated` is a **validity precondition**, never a
ranking input.

**Two gates, both fatal.** `validate_inputs.py` checks the YAMLs (chains, cofactors, Mg count,
sequence identity, MSA-query-vs-sequence, carboxylate present, no competing nucleophile).
`validate_outputs.py` checks the *structures* (per-pose heavy-atom and carboxyl counts against the
input SMILES, ATP completeness, Mg actually coordinating a phosphate) — because Boltz renames all
CCD ligands to `LIG`, renumbers substrate atoms, and **exits 0 on OOM**.

**Non-dockers are reported, never dropped.** Dropping them is how "did the ligand dock at all"
masquerades as a catalytic signal.

## Bugs these gates caught (all in previously-"working" inherited code, none raised an error)

1. **ω-amino donors had the wrong charge state.** `substrate_reference.csv` applies its pH-7 rule to
   carboxylates but not amines, so donor H was `NCCCCCC(=O)[O-]` (neutral −NH₂, −1) instead of the
   zwitterion. On a step-1 donor a neutral −NH₂ is a **second nucleophile competing for Pα**.
2. **The acyl-adenylate builder was wrong for 5 of 13 donors.** It formed acyl-AMP by
   `donor[len("[O-]"):]`, assuming the carboxylate is the first SMILES token — false for every donor
   starting `[NH3+]`, where it excised part of the amine and produced a *different molecule that
   still parses*.
3. **My own mass-balance expectation was off by one H** — the condensation expels **hydroxide, not
   water**. Caught by the gate, not by inspection.
4. **A stale preflight would have idled the entire fleet** — it required `mpi4py`, absent from the
   current env, while `import boltz` succeeds without it.
5. **A figure legend mislabelled the active anchor.** Marker shape keyed to docking while the caption
   read "square = inactive", so **SfaB** (active, 1.0, `n_gated` 0) read as inactive.

## Results so far — negative, and honestly reported

- **Fig 2A** (X+A→XA): no contrast (6/6 detected); among them ρ(`A`, activity) = **+0.09, p = 0.919**.
  `A` ranks AcsA 3rd of 6 though it is ~6× the next best.
- **Fig 5A** (MAM+A→MAMA): detected enzymes in the top 3 by `A` = **NO, p = 0.250**. Expected —
  that panel's discrimination is amine-side by construction, so step-1 `A` should not explain it.
- **A tracks docking, not catalysis, on Fig 2A**: `A` and `n_gated` rank the six enzymes
  *identically* (DesD 33 → AsbB 19 → AcsA 7 → AsbA 4 → DdaG 1 → SfaB 0), and docking tracks **fold
  family** (all NIS above both ANL) — the axis on which `A` is already documented as biased.

**The control has now reported, and it removes the main alternative explanation.** The poses are
right (0.23 Å), so `A`'s failure is a statement about the metric and its premise, not about bad
modelling. Also recorded: DesD occupancy is 50/50 on native HSC, 33/50 on adipate, 1/50 on the AMA
triad — occupancy may track *how native-like* a substrate is (learnable from the PDB) rather than
*whether it reacts* (no training signal at all).

Families tried and their status: `A` (failing, see above); **accommodation PARKED** — TA1 not
evaluable because `buried_frac` saturated at 1.000, and pocket **volume abandoned as ill-posed**
(grid method did not converge: 454/773/1139 Å³ at box radius 10/14/18 Å, and missed the paper's
Fig-3B values by non-constant factors). No fold-derived pocket volume is quoted anywhere.
The design's power ceiling is **p ≈ 0.048** even if `A` were perfect — 6 enzymes per stratum is this
paper's hard limit, so the deliverable is mechanistic, not a p-value.
