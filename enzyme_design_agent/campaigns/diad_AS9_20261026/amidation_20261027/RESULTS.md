# Step-2 (amidation) result: the primary endpoint is null, and the designs do not advance

Scored 2026-10-27 against `PREREGISTRATION.md`, which was written before any step-2 structure
existed. The pre-registered prior was a null; the primary endpoint returned one.

## Joint readout

| gene | step-1 n<4 A | step-1 med d | step-2 n<4 A | step-2 med d2 | step-2 med BD |
|---|---|---|---|---|---|
| WT_AS1 | 35/50 | 3.77 | **0/50** | 30.24 | 44.5 |
| WT_AS9 | 10/50 | 4.26 | **0/50** | 22.56 | 134.9 |
| WT_AS9_rep2 | - | - | **0/50** | 22.31 | 136.8 |
| AS9_diad_pose0_fixed_cap_r2 | 2/50 | 5.10 | **0/50** | 8.82 | 130.8 |
| AS9_diad_pose0_internal_cap_r2 | 1/50 | 5.59 | **0/50** | 8.91 | 131.6 |
| AS9_diad_pose3_internal_cap_r2 | 3/50 | 5.26 | **0/50** | 8.62 | 119.4 |

## Verdict under the pre-registered rule

1. **Primary endpoint `fc2` is 0/50 in all six structures, including both wild-types.** The
   metric has zero dynamic range on this panel, so it cannot rank anything and no design is
   called improved at step 2.
2. `A2` is **not interpreted**, per the standing rule that a proximity-weighted angle over an
   empty gated set is not a ranking. Its values are in `step2_scores.json` and are inert.
3. Advancing required improving at step 2 **and** not regressing at step 1. All three designs
   regress at step 1 (10/50 -> 1-3/50) and none improves at step 2. **No design advances.**

## The folds are sound; the ligands are placed, just in the wrong place

This is not a docking failure, which is what makes it informative:

| gene | ligand_iptm | P to A3 loop | acyl C to His209 | protein atoms within 5 A of HMDA |
|---|---|---|---|---|
| WT_AS1 | 0.932 | 5.58 | 11.80 | 28 |
| WT_AS9 | 0.951 | 5.53 | 7.05 | 42 |
| WT_AS9_rep2 | 0.953 | 5.36 | 3.62 | 34 |
| pose0_fixed_cap | 0.974 | 6.29 | 3.53 | 41 |
| pose0_internal_cap | 0.976 | 5.39 | 3.29 | 41 |
| pose3_internal_cap | 0.979 | 6.81 | 3.55 | 37 |

Adipoyl-AMP sits in the canonical adenylation site in **all six** (phosphorus 5.4-6.8 A from
the A3 loop), at high confidence (ligand_iptm 0.93-0.98). HMDA is **buried** in every
structure (28-42 protein atoms within 5 A), not drifting in solvent. It is simply buried in a
*different pocket*: ~22 A from the electrophile in wild-type AS9, ~8.6-8.9 A in the designs.

This reproduces the PURp1 urethanase and SfaB pattern already on file -- a confidently docked,
buried ligand at a non-catalytic site -- rather than a model that failed to place anything.

## The one real, sub-threshold effect

The WT_AS9 / WT_AS9_rep2 pair (identical sequence, seeds 20261027 and 20261028) gives the
step-2 replicate floor measured in-run:

- `median_d2` 22.56 vs 22.31 -> **|delta| = 0.25 A**
- `min_d2` 20.04 vs 16.13 -> |delta| = 3.92 A, so **min_d2 is noisy; report the median**
- `n_gated` 0/50 vs 0/50

Against a 0.25 A floor, the designs move HMDA **13.6-13.9 A closer** to the electrophile than
wild-type AS9 -- roughly 55x the replicate difference, and consistent across all three designs
(8.62, 8.82, 8.91). So the designs did relocate the diamine substantially toward the active
site.

But 8.6 A is not a near-attack distance; that needs ~3 A. The relocation is real and
reproducible and still an order of magnitude short of productive. It does not satisfy the
pre-registered endpoint, and it must not be reported as a hit.

## Caveats that limit this run

- **Microstate mismatch.** Adipoyl-AMP is modelled here as the **neutral** species, whereas the
  source ensemble that produced the seed poses used **-2**. The state is identical across all
  six jobs, so the design-vs-WT contrast is internally valid, but it is not matched to the
  generation source. Since phosphate charge is what drives the salt-bridge artifact documented
  in `../MOTIF_DIAGNOSIS_AND_FIX.md`, a `-2` arm is the clean test of whether the 8 A plateau
  and the wild-type 22 A site are charge artifacts. **Not run.**
- Mg is present in all six and its absolute effect is untested here by construction.
- Neither axis predicts a rate, and no step-2 metric in this project has been validated
  against a measurement. Step-1 `A` did not separate active from inactive on the regio panel
  (AUROC 0.561, p=0.248).
- `ame.pass` remains uncomputed for all four designs; the two refold criteria were never run.

## What this does and does not establish

It establishes that the designed objective was **not** achieved: the amine does not reach the
electrophile in any design, and the step-1 cost is real. It does not establish that the designs
are inactive, because the readout has no dynamic range -- a metric that reads 0/50 on
wild-type AS9 cannot distinguish a bad design from a bad metric. The third possibility, that
the amidation state is simply not modellable this way, is consistent with everything here and
with the two prior step-2 nulls in this project.
