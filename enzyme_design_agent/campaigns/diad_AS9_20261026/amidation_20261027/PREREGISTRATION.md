# Pre-registration: step-2 (amidation) evaluation of the RFD4 AS9 diad designs

Written **before** any step-2 structure was generated or scored. Registered 2026-10-27.

## Why this run exists

The `review_20261027` evaluation cofolded **ATP + Mg + free adipic acid** — the step-1
(adenylation) state. The RFD4 designs were generated to improve **step 2**: attack of
hexamethylenediamine (HMDA) on adipoyl-AMP. So the existing scores measure collateral
damage to a step the designs were not optimising, and the design objective itself has
never been evaluated. This run supplies the missing axis.

## Why both steps, not step 2 alone

The step-1 result already shows a trade: adipate docking fell from 10/50 poses <4 A in
WT_AS9 to 1-3/50 in all three designs. A diad requires both half-reactions in the same
protein, so a design that wins step 2 while losing step 1 is not a candidate. Both axes
are reported jointly, on the identical 5-gene panel, so they can be joined per sequence.

## Panel (6 jobs)

| job | sequence | seed | role |
|---|---|---|---|
| WT_AS1 | WT AS1 (523 aa) | 20261027 | second WT reference |
| WT_AS9 | WT AS9 (516 aa) | 20261027 | parent / primary control |
| WT_AS9_rep2 | WT AS9 (516 aa) | 20261028 | **step-2 noise floor** |
| AS9_diad_pose0_fixed_cap_r2 | design, 6 muts | 20261027 | design |
| AS9_diad_pose0_internal_cap_r2 | design, 7 muts | 20261027 | design |
| AS9_diad_pose3_internal_cap_r2 | design, 7 muts | 20261027 | design |

`WT_AS9_rep2` is an independent-seed refold of the *same* sequence. The Oct-10 lesson was
that the delta noise floor was free to measure and nobody measured it; a step-2-specific
floor does not exist yet, so it is measured here rather than borrowed from step 1
(step-1 floor was SD 2.67 deg, mean |dA| 2.09, n=7 WT pairs).

## Modelled state

- protein, 50 diffusion samples, FP32, recycling 3, sampling 200 — identical to step 1
- MSAs **reused byte-for-byte** from `review_20261027/msa/` (same sequences), so MSA depth
  cannot differ between step 1 and step 2, nor between designs and WT (11.8-12.8k)
- adipoyl-AMP `Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OC(=O)CCCCC(=O)O)[C@@H](O)[C@H]1O`
  (C16H22N5O10P, neutral microstate, 32 heavy atoms — matches generation-input chain L)
- mono-protonated HMDA `NCCCCCC[NH3+]` (C6H17N2+, 8 heavy atoms — matches chain M)
- **one Mg**, matching the source ensemble that produced the seed poses

Mg and protonation are modelled microstates, not assigned physiological states. Mg is held
**identical across all six jobs**, so it cannot confound the design-vs-WT contrast; its
absolute effect is untested here. This deliberately avoids the earlier error where the
restraint and Mg were varied together and neither could be isolated.

## Endpoints, fixed in advance

Electrophile is identified topologically: the unique carbon in the P-O-C(=O) mixed
anhydride (verified to be unique by RDKit). Attacking nitrogen is whichever HMDA N is
nearer that carbon (heavy-atom output cannot distinguish NH2 from NH3+, and either
terminus may attack).

- `d2` = distance(N_attack, C_acyl), per pose
- `BD2` = angle(N_attack, C_acyl, O_carbonyl), ideal Burgi-Dunitz ~105 deg
- `inline2` = angle(N_attack, C_acyl, O_bridge), ideal in-line displacement ~180 deg
- `fc2` = fraction of 50 poses with `d2` < 4 A  -- **primary**
- `A2` = proximity-weighted BD angle, sum(BD_i/d_i)/sum(1/d_i)  -- secondary

## Decision rule, fixed in advance

1. `fc2` is the primary endpoint; `A2` is reported but is **not** interpreted when
   `n_gated` is low. The standing rule (PylC, 1/50) is that `A` over a near-empty gated
   set is not a ranking.
2. A design is called **improved at step 2** only if `fc2(design) > fc2(WT_AS9)` by more
   than the WT_AS9-vs-rep2 replicate difference measured in this same run.
3. A design **advances as a diad candidate** only if it improves at step 2 **and** does
   not regress at step 1. On the step-1 numbers already in hand (fc 1-3/50 vs WT 10/50),
   **no current design can satisfy this** — so this run tests whether RFD4 achieved its
   stated objective, not whether these three variants are worth making.
4. Expectation set in advance: step-2 cofolds have been uninformative twice in this
   project (regio_desd step-2 BD at chance; the Oct-3 modeling audit). A null is the
   prior, and a null will be reported as a null.

## What this run cannot show

Neither axis predicts a rate. Step-1 `A` did not separate active from inactive on the
regio panel (AUROC 0.561, p=0.248), and no step-2 metric in this project has ever been
validated against a measurement. Forced starting proximity in the generation input is not
independent confirmation of a productive pose.
