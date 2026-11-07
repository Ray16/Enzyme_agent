# AS9 motif-conditioned design panel — refold results (2026-11-07)

Scope: 100 RFD4-Proteina designs (516 aa), 50 independent Boltz-2 refolds each
(fresh batched MSAs, no templates, no restraints, FP32 `stock_fp32_serial`,
`--max_parallel_samples 1`). 5000 structures, 0 scoring errors.
**All geometry below is measured from the refolds only.** The geometry imposed in
the RFD4 input is a design constraint, not validation evidence.

## Headline: the M1 motif did not beat the M0 control

| arm | n | A (deg) | d_h (A) | frac d<4A | near-attack poses | bb RMSD | motif RMSD |
|---|---|---|---|---|---|---|---|
| M0_pose15 | 25 | 91.3 +/- 24.7 | 9.66 | 5.4% | 16 | 2.94 | 3.16 |
| M0_pose6  | 25 | 96.5 +/- 20.2 | 9.29 | 7.0% | 16 | 3.16 | 3.06 |
| M1_pose15 | 25 | 89.5 +/- 20.5 | 8.69 | 4.3% | 13 | 3.76 | 4.13 |
| M1_pose6  | 25 | 86.1 +/- 21.5 | 8.82 | 5.8% | 17 | 2.43 | 3.67 |
| WT_AS9 anchor (Oct 27) | - | 164.7 | 4.29 | 20% | - | - | - |

Pooled M0 (n=50) vs M1 (n=50): A 93.9 vs 87.8, **MWU p=0.078**; d_h p=0.299;
frac d<4A p=0.833. The 6.1 deg angle gap is ~2x the measured 2.67 deg dA noise
floor and is not significant. On the clean-provenance subset the conditioned
motif is reproduced *worse* by M1 (3.74 A) than M0 (2.66 A), p=0.101.

0/100 designs reach the WT angle; 1/100 reaches WT harmonic distance;
62 near-attack poses (d<4 A, theta>150 deg) out of 5000.

## Refolding itself is sound

median backbone RMSD **1.166 A** (IQR 0.89-1.81), 78/100 <2 A, 39/100 <1 A,
14/100 >5 A; pLDDT median 0.904; rho(pLDDT, bbRMSD) = -0.798.
Restricting to the 78 well-refolded designs does not move the geometry
(A 91.9, d_h 8.81, frac 6.4%), so the docking failure is **not** a folding artifact.

Spearman A vs d_h = -0.219 (p=0.029): higher angle tracks *better* docking, so the
Oct-27 off-axis-carboxylate artifact (A rising while docking worsens) is not operating here.

## Four designs do dock at WT level

| id | arm | A | d_h | frac<4A | near | bb | motif | pLDDT |
|---|---|---|---|---|---|---|---|---|
| AS9D_a3b9be538e67 | M0_pose15 | 135.4 | 4.25 | 42% | 10 | 0.72 | 1.27 | 0.928 |
| AS9D_9b0667393066 | M1_pose6  | 144.3 | 4.48 | 46% | 10 | 1.59 | 4.62 | 0.883 |
| AS9D_81a3d52b4a2e | M0_pose6  | 132.6 | 4.41 | 30% |  6 | 0.77 | 1.70 | 0.927 |
| AS9D_149e22a7c2f7 | M1_pose15 | 125.7 | 4.42 | 28% |  4 | 0.86 | 1.61 | 0.944 |

The readout therefore has dynamic range; the ~90 deg / 9 A panel mean is a real
property of most designs, not a floor imposed by the metric. Two of the four are
M0 controls.

## Motif-mapping provenance is arm-confounded but not outcome-biasing

22/100 designs needed motif mapping recovered by ligand-frame superposition because
RFD4 dropped its provenance columns; **all 22 are M1**, 0 are M0. Recovered vs
labelled designs do not differ on any metric (motif RMSD p=0.680, bb p=0.878,
A p=0.541, d_h p=0.106, frac p=0.789), so the recovery path is not distorting
results. Arm comparisons above are also reported on the labels-only subset.

## Open: ligand charge-state confound

The Oct-27 WT anchor used CCD ATP + neutral adipate `O=C(O)CCCCC(=O)O`; this
campaign uses ATP(4-) + adipate(2-), one Mg, no hydration. Recomputing the Oct-27
ensembles with this pipeline's geometry definition reproduces the recorded anchors
exactly (WT_AS9 164.74 / 4.288 / 20%; WT_AS1 143.78 / 3.781 / 70%), so the metric
is not at fault -- but the two setups are chemically different and the current
`molecule_maps()` cannot even parse the neutral ligands.

Single-variable controls (`wt_controls_charged_20261107/`) re-run both WT sequences
with the same Oct-27 MSAs and the same FP32 settings, changing only the charge
states. **Until these return, the absolute comparison of designs against the
164.7 deg anchor is uninterpretable.** The M0-vs-M1 null above is internal to this
campaign and does not depend on the control.
