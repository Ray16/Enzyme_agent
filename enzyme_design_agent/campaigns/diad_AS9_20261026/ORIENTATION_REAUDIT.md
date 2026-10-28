# Orientation re-audit of the 50-pose source ensemble: re-filtering cannot work

Run 2026-10-27 over all 50 poses of
`exp_data/amide_synthetase/results/modeling/boltz2/step2_1mg/results/AS9/boltz_results_step2_AS9`,
the ensemble the r2 seed poses were drawn from. 50/50 parsed. The acyl carbon is located
topologically (unique P-O-C(=O)); the attacking N is whichever HMDA nitrogen is nearer it.

## The two criteria are mutually exclusive on this ensemble

| criterion | passing poses |
|---|---|
| original eligibility: `d_acyl < 4 A` AND angle 90-120 deg | 4/50 - models 0, 1, 3, 11 |
| orientation: `d_acyl < d(N, nearest phosphate O)` | 11/50 - models 4, 8, 10, 12, 30, 38, 40, 41, 43, 44, 47 |
| **both together** | **0/50** |

**Not one pose in the ensemble satisfies both.** Re-filtering therefore cannot produce a usable
seed; the ensemble has to be regenerated.

## Why they are mutually exclusive

Sorting the eight closest poses by `d_acyl` shows a clean anti-correlation between orientation
and angle:

| model | d_acyl | d_phosphate | angle | orientation |
|---|---|---|---|---|
| 41 | 3.02 | 3.51 | 44.2 | on electrophile |
| 1 | 3.45 | 2.65 | 94.6 | **phosphate** |
| 8 | 3.50 | 4.33 | 84.1 | on electrophile |
| 3 | 3.63 | 2.48 | 96.0 | **phosphate** |
| 0 | 3.68 | 2.97 | 97.5 | **phosphate** |
| 12 | 3.69 | 4.74 | 62.8 | on electrophile |
| 11 | 3.88 | 2.60 | 105.7 | **phosphate** |
| 4 | 4.10 | 4.97 | 59.8 | on electrophile |

Every phosphate-parked pose lands at 94.6-105.7 deg, inside the window. Every
electrophile-oriented pose lands at 44.2-85.0 deg, below it. On this ensemble the
**90-120 deg window is a phosphate-proximity detector, not an attack-geometry detector** --
which is the complete mechanism behind the seed-selection failure documented in
`MOTIF_DIAGNOSIS_AND_FIX.md`.

Note what this also means: **the ensemble contains no genuine near-attack conformer at all.**
The best correctly-oriented pose, model 41, is 3.02 A from the acyl carbon but at 44.2 deg --
an edge-on approach roughly along the carbonyl axis, not a Burgi-Dunitz trajectory
(~105 deg). The phosphate-parked poses only look Burgi-Dunitz-like by coincidence. So neither
subset was ever a productive starting point.

## Consequences for the corrected round

1. **Regenerate, do not re-filter.** 0/50 is unambiguous.
2. **Regenerate at the corrected charge.** The source paired adipoyl-AMP at **-2** with HMDA at
   **+1**; that electrostatic pairing is what drives the amine onto the phosphate. The acyl
   phosphate carries one ionizable oxygen (the other three are the C5' ester, the acyl
   anhydride and P=O) and is fully ionized at pH 7 (-1); the distal adipate carboxylate
   (pKa 4.43/5.41) is also deprotonated (-1). So **-2 is correct** and is not the thing to
   change. HMDA at +1 is the *reactive* microstate (both amine pKa ~10-11, so +2 dominates at
   pH 7 but cannot attack), so +1 is also defensible. The charge state is right; the problem is
   that correct electrostatics plus an unanchored amine gives a salt bridge.
3. **The amine needs a protein anchor, not a restraint.** This is the same conclusion the
   contact analysis reached independently: four basic residues within 8.6 A of the cationic
   terminus and no carboxylate inside 9.44 A. Until an anchor exists, any ensemble will put the
   +1 amine on the -2 phosphate, because that is the only anion available to it.
4. **A regenerated ensemble must carry the pocket restraint.** The source used a 19-residue
   HMDA pocket constraint (max_distance 10 A); the `amidation_20261027` and
   `amidation_q2_20261027` evaluation arms do **not**, which is why unrestrained wild-type AS9
   puts HMDA ~22 A away. An unrestrained run will not yield seeds.

## Recommended seed regeneration

AS9 + adipoyl-AMP (**-2**) + 1 Mg + HMDA (**+1**), with the original 19-residue HMDA pocket
constraint re-applied, 50 poses, then filtered on **both** criteria at once plus a
Burgi-Dunitz window centred near 105 deg rather than the 90-120 deg band. If that still returns
0/50, the conclusion is that AS9's pocket cannot hold the diamine productively without first
installing the anchor -- in which case the anchor mutation has to precede seed generation, not
follow it.
