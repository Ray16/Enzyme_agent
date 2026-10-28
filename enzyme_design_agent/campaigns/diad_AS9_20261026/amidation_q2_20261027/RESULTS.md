# Charge-corrected (-2) arm: the charge confound is REFUTED

Run 2026-10-27 after establishing that **-2 is the chemically correct state** and that the
neutral model used in `amidation_20261027` was simply wrong. Same six genes, same MSAs, same
seeds, same settings; the only change is adipoyl-AMP `0 -> -2`. HMDA stays at `+1`, the
reactive microstate, matching the source ensemble.

## Why -2 is correct

The acyl-adenylate phosphorus carries **one** ionizable oxygen -- the other three positions are
the C5' ester, the acyl anhydride and the P=O -- and an acyl phosphate is a strong acid, fully
ionized at pH 7 (-1). The distal adipate carboxylate (adipic acid pKa 4.43 / 5.41) is also
deprotonated at pH 7 (-1). Total **-2**, which is exactly what the source ensemble recorded
(`encounter_audit.json`, `chemical_atom_mapping.L.formal_charge = -2`).

HMDA's two amines have pKa ~10-11, so **+2 dominates at pH 7** -- but the di-cation has no lone
pair and cannot attack. `+1` is the reactive microstate, so the campaign's choice stands.

## Result: no effect

| gene | neutral med d2 | **-2** med d2 | delta | neutral fc2 | -2 fc2 |
|---|---|---|---|---|---|
| WT_AS1 | 30.24 | 29.99 | -0.26 | 0/50 | 0/50 |
| WT_AS9 | 22.56 | 22.56 | -0.00 | 0/50 | 0/50 |
| WT_AS9_rep2 | 22.31 | 22.54 | +0.24 | 0/50 | 0/50 |
| pose0_fixed_cap | 8.82 | 8.80 | -0.02 | 0/50 | 0/50 |
| pose0_internal_cap | 8.91 | 8.94 | +0.04 | 0/50 | 0/50 |
| pose3_internal_cap | 8.62 | 8.72 | +0.10 | 0/50 | 0/50 |

Largest charge-arm shift: **0.26 A**. Replicate floor: 0.25 A (neutral arm), 0.01 A (-2 arm).
`fc2` is **0/50 in all twelve structures**. Every shift is at or below replicate noise.

**The 22 A wild-type HMDA site and the 8.6-8.9 A design plateau are not charge artifacts.**
The confound flagged in `amidation_20261027/RESULTS.md` is refuted, and the step-2 conclusion
stands unchanged at the correct charge state: no design advances.

## The charge really was an input to the model

Checked rather than assumed, because a null is only meaningful if the input changed:

- The RDKit `mols` object differs between arms and carries the charge correctly
  (`total_formal_charge = -2`, negatives on O17 and O27 vs none in the neutral arm).
- The processed `structures/*.npz` `atoms` dtype is `('name','coords','is_present','bfactor','plddt')`
  -- **no charge field**. The 29 atoms that differ there differ only by auto-renumbered *names*
  (N43 -> N41 etc.), a consequence of the deprotonated form carrying fewer hydrogens. That path
  does not carry charge.
- But `boltz/data/feature/featurizerv2.py:1216` does:
  `token_atoms_charge = np.array([a.GetFormalCharge() for a in token_atoms_ref])`, concatenated
  into `ref_charge` and passed to the model (line 1548). Charge reaches the network from the
  RDKit mols, not from the structure npz.

So the two arms genuinely differed in model input and the null is a real null, not an identity.

## The limitation this exposes, which matters more than the result

`ref_charge` is a **reference-conformer** feature describing the ligand's own atoms. Boltz-2 has
no explicit electrostatics term, so "the model is given the formal charge" is not the same as
"the model represents charge-driven intermolecular binding". Two consequences, both of which
should temper the plan:

1. The **phosphate parking** seen in the source ensemble may not be electrostatic at all. If
   Boltz cannot express a salt bridge, that contact is more likely a shape/packing outcome, and
   calling it a "salt bridge" over-reads it. The *observation* -- amine nearer the phosphate
   than the electrophile in 4/4 selected poses -- is unaffected; the *mechanism* is not
   established.
2. More seriously for the next round: **the anchor hypothesis cannot be tested by Boltz
   cofolding.** Installing Ser496Asp/Glu to hold the cationic amine is precisely a
   charge-complementarity hypothesis, and this run shows a two-unit change in ligand charge
   moves nothing measurable. A tool that is insensitive to a -2/0 swap on the ligand should not
   be expected to report whether a new carboxylate retains an amine.

That argues for validating the anchor with a method that has real electrostatics -- the MLMM
pipeline already in this repo, or at minimum a BioEmu2 retention run seeded from the anchored
pose -- rather than by another round of cofold scoring.

## Standing caveats

- Mg is present in all twelve structures; its absolute effect is untested by construction.
- Neither axis predicts a rate; no step-2 metric here has been validated against a measurement.
- `ame.pass` is still uncomputed for all four designs.
