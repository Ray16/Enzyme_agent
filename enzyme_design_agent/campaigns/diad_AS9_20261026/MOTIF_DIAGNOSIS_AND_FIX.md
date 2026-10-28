# Why the r2 motif could not work, and what to change

Written 2026-10-27, after the step-1 review and before any new generation round. Every number
below is measured from files in this campaign; nothing here is an activity claim.

## Summary

The r2 generation round was sound in its bookkeeping and wrong in three specific, fixable
ways. All three are properties of the **motif**, not of RFD4 or of the sampling.

1. The pose-selection filter systematically preferred poses in which the attacking amine is
   **salt-bridged to the phosphate** rather than poised at the electrophile.
2. The protected set froze the only three residues that could have anchored the amine.
3. The run was a tight local refinement (70.8% of protein atoms coordinate-fixed) sampled
   **once per input**, which is not enough sampling to find anything.

## 1. The selection filter chose salt bridges

Eligibility was "neutral N-C distance < 4 A and N-C-O angle 90-120 deg", which passed 4 of
50 source poses. Measuring the amine's nearest contact in all five sub-4-A poses:

| source model | d(N, acyl C) | N-C-O angle | d(N, O2P) | nearest contact |
|---|---|---|---|---|
| 0  | 3.68 | 97.5  | 2.97 | **phosphate** |
| 1  | 3.45 | 94.6  | 2.65 | **phosphate** |
| 3  | 3.63 | 96.0  | 2.48 | **phosphate** |
| 11 | 3.88 | 105.7 | 2.60 | **phosphate** |
| 12 | 3.69 | 62.8  | 5.12 | electrophile |

All four poses that **passed** the angle window have the amine closer to a phosphate oxygen
than to the carbon it is supposed to attack. The one pose where the amine is genuinely
oriented at the electrophile, model 12, is the one the filter **rejected**, on an angle of
62.8 deg. Medoids of the four passing poses gave models 0 and 3, which became all four r2
inputs. (`encounter_audit.json` also carries a stale pre-filter field
`two_medoid_models: [0, 12]`; the inputs built were 0 and 3, matching
`encounter_selection.json`. The stale field should be dropped or renamed.)

Mechanistic cause: the source cofold modelled adipoyl-AMP as a **dianion** (recorded formal
charge -2) against a **mono-cationic** HMDA (+1). A +1 amine placed next to a -2 phosphate
salt-bridges, and sitting on the phosphate incidentally puts the N-C(acyl)-O angle inside the
90-120 deg window. The diagnostic therefore cannot separate "poised to attack" from "parked
on the adjacent anion", and under that charge state the parked geometry dominates 4:1.

Note also that the superseded **r1** inputs had the amine correctly oriented
(d(N,O2P) 4.48 vs d(N,acyl C) 3.00). The r2 "pose hypotheses" round replaced correctly
oriented seeds with salt-bridged ones.

**Fix:** add an orientation criterion to eligibility -- require
`d(N, acyl C) < d(N, nearest phosphate O)` -- and re-derive the seed poses from a source
ensemble whose adipoyl-AMP charge state is stated and defended. Do not rely on the angle
window alone; it is satisfied incidentally by the failure mode.

## 2. The protected set froze the amine site

Residues within 5.0 A of HMDA, with their status in the r2 motif:

| res | aa | d(HMDA) | d(cationic N) | r2 status | has another role? |
|---|---|---|---|---|---|
| 205 | MET | 1.50 | 2.08 | mutable | - |
| 495 | ALA | 2.58 | 2.58 | **protected** | **none** |
| 496 | SER | 2.75 | 2.82 | **protected** | **none** |
| 206 | PRO | 2.89 | 2.89 | mutable | - |
| 209 | HIS | 3.24 | 5.68 | protected | catalytic, keep |
| 210 | ILE | 3.39 | 9.27 | mutable | - |
| 211 | GLY | 3.41 | 7.98 | mutable | - |
| 251 | VAL | 3.55 | 4.79 | mutable | - |
| 212 | GLY | 4.12 | 7.50 | mutable | - |
| 497 | GLY | 4.13 | 4.13 | **protected** | **none** |
| 498 | LYS | 4.13 | 6.78 | protected | AMP contact 2.17 A, keep |
| 249 | PHE | 4.33 | 7.23 | mutable | - |
| 166 | SER | 4.36 | 8.77 | protected | A3 loop, keep |
| 306 | THR | 4.41 | 11.85 | protected | 301-307 block, keep |
| 279 | GLY | 4.90 | 8.79 | protected | 279-281 block, keep |

The C-terminal domain was protected **110 residues wide** (407-516) to protect what turns out
to be two genuine contacts: Lys498 (AMP, 2.17 A) and Arg407 (AMP, 3.55 A). The collateral
cost was Ala495, Ser496 and Gly497 -- the three closest residues in the entire protein to the
diamine's cationic terminus, with no other assigned role.

And the site the amine must occupy is electrostatically hostile to it. Within 8.6 A of the
cationic terminus: Lys498 (6.78), Arg257 (6.91), Arg90 (7.04), Arg493 (8.58). The nearest
carboxylate is **Asp231 at 9.44 A**. Four basic residues close in, zero acidic within 9 A.
The one acidic residue the designs installed, S311E, sits 9.14 A away on the adipoyl side and
cannot anchor anything. This is a sufficient explanation for the diamine not being retained,
and it is consistent with the regio-campaign finding that the donor axis is functional class
rather than pocket volume -- which is also why the Ala/Gly cavity enlargement that the
fixed-cap design chose (N185A, M205A, A215G, V251G, C309A) could not have helped.

**Fix:** move 494-497 into the mutable set; keep 498 and 407 protected (they bind AMP), keep
His209, the A3 loop, 279-281, 301-307 and 392 protected. Ser496 is the natural position to
try an Asp or Glu: it already points at the cationic amine at 2.82 A. Consider Arg493 (8.58 A)
as a second-shell liability. Protect the remainder of 408-492 and 499-516 as before -- keeping
bulk protected is harmless; freezing the pocket lining is not.

## 3. It was a local refinement sampled once

Coordinate conditioning in the r2 inputs, from `condition_CoordinateGroup_1_atom`:

| group | atoms | share | meaning |
|---|---|---|---|
| 0 | 2725 | **70.8%** | coordinates fixed |
| 1 | 420 | 10.9% | cap, rigid-body mobile |
| -1 | 704 | 18.3% | coordinates free |

For comparison, the enzyme-design tutorial's theozyme fixes **16 of 196 protein atoms
(8.2%)** and invents 180 residues of scaffold from noise. The r2 round was therefore not
scaffolding but a tightly constrained local search -- a legitimate mode, and it behaved
accordingly: 6-7 point mutations per design, and the two fixed-cap inputs converging on an
identical sequence.

The decisive problem is sampling depth. `aurora_design.py` records `expected_designs=4`:
**one replicate per input.** The tutorial uses `num_replicates=25` for a single theozyme and
states that de novo scaffolding has the lowest pass rate of its three modes. Four structures,
one per input, cannot distinguish a design strategy from a draw.

**Fix:** 25+ replicates per input, and treat the four r2 structures as a pilot, not a panel.

## 4. The AME criteria were never computed

`generated/structural_qc.json` checked backbone integrity, ligand identity, charge, carbon
stereochemistry, peptide connectivity, protected-sequence preservation and a 1.5 A
protein-ligand clash test. That covers the **third** AME criterion only. Never computed:

- `refolded_backbone_rmsd <= 2.0 A`
- `refolded_motif_allatom_rmsd <= 1.5 A`

Both require refolding each design **from sequence alone** with an independent oracle and
asking whether the generated structure, and specifically the catalytic atoms, reproduce. The
design-side motif fidelity is known and excellent (joint fit of the 2752 fixed atoms,
RMSD 0.061-0.065 A, max error 0.31 A), but that measures input-vs-generated, not whether the
sequence actually encodes the structure. So `ame.pass` is unknown for all four designs.

RFD4-Proteina here is an Aurora XPU port under PBS
(`/lus/flare/projects/FRAME-IDP/rzhu/rfd4_proteina_20261026`); the AME refolding oracle is a
CUDA/apptainer container Aurora cannot run. Any future round is therefore inherently split:
**generate on Aurora, refold and score on lambda.**

## Recommended order

1. Score the existing four designs against the AME refold criteria. Cheap, and it gates
   everything else: if they fail `refolded_motif_allatom_rmsd`, the geometry argument above
   and the step-2 result are both moot.
2. Re-derive seed poses with the orientation criterion of section 1.
3. Re-cut the motif per section 2, raise replicates per section 3, and run partial diffusion
   from the AS9 parent.

## Known limitation of the step-2 run launched today

`amidation_20261027` models adipoyl-AMP as the **neutral** microstate, whereas the source
ensemble that produced the seed poses used **-2**. The state is identical across all six jobs,
so the design-vs-WT contrast is internally valid, but it is not matched to the generation
source. Since phosphate charge is exactly what drives the salt-bridge artifact in section 1,
a `-2` arm would be the clean way to test whether the parking is real or a charge artifact.
That arm has not been run.
