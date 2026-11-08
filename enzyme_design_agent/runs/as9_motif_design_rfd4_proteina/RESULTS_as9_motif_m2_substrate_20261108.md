# AS9 M2 substrate-pocket motif — refold panel results (2026-11-08)

**Campaign:** `as9_motif_m2_substrate_20261107` · 100 RFD4-Proteina designs (516 aa,
poses 15+6 × 50 draws) × 50 independent Boltz-2 refolds = 5000 refolds, **0 scoring errors**.
Pipeline: FP32 motif-conditioned generation → fresh batched MSAs → Boltz-2 refold
(stock_fp32_serial, 50 poses, no templates/restraints) → attack geometry from refolds only.
Chemistry: ATP⁴⁻ + Mg²⁺ + adipate²⁻ (charge shown not to matter — see M0/M1 results).

## Verdict: NULL. Conditioning the substrate pocket did not move docking toward WT.

The M2 hypothesis was that M0/M1 failed because the substrate subsite was never
conditioned. M2 added it (20 residues, 64 atoms, 17 sidechain+sequence-fixed). It made
no difference to docking, and the designs reproduce the larger motif *less* faithfully.

### Docking is statistically indistinguishable from the M0/M1 panel
| arm | n | A (°) | d_h (Å) | frac d<4Å | near-attack poses |
|-----|---|-------|---------|-----------|-------------------|
| M0 | 50 | 93.9 ± 22.5 | 9.47 | 6.2% | 32/2500 (1.3%) |
| M1 | 50 | 87.8 ± 20.9 | 8.75 | 5.0% | 30/2500 (1.2%) |
| **M2** | **100** | **93.2 ± 22.9** | **9.35** | **4.9%** | **47/5000 (0.9%)** |
| *WT_AS9 (charged anchor)* | — | *156.6* | *4.24* | *28%* | — |

**M2 vs M0/M1 pooled:** A +2.35° (MWU p=0.46, below the 2.67° noise floor),
d_h +0.23 Å (p=0.44), frac d<4Å −0.7% (p=0.70). All null. M2 still docks the
substrate ~90°/9 Å off-axis, exactly where M0/M1 did, nowhere near the WT anchor.

### The deeper limit the M1/M0 panel predicted
The M2 plan pre-registered the falsifier: *"If M2 also collapses to ~90°/9 Å, the
limitation is deeper than motif coverage — the scaffold cannot hold the two subsites
in register."* It collapsed. The decisive evidence is motif reproduction:

| arm | median bb RMSD (Å) | designs reproducing motif <1.5 Å | their d_h (Å) |
|-----|--------------------|----------------------------------|---------------|
| M0 | 1.09 | 10/50 | 8.17 |
| M1 | 1.29 | 5/50 | 8.77 |
| **M2** | **1.54** | **0/100** | **—** |

**0/100 M2 designs reproduce the full 20-residue motif within 1.5 Å** (vs 10/50 for
M0's compact 12-residue ATP motif). Loosening to <2.0 Å gives 6/100, and *those* still
dock at d_h 10.55 Å — no better. Adding the substrate pocket did not let the generator
grip the substrate; it gave it a second subsite it could not satisfy simultaneously
with the ATP machinery. (Caveat: M2's motif spans 64 atoms vs M0's 31, so a larger
all-atom RMSD is partly mechanical. But the docking readouts are the primary metric and
they are flatly null regardless — the fidelity drop only explains *why*.)

**Conclusion:** The bottleneck is not which residues are conditioned. RFD4-Proteina
cannot co-register two spatially separated functional subsites (the ATP triphosphate
clamp and the adipate carboxylate pocket) into a single catalytically poised scaffold
for this 516-aa enzyme. The substrate-coverage hypothesis is refuted.

### One honest bright spot (noise-level, not a trend)
`AS9D_21957b74d5eb` (M2_pose6) is the single design that both reproduces the motif
(1.56 Å) and docks well: A 143.4°, d_h 4.28 Å, 42% of poses d<4Å, **12/50 near-attack
poses**, min distance 3.14 Å at 146°. This is a genuine candidate — but it is 1/100,
consistent with the 2–4/100 WT-distance hit rate already seen in the M0/M1 null. It is
a lottery winner, not evidence the motif works.

## Provenance
Distinct new campaign; the M0/M1 campaign at `as9_motif_m0_m1_20261106` is untouched.
Generation 100/100 (0 failures), refold 100/100 scored (0 errors), 11760 s wall.
Analysis: `analyze_m2.py` (tracked copy in `as9_motif_design_rfd4_proteina/`).
