# AS9 motif-conditioned design — campaign reasoning log

One entry per campaign: the hypothesis, the motif rationale, the test, and the
outcome. All campaigns use RFD4-Proteina motif-conditioned generation of 516-aa
AS9 variants (checkpoint `a06e2509…`), then independent Boltz-2 refolding (fresh
batched MSAs, 50 poses, no templates/restraints, FP32), and attack geometry
measured **only** from the refolds. Chemistry: ATP⁴⁻ + Mg²⁺ + adipate²⁻
(the charge state was shown not to matter — see note at end).

Shared frame: the enzyme is a step-1 adenylating enzyme. The reaction needs the
adipate distal carboxylate oxygen to attack ATP's α-phosphorus (Pα) in-line
(~180°, <4 Å) displacing the α/β-bridging oxygen (O3A/O28). The proximity-weighted
attack angle **A**, harmonic distance **d_h**, and near-attack fraction (d<4 Å,
θ>150°) are the readouts. WT_AS9 anchor under this pipeline: A≈157–165°, d_h≈4.24 Å,
~20–28% near-attack. Measured ΔA noise floor SD = 2.67° — gains under ~5° are not real.

---

## M0 — baseline ATP-machinery motif (control)
**Campaign:** `as9_motif_m0_m1_20261106` · **Status:** complete (n=50)

**Reasoning:** Condition only the catalytically essential ATP/Mg apparatus and let
RFD4 design everything else. The minimal test of "does motif-conditioned generation
reproduce a working adenylation site?"

**Motif (12 residues, 31 atoms, all sequence-fixed):**
- 165–173 (T-S-G-T-T-G-A-A-K) — the phosphate-binding loop gripping the ATP triphosphate
- Q301, D392 (Mg-coordinating Asp), K498 (conserved Lys at the Pα/substrate interface)

**Role:** The control arm. Everything M1/M2 is measured against.

**Outcome:** Designs fold well (median backbone RMSD 1.17 Å) but dock the substrate
off-axis: A≈94°, d_h≈9.5 Å, ~6% near-attack — far from the WT anchor. Not a folding
or scoring artifact.

---

## M1 — baseline + weak substrate gesture
**Campaign:** `as9_motif_m0_m1_20261106` · **Status:** complete (n=50)

**Reasoning:** Same ATP machinery as M0, plus a first attempt at acknowledging the
substrate loop near residues 310/311.

**Motif (14 residues, 39 atoms):** M0 set + 310/311 added **backbone-only**
(N,CA,C,O), sequence **not** fixed.

**Outcome / lesson:** M1 was statistically indistinguishable from M0 (attack angle
p=0.078, docking p=0.299, near-attack p=0.833) and reproduced its own conditioned
motif slightly *worse*. Diagnosis: 310/311 were the only substrate-contacting
residues in the motif, but conditioning them backbone-only with free sequence
cannot position a sidechain to grip the adipate carboxylate. M1 gestured at the
right subsite with conditioning too weak to do anything.

**Decisive diagnostic (whole panel, n=100):** the 19 designs that reproduce the
conditioned motif faithfully (<1.5 Å) *still* dock at d_h 6.94 Å / A 98°, only 5%
reaching WT distance. So the failure is not "the motif floats" — it is that the
conditioned residues do not span the catalytic constraint. The substrate-positioning
subsite was never conditioned: in the WT pose every M0/M1 motif residue except K498
sits 8–15 Å from the adipate.

---

## M2 — substrate-pocket motif (the fix M1/M0 implied)
**Campaign:** `as9_motif_m2_substrate_20261107` · **Status:** generating (target n=100)

**Reasoning:** The M0/M1 panel pinned ATP tightly and left the substrate unanchored,
so scaffolds build a good ATP site and park adipate off-axis. Fix the *coverage*,
not the *count*: condition the residues that actually line the adipate nucleophile
carboxylate in the WT holo pose, with **sidechains and sequence fixed**, so the
generator is forced to build a carboxylate-gripping subsite. This is a targeted
addition (~20 residues, 64 atoms, ~5% of a 516-aa protein) — still sparse motif
scaffolding, and a foldable solution provably exists (the WT itself). Over-constraint
was ruled out: the genuine problem was under-constraint of the substrate.

**Motif (20 residues, 64 atoms, 17 sequence-fixed):**
- All of M0 (ATP machinery: 165–173, Q301, D392, K498) — kept
- **Substrate-gripping, sidechain + sequence fixed** (contact adipate 2.9–4.5 Å in
  both WT poses): N185 (polar), I210 (hydrophobic wall), F249 (aromatic wall),
  T306 (polar), **S311 upgraded from M1 backbone-only to sidechain+sequence**
- **Glycine pocket-floor shapers, backbone-only, identity free:** G279, G304, G310

**Residue selection:** every chain-A residue with an atom within 5 Å of the adipate
(and ≤6 Å of the reactive carboxylate head) in `WT_AS9_model_15.pdb`
(sha256 5afe833…), verified present with matching identity in both poses 15 and 6.

**Test:** compare M2 (n=100, poses 15+6) against the existing M0/M1 panels under
identical chemistry and pipeline. Hypothesis: if conditioning the substrate subsite
is what was missing, M2 should raise the near-attack fraction / harmonic docking
toward the WT anchor. If M2 also collapses to ~90°/9 Å, the limitation is deeper
than motif coverage (e.g. the scaffold cannot hold the two subsites in register).

**Provenance guard:** M2 is a **distinct new campaign**; the M0/M1 campaign at
`as9_motif_m0_m1_20261106` is untouched and its results are not M2's.

---

### Cross-campaign note: ligand charge state is not a confound
The Oct-27 WT anchor used CCD ATP + neutral adipate; the campaigns use ATP⁴⁻/adipate²⁻.
Re-running both WTs under the charged chemistry moved them only ~8° in angle and
<0.08 Å in distance (WT_AS9 164.7→156.6°, WT_AS1 143.8→135.8°). Charge state does
not expel the substrate; the design docking failure is real. (A separate house
decision is to use neutral **CCD ATP** + charged adipate SMILES for in-distribution
Boltz folding; the scorer identifies Pα/bridge by P–O–C / P–O–P topology, so the
geometry measurement is charge-independent either way.)
