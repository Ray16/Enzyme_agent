# QM/MM Activation-Barrier Protocol (UMA + PySCF)

**Objective:** a *reproducible computational method* for enzymatic reaction barriers of ATP-dependent
amide/acyl chemistry — not four barrier numbers. Validated on GlnA3 (Q9KZC7) with AS9 as an independent
reference, GlnA2 (Q9RDS6) as a budget-limited substrate-specificity comparison.

**Three quantities, reported separately** (one does not establish another):
1. **Reaction free energy ΔG°** per chemical step (is the step thermodynamically favorable?).
2. **Activation free energy ΔG‡** per step (free-energy cost to the TS).
3. **Catalytic effect** = enzyme ΔG‡ − *solution-phase* ΔG‡ for the same reaction. **Required** to claim
   GlnA3 catalyzes the non-native glutarate reaction; a plausible enzyme-bound barrier alone does NOT.

**Pathway-first.** Establish a chemically valid reactive complex + mechanism BEFORE any TS optimization
or production PMF. A relaxed-scan or PMF maximum is NOT automatically a transition state.

---

## Enzyme roster & budget

| Enzyme | Role | Budget |
|---|---|---|
| GlnA3 (Q9KZC7) | Primary GS-fold acyl-phosphate mechanistic study; native glutamate vs non-native glutarate | Full |
| AS9 | Independent adenylation (acyl-AMP) transferability reference; adipate + glutarate | Full (reuse existing MD prep) |
| GlnA2 (Q9RDS6) | Comparative substrate-specificity (may be the better cadaverine enzyme) | **Secondary — pilot only** until its active-site geometry + metal coordination are shown to support the chemistry; only then promote |

**Order:** (1) this protocol → (2) GlnA3 native + glutarate pilots, both metal configs → (3) GlnA2
comparative pilot → (4) AS9 revisit under the same framework.

---

## Reactions, steps, and collective variables

Compute **BOTH steps** for each enzyme — the rate-limiting step can switch glutamate→glutarate, and human
GS QM/MM found step-2 rate-limiting (19.2 kcal — a mechanistic reference, not a numerical target).

**GlnA3 / GS-fold (acyl-phosphate), 2 or 3 Mg:**
- Step 1 — γ-acyl-phosphate formation: carboxylate O attacks ATP Pγ; O3β leaves to ADP.
  CVs (2D-capable): ξ₁ = r(O_carb–Pγ) [forming], ξ₂ = r(Pγ–O3β) [breaking]. Monitor ξ₁+ξ₂ for
  associative vs dissociative character; collapse to 1D antisymmetric only if it demonstrably captures
  the mechanism.
- Step 2 — aminolysis: cadaverine N attacks the acyl-phosphate carbonyl C; phosphate leaves; amine
  deprotonation. CVs: r(N–C_acyl) [forming], r(C_acyl–O_P) [breaking], **+ proton-transfer CV**
  (N–H → accepting phosphate O / catalytic base). Default 2D (bond + proton); amine pKa≈10–11 so the
  deprotonation penalty is part of the real barrier.

**AS9 / ANL (acyl-AMP), 1 Mg:**
- Step 1 — adenylation: carboxylate O attacks ATP Pα; PPi leaves. ξ₁=r(O_carb–Pα), ξ₂=r(Pα–O3A).
- Step 2 — aminolysis of the acyl-AMP (parallels GlnA3 step 2).

---

## Software stack (verified installed, Nov 10 2026)

| Task | Tool / env |
|---|---|
| Superpose crystal template, graft substrate | PyMOL (`envs/pymol`) |
| Parameterize / solvate (Amber) | tleap, parmchk2, antechamber (`envs/mlmm-toolkit`); **12-6-4 Mg** |
| Protonation states | pypka (`envs/pypka`) + manual active-site inspection |
| Classical MD (equilibration, replicate) | OpenMM (`Tool/envs/processivity-md-20261106`) |
| Reactive-region potential (sampling) | **UMA / fairchem** (`uma-s-1p2`) + ASE, `Tool/envs/processivity-qmmm-20261106` |
| DFT correction / single points | **pyscf** (same qmmm env); ωB97M-V/def2-TZVPD (match AS9 DFT-on-path) |
| Enhanced sampling | PLUMED OPES/umbrella (`Tool/envs/processivity-plumed-20261106`); reuse AS9 OPES drivers |

No turnkey sander-QM/MM or AM1/d-PhoT is installed. Method = **UMA as the reactive-region potential
embedded in explicit solvent** (cures the vacuum-cluster charge-screening artifact that gave the discarded
+55 kcal), with **pyscf DFT** as the accuracy check/correction. Report MM, UMA, and DFT energies distinctly.

---

## The 2-Mg vs 3-Mg problem — handle composition rigorously

2BVC (actinobacterial GS, TS-mimic) shows **three** Mg per site; canonical GS has two principal sites
(n1/n2); the cofold placed two. **Do NOT let a metal "appear" partway along a reaction coordinate.** Two
systems with different metal counts have **different composition** and their free energies are NOT directly
comparable.

Rules:
1. **Composition is fixed within any single pathway/PMF.** Run the 2-Mg system and the 3-Mg system as two
   *self-consistent* models, each with its own reactant reference.
2. **Model selection (pilot):** prefer the config that (a) keeps stable metal coordination through replicate
   equilibration, and (b) reproduces the crystallographic occupancy pattern (2BVC n1/n2/n3 geometry).
3. **Cross-composition comparison requires the Mg²⁺ chemical potential.** To place 2-Mg and 3-Mg ΔG‡ on a
   common free-energy scale — or to model the 3rd metal binding specifically at the TS/intermediate — compute
   the **Mg²⁺ binding/transfer free energy from bulk** (alchemical insertion / TI, or grand-canonical
   treatment) and add it explicitly. Never subtract barriers across compositions without this term.
4. Report every barrier with its metal composition stated.

---

## Gated execution plan (8 stages)

Each stage: **Procedure → Acceptance criteria → If it fails.** Do not advance a stage whose acceptance
criteria are unmet.

### Stage 1 — Structure preparation
**Procedure.** Superpose template **2BVC** (cross-check 1FPY) onto the GlnA3 structural model; transplant
ADP + metal cluster (n1/n2 for the 2-Mg model; n1/n2/n3 for the 3-Mg model) into crystallographic positions;
graft the substrate (P3S marks it: PA/O1A/O2A/O3A = transferring phosphoryl, NE = nucleophile locus,
CA/N/C/OT = glutamate backbone → glutarate keeps the activated carboxylate + chain, deletes the α-amino);
place cadaverine for step-2 systems; resolve protonation (pypka + manual on active-site Asp/Glu/His/Lys,
ATP phosphates, substrate); 12-6-4 Mg; solvate (TIP3P octahedron), neutralize.
**Acceptance.** tleap builds at the intended net charge; no bad contacts; protonation self-consistent and
documented; both 2-Mg and 3-Mg systems built for GlnA3.
**If it fails.** Re-examine pose/protonation; if a metal config cannot be built with sane coordination,
record it as a negative result and carry the other.

### Stage 2 — Chemical validation (mechanism)
**Procedure.** On the carved reactive region, UMA-optimize reactant, putative intermediate (acyl-phosphate /
acyl-AMP), and product; identify nucleophile, electrophile, bonds formed/broken, proton transfers, metal
coordination changes. Enumerate ≥1 alternative mechanism (e.g. stepwise vs concerted; which O leaves).
**Acceptance.** A connected reactant→intermediate→product sequence exists; the intermediate is a genuine
local minimum; metal coordination is chemically sensible at each stationary point.
**If it fails.** Revisit metal config / protonation / pose; test the alternative mechanism; do not force a
single assumed concerted path.

### Stage 3 — Pilot calculations
**Procedure.** Short independent equilibration MD (≥3 replicas, e.g. 3×20 ns) of each pilot system; then
relaxed scans along the Stage-1-step CVs in the solvated reactive region (UMA).
**Acceptance.** Metal coordination number stable (±~0.5) and substrate carboxylate retained in the inner
shell across replicas; relaxed scan yields a **finite, single-maximum** barrier with the intermediate as a
minimum (NOT monotonic, NOT >>25 kcal).
**If it fails.** Substrate drifts / coordination breaks → binding-pose or composition problem (that may itself
be the scientific answer for glutarate — record it); monotonic/absurd scan → CV or mechanism wrong, return to
Stage 2.

### Stage 4 — DECISION GATE
**Criterion.** Proceed to production free energy ONLY if (reactive complex stable, Stage 3a) AND (a
chemically plausible pathway with a finite barrier exists, Stage 3b), for a stated metal composition.
**If it fails.** Halt. Diagnose and loop back; do not spend production GPU on an unvalidated complex.

### Stage 5 — Free-energy profiles
**Procedure.** OPES/umbrella along the chosen CV(s) (2D where proton transfer couples); ≥3 independent
trajectories from independent starting configs; PMF via WHAM/MBAR; UMA potential in explicit solvent.
**Acceptance.** Adequate window/CV overlap; block-averaged statistical error ≲1 kcal; forward/reverse
hysteresis ≲2 kcal; replicate PMFs agree within error.
**If it fails.** Extend sampling; improve CV (add dimension); reseed windows from a maintained reactive
complex (this is where AS9's replicate divergence must be controlled).

### Stage 6 — TS validation
**Procedure.** Sample configs around the PMF maximum; pyscf DFT optimization where feasible; vibrational
analysis; committor analysis where feasible.
**Acceptance.** A single imaginary mode corresponding to the intended bond-making/breaking event; committor
≈0.5 at the putative TS.
**If it fails.** The CV does not capture the TS → add the missing coordinate and return to Stage 5. A PMF max
without a validated saddle is not reported as a barrier.

### Stage 7 — Method validation
**Procedure.** UMA-vs-pyscf-DFT single points along the minimum-energy path (characterize systematic offset;
reference-potential correct if needed); QM/reactive-region size test (e.g. 60–80 vs 120–150 atoms);
protonation-state sensitivity; **metal-count sensitivity (2 vs 3, with the Mg chemical-potential term)**;
alternative-mechanism comparison.
**Acceptance.** Barrier stable to region size within ~2–3 kcal; UMA–DFT offset characterized (systematic, not
erratic); conclusions unchanged by defensible protonation/metal choices, or the sensitivity is quantified and
reported.
**If it fails.** Report the sensitivity as an uncertainty; if UMA is erratic vs DFT on the charged TS, adopt
the DFT-corrected profile.

### Stage 8 — Comparative analysis
**Procedure.** For each enzyme/substrate and step, assemble ΔG°, ΔG‡, TS geometry/mechanism, metal &
proton-transfer details, replicate uncertainty, and sensitivity (region/Mg/protonation). Compute
**ΔΔG‡ = ΔG‡(glutarate) − ΔG‡(native)** (same mechanism/model/protocol so systematic error cancels) and the
**enzyme-vs-solution catalytic effect** (requires the solution-phase PMF of the same reaction).
**Acceptance (three validity milestones).** (1) chemical validity — correct reactive complex + mechanism;
(2) free-energy validity — reproducible converged PMFs with identified, validated TS; (3) predictive validity
— reproduce a reference reaction AND predict the relative barrier for the non-native substrate.

---

## Reproducibility requirements
Every system: record net charge, metal composition, protonation choices, force field + 12-6-4 set, water
model, random seeds, UMA model id (`uma-s-1p2`), DFT functional/basis, CV definitions, OPES/umbrella
parameters, and env paths. Separate **statistical** (replicate/block) from **systematic** (region/Mg/
protonation/UMA-vs-DFT) uncertainty in all reported barriers. State metal composition next to every ΔG‡.

## Explicitly out of scope
Vacuum-cluster electronic-energy barriers; mechanical-embedding ONIOM; reporting a scan/PMF maximum as a
validated TS; cross-composition barrier comparisons without the Mg chemical-potential term; all 59 candidates
(this milestone is method-building on GlnA3/AS9/GlnA2).
