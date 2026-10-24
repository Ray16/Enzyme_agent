# Pre-registration — can the proximity-weighted attack angle `A` separate active from inactive in the regioselectivity paper?

**Written 2026-10-21, BEFORE any structure was folded or scored.** Nothing below may be changed
once folding starts; deviations go in a dated `DEVIATIONS` section at the bottom.

Paper: `paper/Regioselective biosynthesis of oligoamides as precursors for sequence-controlled
co-polyamides.pdf` (bioRxiv 2025.11.03.685623, posted 2025-11-03).
Experimental labels: `exp_data/amide_synthetase/literature_activity/regio_fig2/` (digitized,
independently re-verified → `review.md` verdict **ACCEPT**).

---

## 1. The metric is frozen, not fitted

`A` = `structural_modeling_agent/proxies/attack_angle.py`, `METRIC_ID =
inverse_distance_attack_angle_v1`:

```
A   = Σ(angle_i / d_i) / Σ(1 / d_i)          # inverse-distance-weighted mean attack angle
D_h = n / Σ(1 / d_i)                         # harmonic mean distance, reported alongside
```

per enzyme over its **50-pose** Boltz-2 ensemble, where for each pose
- `d_i` = distance from the nearest substrate **carboxylate oxygen** to ATP **Pα**
- `angle_i` = O(nucleophile)–Pα–O3A angle (O3A = the leaving-group axis, AMP transfer)

No threshold, no gate, no fitting enters `A` — that is the whole point of the inverse-distance
form. `n_gated` (poses with d < 4.0 Å) is reported **only as a validity precondition**, never as a
ranking input. **One metric, one atom map, no variants will be swept.** If `A` fails we report that
it failed; we do not go looking for a descriptor that works (see the six already-null proxy
families in this repo).

## 2. What `A` structurally cannot see — stated up front

`A` scores **step 1 only**: activation of the *carboxylic-acid donor* as the acyl-adenylate. It has
**no amine dimension** (established 2026-10-10). Consequences, accepted in advance:

- Two Fig-2 panels that share an acid donor receive **identical** `A` by construction.
  Panels **C** (H+H→HH) and **D** (M+H→MH) both activate **H**, so `A` cannot explain the C-vs-D
  contrast. This is a known blind spot, not a result.
- The live question is therefore narrower and honest: **does acid-side activation geometry alone
  recover the paper's enzyme ordering?**

## 3. Design — 18 cofolds

6 enzymes × 3 unique acid donors. `Control` is a no-enzyme well and is **not modelable** — excluded.

| Enzyme | Source | Family |
|---|---|---|
| DdaG | UniProt E2JA32 (reused verbatim from the 49-enzyme panel) | ANL |
| SfaB | UniProt A0A2H4T912 (reused verbatim) | ANL |
| AsbA | UniProt A0A0F7RJ52 (Swiss-Prot, spermidine–citrate ligase) | NIS type A |
| AsbB | UniProt Q81RQ8 (Swiss-Prot) | NIS type C |
| AcsA | UniProt E0SIF5 (achromobactin NIS synthase) | NIS type B |
| DesD | **PDB 7TGM entity 1, His-tag stripped**, *S. griseoflavus*, 592 aa | NIS type C |

> **DesD identity is a validated call, not a name match.** The paper quotes catalytic
> **His153/His440/Arg303**. Those indices land on H/H/R in the 7TGM mature sequence **exactly**, and
> on T/Y/G in the *S. coelicolor* DesD (Q9L069) — so the obvious `gene:desD` hit is the **wrong
> protein**. Recorded in `sequences/enzymes_provenance.json`.

Acid donors (SMILES from the canonical `substrate_reference.csv`, deprotonated at pH 7):

| Code | Acid | Used by Fig-2 panel |
|---|---|---|
| A | adipic acid (C6 diacid) | A — X+A→XA |
| Z | azelaic acid (C9 diacid) | B — M+Z→MZ |
| H | 6-aminocaproic acid (ω-amino acid) | C — H+H→HH, and D — M+H→MH |

Cofold composition is held **identical to the existing campaign** for cross-campaign
comparability: chain A = protein + precomputed MSA, B = ATP, C = one Mg²⁺, D = acid donor
(deprotonated), `affinity: binder D`.

## 4. Labels — fixed now, read off the verified digitization

Active = bar rises above LOD in that panel (`below_LOD == False`); for acid **H** a cell counts as
active if it is active in panel C **or** D.

| Acid | Active | Inactive |
|---|---|---|
| A | DdaG, SfaB, AsbA, AsbB, AcsA, DesD (**all 6**) | — |
| Z | SfaB, AcsA, DesD | DdaG, AsbA, AsbB |
| H | DesD | DdaG, SfaB, AsbA, AsbB, AcsA |

Pooled: **10 active / 8 inactive** over 18 cells.

## 5. Hypotheses and the tests that decide them

**Primary (confirmatory), one-tailed — `A` higher in actives:**

- **T1 — acid Z, within-stratum.** Rank the 6 enzymes by `A`; do the 3 actives occupy the top 3?
  Exact p = 1/C(6,3) = **0.0500** if perfectly separated.
- **T2 — acid H, within-stratum.** Is DesD ranked 1st of 6 by `A`? Exact p = 1/6 = **0.1667**.
- **T3 — Fisher combination of T1 and T2.** Ceiling **p = 0.0482**.

T1/T2 are **within-acid**, which is the point: the pooled labels are *confounded with acid
identity* (every acid-A cell is active, 5 of 6 acid-H cells are inactive), so a pooled AUROC could
be driven entirely by chain length rather than by any enzyme discrimination. The within-acid tests
are immune to that; the pooled test is reported as **secondary and explicitly confounded**.

**Secondary (descriptive, no claim rests on these):**

- **T4 — pooled AUROC** of `A` over all 18 cells, with an acid-stratified permutation null
  (permute labels *within* acid only). Reported with the confound named.
- **T5 — panel A continuous.** Spearman ρ of `A` vs normalized MS signal across the 6 enzymes, all
  of which are active, where AcsA is ~6–8× the next best. Perfect ordering would give
  p = 1/6! = **0.00139**. This is the only panel with real dynamic range among actives.

**Power ceiling, acknowledged before looking:** the best achievable primary result is
**p ≈ 0.048** — barely significant even if `A` is *perfect*. n = 6 enzymes per stratum is the hard
limit of this paper. We therefore commit to reporting T1/T2 as **rank patterns with exact
p-values**, and to treating the outcome as *mechanistic evidence*, not as a p-value win. A null
here does not refute `A`; it is simply uninformative at this n.

## 6. Pre-declared failure / abstention conditions

- A cofold whose ensemble has **`n_gated == 0`** (no pose with d < 4 Å) is reported as
  **"did not dock"** and `A` for it is flagged *non-interpretable*, as in prior campaigns —
  but it is **not** silently dropped, since dropping non-dockers is how "did the ligand dock at
  all" masquerades as a catalytic signal (the Oct-6 hydrolase lesson).
- Any run finishing with **< 50 PDBs** is a failure (Boltz exits 0 on OOM), re-queued, and if it
  persists the enzyme is reported as **unfoldable at this setup**, not as a low score.
- If fewer than 16 of 18 cofolds yield 50 poses, the primary tests are **not run**.

## 7. Compute

Boltz-2, `--diffusion_samples 50 --max_parallel_samples 5`, precomputed batched MSAs (never
`--use_msa_server` per sequence). All GPU work goes through
`/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve`. **`--max_parallel_samples` is mandatory here**:
these chains are 592–647 aa, longer than the 396–530 aa the existing fleet worker was built for,
and omitting it OOMs at exit 0 with zero structures. AcsA (647 aa) is above the longest chain known
to fit a V100 in FP32 in this repo (PubC, 629 aa), so it is the run most likely to need an H100.

---

## DEVIATIONS

### 2026-10-21, amendment 1 — added a within-fold stratification (8/58 folds scored)

**Added BEFORE any panel was complete**, on evidence from the docking diagnostic
(`analysis/docking_validity_diagnostic.png`), not from any test result. No primary test is changed
or removed; one stratification is **added**, and the original cross-fold tests stay as pre-specified.

**What prompted it.** The first 8 ensembles show docking quality tracking **enzyme family**:
the three best-docked are all NIS-fold (DesD·A n_gated 33/50 D_h 4.0 Å A 153°; AsbB·A 19/50, 146°;
AsbA·A 4/50, 110°) while both ANL members sit at the bottom (SfaB·A n_gated 0, D_h 12.6 Å — and
SfaB+A is the *active* normalisation anchor). `A` and `n_gated` are covarying.

**Why this is a real threat, not a curiosity.** `A` is already documented in this repo as
**fold-confounded across folds** (GS-fold 148.4 vs 131.1, p=0.0007). Figure 2's enzyme set is
**cross-fold by construction** — 2 ANL (DdaG, SfaB) vs 4 NIS (AsbA, AsbB, AcsA, DesD) — so the
pre-registered within-acid tests T1/T2 compare across exactly the axis on which `A` is known to be
biased. A "win" there could be the fold bias rather than catalysis.

**Amendment.** Each within-acid test is additionally run **within fold family**, and both results
are reported side by side. Where a stratum has too few members to test (ANL has only 2 enzymes),
that is stated as *untestable* rather than reported as a null.

**Also pre-declared now, so it cannot be chosen later:** report the correlation between `A` and
`n_gated` across all scored ensembles. If `A` is largely explained by docking quality, then `A` is
substantially reporting "did the ligand dock", which is the exact artifact that sank the hydrolase
campaign. This is reported **whatever it shows**, and it is a property of the metric on this panel,
not a reason to swap metrics.

**Known tension recorded in advance:** Fig 2A has **all six enzymes active** with **AcsA dominant
(~7.6×)**, yet `A` currently ranks DesD first. If that ordering holds, `A` does not recover the
panel's measured ordering, and T5 (the panel-A continuous test) is expected to fail. Writing this
down before the panel completes so the outcome cannot be reinterpreted afterwards.

### 2026-10-21, amendment 2 — declared sensitivity: tests restricted to docked ensembles

SfaB's adipate contacts **Arg103 in 49/50 poses (98%)** — together with P223/P508/M272/F249 — i.e.
it docks **reproducibly at a surface-proximal Arg site ~12.6 Å from Pα**, with protein and ATP
modelled correctly. This independently reproduces a previously characterised SfaB behaviour (ligand
parking on a conserved Arg, R/K at that aligned column in 42/49 panel members), now seen with a
different substrate in a different campaign. SfaB's `A` is therefore an angle measured **outside
the active site** and is not a catalytic quantity at all.

Because SfaB is experimentally **active**, it alone drags the Fig-2A correlation down for a reason
that has nothing to do with catalysis. The obvious question — "what if the tests are restricted to
ensembles where `A` is interpretable?" — is therefore **declared now, before being run**:

- **Primary tests stay exactly as pre-specified, over ALL enzymes**, non-dockers included.
- **Secondary, explicitly labelled:** each test repeated over the subset with `n_gated > 0`,
  reporting how many enzymes were excluded and which.

This subset analysis is **not** a rescue and must never be quoted as the headline: restricting to
dockers conditions on the model's own success and is precisely the move that manufactured a false
signal in the hydrolase campaign. It is reported to separate two distinct failure modes — "`A`
mis-ranks catalysis" vs "there was no catalytic pose to score" — which is a mechanistic
distinction worth making, not a p-value to be improved.

### 2026-10-21, amendment 3 — the POOLED trend test (added at 19/58, before the panels completed)

**Why.** Every per-panel test here has 4–6 enzymes, so none can demonstrate a trend on its own:
the best attainable p is 0.048 and most panels cannot even reach that. The question "does `A`
capture the trend at all?" is therefore **underpowered by construction panel-by-panel** and needs a
pooled estimator. Declared now, with the data still incomplete, so the pooling rule cannot be
chosen to suit the answer.

**Why not just pool the raw numbers.** Readouts differ across figures — Fig 2A/2B are
*normalised* MS intensity (SfaB ≡ 1), everything else is *log₁₀ relative* MS intensity with a
floor — and an enzyme's `A` is shared across panels using the same acid donor. Pooling raw values
would mix scales and double-count. So the pooling is on **within-panel ranks**.

**TP1 — stratified rank-correlation (continuous trend).** For each panel with ≥3 detected enzymes,
compute Spearman ρ(`A`, measured) among the **detected** enzymes only (censored values are not real
numbers and cannot be ranked against each other). Combine the per-panel ρ across panels by
**Stouffer's z**, weighting each panel by √(n−3). Pre-specified one-tailed, positive direction.
Report the combined z, its p, and the per-panel ρ values so the combination is auditable.

**TP2 — stratified binary separation (detected vs below LOD).** Pool all (enzyme, panel) cells and
compute AUROC of `A` for detected vs below-LOD, with a **panel-stratified permutation null**
(labels permuted *within* panel only, 20 000 draws). Panel stratification is mandatory: the
marginal detection rate differs enormously between panels (2A is 6/6, 5C is 1/4), so an
unstratified null would be badly anticonservative.

**TP3 — the same two tests restricted to `n_gated > 0`**, labelled secondary per amendment 2.

**Pre-specified interpretation, fixed now:**
- TP1/TP2 significant and positive → `A` captures the trend despite per-panel noise.
- Not significant → report as **"no detectable trend at this n"**, explicitly NOT as "`A` is
  refuted". With ~10 strata of 4–6 enzymes the pooled design still has modest power, and a null is
  consistent with a real-but-small effect.
- **Either way, report alongside it:** ρ(`A`, `n_gated`) over all ensembles, and the ANL-vs-NIS
  fold comparison. A positive TP1 that disappears within fold family is a fold artifact, not a
  catalytic signal, and must be reported as such.

### 2026-10-21, amendment 4 — TP2 pooling corrected for non-independence

**A bug, not a choice.** TP2 as first implemented pooled (enzyme, PANEL) cells. But several panels
share a step-1 acid donor and therefore carry **byte-identical `A` values**: 4B, 4C and 5C all use
donor **MA**, and 2C/2D both use **H**. Pooling them as separate strata multiply-counts a single
measurement and inflates the test.

TP2 now pools **unique (enzyme, acid donor) cells**, with the label = detected in *any* panel using
that donor — which is exactly the labelling rule already fixed in §4 for acid H ("active if active
in panel C or D"). So this restores the pre-registered intent rather than altering it.

**Direction of the correction, recorded for integrity:** it made the result **worse**, not better —
AUROC fell from 0.808 to 0.697 and the below-LOD count corrected from 1 to 3. A fix that costs the
favoured hypothesis is evidence it was a genuine bug rather than motivated analysis.

**Note on what TP1 does and does not cover.** Panels 4B/4C/5C each have only ONE detected enzyme, so
no within-panel Spearman exists for them and they contribute nothing to TP1 either way. TP1 remains
driven by the few panels with >=3 detected enzymes.

---

## 2026-10-21, amendment 5 — the ACCOMMODATION hypothesis (new family, declared before computing)

**Status: this is a NEW proxy family, and the multiplicity cost is acknowledged up front.** Six
families are already null in this repository and `broad_proxy_search_20261008` showed that ~3×
top-10 enrichment on panels like this carries a selection-corrected **p ≈ 1.0**. Declaring the
hypothesis and its test before computing is therefore not a formality — it is the only thing that
makes the result interpretable. No descriptor will be added, dropped, or re-thresholded after the
numbers are seen.

### Why this is not the already-refuted near-attack test

The near-attack fraction was tested panel-wide and was null (Spearman vs yield = **−0.072,
p = 0.63**, n = 49). **But that panel fixed the substrate** — amine = HMD, acid = adipate — so it
measured enzyme *main effects* only. Accommodation predicts an **enzyme × substrate-size
interaction**, which a fixed-substrate design cannot see. This campaign varies the donor across
nine substrates spanning 10 to 48 heavy atoms, so the interaction is estimable here and was not
estimable there. That is the specific gap this amendment addresses.

### The hypothesis

From the paper's own Fig 3: DesD has a larger (2497 Å³ vs 816 Å³ for AcsD) and more hydrophobic
acyl-binding pocket, and the authors argue pocket volume and polarity jointly govern which
substrate sizes are accepted. Mechanistically this predicts *accommodation*, not attack geometry:
a substrate too large for the pocket never reaches the reactive centre at all, so the failure is
steric, upstream of any angle.

**H_acc:** the probability a cell is active falls with donor size, and that fall is **steeper for
small-pocket enzymes** — i.e. a negative size × pocket-capacity interaction.

### Pre-specified descriptors (fixed now; no additions later)

Per (enzyme, donor) ensemble, computed on the existing step-1 folds — no new GPU:

1. `donor_heavy_atoms` — substrate size, from the input SMILES (exogenous, not model-derived)
2. `occupancy` = n_gated / 50 — does the donor reach the reactive centre at all
3. `D_h` — harmonic mean donor-carboxylate-O to Pα distance
4. `buried_frac` — fraction of donor heavy atoms with ≥1 protein heavy atom within 4.5 Å,
   median over 50 poses
5. `n_contacts` — median count of protein heavy atoms within 4.5 Å of any donor atom
6. `pocket_apolar_frac` — fraction of residues lining the Pα pocket (any heavy atom within 8 Å of
   Pα) whose type is in {A,V,L,I,M,F,W,P,G}

### Tests

- **TA1 (primary).** Logistic regression of detected/below-LOD on
  `donor_heavy_atoms + enzyme_capacity + their interaction`, where `enzyme_capacity` is each
  enzyme's **mean `buried_frac` on the two monomer diacids (A, G) only** — i.e. capacity estimated
  on *small* substrates and used to predict behaviour on *large* ones, so the predictor is not
  fitted on the cells it must explain. Report the interaction term's coefficient, sign and p.
- **TA2.** Spearman of each descriptor against activity *within* donor (controls size), and
  against activity *within* enzyme (controls pocket), reported as a matrix so no single cell can
  be cherry-picked.
- **TA3 (mandatory, interpretive).** Repeat TA1 **within fold family**. A result that disappears
  within ANL or within NIS is a fold artifact, exactly as for `A`.
- **Selection correction.** Because this is family #8, the headline p is reported **alongside** a
  descriptor-count-corrected threshold (6 descriptors × 2 strata = 12 looks → Bonferroni α = 0.0042)
  and the honest statement that clearing 0.05 but not 0.0042 is **not** a finding.

### Pre-declared interpretation

- Interaction negative and significant past the corrected threshold → accommodation explains the
  size-dependent activity pattern that attack geometry could not.
- Not significant → reported as null, and the accommodation family joins the other seven. It will
  **not** be followed by a further descriptor search on this panel.
- **Anticipated confound, stated now:** `occupancy` and `buried_frac` are both expected to
  correlate with donor size mechanically (a bigger ligand has more atoms to bury and more ways to
  clash), so size must appear as a covariate in every model, never as the sole predictor.

### 2026-10-21 — amendment 5 PARKED (not withdrawn, not reported as a null)

The accommodation family is **parked before completion**, by decision, with its state recorded so
it cannot later be revived as a fresh first look:

- **TA1 is NOT EVALUABLE.** `buried_frac` saturated at 1.000 for essentially every ensemble
  (capacity = 1.000 for 5 of 6 enzymes, AsbB 0.950), because a 4.5 A contact cutoff puts *some*
  protein atom near nearly every ligand heavy atom — even a surface-parked one. The predictor was
  effectively constant, the design matrix near-singular, and the fit degenerated (se 119–834,
  z ~ 0.00, p ~ 0.99). **Those p-values are a collapsed regression, not a null.**
- **TA2 ran as specified** and is recorded in `analysis/accommodation_descriptors.csv`: occupancy
  +0.38 / +0.34 (within donor / within enzyme), D_h −0.66 within enzyme, `n_contacts` −0.39 / −0.52
  (wrong sign), donor size −0.41. The right-signed descriptors are the *docking* family already
  null at n = 49; `n_contacts` runs backwards.
- **Awkward for the hypothesis, recorded now:** AcsA is active on AMA (26 heavy atoms) at
  occupancy 0.00 and DesD is active on AMA at 0.02 — large substrates turned over by enzymes whose
  modelled ensembles never approach Pa.

**Rules for any future resumption.** Re-running TA1 with a tighter cutoff is a **second look** and
must be declared and penalised as such; it may not be presented as the first test of accommodation.
Prerequisites before resuming at all: the 15 outstanding step-1 folds (several are the large-ligand
cells this test most depends on), and a well-posed capacity descriptor — the obvious candidate is
the bottleneck radius from the validated in-house `channel_geometry.py`, run on APO folds, since a
bound substrate plugs its own channel.

**Also recorded: pocket VOLUME was attempted and abandoned as ill-posed.** A grid implementation
failed to reproduce the paper's Fig-3B values (DesD 783 vs 2497 A^3, AcsD 380 vs 816 — ratios 0.31
and 0.47, so not a constant factor) and, decisively, **did not converge**: 454 / 773 / 1139 A^3 at
box radius 10 / 14 / 18 A. With no plateau there is no well-defined volume, so no fold-derived
pocket volume is quoted anywhere in this campaign. The paper's 816 / 2497 A^3 remain attributed to
the authors (PyVOL), not reproduced here. PyVOL, fpocket and freesasa are all absent from this
environment.

---

# Amendment 6 — declared descriptor search over BOTH steps (2026-10-22)

**Status: declared BEFORE any descriptor was computed or any AUROC seen.** Cofolding is complete
(58/58 step 1, 90/90 step 2) and both frozen metrics are null, so the question becomes whether a
*different* pair of geometric descriptors — one per half-reaction — separates active from inactive.

This amendment exists because this is the single most dangerous thing this project does. Two prior
findings in this repo died exactly here: a "3.1x top-10 enrichment, p = 0.02" that was **cutoff
shopping** across nine cutoffs, and `thermo_packing_density` (rho = +0.77 on six members),
**retracted the same day** as a length artifact. A battery searched without a selection-corrected
null will always produce a winner. The null is therefore declared first, and the battery is frozen
here.

## 6.1 The battery — frozen, and no descriptor may be added after results are seen

Every descriptor is computed from the EXISTING folds; no new GPU work, so there is no opportunity
to re-fold toward a better score. All are per-ensemble summaries over the 50 poses.

**Step 1 (adenylation), 13 descriptors** — measured on donor carboxylate O -> ATP Pa:

| # | name | what it measures |
|---|---|---|
| 1 | `s1_A` | the frozen proximity-weighted attack angle (incumbent) |
| 2 | `s1_Dh` | harmonic-mean attack distance |
| 3 | `s1_occupancy` | fraction of poses with d < 4 A (the validity precondition, now AS a descriptor) |
| 4 | `s1_min_d` | closest approach over the ensemble |
| 5 | `s1_nac_frac` | fraction of poses that are near-attack: d < 4 A **and** angle > 150 deg |
| 6 | `s1_best_angle_gated` | best in-line angle among gated poses only (0 if none) |
| 7 | `s1_mg_carboxylate` | min Mg2+ -> donor carboxylate O distance (Mg bridging the nucleophile) |
| 8 | `s1_mg_phosphate` | min Mg2+ -> ATP phosphate O distance (cofactor placement integrity) |
| 9 | `s1_carboxylate_hbonds` | protein N/O within 3.5 A of a donor carboxylate O |
| 10 | `s1_cation_contacts` | Arg/Lys side-chain N within 4.0 A of a donor carboxylate O |
| 11 | `s1_donor_contacts` | protein heavy atoms within 4.5 A of any donor atom |
| 12 | `s1_centroid_spread` | SD of donor centroid across poses (docking convergence) |
| 13 | `s1_pocket_apolar` | apolar fraction of residues lining Pa |

**Step 2 (aminolysis), 13 descriptors** — measured on amine N -> acyl-adenylate carbonyl C:

| # | name | what it measures |
|---|---|---|
| 14 | `s2_BD_dev` | \|Burgi-Dunitz angle - 105\| (incumbent) |
| 15 | `s2_Dh` | harmonic-mean N...C distance |
| 16 | `s2_occupancy` | fraction of poses with d < 4 A |
| 17 | `s2_min_d` | closest approach |
| 18 | `s2_nac_frac` | near-attack fraction: d < 4 A **and** BD in [95, 115] |
| 19 | `s2_best_bd_gated` | best \|BD - 105\| among gated poses only |
| 20 | `s2_oxyanion_hbonds` | protein N/O within 3.5 A of the carbonyl O (oxyanion-hole occupancy) |
| 21 | `s2_leaving_anti` | N-C-O(AMP ester) angle — nucleophile anti-periplanar to the leaving group |
| 22 | `s2_n_burial` | protein heavy atoms within 4.5 A of the attacking N |
| 23 | `s2_acceptor_contacts` | protein heavy atoms within 4.5 A of any acceptor atom |
| 24 | `s2_distal_charge` | min attacking-N-distal cationic N -> Asp/Glu carboxylate O distance |
| 25 | `s2_centroid_spread` | SD of acceptor centroid across poses |
| 26 | `s2_acyl_intact` | fraction of poses where the acyl-adenylate anhydride is resolvable |

**26 single descriptors + 13 x 13 = 169 cross-step pairs = 195 declared looks.**

## 6.2 Endpoint, unit of analysis, and the pairing rule

- **Endpoint:** detected vs below LOD, from the digitised panels. Binary — the readouts are not
  comparable across figures, so no continuous pooling.
- **Unit:** one (enzyme, acid donor, amine acceptor) cell = one reaction. n = 48 labelled cells.
- **Statistic:** AUROC, oriented so that the mechanistically favourable direction is the positive
  one, fixed per descriptor in the code BEFORE running (an unoriented AUROC lets a sign flip count
  as a hit and silently doubles the battery).
- **Pairing rule — deliberately fit-free.** A pair is combined as the **minimum of the two
  within-acid-stratum percentile ranks**: a cell scores well only if it is good at BOTH steps.
  There are **no fitted weights and no tuned thresholds**, so this is not a composite score in the
  sense the project's standing rule forbids; it is the conjunction the two-step mechanism implies.
  Logistic regression on the pair is explicitly NOT run — 48 cells and 195 looks cannot support it.

## 6.3 The null — selection-corrected over the WHOLE battery

A per-descriptor p-value is meaningless with 195 looks. The declared primary null is:

> Permute the detected/below-LOD labels **within acid stratum** (detection rates differ enormously
> between strata: acid A is 6/6 detected, MA is 1/4, so an unstratified null is badly
> anticonservative). For each of 20 000 draws, recompute the AUROC of **every one of the 195
> descriptors and pairs** and retain only the **maximum**. `p_sel` = fraction of draws whose
> maximum AUROC >= the observed best.

**Decision rule, fixed now:** a descriptor or pair is reported as a finding only if
**`p_sel` < 0.05**. A descriptor that clears an unadjusted 0.05 but not `p_sel` is reported as
**not a finding**, in those words. The best observed AUROC will be reported either way, together
with the null's own 95th percentile, so the reader can see how much of it is selection.

## 6.4 Mandatory confound controls — reported for the winner whatever `p_sel` says

Both are known live confounds in this repo, not hypotheticals:

1. **Docking quality.** Report `rho(descriptor, s1_occupancy)` and `rho(descriptor, s2_occupancy)`.
   A descriptor that is a repackaging of "did the ligand dock" is reported as such. `A` already
   fails this: rho(A, n_gated) = +0.314, p = 0.020.
2. **Fold family.** Recompute the winner's AUROC **within** ANL (DdaG, SfaB) and **within** NIS
   (AsbA, AsbB, AcsA, DesD). `A` separates families (NIS 127.1 vs ANL 117.3), so any descriptor
   whose signal vanishes within family is a fold artifact.

Additionally: because the same step-1 ensemble is reused across cells that share an acid donor
(DesD+MA serves 4B, 4C and 5C), the winner is re-evaluated on **unique step-1 ensembles only** as a
non-independence sensitivity check.

## 6.5 What this search cannot do

The power ceiling of amendment 0 still binds: six enzymes per stratum. A null here is **"no
detectable signal in 195 declared looks"**, not a proof that no geometric descriptor exists. And
nothing in this battery can see transition-state stabilisation or domain dynamics — it is 26 ways
of measuring a ground-state pose.

## Amendment 6 — RESULT (2026-10-22): clears the primary test, fails both sensitivity checks

**Verdict: NOT safe to cite as predictive.** The primary test passed and the declared robustness
checks then dismantled it. Both outcomes are recorded because the amendment required it.

### What the search found

Battery actually searched: **167 looks** (13 step-1 + 11 step-2 singles + 143 pairs). Two declared
descriptors were not evaluable on the common cell set — a recorded deviation from the declared 195:

| descriptor | why | searched separately? |
|---|---|---|
| `s2_acyl_intact` | **DEGENERATE — constant at 1.000.** The anhydride resolves in every pose of every ensemble, so it carries no information (the same failure mode as `buried_frac` in amendment 5) | not searchable |
| `s2_distal_charge` | missing for 18/48 cells — structurally, since a mono-amine acceptor has no distal nitrogen | yes: AUROC 0.634, single-look p = 0.074 |

**Primary result (n = 48 cells, 23 detected):**

| look | AUROC |
|---|---|
| **`s1_min_d`** — closest donor-carboxylate-O -> Pa approach over 50 poses | **0.877** |
| `s1_Dh` | 0.811 |
| `s1_mg_carboxylate` | 0.799 |
| `s1_best_angle_gated` | 0.778 |
| `s1_occupancy` | 0.766 |
| best PAIR: `s1_cation_contacts + s2_n_burial` | 0.747 |

Selection-corrected null, 20 000 draws, labels permuted within acid stratum, max taken over all
167 looks: null max median 0.723, **95th percentile 0.797**, p_sel = **0.0003**. By the decision
rule in 6.3 this is a FINDING. For contrast, 54/167 looks clear an unadjusted p < 0.05 against
~8 expected — which is exactly why the unadjusted numbers are unusable.

**No pair beats the best single descriptor.** The best cross-step pair reaches 0.747 against
`s1_min_d`'s 0.877, and every pair in the top 15 pairs a step-1 descriptor with `s2_n_burial`.
Requiring both half-reactions to be good does not add information here.

### Why it does not survive — the three declared controls

1. **It is docking.** rho(`s1_min_d`, `s1_occupancy`) = **+0.864**, p < 1e-4. `min_d` and the
   validity gate are near-duplicates by construction (min_d < 4 A iff occupancy > 0). The campaign
   has always treated occupancy as a validity precondition rather than a ranking input; this says
   proximity is the only thing with signal, and the attack ANGLE adds nothing.
2. **Non-independence (declared in 6.4).** Re-run on unique step-1 ensembles only — cells sharing
   an acid donor reuse one step-1 fold, so DesD+MA was counted three times — gives AUROC 0.805 and
   **p_sel = 0.195. Not significant.** Most of p_sel = 0.0003 was duplicate counting.
3. **It is ONE enzyme, and it is the one in the training set.** Leave-one-enzyme-out:

   | dropped | AUROC | within-stratum p |
   |---|---|---|
   | AcsA / AsbA / AsbB / DdaG / SfaB | 0.84–0.91 | 0.000–0.003 |
   | **DesD** | 0.845 | **0.676** |

   The decisive statistic is the **within-acid AUROC**, which is what the stratified null actually
   tests: **0.848 with DesD, 0.500 — exactly chance — without it.** The reason is visible in the
   raw numbers: in acids **H** and **MA**, DesD is the *only* active enzyme and also has the
   shortest approach (H: DesD 4.36 A vs 6.1–10.9 for the five inactives; MA: DesD 3.02 vs
   6.25–15.07). Those two strata supply 54 of the 69 informative active/inactive pairs. Remove
   DesD and three of six strata lose all contrast, leaving 15 pairs and no signal.

   In the two strata where a genuine multi-enzyme contrast exists and DesD is not the sole active,
   `min_d` runs **backwards**: acid Z's closest approach is AsbB at 3.26 A (below LOD) while active
   SfaB sits at 4.23 A; acid AMA's closest is SfaB at 3.75 A (below LOD) while active AcsA sits at
   5.57 A.

**Full battery re-searched on the DesD-excluded cells** (38 cells, 13 active, same 167 looks):
best is still `s1_min_d` at 0.845, but the null's max has median **0.855** — *above* the observed
value — and **p_sel = 0.665**. Nothing works without DesD.

### Reading

DesD's native complex (PDB 7TGM, released 2022-07-06) is very likely inside Boltz-2's training
set, and the positive control shows near-perfect recall of it (0.23 A ligand placement). A
proximity descriptor whose entire discriminating power is DesD, in the strata where DesD is the
only active enzyme, is most parsimoniously read as **recall of one memorised complex, not
prediction**. It is recorded here as a non-finding and must not be cited as a predictive
descriptor.

**No descriptor and no cross-step pair in 167 declared looks separates active from inactive
once DesD is removed.** Combined with the null results for the frozen step-1 and step-2 metrics,
the conclusion of this campaign is unchanged and is now substantially better supported.

---

## Amendment 7 — the enzyme x substrate INTERACTION (declared 2026-10-23, BEFORE computing)

Every test so far stratified by **acid donor**. That removes the substrate main effect but leaves
the **enzyme main effect** intact, so every contrast was between-enzyme -- and in the two largest
strata (H, MA) "between-enzyme" means DesD against five inactives. Amendment 6 established that
the enzyme main effect is confounded with training-set membership. We have therefore never tested
the component a structural model should actually predict.

Activity in this 6-enzyme x 10-substrate-pair layout decomposes into three parts:

| component | magnitude in these data |
| --- | --- |
| enzyme main effect | DesD 10/10, AcsA 5/10, SfaB 4/10, DdaG 2/10, AsbA 1/4, AsbB 1/4 |
| substrate main effect | acid A 6/6, G 4/4 vs H 1/6, MA 1/4 |
| **enzyme x substrate interaction** | **the untested component -- the endpoint of this amendment** |

### 7.1 Design (the mirror of every previous test)

Stratify by **enzyme**: compare each enzyme's accepted substrates against its own rejected ones.
DesD is 10/10 active and therefore contributes **zero** informative pairs *by construction* --
the memorisation confound is structurally absent, not controlled for after the fact.

```
DdaG 2x8=16 | SfaB 4x6=24 | AcsA 5x5=25 | AsbA 1x3=3 | AsbB 1x3=3 | DesD 10x0=0   -> 71 pairs
```

Endpoint is unchanged: binary detected / below-LOD. Binary detection is comparable across panels
even though the MS readouts are not, so this design does not require pooling incommensurable
yields.

### 7.2 Primary statistic

Within-enzyme concordance C = fraction of (active, inactive) pairs drawn from the SAME enzyme that
the descriptor orders correctly; ties score 0.5. Orientation of every descriptor is fixed by the
`ORIENT` table already frozen in `search_descriptors.py` and is not revisited.

### 7.3 Battery — 26 looks, fixed now

The 13 step-1 and 11 step-2 evaluable singles from amendment 6, plus `A` and `S`. **Cross-step
pairs are excluded a priori**, because amendment 6 established that no pair beats the best single
(0.747 vs 0.877); excluding 143 looks that a prior result already ruled out preserves power and is
a restriction made on a previous finding, not on these labels.

### 7.4 Null — both margins preserved

Permuting labels within enzyme would preserve the enzyme main effect but destroy the substrate
main effect, leaving a descriptor that merely tracks "acid A is easy for everyone" free to score.
The null must therefore hold **both** margins:

- permute the observed activity matrix by **checkerboard swaps** (curveball / swap randomisation,
  standard for presence-absence matrices): pick two enzymes and two substrate pairs, all four cells
  observed; if the 2x2 reads [[1,0],[0,1]] or [[0,1],[1,0]], flip it. Row sums (enzyme promiscuity)
  and column sums (substrate difficulty) are invariant under this move.
- 20000 draws, burn-in 20000 swaps, 2000 swaps between retained draws, seed 20261023.
- for each draw retain the **MAXIMUM** C over all 26 looks -> selection-corrected null.

`p_sel` = fraction of draws whose maximum C is >= the best observed C.

### 7.5 Decision rule

`p_sel < 0.05` is required. Declared in advance, a finding must additionally survive all three:

1. rho(best descriptor, `s1_occupancy`) -- a descriptor that is merely docking propensity is
   reported as docking, not catalysis;
2. leave-one-enzyme-out on the remaining five contributing enzymes;
3. restriction to unique step-1 ensembles.

### 7.6 What this cannot do

Seventy-one pairs come from 38 non-DesD cells with 13 actives and shared step-1 folds; the
effective sample size is far below 71. A null result is consistent with "no signal" and with
"underpowered", and will be reported as the latter where the margin-preserving null's own spread
is wide. A positive result establishes that ground-state geometry tracks substrate specificity
within an enzyme -- it would say nothing about ranking enzymes against each other.

---

## Amendment 7 — RESULT (2026-10-23, written before amendment 8)

**The declared test could not be run. The reason is the finding.**

Deviation from §7.3: the completeness filter admitted **27** looks, not the declared 26 — it kept
`s2_acyl_intact`, which is degenerate (scores exactly 0.5000). Recorded rather than dropped; an
extra look can only make `p_sel` more conservative.

Observed within-enzyme concordance over 71 pairs (DesD contributing zero by construction):
`s1_Dh` and `s1_mg_carboxylate` 0.9366 · `s1_carboxylate_hbonds` 0.9225 · `s1_min_d` 0.9085 ·
`s1_S_frozen` 0.8873 · … · `s1_A` 0.5986 (frozen primary, near chance) · `s2_BD_dev` 0.4789.

The null returned **median = 95th percentile = max = 0.9366**, exactly the observed value, giving
`p_sel = 1.0000`. **This is a point mass, not a statistical test, and must never be cited as one.**

Diagnosis: the activity matrix contains **0 swappable checkerboards in 324 fully observed 2x2
blocks**. The accepted-substrate sets form a strict chain with zero violations —
`AsbA = AsbB {2A} ⊂ DdaG {2A,4A} ⊂ SfaB {2A,2B,4A,5A} ⊂ AcsA {+5B} ⊂ DesD {all 10}` — so the
observed matrix is the *unique* binary matrix with its margins and the swap chain cannot move.
A two-parameter nested model (enzyme promiscuity rank x substrate difficulty threshold) reproduces
**48/48 cells with zero residual interaction**.

The nestedness is not an artifact of the margins: under free permutation of the 23 labels across
the 48 observed cells (20000 draws, seed 20261023), **0/20000** draws were perfectly nested
(median 42 checkerboards, minimum 3) -> **p ~ 5e-5**.

**Consequences, binding on all later analysis.**
1. There is no enzyme x substrate interaction in this panel. Not "undetectable" — absent.
2. The high within-enzyme concordances recover the substrate-difficulty **margin**, which the
   two-parameter model already explains; they are not specificity prediction.
3. The effective sample size is **6 + 10 = 16 rank positions, not 48 cells**. Every p-value in this
   campaign, including the quoted p ~ 0.048 power ceiling, was computed against an inflated n.
4. A nested matrix admits validation of **margin-level** predictors only. A cell-level predictor
   cannot be scored here at all.

**Post hoc observation (hypothesis-generating, not a declared test):** the five DesD-exclusive
panels are exactly the five whose adenylated donor carries a free ammonium (6-aminohexanoate, MA);
the five shared panels are exactly the five with a pure dicarboxylate donor (adipate, glutarate,
azelate, AMA). 5/5 vs 5/5, one-sided Fisher p = 0.004. Donor **size runs backwards** to difficulty
(smallest donor H, 9 heavy atoms, is DesD-only; largest AMA, 26, is not), so the substrate axis is
not steric capacity. DesD's native donor HSC is itself a bifunctional omega-amino-acid donor.
This observation was made on these labels and is **not** evidence; amendment 8 tests a prediction
it makes on an independent axis.

---

## Amendment 8 — the G<->H matched donor-class swap (declared 2026-10-24, BEFORE computation)

### 8.1 Why this design

Amendment 7's observation says the substrate margin is set by donor functional class. That makes a
**directional, within-enzyme, matched-pair** prediction about step-1 adenylation geometry, testable
on folds already in hand. The contrast is a single functional group at constant size:

| donor | class | heavy atoms | SMILES |
|---|---|---|---|
| `G` glutarate | dicarboxylate | 9 | `[O-]C(=O)CCCC(=O)[O-]` |
| `H` 6-aminohexanoate | omega-amino acid | 9 | `[NH3+]CCCCCC(=O)[O-]` |

All six enzymes have both ensembles (12 of 58 step-1 folds). No new compute. This design controls
the confound that destroyed `s1_min_d` in amendment 6: docking propensity cannot drive a *paired
within-enzyme difference* the way it drove a cross-enzyme ranking. DesD is the predicted **positive**
here, so training-set recall of 7TGM argues against the hypothesis rather than manufacturing it.

### 8.2 Carboxylate-count confound — controlled by construction

A dicarboxylate presents **two** carboxylate groups and a mono-acid **one**, so a nearest-oxygen
statistic favours the diacid by chance alone. The primary statistic is therefore the **mean over
carboxylate groups** of each group's closest-oxygen distance to ATP P-alpha, which is unbiased with
respect to group count. Nearest-group is reported as secondary only.

### 8.3 Descriptors — fixed now

Per pose, donor = chain `D`, nucleotide = chain `B`, anchors `PA` / `O3A` as in the frozen
`pdb_geometry`. Carboxylate group = a carbon with >= 2 oxygens within 1.6 A. Basic nitrogen = an N
whose bonded carbons bear no oxygen within 1.6 A (excludes amide N).

1. `d_grpmean` — mean over carboxylate groups of (closest O of that group -> `PA`).  **PRIMARY**
2. `d_nearest` — closest carboxylate O over all groups -> `PA`.  secondary
3. `n_gated` — poses with `d_nearest` < 4.0 A.
4. `f_inverted` — fraction of poses in which a basic N is closer to `PA` than any carboxylate O
   (donor bound backwards). Defined only for `H`; `G` has no basic N and scores 0 by construction.
5. `N_contact` — for `H`, the nearest protein residue to the basic N, with distance.

### 8.4 Prediction, stated before measurement

In the five dicarboxylate specialists (DdaG, SfaB, AsbA, AsbB, AcsA), donor `H` is **worse
registered** on P-alpha than donor `G`: `delta = d_grpmean(H) - d_grpmean(G) > 0`.
In DesD, `H` is **not** degraded relative to `G`.

### 8.5 Primary test

One-sided **sign test** over the 5 paired specialists on the sign of `delta`. With n = 5 the
minimum attainable p is `1/32 = 0.031`, so **5/5 concordant signs are required** at alpha = 0.05.
DesD is an out-of-sample directional check and is **not** part of the test.

### 8.6 Mandatory specificity control — the S<->G swap

`S` succinate (`[O-]C(=O)CCC(=O)[O-]`, 7 heavy atoms) vs `G` glutarate: a one-CH2 **size** change
with **no class** change, folded for all six enzymes. If `delta` merely reflects "any donor swap
perturbs the fold", S<->G must show an effect of comparable magnitude. The donor-class claim
requires `|delta(G->H)| > |delta(S->G)|` in the specialists. A failure here falsifies the
interpretation even if 8.5 passes.

### 8.7 What this cannot do

n = 5 paired enzymes. A pass establishes a reproducible directional geometric consequence of donor
class in ground-state folds; it does **not** establish that this causes the activity difference,
and it cannot, because amendment 7 showed the activity matrix has no interaction term to explain.
`f_inverted` and `N_contact` are descriptive and were not powered for.

### Amendment 8 — RESULT (2026-10-24)

**The pre-registered test PASSES, at the minimum p its own design admits.** Full output:
`analysis/swap_donor_class.txt`; code `swap_donor_class.py`.

Primary (§8.5), `delta = d_grpmean(H) - d_grpmean(G)`, predicted > 0:
DdaG +1.446 · SfaB +0.870 · AsbA +2.191 · AsbB +3.448 · AcsA +1.528 —
**5/5 concordant, one-sided sign test p = 0.0312.**

Out-of-sample directional check, DesD: **delta = -4.010**, i.e. the one enzyme whose native donor is
a bifunctional omega-amino acid registers `H` *better* than `G`. Specialist deltas span
[+0.870, +3.448]; **DesD does not overlap them.**

Specificity control (§8.6) **holds**: mean |delta| for the class swap G->H is **1.897 A** versus
**0.454 A** for the size swap S->G — a **4.18x** ratio. The effect is not "any donor swap perturbs
the fold".

The declared group-mean form is confirmed **conservative**, as §8.2 intended: the biased
nearest-oxygen form gives a mean delta of +3.586 A against the group-mean's +1.897 A, so
controlling the carboxylate-count confound shrank the effect rather than creating it.

Mechanism (§8.3 items 4-5, descriptive): donor `H` binds **backwards** — a basic nitrogen closer to
P-alpha than any carboxylate oxygen — in **1.00 / 0.96 / 1.00 / 1.00** of poses for DdaG, SfaB,
AsbA, AcsA, and 0.22 for AsbB, against **0.04 for DesD**. The ammonium is captured by an acidic
residue in every case (AsbA Glu444 50/50 · AcsA Glu482 45/50 · AsbB Glu569/Glu436/Asp509 46/50 ·
DesD Asp497 46/50). The discriminating feature is therefore not *whether* the omega-amine is
anchored but whether anchoring it is **compatible with the carboxylate reaching P-alpha** — in DesD
it is; in the specialists it is not.

#### Limitations, binding on any citation of this result

1. 🔴 **Donor `H` has `n_gated` = 0/50 in ALL SIX enzymes, DesD included.** No ensemble here is a
   catalysis-competent complex. The result is strictly about *relative* registration of a
   non-productive ground state. It must never be stated as "DesD binds H productively".
2. 🔴 **p = 0.0312 is the floor of this design**, not a measurement; n = 5 admits nothing smaller.
3. **Leave-one-enzyme-out is uninformative at this n** — a 4/4 sign test gives p = 0.0625, so all
   five drops "fail" mechanically. The *direction* is unanimous and the magnitudes are
   well-separated from DesD; only the p-value is fragile. Reported as power, not as instability.
4. ⚠ **SfaB never docks** (all three donors 12-15 A, `n_gated` 0/50 throughout), so its +0.870 is a
   difference between two unphysical ensembles. DdaG and AcsA are marginal (5/50 on the diacids).
   Only AsbA and AsbB dock the diacids well.
5. 🔴 **The DesD check is confounded with training-set recall.** 7TGM is very likely in Boltz-2's
   training set and DesD's native HSC is itself a bifunctional omega-amino-acid donor, so DesD's
   correct placement of `H` may be memorised geometry generalising from HSC. This cuts against the
   check, not for it.
6. Boltz-2 has no energy function and no explicit charges; a "salt bridge" in these poses is a
   learned packing preference, not an electrostatic calculation.
7. **Amendment 7 still applies**: the activity matrix has zero interaction term, so even a real
   geometric consequence of donor class does not explain the activity data. This is a mechanistic
   observation about ground-state registration, not a validated activity predictor.
