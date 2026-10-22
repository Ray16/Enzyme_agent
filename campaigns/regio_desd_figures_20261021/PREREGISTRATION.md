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
