# Structural proxies under consideration — amide synthetase activity surfacing

**Goal.** Rank the amide-synthetase panel by how well each *structural* proxy, computed on
Boltz-2 cofold structures, surfaces the top catalysts (top-3 actives = AS9, AS17, AS23).
All proxies are computable from a **single static structure or the 50-sample diffusion
ensemble** — no MD, no per-candidate QM/MM.

**Reaction (ANL adenylating amide synthetase):** adipate + ATP + HMDA → nylon-6,6 amide.
- **Step 1 — adenylation:** adipate proximal carboxylate O attacks ATP **Pα** in-line,
  displacing PPi, organized by **1 Mg²⁺** → acyl-AMP (adipyl-adenylate) intermediate.
- **Step 2 — aminolysis:** HMDA amine attacks the acyl-AMP **carbonyl C** → tetrahedral
  intermediate → amide + AMP.

**Metric.** Surfacing AUROC in each proxy's activity-consistent direction, at nested cutoffs
(top-3 / top-5 / top-10 most active, and activity > SfaB) + their mean. Source ranking:
`exp_data/amide_synthetase/analysis/feature_surfacing_ranking.csv`. Figure:
`exp_data/amide_synthetase/figures/synthetase_feature_surfacing.png`.

`†` = **conditional on the d<4 Å near-attack gate** (only scored on poses that already dock in
a reactive geometry) → survivorship-biased. All others are **always-defined** (all 48 enzymes).

Determinant key (the physical *why*): **(i)** ground-state positioning/complementarity ·
**(ii)** reactive-conformation attainment (NAC geometry) · **(iii)** electrostatic TS
stabilization/preorganization · **(iv)** nucleophile/base activation · **(v)** leaving-group /
metal-ion stabilization · **(vi)** dynamic sampling & rigidity · **(vii)** integrated barrier.

---

## A. Implemented — on the current figure (6 curated, one per determinant)

| Proxy | Det. | Step | Definition (how computed) | AUROC mean / top-3 | Dir. |
|---|---|---|---|---|---|
| **Distal-carboxyl anchoring** | i | 1 | Shrake–Rupley SASA of adipate's **distal (non-reacting) carboxyl** in the full step-1 complex, averaged over docked poses. Activity-consistent direction: *more exposed → more active* (enzyme grips the proximal carboxyl, distal end faces solvent). | 0.68 / 0.73 | + |
| **ATP adenine burial** | i | 1 | Burial (1 − relative SASA) of the ATP **adenine ring** in the step-1 complex. More buried = cofactor correctly seated. | 0.71 / 0.77 | + |
| **In-line attack angle** `†` | ii | 1 | Among poses with nearest carboxylate-O···Pα distance **< 4 Å**, the mean **O_nuc–Pα–O_leaving angle** (→180° = in-line). The validated near-attack proxy; strong top-3 (0.94) but survivorship-biased (mean collapses to 0.69). | 0.69 / **0.94** | + |
| **Mg²⁺–phosphate distance** | v | 1 | Median distance from the single Mg²⁺ to ATP phosphate oxygens, averaged over docked poses. Best **always-defined** proxy. | 0.76 / 0.81 | + |
| **Reactive-center positional RMSF** `†` | vi | 1 | Positional RMSF of the reactive-center atoms **across the 50-pose ensemble** (lower = more preorganized/rigid). Strongest overall; ensemble-variance analog of PLACER preorganization (PMID 41187076). | 0.79 / 0.85 | − |
| **Acceptor-tunnel enclosure** `†` | i/ii | 2 | Geometric enclosure ("tightness") of the channel along the inferred **amine-approach vector** to the reactive center — can the HMDA amine be delivered to the carbonyl. | 0.71 / 0.77 | − |

Generators: `battery_1mg.py` (angle, Mg), `reactive_sasa_intermediate.py` (distal SASA),
`hypothesis_battery.py` (adenine burial), `reactive_preorganization.py` (RMSF),
`acceptor_tunnel_geometry.py` (tunnel). Figure driver: `figures/make_feature_surfacing_figure.py`.

---

## B. Piloting now (physics-motivated additions — fill the electrostatics & activation gaps)

| Proxy | Det. | Step | Definition (how computed) | Status | Lit anchor |
|---|---|---|---|---|---|
| **ESP at reactive atom** | iii | 1 & 2 | Electrostatic potential at the reactive atom from surrounding **protein partial charges**: ESP = Σᵢ qᵢ / rᵢ over protein atoms within 15 Å (exclude ligand), ε = 1, averaged over the ensemble. Reactive atom = **Pα** (step 1) / **acyl-AMP carbonyl C** (step 2). | running | PMID 35655887 (ESP on catalytic-His Nε → barrier, linear) |
| **Projected local electric field (LEF)** | iii | 1 & 2 | Protein electric-field *vector* at the reactive atom, **projected onto the scissile/forming bond**: O_nuc→Pα (step 1); C=O (step 2, Boxer convention). Warshel/Boxer electrostatic preorganization. *Ranking-only* (cheap Coulomb field ≠ QM/MM magnitude). | running | PMID 38455014 (field on Fe=O axis), 35513508 (VSE) |
| **Active-site pKa (PROPKA)** | iv | 1 & 2 | PROPKA per pose; summarize protonation environment near the reactive atom (Asp/Glu pKa shift, min Lys pKa, His pKa within ~10 Å), averaged over ensemble. Fills the **nucleophile/base-activation** determinant nothing else covers. | running | PMID 30254369 (catalytic Glu101 pKa → efficiency) |

Generators (in progress): `esp_reactive.py` → `analysis/esp_reactive.csv`;
`propka_active_site.py` → `analysis/propka_active_site.csv`.

---

## C. Queued (next, pending pilot-B results)

| Proxy | Det. | Definition | Why | Lit anchor |
|---|---|---|---|---|
| **Reactive-moiety vdW packing** | i/vi | Lennard-Jones (12-6) vdW interaction energy between the reactive moiety (carboxyl/carbonyl) and the protein, generic per-element params (no ligand charges needed), over ensemble. Reframed from "catalytic-base vdW" (we have no classic catalytic base). | Strongest *quantitative, cheap* descriptor in the corpus. | PMID 40533551 (catalytic-Asp vdW → kcat/KM, ρ = **−0.88**) |
| **Active-site pocket volume** | i | Geometric pocket volume (Fpocket/POVME-style) around the reactive center, over ensemble. | Recurring cheap geometric correlate of kcat/KM; not yet in our panel. | PMID 40533551, 26091851 |

---

## D. Considered and excluded (with reason)

| Proxy | Reason excluded |
|---|---|
| **ps-iPTM** (cofold interface confidence) | Physically unmotivated as a catalytic determinant (it's a model-confidence score) *and* weak (mean AUROC 0.55). |
| **Near-attack pose fraction** (`near_attack_frac`) | The unbiased form of the gated angle, but genuinely weak for us (0.59); buries AS9. |
| **Mg²⁺ inner-sphere coordination** | Near-random (0.46) — depends on cofold Mg-placement fidelity the model doesn't get right. |
| **Reactive-carbon burial (resting adipate)** | Measures the wrong species (resting adipate, not the acyl-adenylate); correctly near-dead (0.53). |
| **MM-GBSA binding energy** | Explicit literature null vs activity (PMID 39840811). Keep only as a *binding* proxy, never activity. |
| **QM/MM barrier, EVB, CAMM-DTSS** | The field's most predictive tier, but all require MD/QM per candidate — out of our compute budget. |

---

*Classification note.* The corpus taxonomy (kg-memory-system, 238 records / 79 papers) is
organized by **descriptor type**, which smears single determinants (e.g. "preorganization")
across Geometric/Dynamics/Electrostatics buckets. Here proxies are tagged by **catalytic
determinant × mechanistic step × compute tier** instead, so gap analysis is unambiguous: we
now cover determinants i, ii, v, vi; pilots B add iii and iv; vii (integrated barrier) remains
out of reach without QM/MM.
