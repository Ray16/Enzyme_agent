# Structural proxies — amide synthetase activity surfacing (final panel)

**Goal.** Rank the amide-synthetase panel by how well each *structural* proxy, computed on
Boltz-2 cofold structures, surfaces the top catalysts (top-3 actives = AS9, AS17, AS23).
All proxies are computable from a **single static structure or the 50-sample diffusion
ensemble** — no MD, no per-candidate QM/MM.

**Design rule.** Comprehensive but physically motivated: keep every proxy that is
mechanistically justified and non-redundant; drop the unmotivated, dead, or artifactual.
Proxies span **catalytic determinants** (the physical *why*), one or more per determinant.

**Reaction (ANL adenylating amide synthetase):** adipate + ATP + HMDA → nylon-6,6 amide.
- **Step 1 — adenylation:** adipate proximal carboxylate O attacks ATP **Pα** in-line,
  displacing pyrophosphate (Pβ–Pγ), organized by **1 Mg²⁺** → acyl-AMP intermediate.
- **Step 2 — aminolysis:** HMDA amine attacks the acyl-AMP **carbonyl C** → tetrahedral
  intermediate → amide + AMP.

**Metric.** Surfacing AUROC in each proxy's activity-consistent direction, at nested cutoffs
(top-3/5/10 + activity > SfaB) + their mean. Ranking:
`exp_data/amide_synthetase/analysis/feature_surfacing_ranking.csv`. Figure:
`figures/make_feature_surfacing_figure.py` → `exp_data/amide_synthetase/figures/synthetase_feature_surfacing.png`.

Markers: `†` conditional on the d<4 Å near-attack gate (survivorship-biased); `‡` His-present
subset (n=44 of 48); unmarked = always-defined (48 enzymes).

Determinant key: **(i)** ground-state positioning · **(ii)** reactive-conformation (NAC) ·
**(iii)** electrostatic TS stabilization/preorganization · **(iv)** nucleophile/base activation ·
**(v)** leaving-group/metal stabilization · **(vi)** dynamic sampling & rigidity ·
**(vii)** integrated barrier.

---

## Final panel (6 proxies, on the figure)

Every proxy passes a **direction-vs-mechanism audit**: its activity-consistent direction
matches a clean physical mechanism (not a backwards/confounded correlation).

| Proxy | Det. | Step | Definition (how computed) | mean / top-3 | dir | AS9 rank |
|---|---|---|---|---|---|---|
| **Distal-carboxyl solvent exposure** | i | 2 | SASA of the distal (free, non-adenylated) carboxyl of the adipyl-AMP intermediate. More exposed → more active (distal carboxyl solvent-facing = correct intermediate orientation). | 0.68 / 0.73 | + | 2 |
| **ATP adenine burial** | i | 1 | Burial (1 − rel. SASA) of the ATP adenine ring — cofactor correctly seated. | 0.71 / 0.77 | + | 5 |
| **In-line attack angle** `†` | ii | 1 | Among poses with carboxylate-O···Pα < 4 Å, mean O_nuc–Pα–O_leaving angle (→180°). Validated near-attack proxy; strong top-3, survivorship-biased. | 0.69 / **0.94** | + | 6 |
| **Reactive-center positional RMSF** `†` | vi | 1 | Positional RMSF of reactive-center atoms across the 50-pose ensemble (lower = more preorganized). Strongest overall. | **0.79** / 0.85 | − | 6 |
| **Electric-field preorganization (LEF)** | iii | 1 | Ensemble **spread (std)** of the protein electric field projected on the O_nuc→Pα axis, from FF partial charges (Coulomb, ε=1). Lower variability → more active (Warshel electrostatic preorganization; the field *consistency*, not its cheap magnitude). | 0.70 / 0.71 | − | 5 |
| **Carbonyl-oxyanion ESP (step 2)** | iii | 2 | Electrostatic potential at the acyl-AMP carbonyl C from protein charges (Σ qᵢ/rᵢ, 15 Å), ensemble mean. More positive → stabilizes the developing tetrahedral oxyanion. | 0.65 / 0.67 | + | 13 |

Generators: `battery_1mg.py` (angle), `archive/reactive_sasa_intermediate.py` (distal SASA),
`hypothesis_battery.py` (adenine burial), `reactive_preorganization.py` (RMSF),
`esp_reactive.py` (LEF, carbonyl ESP).

**Determinant coverage:** i, ii, iii, vi covered. **(iv) nucleophile/base activation,
(v) metal, and (vii) barrier are NOT covered** (see below).

---

## Determinant (v) metal — tested and UNCOVERABLE on this cofold

The mechanistically-correct metal descriptor is Mg²⁺ **bridging the β,γ-diphosphate**
(stabilizing the departing pyrophosphate). We tested it (`mg_bridging.py`) and every form of
the metal proxy fails on Boltz-2 structures:

| metal descriptor | best top-3 | ρ | AS9 rank | verdict |
|---|---|---|---|---|
| Mg–phosphate distance (coarse) | 0.81 | +0.35 | 25 | backwards direction; misses AS9 |
| Mg²⁺ β,γ-bridging (frac ≤2.4 Å) | 0.79 | +0.13 | 22 | ρ≈0; 2-of-3 artifact |
| Mg²⁺ β,γ co-coordination tightness | 0.66 | −0.05 | 10 | flat |
| Mg inner-sphere coordination | 0.50 | — | — | near-random |

**Conclusion:** the physics is right, but **Boltz-2 does not place Mg²⁺ with the fidelity needed
to resolve catalytic metal geometry.** Determinant (v) is left uncovered — a structure-prediction
limitation, not a proxy-design failure. Stated plainly rather than shipping a good-but-backwards
number.

---

## Considered and excluded

**Failed the direction-vs-mechanism audit** (good AUROC, but activity-consistent direction
contradicts the stated physics — the "Mg trap"):

| Proxy | AUROC | What the data actually says | Verdict |
|---|---|---|---|
| **Catalytic His pKa** (min\|pKa−7\|) | 0.75 | Top actives (AS9, AS23) have His pKa in the top 10–15% of *deviation* from 7 → His **far** from neutral, not "poised at 7". Falsifies the proton-shuttle motivation; lumps protonated & neutral His. | drop (det. iv uncovered) |
| **Acceptor-tunnel enclosure** | 0.71 | Actives have **more open** tunnels (AS9/AS23 in bottom 5–19% of enclosure) → contradicts the walled-pocket / water-exclusion hypothesis; only rescuable by an ad-hoc accessibility story; weak (ρ −0.13). | drop |
| **Mg–phosphate distance** | 0.76 | Farther Mg → more active (ρ +0.35); ranks AS9 25/48. Backwards for metal catalysis. | drop |

*Note:* the perturbed-His-pKa observation (actives have a strongly shifted active-site His) is
a possible future hypothesis (electrostatic stabilization by a cationic His), but only if the
shift is consistently one-directional — not built (would be post-hoc).

**Other exclusions:**

| Proxy | Reason |
|---|---|
| **Step-2 aminolysis NAC** (amine→carbonyl dist/Bürgi–Dunitz) | Weak (0.55–0.63, AS9 ranks 11–18); step-2 geometry is low-signal on these cofolds. |
| **All Mg proxies** (distance, β,γ-bridging, coordination) | Boltz Mg-placement fidelity (above). |
| **Bare Coulomb-LEF magnitude** | Near-random (~0.53–0.56); MM-charge field magnitude is a crude approximation (only the ensemble *spread* carried signal). |
| **ESP-at-Pα (step 1)** | Direction not mechanistically clean; kept only the physically-clean carbonyl ESP. |
| **`s1_nion` (active-site ionizable count)** | 0.75 but confounded with pocket burial, not a clean base-activation story. |
| **Step-2 pKa** | Null (0.45–0.65) — aminolysis base-activation not supported. |
| **ps-iPTM** | Model confidence, not a catalytic determinant; weak (0.55). |
| **Near-attack pose fraction** | Weak (0.59); buries AS9. |
| **Reactive-carbon burial (resting adipate)** | Wrong species; near-dead (0.53). |
| **MM-GBSA** | Explicit literature null vs activity (PMID 39840811). |
| **QM/MM barrier, EVB, CAMM-DTSS** | Most predictive tier, but require MD/QM per candidate — out of budget. |

*Classification note.* Proxies are tagged by **catalytic determinant × mechanistic step ×
compute tier** rather than the corpus's flat descriptor-type list (which smears single
determinants across buckets). Cross-referenced against the kg-memory-system taxonomy
(238 records / 79 papers); only ~17% of that corpus is cheap-tier (static/ensemble).
