# Structural proxies — nylon-hydrolase activity surfacing

**Goal.** Rank the nylon-hydrolase panel by how well each *structural* proxy, computed on
Boltz-2 cofold structures, surfaces the top experimental catalysts — the exact question and
metric used for the amide synthetases (`analysis/synthetase/STRUCTURAL_PROXIES.md`), so the
two campaigns are directly comparable. All proxies are computable from a single static
structure or the **50-sample diffusion ensemble** — no MD, no per-candidate QM/MM.

**Panel.** 96 enzymes, a **single fold**: N-terminal-nucleophile (Ntn) hydrolase, Peptidase
S58 / DmpA (InterPro IPR005321 + IPR016117) — the NylC family. Cofold = mature (autocleaved)
enzyme, split at the catalytic Thr so chain B residue 1 is the free N-terminal nucleophile,
plus a **nylon-6,6 tetrad** ligand. No ATP, no Mg.

**Reaction (Ntn amide hydrolysis).** The catalytic Thr **Oγ** attacks a scissile amide
**carbonyl C**; the Thr's own free **α-amino group is the general base** (the Ntn analog of a
Ser–His–Asp triad); the tetrahedral intermediate's oxyanion O is stabilized by an oxyanion
hole; C–N cleaves with leaving-amine protonation.

**Activity ground truth.** L2 = linear PA66 **dimer** release (µg/mL; 38/96 significant vs
no-enzyme control) as primary — the product directly matched to the cofolded tetrad and the
best-powered readout — with L1 = PA66 **monomer** release (15/96) reported alongside. Metric:
surfacing AUROC at nested cutoffs (top-5 / top-10 / top-20 most active, and > control) + mean.
Source: `exp_data/nylon_hydrolase/analysis/feature_surfacing_ranking_{L2,L1}.csv`. Figure:
`exp_data/nylon_hydrolase/figures/hydrolase_feature_surfacing.png`.

**Directions are FIXED by mechanism**, never picked from the observed correlation. On a small
positive set (esp. L1's 15) letting the sign follow a near-zero Spearman manufactures spurious
top-k AUROC — verified: the L1 oxyanion/TI/distance proxies read ~0.9 when flipped but are
~0.1 in the mechanistically correct direction. A `*` in the ranking/figure marks a proxy whose
data correlation runs *opposite* to mechanism (its apparent value is a sign-flip artifact).

Determinant key: **(i)** ground-state positioning · **(ii)** reactive-conformation / near-attack ·
**(iii)** electrostatic TS stabilization (oxyanion hole) · **(iv)** nucleophile/base activation ·
**(vi)** dynamic rigidity · **(--)** model-confidence null.

---

## A. Implemented (per the KG-informed hydrolase shortlist)

Shortlist source: `kg-memory-system` agent (391 records / 95 papers), cross-referenced to the
Ntn / Ser-nucleophile amide-hydrolysis literature. Anchors below.

| Proxy | Det. | Definition (ensemble cofold) | Dir. | L2 mean/top5 | L1 mean/top5 | Lit anchor |
|---|---|---|---|---|---|---|
| **Near-attack joint gate** | ii | Fraction of 50 poses with Oγ···C ≤ 3.5 Å **AND** Bürgi–Dunitz ∠(Oγ,C,O) ∈ [90,110]°. | + | 0.58 / 0.69 | 0.39 / 0.23 | PMID 37749089, 40891340 |
| **Bürgi–Dunitz deviation (gated)** | ii | Mean \|BD − 107°\| among poses with Oγ···C < 3.5 Å. | − | 0.49 / 0.54 | 0.32 / 0.24 | 37749089 |
| **Tetrahedral-intermediate template RMSD** | ii | Min RMSD of the reactive tetrad {Oγ,C,O,N} to an idealized sp3-TI template (v1, geometric). | − | 0.52 / 0.62 | 0.30 / 0.11 | 27243223, 39946508 |
| **Oxyanion-hole in-plane donors** | iii | Fraction of ensemble with ≥2 backbone/side-chain donors < 3.2 Å of the carbonyl O in the sp2 lone-pair plane. | + | 0.40\* / 0.37\* | 0.38\* / 0.25\* | 39946508, 34760153, 34085821 |
| **Oxyanion-hole donor count** | iii | Mean donor count < 3.2 Å of carbonyl O. | + | 0.46 / 0.51 | 0.30\* / 0.10\* | 39946508 |
| **Nucleophile activation (α-amine⋯Oγ)** | iv | Fraction of ensemble with Thr α-amino N···Oγ ∈ [2.5,3.4] Å (Ntn analog of the Ser–His H-bond). | + | 0.57 / 0.60 | 0.54 / 0.49 | 39946508 |
| **Reactive-center positional RMSF** | vi | Positional RMSF of Oγ across the 50-pose ensemble (CA-superposed frame); lower = preorganized. | − | 0.63 / 0.73 | 0.51 / 0.42 | 41187076, 29379013 |
| **Oxyanion electrostatic potential** | iii | Coulomb ESP at the carbonyl O from charged protein groups within 15 Å (ε=1; rank-only). | + | 0.44\* / 0.56\* | 0.34\* / 0.20\* | 35655887 |

Confidence **nulls** (expected non-mechanistic): complex pLDDT (L2 0.66/0.72; L1 0.57/0.57),
active-site Thr pLDDT (L2 0.51/0.46; **L1 0.64/0.81**), ligand ipTM.

Generators: `analysis/hydrolase/proxy_battery.py` → `hydrolase_proxies.csv`;
evaluator `analysis/hydrolase/feature_surfacing.py`; figure
`figures/make_hydrolase_feature_surfacing_figure.py`.

---

## B. Verdict — structural proxies do NOT surface hydrolase activity here

Unlike the fold-diverse synthetase panel (Boltz-2 near-attack angle surfaces the top-3
catalysts at AUROC **0.94**), **no catalytic-geometry proxy correctly surfaces the top
hydrolases** for either readout:

- The only feature with correct-direction signal for **both** L1 and L2 is **model confidence /
  active-site pLDDT** (a foldability/quality proxy, *not* catalytic geometry) — the same null
  the synthetase study excluded. Best mechanistic proxy: near-attack joint gate (L2 top-5 0.69),
  but it is matched by pLDDT and vanishes for L1.
- Every enzyme binds the tetrad (median best Oγ···C ≈ 2.5 Å in both actives and inactives) and
  productive near-attack is rare for *all* of them (47/96 have zero joint-NAC poses). The
  catalytic apparatus is essentially invariant across this single fold.

**Interpretation (consistent with the manuscript's own framing).** For a single-fold panel
assayed on **crystalline nylon powder**, the rate-limiting barrier is physical polymer
access / adsorption / thermostability — not catalytic chemistry on a soluble oligomer, which
is what a static cofold reports. Near-attack geometry surfaces catalysts when the panel varies
in *how well the reactive geometry forms* (diverse folds/substrates → synthetases); it cannot
when the fold is conserved and activity is access-gated. This is the informative contrast.

---

## C. Considered / excluded (hydrolase-specific, from the KG negatives)

| Proxy | Reason |
|---|---|
| **Nucleophile–electrophile distance alone** | Documented false negative (PMID 39121211: 3.6 Å in both active and 44,000×-slower states). Kept only inside the joint gate. |
| **Single-pose BD angle** | Productive geometry is an ensemble *frequency*, not a one-shot angle (PMID 40891340). We score the ensemble fraction. |
| **PROPKA on the N-terminal α-amine** | Mechanistically depressed, buried α-amine pKa is unreliable for PROPKA (KG caveat); not run — geometry (α-amine⋯Oγ) used instead. |
| **Coulomb ESP magnitude** | Physics-precedented (35655887) but the cheap MM/Coulomb form is uncalibrated; here it is *backwards* (\*) — ranking-only, and it fails. |
| **MM/GBSA binding** | No correlation with rate (PMID 39840811) — binding proxy only. |
| **Single-state preorganization / TI template** | Only the Michaelis-like cofold state is available; deacylation-state preorg (KG-recommended) is not accessible without a second modeled state. RMSF is single-state and confounded with pLDDT (ρ = −0.63). |

*Classification note.* KG corpus taxonomy is organized by descriptor type; here proxies are
tagged by catalytic determinant × mechanistic role × compute tier, matching the synthetase doc.
