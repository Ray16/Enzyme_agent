# Pre-registration: can an annotation-free structural descriptor recover the native amine class?

**Written 2026-10-10, BEFORE any descriptor value was computed.** Nothing in this file was
chosen after seeing a result. Cohort and labels were fixed first (both derive from
annotation + pose counts only); thresholds and tests are specified below.

## Motivation

The proximity-weighted angle `A` scores **step 1** only — donor carboxylate attacking the
ATP phosphorus. The amine enters at **step 2** and is absent from every ranking we have
produced. Measured consequence: the five candidates `A` ranks at the top are all
**ammonium** enzymes (`substrate_fit` 0.00), while the two that natively acylate an
aliphatic diamine (PubC, PuuA) have the *lowest* `A` of the tier-A set. For the target
reaction (adipate + hexamethylenediamine) this is the wrong way round.

Fixing it by adding an amine term **into** `A` is rejected: `A` carries the project's only
retrospective validation (AUROC 0.714, 2.80x top-10, p=0.037 on the 49-member panel), and
folding in an untested term voids it. The amine dimension must be a **separate reported
axis with its own gate**, never summed into `A`.

For future candidates there will be no native-substrate annotation. So the descriptor must
be **purely structural**. Annotation is used here *only as the held-out label that judges
the descriptor* — never as an input. Same firewall as retrieval.

## The descriptor (annotation-free)

The amine subsite is a geometric consequence of the step-1 pose; no annotation is needed to
locate it. For each of the 50 poses:

1. Identify the donor carboxylate carbon `C*` — the carboxylate carbon whose oxygen is
   nearest the reactive phosphorus `P` (PA for AMP-forming, PG for ADP-forming, taken from
   each campaign `spec.json`; donor chain is **D** for 1-Mg/Pa systems and **E** for
   2-Mg/Pg systems — read it, do not hardcode).
2. `O_lg` = that nearest donor oxygen (it becomes the leaving group after acyl transfer).
3. The step-2 nucleophile must approach `C*` anti to `O_lg`. Place the probe point at
   `site = C* + 3.0 A * unit(C* - O_lg)`.
4. Run `channel_at_site(coords, radii, site)` from
   `exp_data/nylon_hydrolase/code/channel_geometry.py` (CAVER-equivalent widest-path
   bottleneck; validated on 9DYS at 1.38/1.69 A vs the paper's 1.5 A).
   **The ligand being probed is excluded from the occupancy grid**; protein + ATP + Mg are
   retained. Measured on holo poses only (the hydrolase apo tunnel read closed from
   induced fit).

Reported per enzyme, as the median over the 50 poses:

- **`bottleneck_A`** - largest probe radius that reaches bulk solvent from `site`
- **`channel_len_A`** - path length from `site` to bulk solvent

## Cohort (fixed before measurement)

All 36 accessions with a >=40-pose `step1_<ACC>` ensemble. Labels from the Rhea-derived
`native` field of `discovery_shortlist_gs_corrected.csv`.

**Rule:** does the native nitrogen nucleophile sit at the end of an aliphatic chain of
>=3 carbons (an extended channel is required), or is it a bare/point nucleophile?

**BARE (n=13)** - ammonium / NH3:
Q5UR44, Q54WR9, P13564, O22506, A1AHS4, P07694, P41320, Q43066, P00965, P14654, P08281,
P04770, A5UK38

**EXTENDED (n=13)** - amine at the end of a >=3-carbon chain:
P0DXD6, P78061, Q9RDS6 (putrescine) · Q81RQ8, A0A0F7RJ52, O60993, P90518 (spermidine) ·
Q46E79, Q47316, Q47318, P18204 (lysine eps-/N6-amine) · Q2FW70, Q2FW72 (ornithine delta-/N5-amine)

**SHORT / ambiguous (n=10)** - excluded from the primary test, retained as a held-out
intermediate class: A2SQ86, P52420 (glycine) · Q2G1M9, Q2G1M8, E2JA32 (3-amino-L-alanine) ·
C7P8V7 (tyramine) · Q44249 (aniline) · Q936T0 (isopropylamine) ·
A0A0H2WWP1 (D-Ala-D-Ala) · Q2G1N1 (short carbamoyl tether)

## Pre-specified predictions and tests

**Primary (confirmatory), BARE vs EXTENDED, n=13+13:**
- H1: `bottleneck_A` is larger in EXTENDED. One-sided Mann-Whitney, **alpha = 0.05**. Report AUROC.
- H2: `channel_len_A` is larger in EXTENDED. One-sided Mann-Whitney, **alpha = 0.05**.
- Two primary tests -> **Bonferroni alpha = 0.025 each**. Both reported regardless of outcome.

**Pre-specified absolute thresholds** (from vdW requirements, not fitted):
a C6 methylene chain needs **bottleneck >= 1.9 A**; NH3 needs only ~1.5 A. A diamine's
distal NH3+ needs somewhere to sit, so **channel_len >= 7 A**. Fisher exact on the
2x2 of (passes both thresholds) x (EXTENDED).

**Secondary (ordinal, not counted in the alpha budget):** SHORT should fall *between* BARE
and EXTENDED. Jonckheere-Terpstra for the ordered alternative BARE < SHORT < EXTENDED. A
monotone trend across three classes is substantially harder to obtain by chance than a
binary split and is the stronger evidence if it holds.

**Covariate check:** `bottleneck_A` vs protein length (Spearman). If the descriptor is a
size artifact it must survive partialling length out. Also report vs `n_gated`, since site
placement depends on the donor pose.

## Declared failure modes - stated now, not after the fact

1. **Saturation / null.** This same tool on the nylon hydrolase panel returned AUROC 0.596,
   selection-corrected p=0.37-0.56, and the related beam-threading descriptor saturated
   against its crystal controls. If `bottleneck_A` >= 1.9 A for essentially every enzyme it
   separates nothing. That is a real possible outcome and will be reported as such.
2. **Site-placement failure** where the donor pose is unreliable. PylC has n_gated 1/50 and
   PuuA 10/50, so their `site` is poorly determined. Both are in EXTENDED. A sensitivity
   analysis dropping all enzymes with n_gated < 40 is pre-specified, and it will shrink
   EXTENDED badly - reported either way.
3. **Class imbalance in fold.** EXTENDED is NIS/ATP-grasp-rich and BARE is GS-rich, so a
   positive result may be the fold confound again (GS median A 148.4 vs 131.1, p=0.0007)
   rather than the amine. Pre-specified: repeat the primary test **within GS only** and
   **within non-GS only**. If it dies in both strata, it is fold, not amine.

## Commitments

- No threshold, class assignment, or test will be changed after seeing a value.
- If the primary tests fail, the descriptor is reported as **null** and is not rescued by
  trying other site definitions, probe radii, or aggregations. Any such exploration is
  labelled post-hoc and excluded from the alpha budget.
- Success criterion for deployment on unannotated enzymes: **both primary tests pass at
  Bonferroni alpha=0.025 AND the descriptor survives the length covariate AND it does not
  die in both fold strata.**
