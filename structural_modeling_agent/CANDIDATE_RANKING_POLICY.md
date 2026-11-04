# Candidate ranking policy

## Current metric

The current candidate workflow uses `inverse_distance_attack_angle_v1`, defined
in `knowledge/attack_angle_metric.json` and implemented in
`proxies/attack_angle.py`. The ensemble angle is
`A = sum(angle_deg / distance_A) / sum(1 / distance_A)`; larger angles rank higher.
Always report `harmonic_distance_A = N / sum(1 / distance_A)` alongside it.
The angle has no fitted parameters and is invariant to uniform distance scaling.
Absolute proximity, setup validity, full sampling coverage, mechanism applicability
and experimentally established activity remain separate observations.

`integration_examples.campaign_configs()` selects this metric for the current
synthetase application. The integration API retains the older
`geometry_ensemble` kind only for explicitly supplied historical bundles; it does
not substitute that score for the current angle. Historical calibration statistics
below apply to their original definitions, not to the current metric.

## Historical exploratory continuous-score analysis (2026-10-06)

Discovery evidence remains primary. One continuous first-step geometry score is
evaluated for the SfaB assay panel, DdaG donor series, and modeled discovery
candidates. Identical parameters provide consistent scoring; predictive transfer
between enzyme families has not been validated. Scores must not automatically
override native-reaction evidence or reported target activity.

The frozen exploratory implementation is
`analysis/amide_synthetase/continuous_score.py`. For each of 50 predictions:

- Select the nearest carboxylate oxygen to the reaction-specific ATP phosphorus.
- Use P-alpha/O3A for AMP-forming chemistry and P-gamma/O3B for ADP-forming chemistry.
- Set `w = sigmoid((3.5 - distance_A) / 0.25)`.
- Set `q = exp(-0.5 * ((180 - angle_deg) / 30)^2)`.
- Score the ensemble as `sum(w*q) / (0.1 + sum(w))`.

There is no minimum-pose cutoff or compatible/incompatible score threshold.
Report total distance weight and sampling uncertainty alongside the score. A
low score can reflect missing assembly context or inadequate sampling; a high
score does not validate metal coordination or the amine-dependent step.
Metal inputs must follow candidate-specific evidence: the audited archived
ensembles used one or two Mg ions, not two universally.

Selection compared 81 settings against the 48-homolog panel. The selected score
places AS17, AS23, and AS9 at ranks 1, 3, and 5 (calibration AUROC 0.978). This is
an in-sample result, not universal validation. AS41 and AS8, the fourth and fifth
experimental producers, rank 23rd and 43rd. Top-five AUROC is 0.721 and only
three of the five producers enter the top ten. This score does not yet meet the
broader recovery objective and must not be promoted to a general activity ranker. Full selection diagnostics and
scores for all 88 ensembles are in `analysis/amide_synthetase/`.

Plotting scripts write to `figures/candidacy/`. Run
`analysis/amide_synthetase/publish_manuscript_figures.py` to copy only PNG figures
referenced in the manuscript into Overleaf. Code, data, and exploratory figures
remain in this structural-modeling tree.

## Historical threshold analysis

The policy below documents the earlier analysis and its legacy table builder.
It is retained for interpreting archived outputs, not for current score selection.

The discovery-agent evidence rank is primary. Boltz-2 geometry is an orthogonal,
recall-oriented annotation and is compared only within the same mechanistic family.

1. Assign the activation mechanism before scoring.
   - Acyl-adenylate families: adipate carboxylate O attacks ATP P-alpha along
     the P-alpha--O3A leaving-group axis.
   - Acyl-phosphate families: adipate carboxylate O attacks ATP P-gamma along
     the P-gamma--O3B leaving-group axis.
   - An unrecognized family is `unresolved` and receives no structural rank.
2. Generate 50 unrestrained Boltz-2 cofold samples with protein, ATP,
   candidate-specific metal inputs, and adipate. Do not impose a pocket restraint.
3. Retain poses with nucleophile-to-reactive-phosphorus distance `d < 4.0 A`.
4. Fewer than five retained poses is `inconclusive`, not a negative result.
5. For candidates with at least five retained poses, calculate the mean in-line
   angle and rank only within the assigned mechanistic family.
6. Report structural support as:
   - `>=165 degrees`: strong geometric support
   - `150-<165 degrees`: compatible
   - `<150 degrees`: weak geometric support
   - unresolved mechanism or `<5` retained poses: inconclusive

Candidates without completed structural results are omitted from the
structural table. The angle is a coarse competence filter, not a quantitative
prediction of experimental yield, and does not automatically override biochemical
or literature evidence.

Generate the table with:

```bash
python structural_modeling_agent/scripts/build_candidate_prioritization.py
```
