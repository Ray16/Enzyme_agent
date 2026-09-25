# Candidate ranking policy

The discovery-agent evidence rank is primary. Boltz-2 geometry is an orthogonal,
recall-oriented annotation and is compared only within the same mechanistic family.

1. Assign the activation mechanism before scoring.
   - Acyl-adenylate families: adipate carboxylate O attacks ATP P-alpha along
     the P-alpha--O3A leaving-group axis.
   - Acyl-phosphate families: adipate carboxylate O attacks ATP P-gamma along
     the P-gamma--O3B leaving-group axis.
   - An unrecognized family is `unresolved` and receives no structural rank.
2. Generate 50 unrestrained Boltz-2 cofold samples with protein, ATP, two Mg ions,
   and adipate. Do not impose a pocket restraint for this score.
3. Retain poses with nucleophile-to-reactive-phosphorus distance `d < 4.0 A`.
4. Fewer than five retained poses is `inconclusive`, not a negative result.
5. For candidates with at least five retained poses, calculate the mean in-line
   angle and rank only within the assigned mechanistic family.
6. Report structural support as:
   - `>=165 degrees`: strong geometric support
   - `150-<165 degrees`: compatible
   - `<150 degrees`: weak geometric support
   - unresolved mechanism or `<5` retained poses: inconclusive

The five candidates without completed structural results are omitted from the
structural table. The angle is a coarse competence filter, not a quantitative
prediction of experimental yield, and does not automatically override biochemical
or literature evidence.

Generate the table with:

```bash
python structural_modeling/scripts/build_candidate_prioritization.py
```
