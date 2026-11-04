# Production quality and required-partner comparisons

The fresh production run improved evidence publication and found a paper-linked
starting point. It did **not** deliver 48 strong candidates or demonstrate the
requested reaction. The saved ranking needs correction before experimental use.
The machine-readable audit is [production_quality_20261104.json](production_quality_20261104.json).

## What completed

Run `20261104_004330_Find_enzymes_that_ligate_glutarate_and_cadaverine` loaded
source revision `03cc3891` and stopped at its conservative token-admission limit.
It used 2,906,149 reported tokens over 7,511.77 seconds: 1,541 registered records,
204 target assessments, 68 ranked hypotheses and one primary proposal. Six saved
assessments said prioritize, 63 investigate, 128 defer and seven exclude.
These are model decisions, not experimental activity measurements.

Review calls used 58.46% of tokens, down from 81.75% in the earlier cited run;
controller calls used 40.04%. Overall prompt-cache reads increased from 13.74%
to 21.49%, but wall time increased from 113.8 to 125.2 minutes. This comparison
does not establish a speedup; the runs explored different records and used
different execution concurrency. No GPU predictions or MSA jobs were submitted.

The literature-lead counter reads zero registered lead objects. It misses one
actual paper-to-sequence resolution performed through an explicit RefSeq lookup.
It should not be interpreted as zero productive literature work.

## Both required substrates

All 204 saved assessments explicitly mapped target participants T1 and T2.
The saved molecular-property comparisons are less complete:

| Resolved target comparisons | Assessed candidates |
| --- | ---: |
| Both | 89 |
| Only one | 95 |
| Neither | 20 |

The 95 partial cases received zero under the earlier accommodation logic. They
were not awarded an optimistic score based only on the first partner. However,
zero conflated an unresolved comparison with a measured poor analogue, and
per-axis ranking could obscure the weakest partner even for complete records.
Some annotated counterpart structures were unresolved or generic; ontology
preferred names and reaction aliases also failed to exclude products reliably.

The repairs make the following behavior explicit:

- Resolve partners within one native reaction, retaining source stoichiometry.
  Ontology aliases help identify which participants are reactants or products;
  positional chemical labels are preserved when stripping charge notation.
- Use the same one-to-one property-based assignment in family exploration,
  accommodation and candidate evaluation. Distinct recorded occurrences may
  supply separate roles. Descriptor pairing proposes analogues; reacting atoms
  and biochemical roles still require evidence.
- Use the **minimum required-partner fit**, without averaging good and poor
  partners together. Complete records with unequal bottleneck scores follow
  that order within the same scientific-priority stratum. Source-supported
  judgments and observed conflicts remain separate ranking factors.
- Keep incomplete comparisons unknown. Family summaries cannot substitute a
  favorable mean for a missing partner. Unknown and known joint fits remain
  incomparable rather than creating inconsistent dominance relations.
- Ask both controller and assessor to keep a major unsupported partner mismatch
  exploratory, unless candidate-specific evidence supports acceptance of that
  limiting partner. Do not equate molecular descriptor gaps with demonstrated
  inactivity or physical exclusion from a protein pocket.
- Export each partner's fit, comparison coverage, chemical-role assessments and
  an explicit statement that resemblance is not binding or activity evidence.

The requested panel size is capacity, not a ceiling or a quality quota. Missing
annotations remain opportunities for targeted retrieval and exploratory work;
they do not become positive or negative activity results.

## Ranking classification defect

The target contract used descriptive bond labels while a native-evaluation
warning compared them literally with canonical labels. This produced bond-type
mismatch warnings in 189 assessed records, including all six saved prioritize
decisions. Unknown-function records could escape this warning and rise above
better-supported hypotheses.

Native claims and warning checks now share canonical bond normalization, and
target-contract instructions request canonical labels. A diagnostic ranking of
the same 68 saved rows removes or changes 62 such warnings. In that diagnostic,
six rows occupy tier 1, 37 tier 2 and 25 tier 3. This does not validate their
substrate acceptance. The old descriptive contract also mentioned activation
alternatives; normalizing it is not an independent mechanistic adjudication.

The original run's registry, CSV and source packets were preserved. The local
diagnostic is `discovery_validation/20261104_joint_substrates/corrected_diagnostic_ranking.csv`.
It uses saved judgments and corrected mechanical checks; it adds no candidates
or new model assessments and is not a new production result.

## What the useful hypotheses actually support

| Candidate | Recorded support | Remaining target gap |
| --- | --- | --- |
| Paper-associated ShABS / UniProt Fatty-acyl-CoA synthase (A0A1M5ABR5) | The paper names RefSeq WP_073481158.1; independently retrieved RefSeq and UniProt sequences match exactly over 508 residues. | The read source does not establish either glutarate acceptance or cadaverine acceptance. Paper identity is not exact-target activity. |
| Gamma-glutamylpolyamine synthetase GlnA2 (Q9RDS6) | The saved UniProt reaction cites experimental evidence for cadaverine coupling with L-glutamate. | Glutarate replacing L-glutamate remains unsupported. |
| Gamma-glutamylpolyamine synthetase GlnA3 (Q9KZC7) | The saved UniProt reaction cites experimental evidence for cadaverine coupling with L-glutamate. | Glutarate replacing L-glutamate remains unsupported. |
| Gamma-glutamylputrescine synthetase PuuA (P78061) | A characterized native amine-coupling reaction supplies a testable transfer hypothesis. | Neither the change in native acid recognition nor acceptance of the requested amine is established by the inspected evidence. |

These are development observations from the requested campaign, not held-out
generalization evidence. No inspected positive source demonstrates the complete
requested substrate pair. Current names and source identifiers here are audit
outputs; they were not added to runtime prompts, defaults or discovery seeds.

## Verification and limitations

Regression coverage includes weakest-partner ordering, partial and repeated
participants, invalid numeric comparisons, shared assignment across exploration
and evaluation, family-summary consistency, ontology aliases, reaction
stoichiometry, bond-label normalization and incomparable evidence coverage.
The release-suite counts are recorded in [review_validation.json](review_validation.json).

The changes have software and post-run diagnostic validation. They have not had
another paid discovery run, structural prediction campaign or experimental
activity test. Cache reuse and end-to-end throughput still need improvement.
Scores shared by homologues with the same annotated reaction remain shared
analogue context, not independent protein-specific selectivity measurements.
