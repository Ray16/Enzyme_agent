# Pre-registration: can a structural proxy recover measured carbamate-hydrolysis activity?

**Written 2026-10-11, BEFORE any structure was folded and before any descriptor value
was computed.** At the time of writing there are zero cofolds and zero `.a3m` files on
disk; MSA generation is still running. Nothing below was chosen after seeing a result.

## Why this panel is worth the attempt

Six proxy families have already come back null in this project, and two reported hits
were retracted. The difference here is the label, not the optimism:

- **Continuous, 5.2 orders of magnitude** (21.0 down to 0.00014 umol/min/umol).
  The synthetase panel had 3 positives; the hydrolase panel was floor-clustered with
  half the values within 2x of the no-enzyme control.
- **One lab, one protocol, one substrate** (Bendtsen et al., Angew 2026,
  DOI 10.1002/anie.6159643, Table 1), retrieved through
  `enzyme_discovery_agent.tools.measurement_extraction`, not transcribed by hand.
- A continuous label means the test is a rank correlation, so there is **no activity
  cutoff to choose**. Cutoff-shopping is what produced this project's retracted
  "3.1x enrichment, p=0.02" (9 cutoffs tried, 1 cleared 0.05). It is unavailable here
  by construction.

## Cohort (fixed before folding)

n = 11 enzymes with a PURp1 specific activity in `activity_labels.json`. The two rows
with activity "NA" and UMG-SP2 (activity reported as "—") are excluded, and that
exclusion is by missing label only, never by value.

Catalytic nucleophile is resolved for all 26 panel members (`catalytic_residues.json`),
6 from annotation and 20 by alignment transfer; every transfer lands in the expected
motif for its fold (amidase-signature GG-S block, or a GxSxG elbow for the three
carboxylesterases).

## Ligand

PURp1 — the molecule the activity was **measured on**, not the campaign target.
`CCCCN1C(=O)c2cccc3c(NC(=O)OCCOCCOCCOC)ccc(c23)C1=O`, C24H30N2O7, verified against the
paper's own HRMS. It carries exactly one carbamate, so the reactive site is unambiguous.
Folding a different carbamate and comparing to these numbers would test the proxy on the
wrong substrate.

## The descriptors — FIVE, fixed here, no additions

Each is computed per pose over the cofold ensemble and reduced to one value per enzyme.
`O_nuc` is the catalytic Ser OG / Thr OG1; `C_carb` the carbamate carbonyl carbon;
`O_carb` its carbonyl oxygen; `O_lg` the alkoxy (leaving-group) oxygen.

| # | descriptor | definition | predicted direction |
|---|---|---|---|
| D1 | `near_attack_fraction` | fraction of poses with d(O_nuc,C_carb) <= 3.5 A AND Burgi-Dunitz angle in 95-125 deg | higher = more active |
| D2 | `A_proximity_weighted_angle` | the project's proximity-weighted attack angle, nucleophile to carbonyl | higher = more active |
| D3 | `d_nuc_carbonyl` | median d(O_nuc, C_carb) over poses | LOWER = more active |
| D4 | `leaving_group_alignment` | median angle O_nuc...C_carb-O_lg (anti-periplanar departure, ideal 180) | higher = more active |
| D5 | `oxyanion_occupancy` | mean count of backbone/side-chain H-bond donors within 3.5 A of O_carb | higher = more active |

Five is not a budget chosen for convenience. At n = 11 the correlation needed for
p < 0.05 is |rho| >= 0.60 for a single descriptor, 0.74 at k = 5, 0.78 at k = 10 and
0.81 at k = 20. A twenty-descriptor battery would require a correlation no structural
proxy in this project has ever reached, so a broad search is self-defeating rather than
merely risky. **No descriptor may be added after folding.** If one is, the whole test
reverts to exploratory and must say so.

## Primary test

Spearman rho between each descriptor and **log10(specific activity)**, n = 11.

- **Pre-registered primary:** D1, one-sided in the predicted direction.
- D2-D5 are secondary and reported with the same test.
- **Selection correction:** the reported p for the best descriptor is the probability
  that the MAXIMUM |rho| over all five, under label permutation (10,000 permutations),
  exceeds the observed maximum. The winner's nominal p is reported alongside but is NOT
  the headline. This is the step whose absence invalidated earlier results here.

## Covariates and controls, specified now

- **Melting point.** Measured on the same 11: rho(Tm, log activity) = +0.291, p = 0.385.
  Not significant, but a thermostability confound produced the `thermo_packing_density`
  retraction, so any surviving descriptor is re-tested with Tm partialled out.
- **Fold.** 8 amidase-signature, 3 carboxylesterase. Both families span the activity
  range (amidases 10^-3.85 to 10^+1.32, esterases 10^-2.65 to 10^+0.47), so fold cannot
  by itself produce a correlation. Any hit is re-tested within the 8 amidases alone,
  which needs |rho| >= 0.71.
- **Sequence length** and **MSA depth**, as generic confounds.
- **Ensemble validity.** A descriptor is only defined where the ligand actually engages
  the site; enzymes whose poses never place the ligand near the nucleophile are reported
  as such rather than scored as zero.

## Declared failure modes

1. **Null.** The most likely outcome on this project's record, and a publishable one:
   the panel is good enough that a null here is informative about the proxy, not about
   the data.
2. **A hit that dies on Tm or within-family.** Treated as a confound, not a finding.
3. **Degenerate descriptors.** On the hydrolase panel the near-attack gate saturated
   (distance 2.55-3.14 A across all docked members) and had no dynamic range. If D1/D3
   saturate again, that is reported as "no dynamic range", not as a weak correlation.
4. **n = 11 is small.** Even a clean hit is suggestive, not established. Stated in
   advance so it cannot be quietly dropped later.
