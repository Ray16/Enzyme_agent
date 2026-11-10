# AS9 M3/M4 contact audit, 2026-11-10

The two contact grids now show only primary heavy-atom contacts after chemical typing, distance-quality screening and group-level redundancy removal. The method is versioned as `heavy_atom_donor_acceptor_v2` and is documented in [CONTACT_ANALYSIS.md](../CONTACT_ANALYSIS.md). No hydrogens, protonation microstates, refolds, coordinate minimization or new ranking metric were introduced.

| Campaign | Designs audited | Unique poses including WT | Panel cells | Old all-pairs rule (≤3.8 Å) | Primary display lines | Flagged groups omitted | Too-short groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | 100 | 5,050 | 17 | 91 | 34 | 25 | 7 |
| M4 | 100 | 5,050 | 31 | 181 | 85 | 28 | 13 |

The old counts compare the previous geometric all-pairs rule with the new primary group-contact display; the difference includes both redundant lines and contacts quarantined by chemical/geometry checks. The original M4 flat-3.6 Å **distinct-residue** baseline was separately reproduced exactly for all 100 designs. A group with a pair below 2.4 Å remains flagged even if another oxygen permits a longer alternative line. The 1.96 Å Ser-O···O example is retained in the audit, not shown as an ordinary favorable contact.

Each primary residue–carboxylate pair receives one line, with a separate audit preserving all atom pairs and possible independent two-point patterns. His/Cys state ambiguity and anomalously short groups are omitted from primary counts/dashes and explicitly counted in every cell. Candidate membership, representative poses, sequences and the frozen ensemble geometry metric remain unchanged.

M3 uses adipate (C6), ATP supplied as SMILES (−4); M4 uses glutarate (C5), CCD ATP. They are analyzed independently and are not a controlled same-ligand comparison. The source molecular graphs and atom maps were checked rather than trusting the legacy `adipate` field name in M4 metrics.

## Full-campaign associations

The following are exploratory Spearman correlations over 100 designs per campaign. Intervals use 2,000 bootstrap resamples of designs; BH correction covers all reported nonconstant comparisons. Diffusion samples are not experimental replicates. Geometry-based representative selection and shared scaffolds prevent causal interpretation. These results do not establish epistasis.

| Campaign | Predictor | Outcome | ρ | Bootstrap 95% interval | BH-adjusted p |
|---|---|---|---:|---|---:|
| M3 | n_contacts | A | -0.118 | [-0.312, +0.091] | 0.3583 |
| M3 | n_contacts | dh | +0.004 | [-0.185, +0.198] | 0.9964 |
| M3 | n_contacts | nac | -0.089 | [-0.294, +0.141] | 0.4948 |
| M3 | mean_consistency | mean_global_plddt | +0.138 | [-0.059, +0.328] | 0.2919 |
| M3 | mean_consistency | mean_pocket_plddt | +0.289 | [+0.088, +0.466] | 0.01351 |
| M4 | n_contacts | A | -0.028 | [-0.227, +0.161] | 0.8305 |
| M4 | n_contacts | dh | -0.340 | [-0.519, -0.143] | 0.00262 |
| M4 | n_contacts | nac | +0.367 | [+0.177, +0.543] | 0.001184 |
| M4 | mean_consistency | mean_global_plddt | +0.189 | [-0.012, +0.378] | 0.1297 |
| M4 | mean_consistency | mean_pocket_plddt | +0.067 | [-0.139, +0.261] | 0.6096 |

M4 primary contact counts associate with shorter harmonic distance and greater near-attack fraction; M3 does not show the same robust contact-count association. M3 contact prediction consistency correlates with pocket-local pLDDT. Prediction consistency only affects dash style; it never gates the primary count or candidate selection. A flagged group’s frequency describes recurrence of the flagged state, not a favorable-contact frequency.

## Figures and candidate evidence

The overview grids are supplemented by 7.2-inch-wide print PDFs with at most four candidates per page: five pages for M3 and eight for M4. Individual molecular panels are 1500×1150 pixels; overview PNGs are exported at 300 DPI and PDF molecular panels at 400 DPI. Ligand and ATP bond orders were restored from the processed molecular graphs or matching CCD component, with exact atom/bond-count checks. A single WT active-site frame and scale is used within each campaign.

All artifacts are in `enzyme_design_agent/runs/contact_audit_20261110/` (git-ignored):

- `candidate_audit.md`: per-candidate image and typed interaction table, including reasons for every omitted group.
- `group_contacts.csv`: primary/flagged group contacts, distances, two-point patterns, role/fixed-group frequencies.
- `atom_pair_audit.csv`: all nearby accepted/rejected atom pairs and reasons.
- `summary.json`, `consistency_diagnostics.pdf`: complete sensitivity and confidence analyses.
- `figures/M3_essential_contacts.png`, `figures/M4_essential_contacts.png`: overview grids.
- `figures/M3_contact_atlas.pdf`, `figures/M4_contact_atlas.pdf`: print layouts.
- `policy.json`, `provenance.json`, `launch.json`, `render_launch.json`, `validation.json`, `source/`: reproduction and verification records.

## Verification

All 200 designs and both controls have 50 valid, unique poses: 10,100 total, with no pose errors. All representative hashes still match the audited files. All 48 rendered cells match the audited contact and label counts; graph bond counts are 33 for ATP, nine for adipate and eight for glutarate. Twelve regression tests passed, including geometry/chemistry boundaries, frame invariance, group identity, redundancy, short-contact quarantine and sensitivity behavior.

Rendering used ten concurrent CPU shards, three pinned threads each, with a separate coordinator: 31 reserved CPU slots and no GPUs. All observed worker affinities and thread counts matched the allocation. The final 48-cell render took approximately 32 seconds. The original grids are archived before replacement in their run figure directories. The unrelated `M3_vs_priors_metrics.png` summary was not modified.
