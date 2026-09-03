# Annotation evidence — SfaB synthetase & Nyl hydrolase panels

Annotation-evidence sweep (generated 2026-09-03) for the two experimental panels.
Method: **InterProScan** (EBI REST, `iprscan5`) run on each sequence = "query the
sequence in InterPro"; plus **UniProt/UniParc** native functional annotation.

## Amide synthetases (`SfaB_diversity_panel.csv`, AS1–AS48)
- `amide_synthetase_interpro_summary.csv` — one row per candidate: has-evidence, #matches,
  #InterPro entries, entries, family-level entries, Pfam domains, GO terms.
- `amide_synthetase_interpro_evidence.csv` — tidy, one row per signature hit (member DB,
  signature acc/name, integrated InterPro acc/name/type, locations, best e-value, GO).
- Finding: **48/48 have evidence**, all ANL / AMP-binding synthetase superfamily.

## Amide hydrolases (Nyl panel, 96; source = Drufva et al. 2025 Chem Catalysis Table S1)
- `amide_hydrolase_interpro_summary.csv` / `amide_hydrolase_interpro_evidence.csv` — as above.
- `amide_hydrolase_native_annotation.csv` — UniProt/UniParc native annotation per candidate:
  name, EC, catalytic-activity (Rhea) reaction, GO function, organism, keywords, entry status,
  and how each was resolved (`lookup`: direct / merge-demerge / UniParc-by-sequence).
- `amide_hydrolase_panel_sequences.csv` — clean Nyl → UniProt ID → sequence map (96).
- Findings: **96/96 have evidence**, uniformly IPR005321 Peptidase_S58/DmpA (NylC Ntn-hydrolase
  fold). Native reaction is sparse: only NylC (Q79F77) has a curated EC (3.5.1.117); 95/96 have
  predicted names only. 73/96 Table S1 accessions are retired in UniProt but recovered via UniParc.

## Figures (`figures/`)
- `panel_discoverability_summary.png` / `.svg` — 3-panel summary: (1) where each panel lives
  (live UniProtKB reviewed/TrEMBL vs retired→UniParc-only), (2) the agent discovery funnel for
  hydrolases (96 in panel → 96 InterPro family → 28 live UniProtKB → 1 EC-promotable), (3)
  annotation coverage of both panels incl. **native reaction (`cc_catalytic_activity`)**.
- `figure_stats.json` — the exact counts behind the figure.
- Native-reaction (`cc_catalytic_activity`) coverage: **synthetases 2/48, hydrolases 1/96** (NylC).

## Related (not in this folder)
- `../amide_hydrolase_biorep_activity.csv` — primary activity data; now carries `uniprot_id` +
  `sequence` columns (from Table S1).
- `../nylon hydrolase sequences/1-s2.0-S2667109325001563-mmc2.xlsx` — Drufva Table S1 (the source map).
