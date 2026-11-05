COMPREHENSIVE STANDALONE CANDIDACY BACKUP FIGURES

Start with FIGURE_INDEX.txt. It lists every graphic, its interpretation boundary,
and full PNG/SVG/PDF paths, including the earlier framework and chemistry assets.
previews/overview-*.png provide thumbnail sheets. No deck is assembled here.

This folder contains 24 quantitative / source-audit figures and 15 method,
interpretation or proposed-work figures. Their sources and full captions are in
quantitative_figures.json and method_figures.json. quantitative_results.json
records verified calculations and the limited statistical interpretation.

Saved-data evidence:
  All stored assay values; date-label variability; mean/SD summary
  Current angle versus yield and influence of the three strongest producers
  Absolute proximity alongside angle; complete matched baseline comparison
  Shortlist recovery with tie-aware expected recovery
  Weighted-angle scale invariance; full pose heatmaps and sampling sensitivity
  Best-pose versus pose-consistency development examples
  Audited neutral-amine step-2 readout and protonated-N assignment diagnostic
  Full saturation scan tradeoffs; complete AS9/AS17/AS23 mutation heatmaps
  Discovery accounting, activation-route-separated geometry, and proxy inventory
  Source-workbook provenance and candidate-specific modeled-metal audit

Method / proposed-work graphics:
  Assay evidence requirements; exploration/freeze/independent validation
  Multistate design; hypothesis-discriminating experimental panel
  Mismatch-driven follow-up tests; method/readout limitations
  Aim success/failure criteria; source-to-claim reasoning chain
  Design strategy requirements; execution versus scientific status
  Two-step electrophiles; ATP accounting; measurement interpretation
  Archived run cost/scope; native PuuA evidence versus model assumptions

Important evidence boundaries:
  The CSV matches all six values per ID in the 2025 Activity workbook.
  Repeat type is not established by that worksheet. A separate 2023 workbook
  contains biological/technical/calibration notes but is not mapped to this panel.
  Do not transfer its LOD, controls or repeat definition to the 2025 measurements.
  Zeros remain recorded zeros, not confirmed absence or a verified detection limit.
  Model ensembles are technical predictions, not biological repetitions or time.
  The current weighted angle A is not the older calibrated Gaussian score.
  All reported recovery/association is retrospective on an inspected panel.
  Mutant geometry changes are not experimental activity improvements.
  New assay results, biological independence, catalytic barriers, or benchmark
  superiority have not been manufactured to fill missing backup topics.

Rebuild from the repository root:
  /homes/rzhu/miniforge3/envs/palm/bin/python slides-candidacy/backup-evidence-figures/build_quantitative_figures.py
  python slides-candidacy/backup-evidence-figures/build_method_figures.py
  python slides-candidacy/backup-evidence-figures/catalog_figures.py

These scripts read existing data, plot it, and verify the exports. They do not
start inference, schedule fleet work, edit a manuscript, or assemble a slide deck.
