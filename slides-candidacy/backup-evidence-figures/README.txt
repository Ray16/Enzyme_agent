CURATED CANDIDACY BACKUP FIGURES

The active collection contains 14 figures. Start with FIGURE_INDEX.txt.
All retained graphics are in this folder; no slide deck is assembled.

Retained content:
  01  AMP/ADP chemistry and enzyme-family comparison (LaTeX)
  02  AMP-forming AS9 structure
  03  ADP-forming PuuA structure
  04  Exact modeling inputs and assumptions
  05  Structural descriptor inventory (LaTeX)
  06  Stored measurements for key candidates and controls
  07  Matched structural-proxy comparison
  08  Shortlist recovery against baselines
  09  Activation versus aminolysis geometry
  10  Complete mutation-scan tradeoffs
  11  Adaptive agent decision loop
  12  Evidence handoff between agents
  13  Proposed framework evaluation
  14  Proposed multistate catalytic constraints

Layout: concise axis labels, smaller type, separate legend space and rendered
point-label placement checks. Captions retain the detailed metric definitions.
The descriptor bar chart has been replaced by structural_proxy_search_inventory.tex.
Both tables have editable LaTeX sources and compiled PDF/PNG/SVG exports.

The 38 supplementary diagnostics and reference graphics were moved to an archive.
cleanup_manifest.json records the archive path and original locations. No original
assay data, predictions or main-talk assets were removed. The AS9 image here is a
copy of the existing main-talk figure, whose source remains in place.

Rebuild the retained collection from the repository root:
  /homes/rzhu/miniforge3/envs/palm/bin/python slides-candidacy/backup-evidence-figures/build_curated_collection.py

quantitative_results.json retains the saved-data calculations and evidence
boundaries. plot_layout_validation.json records rendered Matplotlib collision
checks; asset_validation.json records cross-format and PDF text-overlap checks.
Proposed-work diagrams do not claim completed experimental results. The current
weighted angle A is distinct from older calibrated Gaussian scores. Model poses
are not biological replicates, equilibrium populations or MD trajectories.
