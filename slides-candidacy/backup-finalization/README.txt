CANDIDACY BACKUP SLIDES — 2026-11-04

LATEST SCOPE: generate standalone visualization assets for the user to assemble.
The expanded family table is updated independently. Earlier assembled review
PDFs below are retained, but have not been updated with the expanded family
inventory. Use ../framework-figures/ and ../ligase-family-comparison/ for the
current independent graphics. No further deck assembly is performed.

Main output: Candidacy slides - with backups.pdf
New backup pack: Candidacy backup slides.pdf
LaTeX table: amp_adp_table.tex / amp_adp_table.pdf
Standalone table: AMP vs ADP synthetases.pdf / .png
Individual slides: individual/backup-01.pdf through backup-10.pdf
Editable framework source: build_backups.py
Preview images and vector exports: previews/
Speaker notes and source interpretation: speaker_notes.txt

Combined deck order:
  1–28  Original main presentation, unchanged
  29    Original backup divider
  30    AMP-forming versus ADP-forming synthetases (LaTeX table)
  31    Structural comparison of AS9 and ADP-forming PuuA
  32    Agent roles, tools, and scientific responsibility
  33    Adaptive action selection and stopping rules
  34    Shared evidence and structured handoffs
  35    Saved decision example: GS-fold annotation and PuuA
  36    Structural validation and follow-up selection
  37    Proposed framework evaluation and ablations
  38    Implementation and validation status
  39    Audited inputs behind the structural comparisons
  40    Original full proxy inventory, footer renumbered

Build the table, then the deck, from the repository root:
  pdflatex -interaction=batchmode -halt-on-error -output-directory=slides-candidacy/backup-finalization slides-candidacy/backup-finalization/amp_adp_table.tex
  python slides-candidacy/backup-finalization/build_backups.py

The original Candidacy slides.pdf is never overwritten. Build validation checks
the original hash and compares text and rendered page digests for all 28 main
slides. The original full proxy inventory is preserved in the review copy.

Current implementation is described separately from the archived September
case study. Comparative framework evaluation and prospective experimental
learning are proposals, not completed performance results. Check sources.json
and the slide-specific speaker notes before making stronger claims.

Two additional backups would be useful when the supporting assay information
is available: experimental product quantification/controls, and retrospective
proxy selection versus independent prospective validation. No assay conditions,
replicate counts, detection limits or predictive statistics were invented here.

Generated PDFs remain local under the repository's existing binary-document
ignore rule. Editable sources, provenance and notes are versioned.
