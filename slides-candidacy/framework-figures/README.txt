CURRENT OUTPUT LOCATION
Four retained framework figures are now exported to ../backup-evidence-figures/:
adaptive_decision_loop, shared_evidence_handoff, proposed_framework_evaluation,
and modeling_setup_audit. Other graphics listed below are historical reference
assets and have been archived; see ../backup-evidence-figures/cleanup_manifest.json.

STANDALONE FRAMEWORK FIGURES

Each figure is available as PNG, editable SVG, and PDF. No slide deck is
modified or assembled by this figure exporter. Titles, slide footer bars and
page numbers are omitted so the graphics can be inserted into your own slides.

  adaptive_decision_loop — six-stage adaptive decision diagram
  shared_evidence_handoff — evidence transfer and handoff contents
  agent_roles_tools — component responsibilities and tools
  evidence_driven_search_example — saved GS / PuuA decision example
  structural_validation — validation stages and consequences
  proposed_framework_evaluation — proposed comparisons and readouts
  implementation_validation_status — demonstrated artifacts versus next tests
  modeling_setup_audit — inputs behind the reported structural comparisons

Read figures.json for the caption and interpretation boundary of each figure.
Source-specific speaker notes remain in ../backup-finalization/speaker_notes.txt.
In particular: comparative evaluation is proposed; the saved search example is
reconstructed; current software interfaces are not attributed retrospectively
to the September run; geometry changes are not experimentally validated activity.

Build: python slides-candidacy/framework-figures/build_figures.py
The script reuses the reviewed artwork functions in memory and writes only
individual figure assets. It does not require the assembled review deck.
