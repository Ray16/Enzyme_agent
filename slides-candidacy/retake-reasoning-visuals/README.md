# Reasoning examples for the candidacy retake

The revised examples replace the initial generic workflow mockups with specific
chemical comparisons. Open **`revised_reasoning_examples.pdf`** for the current
three-slide review set or `revised_preview.png` for the overview. The `v2_*.svg`
files are editable. These remain figure/storyboard drafts, separate from the
existing candidacy deck. They do not introduce new assay values, structural
coordinates or confirmed mechanistic explanations.

The revised figures show:

- **Search:** native ammonia, methylamine and putrescine examples associated
  with GS, GmaS and PuuA, respectively; the target amine is HMDA. The point is to
  inspect reaction-specific evidence rather than assign one specificity to a
  broad fold. Native chemistry motivates a search lead; target activity remains
  unestablished.
- **DesD:** MA donor / XA acceptor versus XA donor / MA acceptor, with their
  different reported product preferences. Neutral drawings show chemical
  identity, not modeled protonation states. Reactive functional groups are
  highlighted; the mechanistic explanation remains open. The qualitative
  higher/trace contrast is based on the accompanying Figure 7 text, recorded in
  the local source review at
  `../../exp_data/amide_synthetase/literature_activity/regio_fig7/review.md`.
- **AS9:** user-reported A215Q improvement for PA66 diad and F249K improvement for
  PA55 diad. The structures show side-chain identities, not predicted poses.
  Questions about positioning and binding/packing are proposals, not verified
  explanations. Experimental values and exact assay protocols are still needed.
  The neutral side-chain drawings do not assert assay protonation states.
  The historical F249K input found in the October 19 saturation campaign contains
  adipate. It therefore cannot substitute for a PA55-specific molecular setup;
  a PA55-matched model is needed before interpreting F249K for that substrate.

Rebuild the revised figures:

```bash
/homes/rzhu/miniforge3/envs/palm/bin/python build_revised_examples.py
```

Native nitrogen-donor references:

- GS: https://iubmb.qmul.ac.uk/enzyme/EC6/3/1/2.html
- GmaS: https://iubmb.qmul.ac.uk/enzyme/EC6/3/4/12.html
- PuuA: https://pubmed.ncbi.nlm.nih.gov/18495664/

## Superseded first mockups

The original three visual mockups below are retained for provenance, rather
than recommended as finished presentation slides.

Open `reasoning_visual_mockups.pdf` for the three slides. Each slide also has
an editable SVG and a PNG. `preview.png` shows all three.

1. **Search:** native putrescine chemistry motivates investigating a PuuA lead
   for the target diamine. Native glutamate acceptance does not demonstrate
   adipate acceptance, and native putrescine acceptance does not demonstrate
   hexamethylenediamine acceptance. The historical audit reconstructs recorded
   decisions; it does not establish a controlled prospective reasoning benefit.
2. **DesD:** the question selects the molecular state and comparison. D and A
   are generic donor/acceptor substituents, not exact MA/XA structures. AMP is an
   abbreviated chemical group. No new mechanism or model geometry is asserted.
3. **AS9:** the user reports yield-enhancing mutations. Identities, raw values
   and experimental repeat types are not supplied here. The diagram shows that
   qualitative observation and competing hypotheses. It does not identify G211V
   or any other mutation as experimentally confirmed. Equal-protein assays,
   initial-rate measurements and abundance/stability controls are proposed.
   These can help separate abundance from catalytic effects; they do not alone
   prove that a predicted geometry change causes the effect.

Use on slides: reveal the observation first, then the hypothesis, then the
changed action. Speak the rationale rather than adding a paragraph. In the
final deck, identify whether a decision was researcher-led or agent-led, and
whether a refinement was completed or remains proposed. Replace the AS9
qualitative comparison with the verified experiment and matched structures.

## Sources

- PuuA native reaction: Kurihara et al., JBC (2008), PMID 18495664:
  https://pubmed.ncbi.nlm.nih.gov/18495664/
- DesD acyl-adenylate mechanism: *An acyl-adenylate mimic reveals the structural
  basis for substrate recognition by the iterative siderophore synthetase DesD*
  (2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC9356276/
- Local decision provenance: `../reasoning-trace/audit/figure_provenance.json`
  and `../backup-finalization/speaker_notes.txt` (saved PuuA decision example).
- Local setup audit: `../structural-modeling-setup/speaker_notes.txt`.
- AS9 retrospective scope: user's clarification in this conversation.

## Rebuild

```bash
/homes/rzhu/miniforge3/envs/palm/bin/python build_visuals.py
```

The script uses RDKit for molecular structures and PyMuPDF for rendering.
All files are local exports; no external image-generation service is needed.
