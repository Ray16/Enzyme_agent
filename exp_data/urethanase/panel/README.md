# Experimentally characterised carbamate/amide hydrolase panel

Validation set for a structural proxy: these enzymes have published activity
data, so a proxy can be tested against them BEFORE it is used to rank anything.

## Files
- `panel_as_published.fasta` — 26 sequences exactly as retrieved (expression
  tags intact), with the source in each header.
- `panel_mature.fasta` — the same sequences with expression tags removed; this
  is what gets folded.
- `catalytic_residues.json` — nucleophile assignment per enzyme, with its basis.

## Provenance of the sequences
Three tiers, because these enzymes are not uniformly in UniProt:
- UniProt / PDB: NylA, NylB, NylB', NylC, NylC-TS, TvgC, SaAmd, CpUrethanase, Aes72
- GenBank only: UMG-SP1/2/3 (WBR49956-58.1); only SP2 has a UniProt cross-ref
- Supplementary information only: the ten Bendtsen enzymes (CCPUR1/2/7,
  MPPUR3, BLPUR4, AXPUR5, AMPUR6, PMGPUR8/9, PYNYL1) and pQR3139/3141/3144

Validated against three independent published identity claims:
CCPUR1 vs CCPUR2 **89.2%** (paper 89%), SaAmd vs UMG-SP3 **61.5%**
(literature ~61.4%), TvgC vs NylC 29.8% (paper 31.6%).

## Catalytic nucleophile
5 from UniProt ACT_SITE; the rest transferred by alignment from the nearest
annotated reference. The reference is chosen by overall identity and THEN
tested, rather than choosing whichever reference happens to yield the expected
residue — that would be shopping for the answer, and a distant reference lands
on a Ser or Thr by chance often enough to matter.

A transfer is downgraded when the reference is distant AND the local motif does
not match it. 22/26 usable; `AXPUR5` and `MPPUR3` are low confidence, `Aes72`
(a different fold, no annotated reference here) and `BLPUR4` unresolved.

The clean amidase-signature transfers all land in the expected motif, which is
the evidence that they are real: `SDIGGSIRVPA` (UMG-SP1/2/3, SaAmd),
`SDLGGSLRWPA` (CCPUR1/2), `NDGGGSIRIPA` (pQR series).

## Activity labels
The usable cross-enzyme label is Bendtsen et al. (Angew 2026, PMC13502555):
specific activity on one probe, one protocol, 11 enzymes, spanning 5 orders of
magnitude (21.0 down to 0.00014 umol/min/umol). That is the comparison a proxy
must survive. Numbers from other studies use different substrates, labs and
assay formats and are NOT comparable across enzymes.

## Not included
AbPURase (Chen et al., Science 2025) — the publisher blocks automated download.
