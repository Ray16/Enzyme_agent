# Experimentally characterised carbamate/amide hydrolase panel

Validation set for a structural proxy: these enzymes have published activity
data, so a proxy can be tested against them BEFORE it is used to rank anything.

## 🔴 2026-10-12 — 15 of the 26 cofold ensembles were ARCHIVED as invalid

See `../archive/20261012_invalid_inputs/README.md` for the per-system reasons.
Summary of what is wrong and what it costs:

- **10 labelled systems had expression-tag residues left on the C-terminus.**
  The trim below removed only `QGLEHHHHHH`; TEV cleaves `ENLYFQ↓G`, so `ENLYF`
  stayed on. Non-uniform across the panel (`ENLYF` ×8, `ENL` ×2, clean ×2).
  CCPUR1 also carried 3 extra N-term residues vs its own crystal 9TW8.
- **4 systems need the assembly, not a monomer** (site 6.5–9.4 Å from a
  neighbouring chain). All unlabelled.
- **CpUrethanase produced 0 structures** in 3 attempts (silent OOM, exit 0).
- `descriptors.json` and the `p = 0.0527` pre-registered result derive from the
  above and are **not results about structural proxies**.

**Consequence: the live labelled set is n = 1.** Only UMG-SP2 and UMG-SP3 are
clean-sequence *and* crystal-validated, and UMG-SP2 has no activity value.

Before re-folding: fix the trim to cut at `ENLYFQ↓G`, and run the positive
controls in `control_out/` against the crystal near-attack distances (8XTB
carbamate C–Ser190 OG **2.87 Å**; 9FZ1 amine N–Ser175 **2.69 Å**). If Boltz
cannot reproduce a known pose on this chemistry, re-folding buys nothing.

⚠ Do not rank ENZYMES on confidence (UMG-SP2 scores pLDDT 0.938 /
ligand_iptm 0.881 with the carbamate mispositioned). But WITHIN an enzyme,
`ligand_iptm` does pick the correct pose (AUROC 0.978) while `pLDDT` is
anti-correlated (0.092) — see the control result below.

## ✅ 2026-10-12 — positive controls PASS, with a sampling caveat

Measured against the crystals (element+connectivity atom selection, not names):

| control | crystal | min | median | <4 Å |
|---|---|---|---|---|
| UMG-SP3 + A1IHK (product) | 2.69 Å | 2.31 | 2.48 | **50/50** |
| UMG-SP2 + A1D5H (intact carbamate) | 2.87 Å | **2.98** | 8.15 | **4/50** |

Boltz CAN reproduce the crystallographic near-attack geometry on this chemistry.
For the intact carbamate the correct pose is a **minority basin (8%)**, so
descriptors must use best-pose or basin-fraction, **never a median over poses**.
Confirms 8XTB Ser190 ↔ our **S175** (motif `GSDIGGSIRVPAH`), so the nucleophile
is experimentally anchored for UMG-SP2/SP3.

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
