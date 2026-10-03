# GS-fold interface re-validation (2026-10-03)

**Question (from audit Finding 1):** 15/36 discovery candidates are glutamine-synthetase-fold
enzymes whose catalytic site sits at a **subunit interface**, but they were cofolded as
**monomers** — so their angle-gate scores were suspected of being open-pocket artifacts, and
"8 of the top-10 docks" were flagged uninterpretable.

## What was tried and what works

1. **Full biological oligomer — INFEASIBLE.** dodecamer P78061 = 476 GB trunk vs 96 GB H100;
   even the smallest full assembly (octamer ~430aa) = ~120 GB. Cannot fold as a whole complex.
2. **De-novo adjacent-pair dimer cofold — FAILS.** `dimer_inputs/` (2 identical chains + ATP +
   2 Mg + adipate, built by `build_dimer.py`; folded with stock boltz-2 2.2.1 on a lambda2 V100,
   `LD_LIBRARY_PATH=$env/lib` for the GLIBCXX fix, chunked samples via `fold_chunked.sh`).
   Boltz dimerizes the chains at a LARGE but **wrong (non-catalytic) interface** — the neighbor
   lands ~12 Å from the reactive center (`validate_dimer.py`). So plain cofold cannot give a
   corrected score; a contact/pocket-constrained or template-guided re-dock would be required.
3. **Template-grafting validation — WORKS, and is decisive.** `../../../../`scratchpad
   `batch_graft.py`: superpose each GS monomer pose onto a real GS oligomer (GS-I→1FPY,
   plant/euk GS-II→human GS 2QC8), graft the neighbor subunit, classify clash (<2.4 Å) vs
   Ser-catalytic contact (<3.6 Å) to the docked adipate carboxylate, over all 50 poses.

## Result

Finding 1's blanket worry is **partially overturned.** Of the 7 proxy-elevated GS (>150°),
**5 survive pocket completion** (Q5UR44, Q54WR9, P13564, O22506, PuuA — neighbor is compatible,
often H-bonding via Ser53′) and only **P08281** is cleanly invalidated (24/50 clash, reliable
2QC8 template); Q44249/Q936T0 suspect on looser 1FPY alignments. glnA2 (Q9RDS6, rank-2 by
substrate_fit) **failed to dock adipate** (0 gated poses).

Reliability: 2QC8 grafts RMSD ~1.1–1.7 Å (trustworthy); 1FPY grafts RMSD 3.3–4.4 Å (indicative).

## Outputs
- `discovery_shortlist_gs_corrected.csv` — full 36-candidate ranking, tiered, with GS pocket
  verdicts and the n_gated≥5 scoreability rule applied.
- `dimer_inputs/`, `smoke_test/`, `fold_chunked.sh`, `validate_dimer.py` — the (disproven)
  de-novo dimer route, kept for provenance.

**Caveat that dominates the shortlist:** even the surviving GS are glutamate/ammonia ligases
(substrate_fit 0.0) — their adipate+HMD candidacy is pure structural analogy, and the angle
proxy is unvalidated on the 2-Mg acyl-phosphate mechanism. The only enzymes with measured
activity on the nylon substrate are the assay hits AS9/AS23/AS17 (see project_synthetase_audit_oct2).
