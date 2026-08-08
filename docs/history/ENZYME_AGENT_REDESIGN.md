# Enzyme Discovery Agent — Architecture Redesign

_Drafted 2026-06-17. Supersedes the MAP-Elites framing for discovery._

## Why redesign

The problem is **retrieval + expansion + ranking over a fixed protein database**, not
generative quality-diversity optimization. MAP-Elites mismatches it on every core
assumption (we retrieve not generate; fitness is a noisy proxy; we want a ranked list
not a feature-space map; ~8 rounds is not 10⁴ evaluations), and its per-cell-elite
archive actively *collapses the family we want to see* and *spreads expansion budget
across irrelevant ligase folds* — the observed "off-target flood / no expansion in the
right direction."

## The determining insight

**Promiscuity conserves catalytic *machinery*, not *substrate*.** An enzyme does a
non-native reaction when its active-site machinery is general enough. Therefore:

- **Recall must run on MECHANISM** (fold / superfamily / active-site / domain) — the
  annotation-independent thing that survives the promiscuity gap. Substrate must be
  kept OUT of recall (substrate-similarity buries promiscuity; so does a learned
  native-function retriever, which is OOD on non-natural reactions).
- **Substrate ACCOMMODATION is the discriminator** — and belongs in **ranking** and in
  **steering expansion depth**, never as a recall gate.

Mechanistic capability is broad (mur / bio / glutathione / NIS are *all* ATP-dependent
C–N ligases), so the off-target families are not a recall bug — they are correct
mechanistic recall, and the job of accommodation ranking is to float the substrate-fit
family (NIS) to the top and to avoid deep-enumerating the rest.

## Target architecture

```
query
  ▼
1. Reaction analysis (LLM + firewall)
     grounded substrates; mechanism hypotheses WITH structural anchors
     (folds/Pfam/active-site signatures) AND the bond-forming atoms (for NAC);
     search plan (EC-subclass, keywords, literature terms, molecules, gene_names)
  ▼
2. Anchored recall — three parallel channels into ONE pool
     • Database : annotated members of each capable superfamily (EC-subclass, Pfam/
                  InterPro, fold, gene_names)
     • Literature: promiscuity precedents + under-annotated/TrEMBL catalysts
     • Structure : Foldseek/Pfam/UniRef expansion = reach annotation-distant relatives
                   (bridging, not a per-round flood; depth steered by accommodation)
  ▼
3. Active-site clustering  →  representatives
     cluster the pool by active-site/structural similarity (Foldseek/MMseqs2);
     accommodation is a SUBFAMILY property → score one representative per cluster,
     propagate to members. (Also the family-aware dedup the report needs.)
  ▼
4. Scalable accommodation ranking (precision) — scores that run on AF structures:
     • catalytic-residue TEMPLATE geometry (do the catalytic residues + cofactor motif
       sit in the right 3-D arrangement?) — mechanism capability at residue resolution
     • pocket FIT (volume/shape vs target substrate; physicochemical complementarity)
     • substrate-character prior (FG type + aliphatic/aromatic) — cheap prior only
     → rank representatives; propagate to cluster; family-aware shortlist
  ▼
5. Precision cap (gated, top ~20–50 only)
     NAC geometry via POSE-TRANSFER from a holo homolog where available, else CO-FOLD
     (enzyme + target substrate(s) + cofactor; confidence-gated by ligand pLDDT/PAE).
     Co-folding is a top-of-funnel confirmation, NEVER a pool scorer.
  ▼
6. Family-aware ranked report (chemistry + evidence views)
```

### What the NAC metric measures and why it's the right one
Substrate accommodation must measure **catalytic competence on the target substrate**,
not substrate similarity (buries promiscuity) and not binding affinity ("binds" ≠
"reacts"). NAC = the geometry of the bond-forming atoms in a predicted enzyme–
substrate–cofactor complex (reacting-atom distance + attack angle + catalytic-residue
positioning). It is substrate-conditioned (→ non-natural-friendly), mechanism-aware,
and structure-based. The **mechanism hypothesis (stage 1) defines which atoms the NAC
checks**, so recall and precision share one mechanistic model. Validate the whole
accommodation stack on the built-in control: **AsbA (active) vs AsbB (in-family,
inactive)**, and actives vs mur/bio/glutathione decoys.

## What we keep vs. drop

- **Keep:** reaction analysis + firewall, grounded-substrate scoring, Pfam/UniRef/
  Foldseek tools, literature channel, EC-subclass triage, LLM ranking, the leakage
  firewall + policy.
- **Drop:** `discovery/archive.py`, `descriptor.py`, MAP-Elites selection/curiosity in
  `controller.py`. Replace with a retrieval-expansion controller (pool + provenance, no
  archive) and a clustering + accommodation ranking stage.

## Phased implementation (each phase testable; structural phases are pluggable/gated)

- **P0 — Clustering + accommodation-ranking restructure (no new heavy deps).**
  Add `scoring/cluster.py` (active-site/structural representatives) and
  `scoring/accommodation.py` (graded substrate-accommodation score = current
  substrate-character signal, interface ready for template/pocket/NAC terms). Re-rank
  the existing discovery pool family-aware. **Directly fixes the flood; no co-folding.**
- **P1 — Retrieval-expansion controller.** Replace the MAP-Elites loop: anchored
  enumeration + accommodation-steered expansion depth, no archive. A/B vs current.
- **P2 — Catalytic-residue template geometry** on AlphaFold structures (biopython);
  pluggable accommodation term. Validate on AsbA/AsbB.
- **P3 — Pocket fit** (fpocket/P2Rank) term.
- **P4 — NAC precision cap:** pose-transfer from holo homolog; co-folding (Boltz/Chai/
  AF3) gated on top-K. GPU; `Scorer.enabled()` pattern, off until wired.

## Open decisions
- Clustering granularity (sequence-identity threshold / Foldseek-TM that conserves the
  active site) — pick empirically so each cluster = one substrate scope.
- Co-folding engine for P4 (AF3-accuracy vs Boltz/Chai throughput) — gated by GPU.
