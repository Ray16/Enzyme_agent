# Reaction-anchored retrieval — standalone validation

**Question:** does anchoring discovery on the *target reaction* (ATP-dependent
amide ligation) recover annotated, on-target amide ligases that the current
EC 6.3.- / InterPro sweep misses — from an answer-agnostic query?

**Method (fully deterministic, no answer seeding):**
1. `detect.py` — RDKit detector for ATP-driven amide ligation from a Rhea
   reaction SMILES: ATP substrate → AMP/ADP product **and** a new C(=O)–N amide
   bond formed from a free carboxylate + free amine. Self-test passes
   (glutamine-synthetase & PuuA-like positives fire; hydrolysis & kinase reject).
2. `run_retrieval.py` — scan all Rhea reactions → matched reactions → map to
   UniProt (reviewed + unreviewed) → overlap with a prior agent run → annotate.
3. `analyze.py` / `ontarget.py` — characterize and isolate the genuinely
   on-target subset (small-molecule carboxylate+amine, excluding protein/tRNA/
   cofactor acceptors).

## Result (target: adipate + HMDA → nylon-6,6 amide; prior run = 20260911_121301)

- 36,552 Rhea reactions → **175** ATP-amide-ligation reactions.
- → **21,444** UniProt members (632 reviewed, 20,812 unreviewed).
- **355 reviewed (Swiss-Prot) members are NEW** (absent from the prior run's 1290).
- Of those 355: ~201 are off-target cofactor/protein/tRNA amide chemistry;
  **36 rows / 18 enzyme types** are on-target small-molecule amide ligases.

## Verdict: **WEAK / INCONCLUSIVE on reviewed annotated space**

IMPORTANT CORRECTION (an earlier draft overstated this). "On-target" must be
judged on the SUBSTRATE axis (does the native chemistry resemble a C6 diacid +
C6 diamine), not on mechanism/family. Re-scored strictly on substrate:

- Of 355 new reviewed members, only 9 mention any aliphatic diamine/diacid, and
  8 are keyword coincidence or right-mechanism/wrong-substrate (dethiobiotin
  ureido-ring, PAICS/ADE1 purine biosynthesis w/ L-aspartate, UDP-sugar+2-oxoglutarate).
- **Exactly ONE genuine substrate-matched find: bibC (B6EP94)** — bisucaberin
  synthetase, succinyl (C4 diacid) + cadaverine (C5 diamine) → macrocyclic
  amide. Closest native nylon-6,6 analog found. Missed by the pipeline DESPITE
  its family IPR037455 (IucA/IucC NIS) already being surveyed → a
  **member-selection** gap, not a coverage gap.
- **cocJ/uncG (ANL) and ankE (NIS) are NOT on-target** — right mechanism/family,
  but bulky peptidic/aromatic substrates. Engineering scaffolds at best (the
  Tier-2 weak-fit bucket the pipeline already fills). Earlier labelling them
  on-target conflated the mechanism and substrate axes.

The substrate-matched leads the target actually wants (PuuA, GlnA2, AsbA/B,
SbnE polyamine/diamine ligases) were ALREADY found by the existing pipeline.
So reaction-anchoring did NOT beat it on annotated substrate-matched recall.

What holds up: (1) bibC is a real missed member worth diagnosing; (2) the
channel crosses EC boundaries (no-EC, 6.2.1.-, 6.-) the 6.3.- sweep cannot;
(3) untested upside is the 20,812 UNREVIEWED members, which annotation
inspection cannot judge (needs substrate-fit/structure eval — a different test).

**Status:** validated out-of-tree; verdict is NOT a clean win. Points toward a
**member-selection** investigation over a new retrieval subsystem. NOT wired in.
