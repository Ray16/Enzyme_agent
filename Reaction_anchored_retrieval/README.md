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

## Verdict: **CONCEPT VALIDATED (as a retrieval channel, not a ranker)**

Genuinely on-target annotated candidates the pipeline MISSED, e.g.:
- **bibC (B6EP94)** — bisucaberin synthetase: ligates *succinyl (C4 diacid) +
  cadaverine (C5 diamine)* → macrocyclic amide. Closest native analog to
  nylon-6,6 chemistry found to date. Family IPR037455 (IucA/IucC NIS) was
  ALREADY surveyed by the pipeline — so reaction-anchoring also fixes
  **member selection within surveyed families**, not just family coverage.
- **cocJ / uncG (6.2.1.-)** — "amide-bond synthetase", ANL family (IPR000873),
  the same mechanistic family as ground-truth SfaB/DdaG/McbA.
- **ankE** — NIS siderophore synthetase (citrate+ATP amide), on-target family.

It reaches enzymes across EC boundaries the 6.3.- sweep cannot (no-EC, 6.2.1.-,
6.-.-.-), which is exactly the annotated-recall gap.

**Caveat / integration shape:** high recall, low precision (~95% of the 355 are
off-target amide chemistry). This is a *discovery channel* that must feed the
existing substrate-fit / family-relevance gate — never the shortlist directly.

**Status:** validated out-of-tree. NOT wired into the agent pending decision on
integration shape.
