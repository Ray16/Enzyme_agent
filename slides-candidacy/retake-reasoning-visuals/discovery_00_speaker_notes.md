# Discovery hero example — speaker notes

**Slide:** `discovery_00_topology_over_similarity.(png|pdf|svg)`
**Answers the committee's "couldn't you just compare SMILES?" directly.**

## The 20-second delivery (question → evidence → interpretation → next)

> "We want an enzyme that joins a **free diacid and a free diamine**. Several
> ATP-dependent amide-ligase families score well on substrate structure. The
> one that scored **highest** — a tetrahydromethanopterin:glutamate ligase —
> actually ligates its acid while it is **tethered to a bulky cofactor**, so a
> free diacid has nowhere to bind. We **stopped** it and **expanded** a
> lower-scoring NIS synthetase that condenses a **free** diamine and **free**
> diacid. Substrate similarity would have taken us to the wrong family; the
> decision came from reasoning about how the enzyme presents its substrate."

## Why this beats a similarity baseline (the whole point)

- Both families make amide bonds via ATP; both act on a small carboxylic
  acid + amine. By **substrate structure + size**, the carrier-tethered family
  (MptN, IPR041107) scored **0.83**; the correct free-substrate family (AsbA,
  IPR022770) scored **0.40**.
- A SMILES/structure-similarity ranker compares the *named small-molecule*
  substrate (glutamate vs. citrate). It **cannot see** that glutamate is only
  accepted while covalently fixed to a carrier protein / bulky cofactor. That
  topology fact lives in the mechanism, not in the molecule's structure.
- The decision rule ("target inputs are free small molecules → a catalyst
  requiring a different substrate topology may not accept them") was **derived
  from the target up front** (`target_context.json → target_reasoning`), then
  applied — so this is a recorded decision, not a story told after the fact.

## Honest boundaries (say these, don't let them be asked)

1. **Recorded, not retrospective.** The free-topology rule precedes the family
   scores in the logs.
2. **Chemical reasoning > substrate similarity** is what this shows. It does
   **not** by itself show **LLM agent > a rule-based workflow** that also encodes
   a free-vs-tethered filter — that is exactly what the proposed controlled
   comparison (Aim on agent evaluation) tests.
3. **Prioritization, not proven exclusion.** No stopped member was experimentally
   tested against free adipate + HMDA; "stop" reorders the search, it does not
   claim zero activity.
4. The "fit" number is the agent's own substrate/size feature, used here as a
   stand-in for a similarity ranker; the qualitative point (structure favors the
   tethered family) is robust to the exact score.

## Provenance (canonical run 20260917_135330)

| What | File |
|---|---|
| Substrate-fit scores (0.83 / 0.40) | `family_landscape.json → holistic_fit.median` |
| Stop / expand + reasons (timestamped) | `family_decisions.json` |
| Free-topology rule derived up front | `target_context.json → target_reasoning` |
| Representative members | `candidate_registry.json` |

- STOP: IPR041107 — MptN, tetrahydromethanopterin:α-L-glutamate ligase (Q8Q0M5)
- EXPAND: IPR022770 — AsbA, spermidine–citrate ligase, *B. anthracis* (A0A0F7RJ52)

## Where the old PuuA slides go

`discovery_01_search_decision` (PuuA/putrescine ≈ HMDA) is **reproducible by
similarity** — demote to backup, or use only as a warm-up before this slide.
This slide (`discovery_00`) is the one that carries the argument.
