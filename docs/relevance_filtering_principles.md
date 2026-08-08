# Relevance filtering — design principles

Principles governing how the agent narrows retrieved literature down to the
papers worth spending tokens on (reading full text, feeding to reasoning).
These are **principles**; the ranking/gating code implements them incrementally.

## The funnel (recall first, precision last)

Cast a wide, high-recall net, then narrow in **cascading stages — cheapest and
most forgiving first, expensive and most decisive last**. Never let a cheap
early stage make an irreversible high-stakes drop. Current/planned stages:

1. **Structural/lexical pre-filter** (free): require an abstract + DOI; score
   overlap against campaign-extracted entities.
2. **Rank fusion** (free): fuse each source's native relevance order with a
   semantic (embedding) rank via Reciprocal Rank Fusion — see
   `tools/literature.py::_rrf_fuse`. Do **not** discard the source's native
   order (the retrieval engine already did lexical work).
3. **LLM abstract-triage** (cheap, batched, survivors only) — later.
4. **Full-text fetch + section-extract/summarize** (expensive, capped) — later.

## Principle 1 — Leakage discipline (non-negotiable)

The relevance signal must be derived from the **campaign target** — the
reaction / bond formed, the substrate (and close analogs), the required
function / EC class — and **never** from the known answer family.

- Correct: "does this paper bear on *amide-bond formation for nylon-6,6
  monomers*?"
- Leakage: "is this paper about *NylC / DdaG / this UniProt accession*?"

A relevance filter is a prime place for the answer to leak in through the back
door (a rubric, a seed list, an example). The search must not be seeded with
the answer family; the LLM proposes *strategies*, tools find *entities*. This
is enforced elsewhere by a firewall + path-provenance metric + tests; the
relevance layer must not become a hole in that firewall. When calibrating
against the confirmed benchmarks (amide synthetase, nylon hydrolase), the
answer accessions are used **only** to score the filter after the fact
(did answer-bearing papers survive?), never as an input to the filter.

## Principle 2 — On-topic vs. decision-relevant

"Relevant" has two bars, and the higher one is what saves tokens:

- **On-topic** — the paper concerns the target chemistry/enzyme space.
- **Decision-relevant** — the paper carries evidence that would *change a
  candidate's ranking*: kinetics on the target/analog reaction, a
  structure–function or mechanism result, a directed-evolution / engineering
  precedent, an activity measurement, a negative result that rules a family out.

A review that is on-topic but adds no discriminating evidence is low value for
full-text spend; prefer papers that would move an evaluation. Triage should
capture *which axis* matched (substrate / reaction / function / evidence-type),
not just a scalar — a multi-axis judgment is more robust and more auditable
than a single "relevance" number, and it makes the on-topic vs.
decision-relevant distinction explicit.

## Calibration philosophy

Thresholds are **calibrated against ground truth, not guessed** — but with only
two confirmed cases (n=2), a calibrated number is a **sanity floor** ("don't go
above X or we start dropping known answers"), not a tuned optimum. Always report
each known-positive's individual score alongside any chosen threshold so the
number is auditable, and resist adding cascade stages (cross-encoder,
LLM-judge, MMR) until the cheap layer demonstrably leaves a measurable gap —
tuning a four-stage cascade against n=2 is the guessed-threshold problem one
level up.
