# Enzyme Discovery Agent — TODO / Roadmap

_Last updated: 2026-08-16. Working branch: `feature/retrieval-expansion`._

Guiding principles (from the design discussion):
- **Code extracts evidence, LLM only reasons.** Functional groups, size, weight,
  substrate↔substrate correspondence, direction — all extracted deterministically;
  the model reasons over the grounded facts.
- **No tuned relevance thresholds / magic numbers.** Prefer relative ranking +
  reuse of existing bounds over new empirical cutoffs.
- **A tool existing ≠ the agent using it.** Wire capabilities into the deterministic
  pipeline (gates), and make sure their output flows to the final answer.
- **Data-leakage firewall** (see `CLAUDE.md`): never seed search with the answer.

---

## ✅ DONE — Substrate-fit family expansion (tags `substrate-fit-promotion-stage1/2`)

- **Stage 1**: family survey now selects the members whose native substrate most
  RESEMBLES the target (multi-dimensional: FG, MW, volume, chain length, HBA/HBD,
  polarity; relative ranking, no cutoff) and PROMOTES them to candidates. Gated
  TrEMBL deepening: pull a fold's unreviewed members of the best-fit reviewed
  member's EC (targeted, not blind paging).
- **Stage 2**: promoted candidates are PROTECTED in `auto_screen` → always
  screened→evaluated→ranked, never dropped by fold budget/stop or ortholog collapse.
- Validated by oracle test: surveying DdaG's (IPR053158) and PuuA's (IPR027303)
  folds retrieves + promotes both (the enzymes "mostly uncoverable via DB search").

## 🔄 IN PROGRESS — End-to-end validation run
- Full agent run on the amide/nylon benchmark, monitored live.
- **Success criteria**: survey→promote→protect→rank fires in a real loop; check
  whether substrate-fit-promoted candidates (esp. DdaG/PuuA-like folds) reach the
  final ranking; compare ranking vs. the 20260812 baseline (DdaG #6, SfaB demoted).
- If good → start the literature axis below.

---

## 1. Literature axis (NEXT major — the other half of "both, substrate first")
Goal: raise cold-start recall and recover **literature-only** candidates like
**SfaB (A0A2H4T912)** — TrEMBL, no EC, no annotated substrate → unreachable by
substrate-fit, only reachable via its paper.
- [ ] **Resolver breadth** (`tools/fulltext.py::_resolve_named_enzyme`): extend
      accession extraction beyond GenBank/EMBL — PDB IDs, RefSeq/`WP_`, and
      `EC + organism` → UniProt.
- [ ] **Confirm-by-reaction resolution**: let the resolver admit more name hits but
      disambiguate using the substrate/reaction the paper already extracted (does the
      resolved accession's catalytic activity match?) — trade a blind precision
      filter for a grounded one.
- [ ] **Citation second-hop**: a cited homolog resolved only by name triggers a
      fetch of ITS primary paper to recover a deposited sequence.
- [ ] **Cross-feed**: feed ChEBI/substrate-class names from the survey into the
      literature harvest query terms.
- [ ] Re-validate SfaB recovery likelihood.

## 2. Strengthen fold-seeding (the survey's seed dependency)
The survey only fires on a fold it has a seed into (derived-family keyword → interpro,
or a discovered member's most-specific fold). Obscure answer-folds may never be seeded.
- [ ] Audit `gates/family_relevance.py::collect_survey_folds` coverage on the benchmark.
- [ ] Improve mechanism-family → `search_interpro` resolution so generic derived
      families reach the specific relevant folds (or their surveyable parents).
- [ ] Consider seeding folds from high-substrate-fit hits discovered by any channel.

## 3. Substrate-correspondence evidence (code-extracts-evidence refinement)
`_substrate_facts` picks, per target substrate, its best-overlapping native
INDEPENDENTLY (can map two targets to one native, or acid↔amine). Make the
correspondence explicit, role-consistent evidence the LLM reasons over.
- [ ] Extract a one-to-one target↔native assignment (respect functional-group roles),
      expose `substrate_correspondence` (per-pair dimensions) in the member evidence.
- [ ] Keep it pure/offline-testable; no new tuned weights.

## 4. Performance / cost
- [ ] **Front-loaded literature exploration** (observed 20260816 run: at iter 5 /
      ~7 min, still 0 candidates — harvest pulled 160 papers from 8 mechanism
      phrases, then reads full text before any candidate exists). Substrate-fit
      machinery can't start until this clears. Ideas: cap/parallelize full-text
      reads; interleave discovery with harvest so candidates form earlier; only
      deep-read papers that pass a cheap relevance screen; time-box the initial
      harvest.
- [ ] RDKit 3D embedding is slow on CoA-ester natives (volume estimate). Cache/2D
      fallback; parallelize survey calls across folds.
- [ ] Runs cost ~2.5M tokens — prune prompt re-sends, fewer finalization LLM passes,
      cheaper model for sub-steps (see `project_token_efficiency`).

## 5. Benchmark harness
- [ ] Turn the oracle spot-check into a repeatable metric: for each answer enzyme,
      does the pipeline retrieve + promote + rank it? Report recall@k + provenance.
- [ ] Guard: keep `eval/benchmarks.py` the only place ground truth lives.
