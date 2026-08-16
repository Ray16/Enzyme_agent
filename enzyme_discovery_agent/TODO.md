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
- **Stage 3**: mechanism-family derivation code-enforces the direction + free-substrate
  constraints (drops inverse-direction / protein-bound families the LLM listed anyway);
  precision-first prompt, no breadth quota. `family_passes_constraints()`.
- **Stage 4**: initial full-text harvest capped to the top-40 relevance-ranked papers
  (`FULLTEXT_HARVEST_MAX_PAPERS`) — early effort goes to candidate search, not reading
  ~160 papers. Targeted per-candidate reading stays uncapped.

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
- [ ] **Role-aware correspondence** (motivated by AsnA/AsnB): map amine↔amine,
      acid↔acid explicitly, so the target amine is always compared against the
      member's amine partner. Then e.g. "HMDA vs ammonia = poor" is guaranteed
      regardless of cofactor-filtering. One-to-one target↔native assignment.
- [ ] Expose `substrate_correspondence` (per-pair dimensions) in the member evidence.
- [ ] **Cofactor-vs-substrate consistency**: `_COFACTOR_NAMES` (reasoning.py) filters
      "ammonia"/"nh3" as a cofactor, but ammonia is a genuine N-source SUBSTRATE for
      some amide synthetases (AsnA). The survey path (COFACTOR_CHEBI_IDS, 18 ids)
      correctly keeps it; the per-candidate path drops it → hides the amine mismatch.
      Make the two paths consistent (don't drop a species that IS the reaction's
      nucleophile/amine partner).
- [ ] Keep it pure/offline-testable; no new tuned weights.

## 3b. Stratified literature harvest (guarantee class coverage) — Stage 5 candidate
The Stage-4 cap takes the top-40 by GLOBAL relevance; on some queries that could
collapse onto one enzyme class and starve the others (defeating the point of
deriving N families). Observed OK emergently on the 20260816 run (read set spanned
GS/ATP-grasp, NIS, ANL/peptide), but not guaranteed.
- [ ] Tag each pooled paper with the family/search-term that surfaced it
      (seed_mechanism_literature), then allocate the harvest budget ACROSS families
      (top-K per family) instead of a global top-N → explicit class coverage.

## 3c. Budget-conscious family expansion — concentrate spend on relevant families (Stage 5)
Don't just RANK irrelevant families low — avoid SPENDING compute expanding +
evaluating them. Current leaks: (1) Stage-2 protection force-evaluates the top-2
of EVERY surveyed fold, incl. low-fit families (AsnA/AsnB); (2) TrEMBL deepening
runs for all surveyed folds; (3) harvest relevance gate read an off-target protease
paper (E-64) that the strict family filter would exclude.
- [ ] **Promote/protect by GLOBAL substrate-fit** across folds (pool all folds' best
      members, rank, concentrate budget on the globally-best) instead of per-family
      top-2 unconditional protection. Relative — no magic threshold. Preserves
      Stage-2's goal (high-fit promotions reach ranking) while starving low-fit ones.
- [ ] Gate the expensive TrEMBL deepening to competitive folds (relative top-K).
- [ ] Align the fulltext harvest relevance gate with the family direction/chemistry
      constraints (no protease/hydrolase papers).
- [ ] MEASURE first: from the 20260816_132034 run, count evaluations that went to
      adjacent (AsnA/AsnB-type) families to size the actual waste before fixing.

## 4. Performance / cost
- [ ] **Substrate-fit survey engages late** (20260816_132034 run: survey couldn't
      fire until iter ~26 / ~19 min because `candidate_count` counts screened+ only,
      and screening didn't run until then — 102 discovered, 0 screened before that).
      The narrowing engine is dormant for the whole early phase. Levers: screen
      earlier / trigger the survey on DISCOVERED folds too (it already resolves
      discovered candidates' folds), so substrate-fit narrowing starts sooner.
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
