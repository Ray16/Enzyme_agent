# Enzyme Design Agent — Build Report

Status report of the clean-slate rewrite for your review. Previous working
version preserved untouched at `enzyme_design_agent_prev/`. New build in
`enzyme_design_agent/`. This file lives at the repo root (not under the agent
package) because it names benchmark targets, which the code-leakage guard
forbids inside the package.

## 1. Objective & approach

Discover enzymes for non-natural reactions where annotation is sparse. The
previous agent fails on the *promiscuity* case: validated catalysts whose native
annotation has no textual/EC path from the query are never found. The rewrite
replaces the linear pipeline with a **MAP-Elites discovery archive** that
expands by *mechanism/structure homology*, so it can traverse to such enzymes,
with a **path-aware fitness** so an annotation-blind-but-homology-reached
candidate is not buried.

## 2. What is built and verified

### Phase 0 — benchmark harness + leakage firewall (DONE)
`enzyme_design_agent/eval/` — self-contained, imports nothing from the agent.
- Two labeled tasks (`datasets/tasks.py`): amide-synthetase ligation (promiscuity
  / recall case) and nylon-66 hydrolysis (precision case). Ground truth resolved
  to UniProt.
- Pure metrics (`metrics.py`): pool-recall, recall@k, precision@k, hard-negative rank.
- Leakage audit + discovery-path provenance (`leakage.py`).
- Harness/loader/report/CLI; `reachability.py` (live probe); `production.py`
  (MAP-Elites vs linear driver).
- **Baseline locked** (grading the preserved `_prev` runs): ligation = 0%
  pool-recall + leaked search plan (25 LLM-named gene seeds); nylon = recall OK
  but cascade decoys out-rank the true family AND the recall was leakage-assisted
  (LLM named the answer gene).

### Phase 1 — MAP-Elites discovery engine (DONE; live-validated)
- `core/` — `models.py` (Candidate with discovery provenance, Score, Elite),
  `reaction.py`, `llm.py`, `state.py` (state.json adapter, eval-gradeable).
- `discovery/` — `firewall.py` (answer-blind entity strip), `descriptor.py`,
  `fitness.py` (path-aware), `archive.py` (MAP-Elites + curiosity selection),
  `variation.py`, `controller.py`, `moves.py`, `real_moves.py` (UniRef + Foldseek),
  `seed_moves.py` (DatabaseSeedMove).
- `tools/` — http, uniprot, structure_db, foldseek (subdir-aware, DB-selectable),
  rhea, interpro.
- `stages/` — `reaction_analysis.py` + `prompts.py` (answer-blind).
- `orchestrator.py` + `__main__.py` + `reporting.py`.
- **35 tests pass** (`python -m enzyme_design_agent.tests`): engine units,
  firewall, descriptor, archive, the synthetic recovery integration test, the
  orchestration seam, the code-leakage guard, and the baseline regression.

### Live validations performed
- **reaction_analysis + firewall** on the real query: produced **zero entity
  seeds** (vs 25 in the baseline) and derived EC 6.3.1/6.3.2 from the bond
  chemistry. Firewall is defense-in-depth.
- **Homology reachability probe**: real Foldseek recovers an annotation-blind
  promiscuous target via structural homology (TM ~0.6) from ANL stepping stones —
  the core architecture claim, on real data. (Swiss-Prot DB; ~14 s/query.)
- **Seed bug found & fixed**: the database seed truncated specific keyword/family
  hits under the broad-EC flood; fixed with priority retention (specific queries
  ahead of broad EC). After the fix the seed surfaces on-target families at the top.

## 3. Leakage controls (your hard requirement)

1. **Answer-blind firewall** (`discovery/firewall.py`): in benchmark mode strips
   LLM-named gene/accession seeds and accession-shaped tokens; the prompt also
   instructs the model to emit none. Verified live (0 seeds).
2. **No answer terms in agent source** — enforced by a guard test
   (`tests/test_no_leakage_in_code.py` + `eval/policy.py`) scanning the whole
   package except `eval/`/`tests/`. Green.
3. **Answer-aware audit** (`eval/leakage.py`): post-hoc, flags any ground-truth
   token in the plan and classifies *how* each candidate was reached
   (homology/derived = clean; gene-seeded = leaked), so leaked recovery is not
   miscredited.

## 4. Key findings / decisions

- Recovery of specific labeled catalysts (DdaG/SfaB) is a **recall sanity-check,
  NOT the optimization target** — chasing specific accessions overfits the answer
  key. The real assessment is breadth/depth of exploration + candidate quality.
- The DB upgrade (Swiss-Prot → UniProt50) is **blocked by a node network
  restriction** (Foldseek's download host is unreachable here); per your call we
  stay on Swiss-Prot for now. `foldseek_db/` reorganized into per-DB subdirs;
  code auto-selects the broadest present.

## 5. Critical review — what needs improvement (per phase)

**P1 engine**
- Descriptor is too coarse on real data (~10–14 cells from hundreds of
  candidates): uncharacterized hits collapse into `fold:unknown`. The Foldseek
  move does not yet populate `fold`, so structural neighbors don't get distinct
  cells. → **P2 priority**: ESM2-embedding (CVT) descriptor and/or Foldseek-
  derived fold, so breadth is real.
- Path-aware fitness gives every homolog the floor regardless of plausibility →
  risk of homology noise. → gate the floor on functional consistency, or balance
  with promiscuity-potential (P2).
- `ExpandAllProposer` is a placeholder; the mechanism-guided LLM variation
  operator is not built (must stay answer-blind).
- Residual family-keyword leak: the LLM derives the answer-family *name* as a
  keyword at runtime (not in code). Eval flags it (review-level); the hard entity
  leak is fixed. Open question for discussion.

**Deferred (not yet built)**
- P2: promiscuity-potential scorer; ESM2 embedding descriptor.
- P3: reaction-compatibility evaluator (CLIPZyme/EnzymeCAGE) + geometric (NAC)
  competence scorer + a real ranker; the literature seed move (a second path to
  sparsely-annotated targets).
- Evidence-gathering stage.

## 6. Production run (MAP-Elites vs linear)

Driver: `python -m enzyme_design_agent.eval.production <task>`. Runs both modes,
writes `runs_new/<task>/{mapelites,linear}/{state.json,report.md}` +
`comparison.json`. Covers both amide-synthetase ligation and nylon-66 hydrolysis.

_Results pending the in-flight background run; to be filled in. Inspect:_
- `runs_new/<task>/mapelites/report.md` — ranked candidates + breadth/depth.
- `runs_new/<task>/comparison.json` — MAP-Elites vs linear on pool size, archive
  cells, distinct EC subclasses, homology-reached count, recall sanity.

## 7. How to run

```
python -m enzyme_design_agent.tests                       # 35 tests
python -m enzyme_design_agent "<query>" --run-dir runs_new/x --max-rounds 2 --beam 6
python -m enzyme_design_agent.eval.production amide_synthetase_ligation
python -m enzyme_design_agent.eval <task> [run_dir]       # grade a run
```
Env: `ARGO_BASE_URL=http://lambda5.cels.anl.gov:52713/v1`, `ARGO_API_KEY=rzhu`.
