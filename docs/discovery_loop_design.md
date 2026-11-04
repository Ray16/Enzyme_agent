# Closed-Loop Enzyme Discovery: Design Document

## Current decision-quality program — 2026-10-22

The immediate goal is a defensible experimental proposal from the existing pool,
alongside incremental improvements toward continuous autonomous discovery.
The canonical experimental proposal lives in `suggested_experimental_panel/`.
The proposal is a versioned snapshot of a campaign with multiple documented
sessions; source sessions, experimental outcomes and earlier selections remain
traceable when priorities change.

The previous robustness pass established storage, interface and validation
properties. It did not prove that available models make good scientific
decisions over a long campaign. Stronger-model availability is a hypothesis to
test, not a sufficient explanation for a narrowly configured retrieval space.

### Workflow invariants

- Preserve existing candidate identities, evidence, registry lifecycle, run
  layouts, reports, resume checkpoints and explicitly selected compute providers.
- Keep the established structural metric and its companion distance unchanged.
  Retrieval priorities may evolve; scientific validity requirements and frozen
  score definitions must not be relaxed to obtain favorable candidates.
- A model proposal, a family name, an annotation group, an exact-sequence hash,
  a predicted solubility score and an experimentally measured activity are
  different kinds of information. Unknown evidence remains unknown.
- New diagnostics do not mutate family decisions, override the existing ranking,
  establish global exhaustion, evaluate user permission or grant compute access.
- No answer-oracle imports, historical answer lists or target-specific defaults
  enter fresh discovery. Explicit historical-pool reuse must be recorded.
- Introduce policy changes through shadow evaluation first. Retain versioned
  before/after source and test records; avoid broad staging of unrelated work.

### Ordered implementation and acceptance gates

| Stage | Required work | Acceptance condition | Current scope |
|---|---|---|---|
| 1. State and feedback visibility | Include imported modeling/proxy observations in progress and bounded working context; preserve units and source locators | New observations change progress; repeated wrapper/prose updates do not; full evidence is retrievable | Implemented; regression checks required below |
| 2. Exploration frontier | Inspect all saved families, unmapped hypotheses, method outcomes, unresolved identities and expression/modeling questions | Pagination reaches records omitted from context; diagnostics are run-local and read-only | Implemented as observational state |
| 3. Novelty and saturation | Separate exact duplicates, heuristic annotation groups, near-sequence clusters, functional evidence, retrieval failures and method-local yield | Errors/partial results never imply depletion; lack of known sequences stays unassessed | Exact duplicates/annotation groups/method diagnostics implemented; near-sequence clustering and semantic novelty remain pending |
| 4. Decision contracts | Record the question, action, subject, evidence locators, alternatives, expected information and revision condition | The model can choose several defensible actions; unsupported actions can be identified without an exact query oracle | Offline rubric and public fixtures implemented; production contract/admission remains pending |
| 5. Model-dependent delegation | Route consequential interpretation separately from extraction or formatting; log actual serving models and transitions | Matched-state/budget comparisons establish task capability; unsupported critical judgments remain unresolved without creating new approval requirements | Design/evaluation pending; provider ladder unchanged |
| 6. Modeling admission and feedback | Select a specific modeling question, validate scientific setup, dispatch through the selected provider, reconcile results and reconsider discovery priorities | A bounded live cycle uses new results in its next decisions, without duplicate jobs or treating pose failures as inactivity | Export/import exists; continuous campaign scheduling and automatic feedback policy remain pending |
| 7. Continuous and experimental evaluation | Assess trajectories, scientific promotion quality, opportunity coverage, cost and later experimental feedback | Success requires useful grounded decisions, not larger raw pools or a polished final report | Pending; historical development cases are not held-out validation |

### State contract and diagnostic limitations

`models/exploration_state.py` reads the explicit campaign's current registry,
family decisions, saved family-learning records, expansion records and event log.
It makes no provider or database calls and writes no campaign state. It exposes:

- Canonical-format family labels separately from fallback gene/name partitions.
- Heuristic annotation-group counts and exact duplicates among known sequences.
  These are not verified biological independence or near-homology clusters.
- Per-family/method records, including failed, partial, successful and unresolved
  outcomes. Repeated successful records with known returns and no other annotation
  groups trigger a review question, not a global exhaustion verdict.
- Reported modeling status, setup/pose status, per-site angle and distance, and
  proxy units/transfer scope. This projection does not rerun physical validation.
- Questions about unresolved identities, functional support, expression,
  modeling readiness, contradictory family stops and unexplored hypotheses.

`get_exploration_state(section, offset, limit, query)` exposes the full saved
frontier beyond bounded working-context examples. It is not an active action
selector. A new family can still be missed if hypotheses/retrieval are narrow;
separate chemistry- and literature-based discovery routes must be evaluated.

`models/modeling_feedback.py` projects scientifically relevant saved observations
for progress tracking. It excludes report-wrapper paths, timestamps and rewritten
interpretation prose. A changed input/ensemble/check/metric result is retained;
an unchanged result re-exported under a new report wrapper is not new progress.

### Decision-quality evaluation before policy enforcement

`python -m enzyme_discovery_agent.eval.decision_evals` checks nine synthetic
rubrics: new structural observations, redundant neighborhoods, access failures,
unresolved identity, contradictory family stops, unknown expression, modeling
readiness, weak-model uncertainty and premature completion.

The default is a rubric self-check, not model evaluation. `--export-public-cases`
writes observations and a proposal schema without acceptable-action keys.
`--responses` grades a complete set of separately recorded proposals. Grading
checks admissible action categories, existing source references and required
decision fields; it does not establish source entailment, provider identity or
the quality of a scientific interpretation. Those need independent expert review.

The next evaluation should compare models on the same saved states, tool
responses and budgets, preserving actual serving-model identity and failed
requests. Include both isolated decision tests and action-dependent multi-turn
replays; a fixed replay must never return an unrelated tool result for an
unrecognized query. Use multiple acceptable actions, distinguish API errors from
negative evidence, and test compaction, resume, provider changes and stale inputs.

Compare current control, richer-state control and proposed shadow-policy control
before enforcing new scheduling rules. Report unsupported promotion, relevant
independent leads, duplicated effort, hypothesis coverage, valid modeling
admission, recovery behavior and cost. Do not infer a model's reasoning quality
from a schema-valid answer or a successful software regression suite.

### Deadline-aware experimental panel

The end-of-week proposal should primarily consolidate existing evidence rather
than depend on a new large run. Keep sequence identity, native-reaction support,
step-specific geometry/setup, experimental overlap, expression/construct scope,
diversity and uncertainty as separate columns. Reuse compatible existing
predictions; targeted new work should resolve identified gaps.

Keep current panel files and a machine-readable policy in
`suggested_experimental_panel/`, with immutable published versions and explicit
added/removed/changed comparisons. A changed policy must not rewrite earlier
experimental labels, claim previously unmeasured target activity, combine
incompatible modeling protocols, or imply that all sessions were one uninterrupted
autonomous run. Sequence and identity checks precede experimental handoff.

## Historical design snapshot — 2026-09-25

The sections below preserve the earlier design and its historical observations.
They describe dated behavior and proxy definitions, not the current controller
or a source of updated validation claims.

## 1. Problem

The current agent (`enzyme_discovery_agent_2026_0916`) runs a **fixed number of
iterations**, surfaces a ranked candidate list, and stops. It does not *keep
exploring*: it never uses what it learned from modeling or experiment to decide
where to look next. We want to reframe discovery as a **closed feedback loop**:

> structural modeling + a structural activity proxy → validate against experimental
> data → propose new candidates (ESM embeddings / ESM Metagenomic Atlas) → repeat.

This document specifies that loop, the reward, the exploration mechanism, and the
guardrails, and maps each part to the code that already exists or is planned.

## 2. Framing: active search, not (yet) reinforcement learning

The natural temptation is "RL over sequence space." We should **not start there**:

- The reward is **sparse and expensive** (wet-lab assays), the action space (sequences)
  is astronomically large, and the cheap proxy is **noisy and weakly correlated** with
  truth (ρ≈0.09 raw on our panel; see `proxy-eval-negatives-hydrolase`). Policy-gradient
  RL under those conditions is sample-inefficient and actively invites reward hacking.
- The right first framing is **multi-fidelity active search / Bayesian optimization**:
  - **State** = everything known so far: the pool of proposed candidates and their
    features, all proxy scores, all experimental measurements, and the running
    proxy↔assay calibration.
  - **Actions** = which candidates to *propose* (exploration), which to *model*
    (cheap fidelity: cofold + proxy on GPU), and which to *assay* (expensive fidelity).
  - **Reward** = activity, observed at two fidelities — a cheap structural proxy
    surrogate and a sparse, expensive experiment.
- RL remains a valid lens **later**, for the acquisition *policy* only, once we have a
  calibrated surrogate and enough experimental feedback to train against.

## 3. Reward: multi-objective and evidence-grounded

A single proxy score is a Goodhart trap. The reward is a **product of three
evidence-grounded factors**, each of which can veto (fail-closed):

```
reward(candidate) = mechanistic_plausibility
                  × calibrated_activity_proxy
                  × developability
```

1. **Mechanistic plausibility** — do we even have the reaction coordinate for this
   candidate's family? Supplied by the `modeling_reasoning` evaluator
   (`modeling-reasoning-bridge`): it resolves the discovery agent's schema-2.0
   handoff (metal per active-site × oligomer, step-specific geometry, cofactor mass
   balance) and runs a **fail-closed step-consistency gate**. A candidate whose
   inputs are internally inconsistent, or whose mechanism the literature does not
   support, is *abstained*, not scored.

2. **Calibrated activity proxy** — the NAC/angle-gate structural proxy. Treated as an
   **uncertain surrogate**, KNOWN WEAK (restraints invert the signal; ipTM does not
   capture selectivity; calibrated on only ~3 positives). It is continuously
   **recalibrated against experiment**; we track the proxy↔assay correlation over time
   in a **calibration ledger** (§5). Abstained / fail-closed candidates contribute **no
   reward and no calibration data** — they must not poison the surrogate.

3. **Developability** — solubility / expression / stability, **first-class**, not an
   afterthought. A proxy-only ranking could surface a poorly-expressing enzyme at the
   top. The layer combines (a) a **real predictor** — **NetSolP** (Thumuluri et al.,
   *Bioinformatics* 2022; ESM-based *E. coli* solubility/usability), run via the authors'
   own quantized models, with grounded Biopython descriptors (Kyte-Doolittle GRAVY,
   Guruprasad instability index) as context — and (b) a **curated liability store**
   (`liability_evidence`) where every record carries a verifiable primary citation. A
   documented liability is authoritative (fail-closed veto); a low predicted solubility
   flags the candidate and becomes an **exploration trigger**: find a soluble homolog of
   the same family (the mechanism transfers — see §4).

   *Grounding correction (2026-09-25):* an earlier draft used "PylC (Q46E79) is
   insoluble" as the motivating case. A primary-literature check **refuted** this —
   Quitterer et al. 2012 (PMID 22985965, PDB 4FFL) expressed PylC in standard *E. coli*
   and solved six 1.5 Å co-crystal structures, and NetSolP independently predicts it
   soluble (0.71). The claim was memory-based, not cited; it has been removed. Nothing in
   the reward may rest on an uncited claim.

**Guardrails:** fail-closed everywhere (abstain over guess); every factor grounded in
computation or a cited source, never in model memory; the proxy is a surrogate to be
distrusted and recalibrated, never an oracle; abstained candidates earn no reward and
feed no calibration.

## 4. Exploration: ESM Atlas seed-and-expand + family transfer

Novel candidates come from the **ESM Metagenomic Atlas** (~617M→700M+ ESMFold
metagenomic structures — dark matter beyond annotated UniProt). We do **not** bulk
download it (HQ subset ~1TB, full ~20TB; shared NFS has ~4.3TB free — discourteous).
Instead we use **real-time seed-and-expand**:

```
top candidate structure (seed)
   └─► structural neighbors in the Atlas        [esm_atlas_expand.py]
          (Foldseek public server, DB mgnify_esm30; api.esmatlas is gated)
       └─► filter (prob, pLDDT, dedup)
          └─► FAMILY ASSIGNMENT                  [family_assign.py]   ← makes hits actionable
                 (InterProScan on the returned sequence; or Foldseek
                  vs annotated AFDB/SwissProt → UniProt → InterPro)
             └─► reaction-coordinate transfer    [modeling_reasoning/reaction.py]
                    developability + proxy apply
                └─► new candidates → back to the loop
```

**Why family assignment is load-bearing.** An Atlas neighbor is an *unannotated*
metagenomic sequence. It only becomes actionable once assigned a family, because the
reaction coordinate, developability priors, and proxy calibration are all **family-level
properties that transfer to a homolog of a known family**. The reaction coordinates are
keyed by InterPro family, and `reaction.select()` picks family-first, so an assigned hit
inherits its family's mechanism. *Validated 2026-09-25:* PylC (Q46E79) → InterProScan →
`IPR011761`/`IPR003806` (ATP-grasp) → correctly selects the `two_metal_acyl_phosphate_ligase`
coordinate, **not** the adenylate route — the transfer distinguishes mechanism classes.

Exploration stays a small, storage-light, on-demand subspace that grows only around
good seeds.

## 5. Calibration ledger

The surrogate is only as trustworthy as its agreement with experiment. The ledger
records, per experimental round: candidate id, family, proxy score (+ proxy version),
assay result (+ conditions), and whether the candidate was abstained. From it we compute
a running proxy↔assay correlation **per family** (mechanism transfers by family, so
calibration should too) and expose it to the acquisition function as the surrogate's
uncertainty. This is the mechanism that keeps the proxy honest and detects Goodhart
drift (proxy climbing while assays flatten).

## 6. Cost tiering

```
ESM Atlas reservoir  (cheap, remote)
   → embedding / structure prefilter        (cheap, precomputed/remote)
      → family assignment + coordinate transfer   (cheap; EBI / local Foldseek)
         → cofold + structural proxy         (GPU — the first real cost)
            → experiment                     (expensive, sparse — the true reward)
```

Per-candidate **mechanistic follow-up search** is decoupled: shortlist-only, **off the
discovery hot loop**, reusing the discovery agent's `extract_reaction_mechanism` as a
library. Atlas search itself is *not* the bottleneck (~<2 min/seed wall-clock);
cofold+experiment are, which is why the acquisition function must be selective about what
it promotes to those tiers.

## 7. Architecture / component map

| Loop role | Component | Status |
|---|---|---|
| Exploration (propose) | `structural_modeling_agent/exploration/esm_atlas_expand.py` | built; async batch added |
| Make hits actionable | `structural_modeling_agent/exploration/family_assign.py` | **built + validated 2026-09-25** |
| Mechanistic plausibility + input compilation | `structural_modeling_agent/modeling_reasoning/` | built (17/17 tests) |
| Cheap fidelity (cofold + proxy) | `structural_modeling_agent/scripts/build_step1_inputs.py`, step1 NAC proxy | built (step1) |
| Developability veto | `developability/` (NetSolP + curated liabilities) | **built 2026-09-25** |
| Surrogate + acquisition | `loop/` (reward, acquisition) | **built 2026-09-25** |
| Calibration ledger | `loop/calibration.py` (proxy↔assay tracker) | **built 2026-09-25** |

## 8. Open questions for the PI

1. **Experimental cadence** — how many candidates per assay round, and turnaround? This
   sets the batch size for the acquisition function and how aggressively we can explore.
2. **Developability threshold** — hard veto vs soft penalty for predicted-insoluble
   candidates? (PylC is the test case.)
3. **Family scope** — restrict the first closed loop to one mechanism class (e.g. the
   amide ligases we already have coordinates for), or run family-agnostic from the start?
4. **Proxy trust floor** — with ρ≈0.09 raw, do we require a minimum per-family
   calibration before the proxy is allowed to influence ranking at all?

## 9. Evidence-quality principle (added 2026-09-25)

Every reward factor must be grounded in **computation or a cited primary source**, never
in model memory. And not all citations are equal: **weight evidence by source authority**.
A paper that *experimentally characterised* the protein (expressed, purified, assayed,
crystallised) is authoritative; a review or a paper that merely *mentions* it is not, and
co-occurrence of a term ("insoluble") near a protein name is not evidence the term applies
to it. The PylC case is the worked example: the 2012 structural paper (which purified and
crystallised PylC) settles solubility; a naive co-occurrence read of secondary sources
"gets it wrong". The curated evidence layer should therefore rank/annotate records by
evidence type and defer to primary characterisation.

## 10. Near-term build order

1. ✅ Family-assignment front-end (`family_assign.py`) — done.
2. ✅ Async batch parallelization of seed searches — done.
3. ✅ Developability layer — done (NetSolP + curated, citation-required liabilities).
4. ✅ Surrogate + acquisition + calibration ledger (`loop/`) — done.
5. Wire the components into a single driver and run one closed iteration end-to-end.
6. Add source-authority weighting to the curated evidence layer (§9); populate liabilities
   only from primary-characterisation sources.
