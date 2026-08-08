# Enzyme Discovery Agent — Implementation Summary

_Last updated: 2026-06-17 (session: MAP-Elites descriptor + Pfam-domain expansion rework)_

This document describes the full agentic workflow in `enzyme_design_agent/`, the
design decisions behind it, and the changes made in this session to fix a
candidate-pool flood and to recover the validated active enzymes for the
amide-synthetase ligation task. It is written for review; the package source
itself never names target enzymes (the leakage firewall forbids it), so the
worked example and ground-truth accessions live here (repo root) and under `eval/`.

---

## 1. What the agent does

Given a natural-language reaction query (e.g. *"find enzymes that can ligate
hexanediamine and adipic acid"*), the agent discovers and ranks candidate
enzymes that could catalyse that transformation — including **promiscuous**
catalysts whose native annotation is unrelated to the query, which a
similarity/keyword search misses.

The thesis: **direct the search by the transformation + catalytic module
(recall), gate precision by substrate character, and use structural/sequence
homology expansion to reach promiscuous catalysts that linear search cannot.**

---

## 2. Pipeline (end to end)

```
query
  │
  ▼
Reaction analysis (LLM + leakage firewall)        stages/reaction_analysis.py
  • ground substrates via PubChem -> SMILES -> RDKit functional-group profile
  • derive the bond made/broken + mechanism(s) from grounded chemistry
  • emit a search_plan: EC prefixes, keywords, literature terms,
    bond-bearing molecules, (production) candidate gene names
  • firewall strips answer-family/gene terms in benchmark mode
  │
  ▼
MAP-Elites discovery loop                          discovery/controller.py
  round 0  seed: database (UniProt EC/keyword/InterPro) + literature
                 (PubMed/OpenAlex -> snowball -> gene mining -> resolve)
  rounds   select elites (fitness + curiosity) ─► substrate-scope gate ─►
           expand (Pfam-domain | UniRef | Foldseek) ─► admit (dedup,score,place)
           every 2 rounds: LLM reflection runs literature gap searches
  │
  ▼
Holistic ranking (LLM judge, NO composite score)   scoring/judge.py
  • collapse orthologs, present top-N shortlist with native reaction / EC /
    substrate-match references; LLM returns a ranked order + tier + rationale
  │
  ▼
Two report views + artifacts                       reporting.py
  report_chemistry.md (native-chemistry suitability) | report_evidence.md
  candidates.csv  events.jsonl  state.json  run.log
```

### Discovery channels (all answer-blind / firewalled)

- **Database seed** (`discovery/seed_moves.py`): UniProt search over the plan's
  EC prefixes (Rhea-expanded), protein-name keywords, InterPro families, and —
  in production mode — the LLM's candidate gene names. Specific queries are
  retained ahead of the broad EC flood under the per-move cap.
- **Literature seed** (`discovery/literature.py`): PubMed + OpenAlex over the
  reaction's molecule/mechanism terms; snowballs citations; mines gene symbols
  from abstracts; resolves them to UniProt accessions; the LLM refines next-round
  queries onto the *families it just discovered* (never named targets).
- **Homology expansion** (`discovery/real_moves.py`): from each scope-vouched
  elite —
  - **Pfam-domain** (primary): proteins sharing the elite's catalytic-module
    *architecture* (co-occurring Pfam domains).
  - **UniRef**: sequence-cluster orthologs (≥50% identity).
  - **Foldseek**: structural neighbours (TM ≥ 0.5) — the long-range promiscuity
    jump that bridges fold-similar enzymes with divergent sequence/annotation.

---

## 3. The MAP-Elites core (why, not just what)

Discovery is an **illumination** loop (Mouret & Clune 2015), not a greedy search.
An archive keeps the best (fittest) candidate per **behavior cell**; selection
mixes exploitation (fitness) with curiosity (UCT bonus for under-visited cells),
so the search keeps expanding from currently-mediocre **stepping-stone** cells
that lead to promiscuous targets, instead of collapsing onto the obvious region.

- **Fitness** (`discovery/fitness.py`) is **path-aware**: a candidate reached by
  homology earns "discovery-path" credit *independent of its own annotation*, so
  a promiscuous enzyme whose native reaction is unrelated to the query is not
  buried at fitness 0. Credit scales with structural closeness (Foldseek TM /
  UniRef identity). A candidate whose **own EC** says it does a conflicting
  reaction class is *not* rescued (we know it does the wrong reaction); blind
  (no-EC) candidates are exempt — they may be the promiscuous catalyst.
- **Low-volume regime**: canonical MAP-Elites runs 10⁴ cheap iterations; our loop
  runs ~6–8 expensive rounds. Reconciliation: the LLM's biochemical knowledge
  **seeds the right families directly** (replacing the volume needed to find
  them); MAP-Elites then does focused local diversity-expansion. Principle: keep
  the LLM at the boundaries (analysis, reflection, ranking); keep the inner loop
  cheap + robust (pure compute + cached lookups + one cheap scope gate).

### The descriptor decision (this session)

The descriptor is the set of axes we **refuse to converge on**. Two principles
from the paper decide the axes:

1. The descriptor must be the **behavioral** "dimensions of variation of
   interest" — what the enzyme *does* (mechanism × substrate character), not its
   genotype (fold). Fold-as-axis is the competing-conventions anti-pattern: two
   enzymes with the same fold can do completely different chemistry (e.g. the ANL
   superfamily holds both aliphatic NIS synthetases and aromatic aryl-acid
   adenylases), and collapsing them to one cell discards the relevant one.
2. The descriptor must let the variation operator **cross cells**. Foldseek/Pfam
   expansion moves *within* a fold/module; if fold is also the cell axis, every
   offspring lands in its parent's cell → no illumination, just dedup (observed:
   "135 homologs collapsed into +4 cells"). Decoupling operator (structure) from
   descriptor (behavior) restores the stepping-stone dynamic.

**Resolution — the blind frontier.** For a blind hit reached structurally:
- **impute mechanism** from the shared catalytic (Pfam) module — mechanism *is*
  conserved along structural homology, so this is read off the expansion edge,
  not guessed;
- **leave substrate `unknown`** — substrate scope is the *divergence* axis we are
  hunting; copying the neighbour's substrate would collapse the promiscuity we
  paid to discover;
- **push the structural-proximity signal into FITNESS** (capability toward the
  target), not the descriptor. "Structural homologs have activity" is a
  *capability* statement, not a *niche* statement.

> Status: the descriptor is currently still `FoldMechanismDescriptor`; the
> mechanism×substrate swap is the remaining Phase-2 change. The **operator** and
> **gate** changes below (which drove the flood + recall fixes) are implemented.

---

## 4. The flood + recall fix (this session's changes)

### Symptom
On the baseline run the pool was dominated by **correct-EC-class but
substrate-wrong** ligase ortholog clusters (glutamine synthetase ×47, carbamoyl-P
synthase ×90, Pup-ligase ×123, Mur ligases, …). All are EC 6.3.x, so EC-class
triage called them "correct"; each entered as an ortholog cluster via homology
expansion.

### Root cause
The relevance/vouch unit was **EC class + SCOP superfamily** — too coarse. EC
`6.3.x` matches every ligase; one glnA vouched its whole superfamily, and homology
pulled all its orthologs. **EC says yes, substrate says no, and nothing made
substrate the gate.**

### The director-vs-gate split (the governing principle)
| signal | role |
|---|---|
| **EC** | coarse **class filter** only (exclude wrong reaction class) |
| **Mechanism + Pfam catalytic module** | **directs the net** (recall) — module is specific yet substrate-blind, so it still reaches promiscuous catalysts |
| **Substrate character** | **vouch / suppress / rank** (precision) — applied only to *annotated* hits, never as a recall gate on blind hits |

Substrate must not *direct* the search (that buries promiscuous enzymes whose
native substrate is unrelated) — it *gates precision* on what the net catches.

### Implemented
1. **`Candidate.domains` + `pfam_domains` + `scope_key`** (`core/models.py`):
   domain architecture is now first-class; `scope_key` = Pfam modules if known,
   else fold (graceful degradation). Threaded onto seed / literature / homology
   candidates.
2. **`PfamDomainExpansionMove`** (`discovery/real_moves.py`) + **`proteins_by_pfam`**
   (`tools/uniprot.py`): expand an elite by its **co-occurring Pfam architecture**
   (`xref:pfam-A AND xref:pfam-B`). The multi-domain signature is the specificity
   guard done right — a single module (e.g. PF04183) is shared by thousands, but
   the NIS architecture PF06276+PF04183 is the family itself. **TrEMBL-inclusive**
   (validated catalysts often live in TrEMBL); single-domain queries are kept
   shallow to avoid ubiquitous-domain floods.
3. **Substrate-scope gate at module granularity** (`orchestrator.py`): the
   previously-*dead* `_scope_gate` is now wired as the controller's `expand_gate`,
   vouching **Pfam modules** (not folds) and **on substrate scope** (not EC).
   - annotated + scope-OK → vouch its modules, expand;
   - annotated + clear substrate mismatch → **skip expansion** (suppresses the
     ammonia-ligase flood);
   - blind → expand only if its module was vouched by a scope-OK enzyme (protects
     promiscuity targets, stops blind drift).
4. **Literature gene resolution made TrEMBL-inclusive** (`discovery/literature.py`):
   mined gene symbols were resolved with `reviewed:true`, silently dropping the
   many validated catalysts in TrEMBL. Now reviewed-inclusive with a deeper
   per-gene fetch, and `domains` threaded onto literature candidates so a
   literature-found family member can itself be scope-assessed and Pfam-expanded.
5. **Candidate list 50 → 100** (judge `top_n`, report `top_n`).
6. **Eval positives extended** (`eval/datasets/tasks.py`) to the full validated
   active set + AsbB negative control; `eval/check_actives.py` grades recall +
   provenance against them.

All 62 unit tests pass (5 new for the module/scope logic); leakage guard green
(no answer terms in package source).

---

## 5. Worked example — amide-synthetase ligation (validation target)

Query: *find enzymes that can ligate hexanediamine and adipic acid.*
Validated actives (ground truth, Qian et al. bioRxiv 2025.02.28.640842 + user):

| accession | gene | family / note | reachability |
|---|---|---|---|
| A0A0F7RJ52 | AsbA | NIS synthetase (PF06276+PF04183), reviewed | Pfam-architecture expansion |
| Q93AT8 | AcsD | NIS synthetase (PF06276+PF04183), TrEMBL | Pfam-architecture (TrEMBL-inclusive) |
| J7H0W7 | DesD | NIS synthetase (PF06276+PF04183), TrEMBL | literature gene resolution (`gene:desD`) |
| A0A2H4T912 | SfaB | ANL/adenylation (PF00501, ubiquitous), TrEMBL | literature (hardest) |
| E2JA32 | DdaG | amide synthetase (PF27279, rare), EC 6.3.2.46 | exact-EC / rare-Pfam seed |
| Q81RQ8 | AsbB | NIS, in-family but **inactive** | negative control |

The NIS family is found via Foldseek from a single stepping-stone seed (B6EP94 →
aerobactin synthase iucC / putrebactin synthase pubC), confirming the
structural-homology thesis; the actives are then recovered by deepening the
catalytic-module (Pfam-architecture) expansion and the literature gene route.

### Results by iteration (recall into the candidate pool)
- **iter1** (Pfam move + scope-gate wired at module granularity): **0/5** (pool 840).
  Diagnosis: the NIS family *was* found — `iucC` (aerobactin synthase) / `pubC`
  (putrebactin synthase) reached via Foldseek from stepping-stone B6EP94, and the
  judge ranked iucC the closest analog — but the specific actives were beyond the
  Pfam/Foldseek depth caps and the literature filter was reviewed-only.
- **iter2** (Pfam by co-occurring *architecture* `xref:pfam-A AND B`, TrEMBL-inclusive;
  literature gene resolution made TrEMBL-inclusive + deeper + domains threaded):
  **1/5** — DesD recovered via literature (`gene:desD` → J7H0W7). pool 3322. The NIS
  synthetases were still missed because the **scope-gate skipped them** as
  "substrate scope mismatch": it required a *distinct* native per slot, but NIS
  enzymes use one bifunctional substrate. Substrate-gating was burying the
  promiscuous family — the recall-failure mode itself.
- **iter3** (loosen the *expansion* scope-gate to `require_distinct=False` so a
  bifunctional native covers multiple slots; keep the strict view for *ranking*):
  **3/5** (pool 5650). **AsbA** ← seed iucC via **Pfam-architecture**; **AcsD** ←
  iucC via Pfam+UniRef; **DesD** via literature. The full thesis works end-to-end:
  stepping-stone (Foldseek) → scope-gate pass → catalytic-module expansion → the
  validated actives. AsbA surfaces at chemistry-report **rank 50/100**.
  Remaining: ranking (the judge ranks clean two-substrate ligases — glutamate–cysteine
  ligase, MurE — above NIS; a defensible heuristic ≠ experimental truth) and pool
  flood (deep Pfam expansion buries AcsD/DesD).
- **iter4** (exact-EC enumeration seeding + Pfam `per_arch` 300→150): **0/5 —
  REGRESSION**. `per_arch=150` is too shallow for AsbA (needs ~200), and the ~1900
  exact-EC reviewed seeds crowded archive selection and starved the NIS stepping
  stone. Lesson: chasing an obscure singleton destabilised the robust family
  recovery. Reverted.
- **iter5** (reverted to the iter3 config): **3/5 reproduced** — AsbA via Pfam from
  a *different* stepping stone (`Q2G1M9`, vs iter3's `Q47318`), confirming the
  mechanism is robust to which NIS seed is found, not a lucky draw. AsbA ranks
  **41/100**.

**Final: 3/5 reproducible** — the validated NIS synthetase *family* (AsbA, AcsD,
DesD) recovered across independent runs via the demonstrated mechanism. The
remaining two are obscure promiscuous singletons:
- **DdaG** (E2JA32): recoverable in principle by exact-EC enumeration (`ec:6.3.2.46`
  returns it at rank 1, where the flat `ec:6.3.2.-` reviewed fetch buries it past
  rank 500) — but only with its own seed budget that cannot crowd the expansion
  path (the shared seed-cap version regressed the family). Left as a guarded
  next-step, not bolted on at the cost of the robust result.
- **SfaB** (A0A2H4T912): TrEMBL, ubiquitous AMP-binding domain, strain-specific
  accession — not reachable by architecture, gene symbol, or the swissprot-only
  Foldseek DB. Only a literature hit printing its accession would surface it (as you
  anticipated, the hardest case).

**Engineering judgment (why stop at 3/5):** overfitting the search to recover two
specific labelled accessions is leakage-adjacent and, as iter4 showed, trades away
the robust recovery of the genuine validated *family*. The biologically meaningful
result — the NIS diamine-citrate ligase family, recovered reproducibly by
stepping-stone structural homology + catalytic-module expansion — is achieved. DdaG
needs a budget-isolated exact-EC seed; SfaB needs a broader (TrEMBL) Foldseek DB or
a targeted literature accession.

---

## 6. Known gaps / next steps

- **Descriptor swap** to mechanism×substrate cells (Phase 2) — improves shortlist
  diversity (keeps aliphatic NIS and aromatic adenylases in separate niches).
- **DdaG** via exact-EC (`ec:6.3.2.46`) seeding + deeper EC fetch, or its rare
  Pfam (PF27279); the broad-EC fetch cap currently drops it.
- **SfaB** lives in a ubiquitous Pfam (AMP-binding) and TrEMBL — literature is the
  realistic route; structural reach needs a broader Foldseek DB than swissprot.
- **Foldseek DB** is swissprot-only; a full AFDB/UniProt50 index would make
  TrEMBL targets structurally reachable.
- **Run env**: conda `palm`; Argo proxy `http://lambda5.cels.anl.gov:52713/v1`
  (model `claudesonnet46`); `EDA_BENCHMARK=0` for production; Foldseek
  `EDA_FOLDSEEK_DB=swissprot`.

### Repro
```
ARGO_BASE_URL=http://lambda5.cels.anl.gov:52713/v1 ARGO_API_KEY=rzhu \
EDA_FOLDSEEK_DB=swissprot EDA_BENCHMARK=0 \
palm/bin/python -m enzyme_design_agent.eval.live_run amide_synthetase_ligation \
  --run-dir runs/<name> --max-rounds 8
palm/bin/python -m enzyme_design_agent.eval.check_actives runs/<name>
palm/bin/python -m enzyme_design_agent.tests
```
```
