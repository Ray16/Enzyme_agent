# Enzyme Discovery Agent — Development Log

**Append-only.** Newest entries at the bottom. One entry per change/experiment:
what was done, why, what it validated, and any failure modes found. Do not edit
past entries — correct them by appending a new dated entry.

**Scope & epistemics (read before adding entries):**
- This file is a **human-facing evidence record only.** It is NOT read by the
  agent and must NOT be fed into any run — it names answers, so doing so would
  violate the data-leakage firewall (see `CLAUDE.md`).
- Record findings as **specific, dated observations** (one record, one target, one
  point in time), NOT as category-level laws. A specific failure does not
  generalize: "this record has no annotated substrate today" ≠ "this class is
  unreachable." External annotations change; a different target or member may
  behave differently. Do not let a past negative bias how a new run is designed.

---

## 2026-08-16 — Substrate-fit family expansion, Stage 1 (`substrate-fit-promotion-stage1`)

**What & why.** The family survey computed rich per-member native-substrate
resemblance but only *displayed* it to the LLM — strong members were never turned
into candidates, and enumeration was reviewed-only. Implemented family-anchored
substrate-fit **selection + promotion** so the relevant enzymes in a family
actually enter evaluation.

- `select_relevant_members()` — ranks same/unknown-direction members by
  multi-dimensional native-substrate resemblance (FG, MW, volume, chain length,
  HBA/HBD, polarity), RELATIVE (no tuned cutoff), bounded by reusing
  `MAX_REPRESENTATIVES_PER_FAMILY`.
- Gated TrEMBL deepening (`_unreviewed_members_by_ec`) — after the reviewed pass
  finds the best-fit reaction, pull that fold's unreviewed members of the SAME EC
  (`reviewed:false AND ec:<EC>`), capped. Turns TrEMBL from a 10^5 noise dump into
  targeted deepening of the relevant reaction.
- Design principle enforced: **code extracts evidence, LLM only reasons**; no new
  empirical thresholds; deterministic (no LLM calls in this path).

**Validated.** Full suite green (183). Live smoke on IPR000291: 120 reviewed + 120
gated-TrEMBL, 2 promoted. Firewall grep clean.

**Observations (specific, not generalizations).**
- The **A0A2H4T912** UniProt record, *as annotated on 2026-08-16*, carries no EC and
  no catalytic-activity comment → this specific record exposes no native substrate
  for the survey to compare. So for THIS record via THIS path it is not reached by
  substrate-fit today. (Not a claim that its class is inherently unreachable — a
  future annotation, or a related member, could change this; literature remains a
  route to it.)
- For the specific query substrate adipate, `chebi:30832` returned 0 reviewed
  UniProt enzymes on 2026-08-16 — i.e. adipate is (at present) no reviewed enzyme's
  annotated native substrate. This motivated resemblance-based (not exact) matching
  for this target. (A different target may have exact hits; do not assume 0.)

## 2026-08-16 — Ensure the agent USES it, Stage 2 (`substrate-fit-promotion-stage2`)

**What & why.** "A tool existing ≠ the agent using it." Traced promotion → final
answer and found a hole: `candidate_count` counts only {screened,evaluated,ranked}
(not `discovered`), and `auto_screen_candidates` screens a fold-relevance-BUDGETED
subset — so a promoted `discovered` candidate could stall, never evaluated.

- `auto_screen` now treats substrate-fit-promoted candidates (`source_tool`
  startswith `family_substrate_fit`) as PROTECTED (union with literature_named):
  always screened→evaluated→ranked, never dropped by fold budget/stop, never
  collapsed behind a longer ortholog.
- Promotion writes `protein_name` (not `literature_name`) — avoids corrupting
  provenance-by-name.

**Validated.** `test_promoted_candidate_screened.py` proves a promoted candidate
survives an allocator that drops everything else. Full suite green (184).
Oracle test: surveying DdaG's fold (IPR053158) and PuuA's fold (IPR027303)
retrieves + promotes both — the enzymes "mostly uncoverable via DB search."

**Failure mode / boundary.** Survey only fires on a fold it has a SEED into
(derived-family keyword → interpro, or a discovered member's most-specific fold,
which must satisfy 1<count<500 — so a superfamily-sized parent like IPR042099/500
is skipped). Cold-start reach of an obscure answer-fold is unsolved here → the
literature/mechanism-derivation axis (see TODO.md).

## 2026-08-16 — End-to-end validation run (in progress)

Launched a full monitored agent run on the amide/nylon benchmark (Run ID
`20260816_125338_...`) to confirm survey→promote→protect→rank fires in a real loop
and to compare ranking vs. the 20260812 baseline (DdaG #6, SfaB demoted, PuuA
absent). Result + observations to be appended here.

## 2026-08-16 — Mechanism-family derivation: code-enforced precision (`substrate-fit-promotion-stage3`)

**Evidence used (20260816 run, stopped early).** Derived family list had 10
families including inverse-direction ones (serine hydrolase, cysteine protease —
native CLEAVAGE, wrong for a formation target) and a non-free-substrate one
(BPL/LplA — protein-bound lysine acceptor). Also observed: heavy front-loaded
literature reading (160 papers, 0 candidates at iter 5 / ~7 min) that did NOT
sharpen the family list — derivation is a pure abstract LLM brainstorm from the
query, and its direction/substrate constraints were prose the model ignored.

**What & why.** Moved the hard constraints from LLM self-policing to CODE.
- Prompt: removed the breadth quota ("≥4, ideally 5-8; a single-family answer is
  wrong") → precision-first, no minimum, explicit "do not pad; a wrong family is
  worse than omitting it."
- Prompt: each family must now emit grounded evidence — `native_reaction`
  (physiological direction, generic), `native_direction`, `substrates_free`.
- Code filter `family_passes_constraints()` drops, deterministically: (a) families
  whose native direction is OPPOSITE the target — incl. the robust signal "native
  reaction consumes H2O ⇒ hydrolysis ⇒ cleavage", which catches hydrolases even
  when the model mislabels them 'formation'; (b) families with `substrates_free=
  false`. Firewall-safe: phrased by direction + substrate-freeness, never by naming
  enzyme classes.
- Dropped names are fed back into the retry `exclude` list so re-samples broaden
  instead of re-proposing ruled-out families (also saves tokens).

**Validated.** `test_mechanism_family_filter.py` (+ existing derivation tests);
full suite green (194). Smoke on the run's actual 10 families: keeps the 3 core
(ANL / ATP-grasp / GS-fold), drops the two inverse-direction + the protein-bound
one; adjacent formation-but-imperfect families (AsnB, dethiobiotin) correctly pass
the HARD filter and are left for the substrate-fit RANKING to demote (right
division: hard constraints in code, substrate nuance in ranking).

**Observation (not generalized).** For THIS query the target direction resolved to
'formation' via the word "synthesis"; if a query gives no direction cue the filter
skips the direction axis (never empties the list on an undetermined target).

## 2026-08-16 — Harvest efficiency: cap the initial full-text sweep (`substrate-fit-promotion-stage4`)

**Evidence used (20260816 run).** At iter 5 / ~7 min the run had 0 candidates and
was in `LiteratureHarvestGate`, having pooled ~160 papers (8 mechanism phrases)
and running one `is_paper_relevant` LLM call PER pooled paper before any candidate
existed. Effort was going into reading, not candidate search.

**What & why.** The relevance screen (read-before-mine) already existed; what was
missing was a CAP. Added `FULLTEXT_HARVEST_MAX_PAPERS = 40`: the pool is already
relevance-ranked, so the harvest now reads the top-40 most-relevant papers (the
mechanism-relevance LLM gate still filters within those), instead of gating +
reading the whole ~10^2 pool up front. Extraction was already bounded
(`_MAX_NAMED_PAPERS = 15`). Pure helper `_select_harvest_papers` (offline-tested).

**Design intent (per user).** Keep reading enough papers to understand context and
narrow down — but not too many at the INITIAL stage. Targeted deep reading around
specific candidates is a different, uncapped path, so this only bounds the broad
upfront sweep. 40 is one tunable config value.

**Validated.** `test_fulltext.py::TestHarvestCap`; full suite green (198).

**Not yet measured.** The actual wall-clock / token reduction on a real run — to be
recorded when the next end-to-end run is done.

## 2026-08-16 — Budget-conscious protection, Stage 5a (`substrate-fit-promotion-stage5a`)

**Concern (user).** Ranking irrelevant families low still WASTES compute expanding +
evaluating them. Stage-2 protection force-evaluated the top-2 of EVERY surveyed
fold — incl. low-fit families (e.g. ammonia-amine synthetases AsnA/AsnB whose
native amine, NH3, is nothing like a C6 diamine).

**What & why.** Protect only the GLOBALLY best-fit promotions across ALL families
(sort by `substrate_fit` metadata, cap at `FAMILY_PROTECT_BUDGET=8`), not per-fold
top-2. Relative fit rank — no relevance threshold; the cap is a resource budget
like the existing FAMILY_EVAL_* budgets. Promotions beyond it stay registered and
eligible via the normal relevance-budgeted screen; they just don't FORCE-consume
evaluation budget. So a low-fit family's members no longer guarantee-evaluate.

**Validated.** `test_promoted_candidate_screened.py::TestGlobalFitProtection`
(top-n-by-fit protected, low-fit excluded); full suite green (199).

**Still open (Stage 5b/5c).** Gate the expensive TrEMBL deepening to competitive
folds; align the fulltext harvest relevance gate with the family direction/chemistry
constraints (the E-64 cysteine-protease paper was read despite proteases being
excluded at the family level).

## 2026-08-16 — Run 20260816_132034 outcome + Stage 5 rework (`substrate-fit-promotion-stage5`)

**Run outcome (stages 1-4, old blanket protection).** reports_delivered, 47 iters,
103 candidates evaluated, **5.74M tokens** (vs ~2.5M / 73 evals baseline). Family
list clean (6, on-target incl. NIS); harvest capped (40 of 292). DdaG/E2JA32
PROMOTED by substrate-fit (fit 0.655, fold IPR053158) and screened — core mechanism
validated end-to-end. BUT cost ~2.3x baseline: 45 of 97 evaluations were substrate-
fit promotions (blanket-protected → all force-evaluated), incl. off-target
hydrolase/esterase/oxidoreductase folds that leaked in as 'unknown' direction.
DdaG ranked ~#11 (not improved vs #6 baseline); PuuA not recovered; user flagged
weak top candidates (CARNS1) and a relevant enzyme (DesD) not evaluated.

**Diagnosis.** The off-target promotions all had direction 'unknown' (no clean EC);
every on-target ligase (DdaG, ATP-grasp, glutamate/citrate ligases) had a 6.3.x EC
→ 'same'. So 'same' vs 'unknown' is the clean separator. Fit-rank alone was NOT
(hydrolases 0.49-0.54 interleave glutathione-synthetase 0.585 etc.), and the
Stage-5a budget=8 would have dropped DdaG (rank ~10).

**Fix (gate + generous cap, user-chosen).**
- GATE: `select_relevant_members` now promotes only direction=='same' (grounded
  same-direction), dropping the 'unknown'-direction off-target leak at the source.
- CAP: `FAMILY_PROTECT_BUDGET` 8 -> 12 (backstop above where real answers rank).

**Validated.** tests updated (unknown no longer promoted); suite green.

**Still open.** DesD-not-evaluated (recall: NIS fold reached but member not
promoted/found — direct search returned 0); CARNS1 ranked too high (ranking/
confidence-report quality, separate from promotion); fold-level survey-expansion
gating (Stage 5b) to also save the survey compute, not just eval.

## 2026-08-16 — UniProt query relaxation (search-quality fix) (`uniprot-query-relax`)

**Evidence.** A real, structurally-characterised NIS-family synthetase (PDB 7TGN)
returned 0 from `search_uniprot` because the agent ANDed name + full description +
organism + a keyword tag (5 terms) — UniProt AND-s free text, so one non-matching
term zeros the query, though the core 2-3 terms have 80-130 hits.

**Fix.** `search_uniprot` now RELAXES on empty results: retry progressively simpler
variants (drop trailing qualifier terms, down to the first two) until hits appear,
mirroring the InterPro simplifier. Structured `gene:` queries left as-is. The
effective (relaxed) query is reported back in the summary. `_relax_uniprot_query`
is pure/offline-tested.

**Validated.** `test_uniprot_query_relax.py`; live: the 5-term over-specified query
now recovers the family (relaxed to the 4-term core). Full suite green (204).

**Bug caught in review.** First placement put the helper between the `@tool`
decorator and `search_uniprot`, so the decorator wrapped the helper and dropped
`search_uniprot` from the registry (test_bootstrap caught it). Moved helper above
the decorator. Also fixed a firewall leak (enzyme name in a comment).

**Note.** family_decisions.json from the run shows the LLM's expand/stop calls over
the injected landscape were excellent (EXPAND the NIS + DdaG + L-amino-acid-ligase
folds, STOP the cofactor-glutamate + glycine-oxidase folds) — the survey→landscape
→set_family_focus mechanism is sound; the cost leak was promotion+protection BEFORE
those stops (Stage 5 gate fixes that).

## 2026-08-16 — Substrate score: worst-substrate bottleneck, not geometric mean (`substrate-score-min`)

**Evidence.** In the 20260816_132034 confidence report, a small-substrate amide
ligase (β-alanine+histidine, CARNS1) ranked #3, ABOVE genuine diamine ligases
(D-ornithine+citrate, spermidine+citrate) and above DdaG (#11). Its per-substrate
matches were [0.286 acid, 0.091 diamine] but the geometric-mean aggregation scored
it 0.161 — deceptively high.

**Root cause (user).** Geometric mean lets a one-sided candidate sneak in: fitting
ONE target substrate well masks failing the other. A two-substrate ligation needs
the pocket to host BOTH, so the honest score is the WORST substrate (bottleneck),
which is also what report_generator._accommodation_score already uses.

**Fix.** `tools/reasoning.py` substrate_similarity base_score: geometric mean ->
`min(per_target_best)`. CARNS1 now scores 0.091 (its diamine mismatch), not 0.161.
GEOMETRIC_MEAN_FLOOR now unused. Suite green (204).

**Still open (user: "the agent should reason over top candidates and their
substrates").** Deeper ranking work: ensure the worst-substrate accommodation
actually DRIVES the final order (CARNS1 outranking diamine ligases suggests the
ranking still under-uses it), and add an explicit substrate-aware reasoning pass
over the top-N candidates × BOTH target substrates (acid partner + amine partner),
role-aware so the diamine is compared to the member's amine partner.

## 2026-08-16 — Ranking discrimination: count-aware functional-group similarity (`accommodation-fg-counts`)

**Diagnosis (why CARNS1 outranked genuine analogues).** Measured accommodation for
the run's top candidates: ALL returned 0.5 — no discrimination. Cause: the
functional-group axis of `compare_substrate_properties` was a binary SET-Jaccard,
which saturates near 0.5 whenever the native carries the target's reactive group
plus one extra (shared/union = 1/2). Because `_accommodation_score` takes the MIN
across axes, that flat 0.5 capped every same-reaction candidate and ERASED the
chain-length/MW signal that actually differs between a true structural analogue and
a small-substrate enzyme. So ranking fell through to curation/characterization →
the well-studied but poor-fit candidate won.

**Fix.** Count-aware (Ruzicka / weighted-Jaccard over per-group COUNTS) fg
similarity. A target bearing N of a reactive group vs a native bearing N now scores
higher than vs a native bearing one. Verified: the genuine two-partner structural
analogue now scores 0.5 (worst-substrate) vs the small-substrate enzyme's 0.333 —
correct ordering restored. Combined with the earlier worst-substrate (min, not
geometric-mean) aggregation, the both-substrate bottleneck now DRIVES the order.
Suite green (204).

**Still open.** Role-aware correspondence (match target acid↔native acid,
target amine↔native amine explicitly rather than best-fingerprint-overlap); and a
validation run to confirm the whole batch (family precision + cost + DesD recall +
substrate-min + fg-counts) end-to-end. Deferred pending user go-ahead (cost).

## 2026-08-16 — Role-aware substrate correspondence (`role-aware-correspondence`)

**What & why.** substrate_similarity matched each target substrate to its
best-FINGERPRINT native, which could map both targets onto one native or pair an
acid target with an amine native. Now each target is matched only against natives
that SHARE a functional group (role) with it — role read GENERICALLY from the
substrate analyser's own `functional_groups` output (NOT a hand-listed subset;
generic per CLAUDE.md), falling back to all natives if a role match is
unavailable. Verified: adipate->citrate, HMDA->spermidine (distinct, role-correct).
Suite green (204).

## 2026-08-16 — Accommodation aggregation: physics-shaped, not blind min (`accommodation-physics-aggregation`)

**Concern (user).** `min(fg, chain, mw)` across mixed axes introduces blind
inclusion/bias — whichever axis is lowest (often a coarse/saturating chemical
metric) dominates and erases the rest; neither min nor geometric-mean of raw axes
is defensible.

**Fix.** Split the per-pair axes by PHYSICS: STERIC/SHAPE fit (MW, chain, volume +
aromatic/cyclic caps) is a genuine hard bottleneck (a too-big/wrong-shape substrate
can't bind) -> WORST of those; CHEMICAL compatibility (fg profile, polarity) is
graded/promiscuity-tolerant -> AVERAGE (no single coarse axis dominates).
Accommodation = worst of {steric, chemical} per substrate, then worst across the
two target substrates. No tuned weights. Verified ordering: genuine diamine ligase
0.344 > small-substrate 0.30 > aromatic-partner 0.25 > ammonia 0.00. Suite green
(204).

**Still open.** Make the LLM's whole-package reasoning over the (now role-aware,
count-aware, physics-grouped) per-substrate facts the PRIMARY ranker, with this
deterministic score as the fallback/tiebreak. And substrate-aware family
expansion (deprioritize ammonia-type families up front). Validation run in flight.

## 2026-08-16 — Family survey fires as soon as families are derived (`survey-fires-on-derivation`)

**Concern (user).** Systematic substrate-driven registration should happen as soon
as the agent deems a family relevant — not wait. Previously the survey's should_fire
required candidate_count >= 3 (SCREENED), so a reviewed enzyme with a perfect fold+
substrate match (e.g. PuuA/GS fold) was registered only mid-run, after screening.

**Fix.** FamilySurveyGate.should_fire now fires on the FIRST pass as soon as
mechanism_families_done (families derived = deemed relevant) — the derived families
seed the survey via search_interpro, no discovered candidates needed. Re-fires as
the pool grows (unchanged). So top substrate-matched reviewed members are registered
immediately when their family is known, not by later chance/timing. Suite green (204).

## 2026-08-16 — Substrate-aware family derivation (`substrate-aware-family-derivation`)

**Concern (user).** Families whose native substrate is a poor analog of the target
(ammonia-amine AsnA/AsnB, large-substrate Mur) still entered the expansion set,
wasting survey compute before being demoted.

**Fix.** The derivation now asks the LLM for each family's `archetypal_native_
substrates` (specific resolvable compound names). Code resolves them and scores the
family's OWN native substrates against the target — role-consistent, WHOLE-PACKAGE
(shared `chemistry_logic.whole_package_pair_fit`: steric bottleneck + graded
chemical average), via ONE-TO-ONE assignment (each native serves one target, so a
two-partner target can't flatter itself; forces the amine partner onto the family's
amine partner even when it is a tiny FG-less molecule like ammonia). Families
scoring <= a near-zero structural dead-end (0.1) are DROPPED from expansion; the
rest are REORDERED best-fit-first; unresolved broad-class natives score None and are
kept (never wrongly dropped). Verified: AsnA/AsnB (aspartate+ammonia) -> 0.0 DROP;
NIS (citrate+spermidine) 0.34, ATP-grasp 0.35 -> keep. Suite green (204).

**Bug caught in own testing (important).** First version used independent best-
native-per-target and scored AsnA 0.48 (the diamine matched aspartate, the acid
donor, because ammonia has no functional group). One-to-one assignment fixed it.
NOTE: the survey (`_substrate_facts`) and ranking (`substrate_similarity`) still use
independent best-per-target and share this blind spot — apply one-to-one assignment
there too (queued).

Also refactored: `_accommodation_score` now calls the shared `whole_package_pair_fit`
(no duplicated physics).

## 2026-08-16 — Family-fit plot layout fix (`family-fit-plot-layout`)

The in-axes legend (bbox 1.28,1.12) overlapped the radar title and the "Molecular
weight" axis label. Moved to a SINGLE figure-level legend below both panels (2-col,
fuller names to max_len 48, ✦ note in the legend title), widened figure to 14x7.2,
tight_layout rect leaves bottom room. Verified render: clean, no overlaps.

## 2026-08-16 — Firewall/genericity cleanup (`firewall-generic-examples`)

Audited all non-test source for hardcoded answer/benchmark-specific strings (user:
"we want the agent to be generic"). Core LOGIC was already generic (bond type +
direction derived dynamically). Found 5 pre-existing leaks — all EXAMPLES in tool
descriptions / prompts — and genericized them to pure format/placeholder syntax:
- database.py search_uniprot query: 'ec:6.3.2 AND synthetase'/'amide bond ligase'
  -> abstract syntax ('ec:<number>', 'gene:<name>', ...).
- kbase.py ec_class: '6.3'/'6.3.2'/'3.5.1 amide hydrolase' -> '<n>.<n>.<n>' format.
- structural_evidence.py accession: "'A0A0F7RJ52' for AsbA" -> 'P12345'.
- prompts/sections.py UniProt hyperlink: [P78061] -> [P12345].
- literature_extraction.py docstring: "amide bond formation" -> "<bond-type> bond".
Kept static/ec_bond_map.py (EC 6.3 -> amide is a generic IUBMB reference table).
Note: PuuA was NOT hardcoded — the LLM proposed `search_pubmed(glutamate putrescine
ligase PuuA)` / `gene:puuA` from its own knowledge; my code has zero answer
accessions. Leakage-firewall test + full suite green.

## 2026-08-16 — FIX: derivation empty-response + revert archetype-based family dropping (`fix-derivation-empty-and-archetype-drop`)

**Regression found (run 193505).** "mechanism-family derivation returned no
parseable families" — the LLM returned an EMPTY (len-0) response. Cause: the richer
prompt (archetypal_native_substrates + constraints) plus the provider's default ~4k
completion cap left a reasoning model no tokens for output. Fix: pass max_tokens=8192
to the derivation call (quality of discovery first — the extra output tokens are cheap).

**Deeper flaw found + reverted.** The archetype-based grounded family scoring would
DROP good families: a fold's ARCHETYPE (textbook substrate) is not its MEMBER range —
e.g. the GS fold's archetype is an ammonia-amine reaction (scores ~0), but the same
fold contains a member acting on a diamine (a strong target analogue). Dropping the
family by archetype would discard that member — a discovery-quality regression.
Reverted the drop/reorder + removed grounded_family_substrate_fit/_FAMILY_DEAD_END_FIT.
Substrate-aware selection stays at the MEMBER level (the survey), which already scores
each member's REAL native substrate and promotes the good ones. Archetype field kept
as informational only. Suite green (204).

**Lesson:** family-level archetype scoring conflates textbook substrate with member
diversity; member-level scoring is the correct grain.
