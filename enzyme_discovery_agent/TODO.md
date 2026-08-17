# Enzyme Discovery Agent — TODO / Roadmap

_Last updated: 2026-08-16 (end of session). Branch: `feature/retrieval-expansion`._
_Full blow-by-blow with rationale + failure modes: see `DEVLOG.md`._

## Guiding principles (from this session, keep enforcing)
- **Quality of discovery FIRST, token cost second.** Never sacrifice recall/precision to save tokens.
- **Code extracts evidence, LLM only reasons.** FG/size/correspondence/direction extracted deterministically.
- **No hardcoded relevance thresholds / magic numbers.** Prefer relative ranking; resource *budgets* are OK.
- **Substrate-awareness belongs at the MEMBER level, not the family archetype level** (a fold's archetype ≠ its member range).
- **Data-leakage firewall** (`CLAUDE.md`): no answer enzymes/accessions/substrates or amide-specific examples in source/prompts. Audited clean this session.
- **A tool existing ≠ the agent using it** — wire capabilities into deterministic gates; make their output reach the final answer.

---

## ✅ DONE THIS SESSION (all tagged, tested, firewall-clean; suite 204 green)

**Substrate-fit family expansion**
- `substrate-fit-promotion-stage1` — family survey SELECTS members by multi-dim native-substrate resemblance and PROMOTES them to candidates (was advisory-only). Gated TrEMBL deepening (EC-anchored).
- `substrate-fit-promotion-stage2` — promoted candidates PROTECTED in auto_screen → always reach evaluation/ranking.
- `substrate-fit-promotion-stage5` / `-stage5a` — promotion gated to grounded **same-direction** (drops unknown-direction off-target leak that doubled eval cost); protection budget-capped by global fit.
- `survey-fires-on-derivation` — survey fires as soon as families are DERIVED (not gated behind screened candidates), so top substrate-matched reviewed members register early, not by chance/timing.
- `full-fold-reviewed-coverage` — score the WHOLE fold's reviewed members (cap 500 = surveyable ceiling), not the first 120 by UniProt default order.

**Family derivation precision**
- `substrate-fit-promotion-stage3` — code-enforces DIRECTION + FREE-SUBSTRATE constraints (drops inverse-direction hydrolases/proteases the LLM listed anyway; H2O-reactant ⇒ cleavage signal). Precision-first prompt, no breadth quota.
- `fix-derivation-empty-and-archetype-drop` — (a) `max_tokens=8192` on derivation call (richer prompt starved reasoning model → empty response bug); (b) REVERTED archetype-based family dropping (would drop good folds whose archetype ≠ member range, e.g. GS/PuuA).

**Ranking / substrate scoring quality**
- `substrate-score-min` — worst-substrate (min across the two target molecules), not geometric mean (one-sided fits sneak in).
- `accommodation-fg-counts` — count-aware functional-group similarity (binary set-Jaccard saturated at 0.5 and, under min, erased size/aromaticity signal → CARNS1-type over-ranked).
- `accommodation-physics-aggregation` — physics-shaped per-pair fit (steric bottleneck + graded chemical average), shared `chemistry_logic.whole_package_pair_fit`, not blind min over mixed axes.
- `role-aware-correspondence` / `role-aware-promotion` — match target↔native by shared functional group (generic from `functional_groups`, non-gating fallback), in BOTH ranking and promotion paths.

**Efficiency**
- `substrate-fit-promotion-stage4` — cap initial full-text harvest to top-40 relevance-ranked papers (was ~250).
- `uniprot-query-relax` — `search_uniprot` relaxes over-specified AND-ed queries on 0 results (recovered DesD/McbA/HolE, previously 0).

**Firewall / genericity**
- `firewall-generic-examples` — genericized 5 pre-existing hardcoded examples (answer accessions P78061/A0A0F7RJ52, amide-specific EC/query examples) → pure format placeholders. Core logic already generic.

**Plot**
- `family-fit-plot-layout` — legend moved below panels (was overlapping title/axis labels), fuller names.

---

## 🔜 REMAINING (priority order)

### 1. One-to-one substrate assignment in survey + ranking (HIGH — correctness)
The grounded family scorer got this right, but `family_reactions._substrate_facts` (survey/promotion) and `reasoning.py substrate_similarity` (ranking) still use **independent best-native-per-target**, so they only weakly demote ammonia-type members (target amine matches the amino-acid *acid* donor, since ammonia has no FG). Apply the same one-to-one assignment (each native serves one target) there. This is what makes the member-level substrate-awareness actually demote ammonia.

### 2. Literature axis (recall of literature-only candidates like SfaB)
- Broaden `_resolve_named_enzyme` accession extraction (PDB/RefSeq/EC+organism).
- Confirm-by-reaction resolution (disambiguate with the paper's extracted substrate/reaction).
- Citation second-hop (fetch a cited homolog's own primary paper).
- Cross-feed survey substrate-class terms into the harvest.
- Make the harvest DECISION less random (user flagged peripheral downloads): the relevance gate should align with the family direction/chemistry; consider stratifying the top-40 harvest ACROSS derived families so class coverage is guaranteed, not emergent.

### 3. Deterministic member coverage (finish the "not by chance" goal)
Full-fold coverage is in, but promotion is still top-2/fold (`MAX_REPRESENTATIVES_PER_FAMILY`). Consider global-fit selection across folds so the strongest matches always register regardless of per-fold quota.

### 4. Validation harness
Turn the oracle spot-checks into a repeatable metric (recall@k for the answer enzymes + provenance). Ground truth stays ONLY in `eval/benchmarks.py`.

### 5. Cost / efficiency (AFTER quality)
- The substrate-fit survey scoring on large folds (500 members × ChEBI+RDKit) is latency; cache/parallelize, skip slow 3D for cofactor-like natives.
- Fewer finalization LLM passes; cheaper model for sub-steps.

---

## ⚠️ Open validation gap
**No clean end-to-end run completed with the FULL fix set.** Runs were repeatedly restarted as fixes landed; the last attempt (195127) was stopped for cost. **Next session: run ONE full validation** and check: family list clean (no inverse-direction), DdaG/PuuA/DesD recovered, CARNS1-type demoted below genuine analogues, token cost back toward ~2.5M (from the 5.7M pre-Stage-5 blowup). Baseline for comparison: run `20260816_132034` (complete, 5.7M tokens, DdaG #6).
