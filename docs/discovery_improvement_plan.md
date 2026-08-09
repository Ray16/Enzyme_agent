# Enzyme-discovery improvement plan

Grounded in measured runs (see experiment_log.md). Each item names the diagnosed
root cause, the concrete change, and how we measure it. Discipline: no change is
believed on a single run — REPLICATES (n>=3), rates not outcomes.

Current state (good, improving): run cost 1.78M → **817K tokens**; NIS-synthetase
family + SfaB (A0A2H4T912) now reach candidates; literature-first ordering in.
Remaining problems below are all root-caused from run artifacts.

## Target pipeline shape (the intended flow)
understand reaction → **read literature to learn mechanism/families** → resolve
in databases → **explore each family** → **register EVERY found enzyme** →
**triage for relevance** → rank → **final list = relevant only** → fetch full
text of mechanism + candidate papers (for downstream mechanism/measurement
extraction).

---

## A — Candidate flow: register-ALL → triage → relevant-only  (HIGHEST: "candidates aren't good")
CORE PRINCIPLE (user): **the agent must NOT downselect at the screening step.**
Every found enzyme is registered; downselection happens ONLY at triage/relevance.
Root cause of the current failure: the agent hand-picks ≤20 accessions to screen
(`MAX_SCREEN_REPRESENTATIVES=20`; evidence-selection gate says "select at most 20
accessions"), so found enzymes (DesD, resolved from 50 UniProt hits) are thrown
away BEFORE registration → never triaged → invisible.
- **A1 No downselection at screening.** Register EVERY found enzyme (every
  resolved gene / evidence-backed accession). Remove/neutralize the ≤20 cap and
  the "pick at most 20" gate instruction. (Nuance to confirm: "found" = enzymes
  discovered via evidence, e.g. DesD/PuuA/McbA — a raw 500-hit UniProt *result
  page* is still a lead; but a resolved gene like DesD MUST get its representative
  registered, never dropped.)
- **A2 Triage = catalytic-MECHANISM feasibility (NOT promiscuity prediction).**
  Predicting whether an active site will accept a non-native substrate is a hard,
  unreliable problem even computationally. So relevance is instead: **can the
  reaction proceed GIVEN THE CATALYTIC MECHANISM?** Primary, data-answerable signals:
    (i) direction (formation, not cleavage);
    (ii) bond chemistry (amide via an ATP-activated acyl-phosphate/adenylate);
    (iii) substrate TOPOLOGY — the mechanism must ligate TWO FREE small molecules
         (free diacid + free diamine). Mechanisms needing a carrier-protein-tethered
         substrate (NRPS adenylation) or a polymer primer (cyanophycin) CANNOT do
         this and are downranked, regardless of substrate identity.
  Substrate-similarity facts (MW/Tanimoto/FG, already computed) stay INFORMATIONAL
  context, not a hard predictor. Most of these mechanism signals already exist in
  evaluate_candidate as soft warnings (direction reject, bond-type, `substrate_class`
  = polymer/carrier_protein_tethered) — A2 ELEVATES them to the primary ranking axis.
- Measure: every explored/resolved family has ≥1 registered rep (0 "resolved but
  unregistered" leaks); `family_coverage.candidates`==1.0; final list substrate-relevant.

## B — Full-text acquisition wired correctly (feeds downstream mechanism/measurement extraction)
Root cause: after the evidence-selection gate, `agent.py` restricts tools to a
whitelist that OMITS `fetch_full_text` → it is stripped, so it never fires
(0 calls every run) regardless of the prompt.
- **B1** add `fetch_full_text` to the post-gate whitelist (or fetch pre-gate).
- **B2** fire it on each family's mechanism paper + registered candidates.
- Measure: `fetch_full_text_calls>0`, `artifacts/fulltext/` populated.

## C — Token efficiency (user priority)
Root cause: 98% of tokens are PROMPT; context re-sent every call grows 18k→148k;
NO prompt caching. (817K win was fewer iterations, not lower per-iteration cost —
still ~90K/iter.)
- **C1** verify Argo passes Anthropic `cache_control`; if so cache the static
  system prompt (+ message prefix). ~10x lever on the repeated prefix.
- **C2** context compaction: evict raw per-source search results once
  `consolidate_literature` has run; window old turns.
- **C3** tighten the system prompt (grown to ~7.8k tokens via steering).
- Measure: tokens/run, prompt-token share, per-iteration prompt size.

## D — Measurement rigor (so we trust A–C)
- **D1** replicate-run harness (n>=3/config), compare rates.
- **D2** tighten `family_coverage`: resolve REGISTRY accessions → family (SfaB is
  registered but showed adenylation "dropped" — text-scan misses it).
- **D3** path-provenance metric: which tool+query first surfaced each answer
  (from the E11 attribution error — never attribute from n=1 again).
- **D4** keep the leakage firewall (answers only in eval/).

## E — Quick fixes
- **E1** `report_confidence.md` truncated (report-gen completion token cutoff) —
  raise max_tokens or generate in sections.

---

## Recommended sequence
1. **D2 + D1** (fix the metric + replicate harness) — measure correctly first.
2. **A** (register-all → triage → substrate-relevant final) — the candidate-quality core.
3. **C** (token efficiency: caching + compaction).
4. **B** (full-text wiring) and **E1** (report) — quick, independent.
Then validate A/C with n>=3 replicate runs on the amide + nylon-hydrolase benchmarks.
