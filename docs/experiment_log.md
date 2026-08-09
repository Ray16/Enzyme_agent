# Experiment log

Every workflow change is recorded here with **what changed, the hypothesis, how
it was measured, and the result** — so we know whether a change actually helps,
not just that it shipped. Entries are append-only; newest first.

Result legend: ✅ measured improvement · ➖ neutral/no measurable effect ·
❌ regression (reverted or gated off) · ⏳ implemented + unit-tested, agent-level
A/B pending · 🔬 measurement only.

Measurement tiers:
1. **Unit tests** — logic correctness (in `enzyme_discovery_agent/tests/`).
2. **Deterministic live demo** — real APIs, scripted (no LLM); proves mechanics.
3. **Agent A/B** — full campaigns via Argo, old-vs-new config, scored by
   `enzyme_discovery_agent/eval/` on the ground-truth benchmarks
   (amide synthetase → DdaG/E2JA32, SfaB/A0A2H4T912; nylon hydrolase → NylC).
   This is the only tier that proves a change improves *agent behavior*.

---

## 2026-08-08

### E8 — Parallel multi-source steering (prompt + gates) — ✅ use / ❌ recall
- **Change:** rewrote LITERATURE_GUIDANCE / Phase-2 / ROLE / reflection so the
  agent fans out across all five sources in parallel, then consolidates; evidence
  gate now counts *unique* papers; `consolidate_literature` exempt from query-dedup.
- **Hypothesis:** the agent will use the new tools coherently (fan-out → consolidate
  → mine) AND recover more of the answer-family.
- **Measured — first full agent run** (amide campaign, rerank on, `eval/score_run`):
  `max_fanout=10`, `n_sources_used=5` (all sources incl. Europe PMC + Semantic
  Scholar), 88 unique papers (6.4% redundancy), 12 iters, 57 tool calls, **1.47M
  tokens**, 9 candidates (siderophore synthetases DesD/IucA/IucC — real amide-bond
  synthetases but NOT the answer-family). `consolidate_literature calls = 0`,
  `fetch_full_text calls = 0`, `answer_marker_recall = 0.0`, `found_answer = False`.
- **Adaptive recall breakdown** (eval targets): `mechanism_class` recall = **1.0**
  (found the ENTIRE ATP-dependent amide-synthetase space: atp-grasp, amide ligase,
  adenylation, nis/siderophore synthetase, nonribosomal, amide bond synthetase);
  `specific_member` (dapdiamide/DdaG/SfaB) = **0.0**; `specific_accession`
  (E2JA32/A0A2H4T912) = **0.0**.
- **Result:** ✅ the agent now genuinely USES the new tools — parallel fan-out across
  all 5 sources — AND its mechanistic reasoning is excellent (100% mechanism-class).
  ❌ two gaps: (a) `consolidate_literature` **never called** despite the prompt →
  steering-only isn't enough, needs a GATE; (b) the **specific** answer members
  (DdaG/SfaB) are still not surfaced from literature/database — the confirmed
  direction (per user) is to make those appear. Query planning + forward-citation
  chasing are the levers. Drives E9 (consolidation gate), E10 (forward-citation),
  E11 (query planning) — each A/B'd against `eval/`.

### M-infra — Adaptive, leakage-safe measurement 🔬
- **Adaptive recall:** benchmarks declare named `recall_targets` (scope
  papers/artifacts, any values) with `primary_targets` defining strict success;
  the scorer computes them generically, so a changed goal is a benchmark edit, not
  a code change. Separates "found the mechanism class" from "found the specific answer".
- **Leakage firewall (enforced by test):** the specific answer identifiers live
  ONLY in `eval/`; `test_leakage_firewall.py` asserts none appear in the agent
  runtime and that the agent never imports `eval` — so recall numbers can't be
  inflated by the answer leaking into the search. Verified clean.

### E7 — Full-text acquisition layer ✅ (mechanics) / ⏳ (agent value)
- **Change:** `tools/fulltext.py::fetch_full_text` downloads verified PDF/XML for
  top-ranked papers into `run_dir/artifacts/fulltext/` (+manifest, +extracted .txt).
  Chain arXiv→bioRxiv→EuropePMC-XML/PDF→oa_url→Unpaywall; %PDF/JATS verified.
- **Measured:** live demo downloaded 3 real papers (EuropePMC PDF 49k chars, bioRxiv
  PDF 52k, EuropePMC XML 76k); paywalled skipped; 11 unit tests incl. HTML-rejection.
- **Result:** ✅ downloads + verification + text extraction work; ⏳ effect on
  candidate reasoning pending (feeds later mechanism/kinetics extraction → predictive models).

### E6 — Pool consolidation: dedup + record-merge + single RRF ✅
- **Measured:** live campaign pool 163 raw → 145 unique (18 dups merged, 10
  multi-source); provenance preserved. 7 unit tests.
- **Result:** ✅ removes near-duplicates before reasoning; global ranking.

### E5 — Embedding cache ✅
- **Measured:** re-consolidation 0.16s → 0.02s (8×), 450 vectors cached.
- **Result:** ✅ repeated-work cut for multi-search campaigns.

### E4 — RRF rank fusion (stop discarding native retrieval order) ⏳
- **Change:** `_semantic_rerank` fuses native + cosine rank via RRF (order), cosine
  floor gates inclusion; sklearn eliminated (numpy dot); loud-vs-quiet failure split.
- **Measured:** 6 unit tests incl. "native #1 rescued above lower-cosine paper".
- **Result:** ⏳ correct + tested; rerank still default-OFF pending calibration + A/B.

### E3 — Rate-limit calibration + API keys ✅
- **Measured (live headers, not docs):** OpenAlex keyless 1000 credits/day (~100
  searches) → keyed 10,000/day; NCBI 3→10 req/s with key; S2 keyed 1 req/s. Keys in
  gitignored .env; secret redaction in HTTP errors (tested).
- **Result:** ✅ avoids silent OpenAlex 409 degradation; politeness correct.

### E2 — Client-side per-host throttle ✅
- **Measured:** N sequential calls spaced (N−1)·interval; concurrent burst serialized
  (real-threads test). 5 unit tests.
- **Result:** ✅ prevents 429 stampede across the tool thread-pool.

### E1 — Multi-source discovery (Europe PMC, Semantic Scholar, Crossref) ⏳
- **Measured:** live probes — each returns real IDs/full-text; S2 gives DOI+PMID+PMCID
  +arXiv in one call. Dropped DataCite. Key diagnostic: generic queries recover 0/131
  answer papers, targeted trivial-name queries recover DdaG/SfaB → **recall is the
  bottleneck, not narrowing**.
- **Result:** ⏳ sources work; agent-level recall gain pending A/B (+ query planning).

---

## Open / queued (each to be logged with its result)
- Agent A/B harness (`eval/`) — the instrument; makes every entry above measurable.
- Forward-citation chasing (multi-source snowball + `cited_by`, 60/40 budget).
- Saturation as a measured novelty-rate gate.
- Query planning as structured, axis-diverse Step-D output (leakage-safe).
- Then: revisit non-literature layers (reaction/substrate analysis, candidate
  reasoning, structural) — same measure-then-keep discipline.
