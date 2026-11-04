# Enzyme Discovery Agent — Framework Description (for the Preliminary Results write-up)

Historical architecture description of `enzyme_discovery_agent_2026_0916/`.
This snapshot predates rolling evaluation, persistent research recovery, user
steering and the independent modeling controller. Tool counts and strict gate
ordering below are historical. Use the maintained package READMEs and current
source for implementation claims; do not treat this snapshot as a current audit.

---

## 1. What it is (one-paragraph framing)

The enzyme discovery agent is an autonomous system that, given only a natural-language
description of a target reaction, proposes and returns a ranked, evidence-backed set of
candidate enzymes — without ever being told the answer family. Its design principle is a
strict division of labor: **a large language model (Claude Sonnet 5, served via ANL Argo)
proposes *strategies* — the target's chemistry, plausible mechanism/fold families, search
terms — while deterministic Python tools and "gates" *find and adjudicate* the specific
entities** (genes, accessions, family members). This separation is enforced by a
**data-leakage firewall**: neither the agent's code, prompts, tool descriptions, nor
injected guidance may name the answer enzyme, its substrates, or answer-adjacent families;
they are written to be correct for *any* reaction (`CLAUDE.md`). The result is a discovery
process where every ranking decision is traceable to explicit, grounded criteria rather
than an opaque score.

## 2. Inputs

- A high-level query, e.g. *"find amide synthetases that ligate adipic acid and
  hexamethylenediamine"* (`__main__.py`).
- Optional target substrate(s): `--substrate` SMILES, `--structure-file`, or a sketch; if
  absent, the agent derives substrate SMILES from the query with an LLM call + deterministic
  name→SMILES fallback (`agent.py:573-592`).
- Optional `--target-profile` JSON (reaction/substrate/required-evidence spec).
- Budget: `--max-iterations` (default 100), of which the last 8 are reserved for evaluation +
  reporting (`config.py`, `FINALIZATION_RESERVE_ITERATIONS`).

## 3. The target-reasoning "contract" (chemistry-first)

Before any database query, one LLM call at run start derives a shared **chemistry contract**
(`models/target_reasoning.py::derive_target_reasoning`), persisted to `target_reasoning.json`
and read by every downstream stage so no stage re-invents the chemistry. It is built in two
layers:

- **Deterministic layer:** bond type(s), reaction direction, substrate functional groups
  (RDKit/SMARTS), and substrate topology parsed mechanically from the query/substrate.
- **LLM layer:** the activation problem (why the reaction is hard), plausible activation
  strategies, implied cofactors/leaving groups, disqualifiers, and — for polymer targets — a
  representative oligomer SMILES (RDKit-validated).

Fields and authority: `reaction_direction` and `substrate_topology` are **LLM-authoritative**
(the model's semantic read, with the deterministic parser as fallback — this fixes brittle
keyword matching, e.g. recognizing "hydrolyze/hydrolases" → cleavage); `bond_types` stay
deterministic-primary; functional groups/activation/cofactors come from their respective
layer. The contract's controlled vocabularies: direction ∈
{cleavage, formation, transfer, redox, other}; topology ∈
{free_small_molecule, carrier_protein_tethered, polymer, macromolecule, mixed}.

## 4. Architecture (four cooperating parts)

1. **LLM reasoning core** (`llm_client.py`, `agent.py::EnzymeAgent.run`) — an iterative
   agent loop. Each iteration: build a compact state header, call the LLM with the available
   tool set, execute the tool calls it chooses (parallel-safe ones concurrently), feed
   results back. The LLM freely chooses *which* tools to call; everything else is
   deterministic.
2. **Tools** (`tools/`, 34 registered) — the agent's "hands": chemistry analysis, sequence/
   family/reaction databases, federated literature + full-text mining, structured candidate
   screening/evaluation, structural-homolog search, and a persistent research notebook.
   (Full inventory in §8.)
3. **Deterministic gates** (`gates/`) — workflow policy that fires *between* iterations,
   inspects run state, and either advances a phase or "intercepts" the turn to inject
   grounded guidance back to the LLM (steer, don't compute). (Details in §6.)
4. **Persistent state** — a candidate registry (lifecycle-tracked), a research notebook,
   an append-only event log + human-readable exploration log, and per-candidate provenance
   files. All on disk, making runs auditable and resumable (`--resume-run`).

## 5. The staged workflow (what a run does, in order)

The phases are partly labeled and partly emergent from the gate sequence:

1. **Reaction analysis.** Derive the chemistry contract (§3). The agent converts the target
   transformation into bond type, direction, functional groups, and required activation.
2. **Literature-led discovery.** The agent searches the primary literature across six
   independent fronts (PubMed, OpenAlex, Europe PMC, Semantic Scholar, Crossref, plus
   citation snowballing), follows citation networks to specific enzymes, and mines *full
   text* (not just abstracts) to register characterized enzymes as discovery **seeds** — so
   literature leads discovery rather than only supplying keywords
   (`gates/literature_harvest.py`).
3. **Mechanism-family enumeration.** From the reaction's bond chemistry (not the answer),
   the LLM derives the distinct mechanism/fold families that could catalyze the target,
   ranked by how closely their native substrates resemble the target
   (`gates/mechanism_family_coverage.py` → `mechanism_families.json`). This turns the target
   into concrete, testable fold-family hypotheses.
4. **Family survey & fold landscape.** Each hypothesized family — and the fold of every
   distinct discovered gene — is resolved to its most-specific InterPro fold and *surveyed*:
   the agent enumerates the fold's reviewed members, reads each member's native reaction, and
   grounds every member in facts — reaction direction vs. target, and a decomposed
   substrate-fit profile (functional-group / size / polarity similarity). This produces a
   grounded "family landscape" (`gates/family_survey.py` + `tools/family_reactions.py` →
   `family_landscape.json`), injected to the LLM as numbers to reason over (no cutoff
   applied). The agent then records EXPAND/STOP decisions per fold (with a required reason)
   via `set_family_focus` (→ `family_decisions.json`).
5. **Evidence-based registration.** Discovered enzymes are registered into the candidate
   registry as `discovered`. Registration collapses orthologs to representatives and refuses
   any enzyme whose native reaction runs *opposite* the target direction (direction firewall,
   `candidate_registry.py`). A coverage gate blocks premature evaluation until a
   target-defined evidence-coverage contract (min candidates, min distinct families) is met
   (`gates/discovery_coverage.py`).
6. **Deterministic relevance-budgeted evaluation.** Rather than leaving evaluation to LLM
   discretion, a triage gate (`gates/ec_triage.py` + `gates/family_relevance.py`)
   deterministically: collapses orthologs to one representative per gene; gives every fold a
   breadth floor (so even off-target folds appear as documented negatives) except
   agent-stopped folds (waived to zero); and allocates the remaining evaluation budget by
   grounded relevance (same-direction and target-resembling folds first). Each selected
   representative is then evaluated (network-only, no LLM) into a structured evidence record.
7. **Structural & systematic expansion.** Qualified evaluated seeds are expanded by Foldseek
   structural-homolog search (`gates/homolog_expansion.py`) and by following literature links
   out of evaluated families (`gates/systematic_coverage.py`), widening the pool with new
   leads that are then evaluated.
8. **Ranking.** Candidates are ordered by a decomposed, traceable scheme (§7).
9. **Finalization.** Two-pass reports, a ranked CSV, the family landscape, an ESM-embedding
   candidate map (optional), and a structural-modeling handoff manifest are written (§9).

## 6. How the gates steer the LLM (deterministic policy)

Gates share one protocol (`should_fire`, `execute` → `GateResult`; `gates/__init__.py`).
When a gate "intercepts," the agent discards the LLM's chosen tool calls for that turn,
injects the gate's guidance (as if the agent had reasoned to it), and refunds the iteration
so steering never shortens the campaign (`agent.py::_apply_gate_result`). Firing conditions,
ordering, budgets, and persistence are **all deterministic**; the LLM only supplies (a) the
derived mechanism families, (b) full-text enzyme extraction, (c) search queries/accessions in
response to guidance, and (d) the `set_family_focus` stop/expand judgments (which the
deterministic budget then honors as an override layer). The gate set, in firing order:

| Gate | Fires when | Does |
|---|---|---|
| `literature_harvest` | ≥16 papers, ≥3 lit rounds | mines full text, registers tested enzymes as seeds |
| `mechanism_family` | ≥3 lit rounds | LLM-derives + ranks mechanism families; SEED→SURVEY→DEEP workflow |
| `family_survey` | mechanism families derived; re-fires as pool grows (≤6×) | surveys InterPro folds, builds landscape, promotes substrate-fit members |
| `evidence_selection` | enough papers but 0 candidates | forces comprehensive registration, then screening |
| `systematic_coverage` | after homolog expansion | follows literature out of evaluated families |
| `EntityExtraction` | first eval call | pulls genes/ECs/accessions from papers + citations |
| `DiscoveryCoverage` | candidates>0, pre-eval | blocks evaluation until coverage contract met |
| `ECTriage` | coverage met | ortholog-collapse + budget allocation + auto-screen + auto-evaluate → "proceed to ranking" |
| `HomologExpansion` | after triage | Foldseek expansion of qualified seeds (≤6) |

## 7. Candidate evaluation and ranking (decomposed, no composite score)

- **Per-candidate evaluation** (`tools/reasoning.py::evaluate_candidate`) gathers structured
  evidence — UniProt annotations + mutagenesis, InterPro domains, characterization/engineering
  paper counts, PDB/AlphaFold availability, bacterial-expression check, and ChEBI-resolved
  native substrates with functional-group comparison to the target. It computes **no composite
  score**; it returns facts.
- **Claim-level promotion validation** (`models/claim_validation.py`) adjudicates each
  candidate against its UniProt record with atomic, fail-closed verdicts
  (supported/contradicted/insufficient): accession identity, active record, native reaction,
  reaction direction vs. target, bond type, and substrate topology. A candidate is
  promotion-eligible only if every *required* claim is supported. **Substrate topology is
  advisory** — it blocks only on a definite contradiction (a monomer-only enzyme for a genuine
  polymer target; a carrier-tethered vs. free mismatch), never on unparseable evidence, and a
  polymer/oligomer-native enzyme is accepted for a small-molecule *proxy* target
  (`topology_compatible`).
- **Ranking** (`output/report_generator.py`) orders candidates by a **lexicographic spine over
  grounded facts** (never a weighted blend): hard gate on critical contradictions →
  fold/family relevance → substrate-fit → mechanism evidence class → Swiss-Prot →
  characterization → structure → engineering. A small LLM "focus rank" reorders the top
  shortlist by reasoning over the same grounded evidence, with the deterministic spine as the
  reliable fallback. **Substrate-fit is physics-shaped**: steric/shape axes (MW, chain length,
  volume) form a hard bottleneck (taken as the worst, with aromaticity/cyclicity caps), while
  chemical axes (functional group, TPSA, logP) are graded/promiscuity-tolerant (averaged); the
  overall fit is the worst across the target's substrates (an enzyme must accommodate *every*
  substrate — for multi-substrate reactions this is a one-to-one assignment scored by the
  worst-matched substrate). **Orthologs collapse** to one representative (keeping the most
  E. coli-expressible member; the rest listed as alternatives). Candidates are reported in
  tier bands (strong lead / worth testing / weak-gated) with grounded columns instead of a
  number; members of a fold stopped late are sunk below the reporting line but retained for
  provenance.

## 8. Tool inventory (34 registered; `tool_registry.py`, `bootstrap.py`)

- **Chemistry (RDKit, local):** `validate_smiles`, `get_molecular_properties`,
  `compute_fingerprint_similarity`, `load_structure_file`, `analyze_substrate`.
- **Databases:** `search_uniprot` (+ query relaxation), `fetch_uniprot_sequence` (+ identity
  validation), `search_kegg_enzyme`, `search_interpro` (+ query simplification),
  `get_protein_domains`, `lookup_genbank_to_uniprot`, `search_modelseed_reactions`,
  `get_modelseed_compound`, `search_structural_homologs` (Foldseek + AlphaFold),
  `map_candidate_neighbors` (ESM2 in-pool neighbors, opt-in), `expand_candidates_esm`
  (ESM2 expansion beyond the pool via wider family membership, opt-in).
- **Literature:** `search_pubmed`, `search_openalex`, `search_europepmc`,
  `search_semantic_scholar`, `search_crossref`, `extract_entities`, `deep_literature_search`
  (citation snowball), `sequence_literature_search`, `extract_experimental_measurements`
  (JATS tables + LLM vision), `extract_reaction_mechanism`.
- **Reasoning / synthesis:** `consolidate_literature`, `screen_candidates`,
  `evaluate_candidate`, `survey_family_reactions`, `set_family_focus`,
  `get_structural_evaluation`, `fetch_full_text` (real PDF/XML verification),
  `update_notebook`, `get_notebook`.
- Query deduplication prevents repeated identical searches; heavy actions
  (`set_family_focus`, `fetch_full_text`, structural/embedding tools) run serially, the rest
  in parallel.

## 9. Outputs (per run directory)

- `report_confidence.md`, `report_explorative.md` — two-pass ranked reports (confidence +
  suitability re-rank).
- `pass1_ranking.json` — the ranking of record.
- `ranked_candidates.csv` — decision list, one row per representative with grounded columns.
- `structural_modeling_handoff.json` — schema-2.0 shortlist manifest (top ~20) handed to the
  structural-modeling pipeline. Each candidate has three explicit sections: `evidence` contains
  retrieved identity, reaction, substrate, family/domain, structure, cofactor, and mechanism
  facts; `modeling_input` contains the agent-derived and reviewable cofold specification; and
  `selection` records why the candidate entered the panel, including ortholog and ESM-neighbor
  context. The top-level accession remains the stable candidate key.
- `family_landscape.json`, `family_decisions.json` (expand/stop with reason + InterPro name),
  `mechanism_families.json`, `family_budget.json` — the discovery/decision trail.
- `exploration_summary.json`, `exploration_log.md`, `events.jsonl` — run rollup + audit trail.
- `family_substrate_fit.png` and (optional) `candidate_map.png` — figures.

## 10. Safeguards (worth stating explicitly)

- **Data-leakage firewall:** the agent must propose families from the target chemistry; code/
  prompts encode no answer families/substrates/enzyme names; enforced by a test that greps the
  runtime for benchmark answer tokens.
- **Direction firewall:** enzymes whose native reaction runs opposite the target direction are
  rejected at registration and blocked at promotion.
- **Coverage before commitment:** evaluation is blocked until the discovered pool meets a
  target-defined breadth contract.
- **One-to-one multi-substrate scoring:** a strong match to one substrate cannot mask a poor
  match to another (worst-substrate scoring).
- **Provenance-by-construction:** every promotion is tagged with why (source tool + the
  substrate-fit or literature name that justified it); every family decision carries a reason
  and the resolved InterPro name.

---

### Worked example available in the draft
The amide-synthetase-for-nylon-6,6 run (main.tex §"Amide synthetase discovery") is the
end-to-end illustration: from the prompt alone the agent inferred bond *formation* of an amide
between a carboxylate and a primary amine, enumerated eight activation-competent fold families,
evaluated 63 candidates → ranked 44 across 18 InterPro families, and surfaced the
γ-glutamylputrescine synthetases (GlnA2, PuuA) and DdaG as the closest natural analogues. The
amide-hydrolase-for-nylon-6,6 run is the complementary (cleavage-direction) example: it
recovers NylB′/NylB/NylA and NylC as the top nylon-oligomer hydrolases.
