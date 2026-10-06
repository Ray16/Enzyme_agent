# enzyme_discovery_agent — working rules

## DATA-LEAKAGE FIREWALL (highest priority — do not violate)

This agent must DISCOVER enzymes without being told the answer. The whole point
is that the LLM proposes *strategies* (reaction type, mechanism, search terms)
and the *tools* find the specific entities. So the agent's own code and prompts
must never encode the answer, the benchmark substrates, or answer-adjacent hints.

**NEVER put any of the following into a system prompt, a tool/@tool description,
a tool PARAMETER description, injected guidance text, OR code comments/docstrings:**

1. **The benchmark query's substrates or product** — e.g. `nylon`, `adipic acid`,
   `hexamethylenediamine`, `diacid`, `diamine`, `polyamide`, `HMD`, `amide diads`.
   Use generic placeholders: "the target substrate(s)", "the target product",
   "[product]", "[substrate A]/[substrate B]".
2. **Answer enzyme/gene names or their products** — e.g. `DdaG`, `SfaB`, `McbA`,
   `PuuA`, `DesD`, `IucA`, `sbnE`, `dapdiamide`. Ground truth lives ONLY in
   `eval/benchmarks.py` (the test oracle the agent never sees).
3. **Answer-ADJACENT mechanism/family examples that presume the answer** — for
   an amide-synthesis benchmark this means `ATP-grasp`, `adenylation domain`,
   `acyl-phosphate`, `aminoacyl-adenylate`, `NRPS`, `condensation domain`,
   `cyanophycin`, `siderophore`/`NIS synthetase`, or using `amide` as an EXAMPLE
   bond/functional group. Enumerating "the classes that are wrong" (transferase,
   hydrolase, protease…) is also leakage — it implicitly names the answer's class.

**What IS allowed (generic, reaction-agnostic):**
- Direction language derived from the query: "bond FORMATION vs hydrolysis/
  cleavage", "ligation/condensation/synthesis". This describes the target
  direction, not the answer.
- A bond-TYPE ENUM for classification (`bond_formed: amide|ester|thioester|…`) —
  the LLM picks one; it is not seeded.
- Generic domain/fold vocabulary as *shape* ("a catalytic domain", "a fold
  family", "a cofactor-binding motif") with no specific family named.
- Resolving the USER'S OWN query substrates to SMILES/synonyms (that is task
  setup, the user provided them).

**When writing a prompt, write it for ANY query.** Ask: "would this sentence
still be correct and non-leaky if the target were a completely different
reaction?" If it only makes sense for the amide/nylon benchmark, genericize it.

**Before finishing any prompt/code change, grep for leakage:**
```
grep -rniE "nylon|adipic|hexamethylene|diacid|diamine|polyamide|ddaG|sfaB|mcbA|puuA|desD|dapdiamide|acyl-phosphate|aminoacyl-adenylate|ATP-grasp|cyanophycin|siderophore|NIS synthet" \
  enzyme_discovery_agent --include=*.py | grep -viE "/runs/|test_|eval/benchmarks"
```
It must return nothing (bare acronyms like `NRPS`/`ATP` inside generic
acronym-stopword lists alongside `KEGG`/`BLAST` are the only exception).

## Env / running
- Python: `/homes/rzhu/miniforge3/envs/palm/bin/python`
- Tests: `python -m unittest discover -s enzyme_discovery_agent/tests` (keep green)
- LLM via Argo (`argo:claude-5-sonnet`); runs land in `enzyme_discovery_agent/runs/`.
