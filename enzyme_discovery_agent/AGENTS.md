# Generic enzyme discovery: cautions for code changes

Read `CLAUDE.md` before modifying this package. Preserve its data-leakage firewall
and the user's stricter instructions. The discovery engine must work across
unrelated enzyme systems.

## Runtime inputs and evidence

- Never put specific enzyme/gene names, accessions, organisms, benchmark
  substrates/products, expected mechanisms, or benchmark publication identifiers
  in runtime scripts, prompts, tool descriptions, defaults, comments or docstrings.
- Construct search terms from the user's input, its derived chemistry contract,
  and explicitly retrieved evidence. Do not supply a preferred reaction class,
  cofactor, protein family, pathway or activation mechanism as a default.
- Names may occur in user input, retrieved text and generated outputs. Preserve
  their provenance. A sequence/name association is a discovery lead, not proof
  of target activity.
- Shared chemical classification vocabularies and molecular patterns can
  recognize diverse input structures. They must not become target-specific
  prompting examples or preferred search strategies.

## Evaluation and memory

- Keep answer oracles in the evaluation namespace. Production imports must not
  reach evaluation code or load its answers, historical reports or development
  search artifacts as initial discovery context.
- Once a missed answer has informed a fix, that example is a development case.
  Do not describe its recovery as held-out generalization. Use independent tasks
  and separately maintained evaluation data for that claim.
- Cache identities must include all evidence that affects the result, the target
  contract and serving model where relevant. Explicit resume may reuse its own
  evidence; new runs must not silently inherit another campaign's findings.

## Controller behavior and verification

- Do not equate a finished task list or a model completion message with adequate
  scientific coverage. Follow complementary sources, relaxed searches, citation
  graphs and unresolved retrieved identifiers. Save an incomplete, resumable
  checkpoint when those routes cannot progress.
- Ranking and pruning need recorded evidence. Unknown evidence is not a negative
  result. Preserve exploration and allow new evidence to revise family priorities.
- Test control flow with synthetic reaction types and synthetic identifiers,
  including models that return empty extractions or repeatedly claim completion.
- Before finishing, audit Python literals, prompt text, runtime defaults, file
  reads, imports and cache dependencies. Record scope, exceptions, actual fixes
  and limitations; a keyword scan alone does not establish absence of leakage.
- Keep required tests passing and retain separate live-service checks. Do not
  claim tested behavior ran in an earlier campaign that loaded older code.
