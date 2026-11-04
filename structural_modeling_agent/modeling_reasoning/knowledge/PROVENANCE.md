# Vendored literature knowledge

`cofolding_rules.json` is a **snapshot** of the reasoning module's PDF-extracted,
exact-grounded cofolding guidance. It is vendored here so `structural_modeling_agent`
stays runnable without the `kg-memory-system` repo present, and so every resolved
handoff records exactly which knowledge version justified its decisions.

- Source repo: `kg-memory-system/reasoning`
- Source file: `results/v2/cofolding_batch/reasoning_synthesis.json`
- Snapshot checksum: see `SOURCE.sha256`
- Schema: `{ candidate_rules: [{ decision_point, applies_when, evidence: [{claim_id, paper_id, quote, claim_type}] }], corpus, coverage, caveat }`

Each rule is an **atomic, exact-grounded** candidate rule: its `evidence` quotes
map verbatim into the cited paper (`paper_id`) at extraction time. Rules are
candidate guidance, not verified defaults — a `single-paper-candidate` is weaker
than a `cross-paper-candidate`. Do not treat the presence of a rule as proof.

## Refresh

Run `refresh.sh` to re-vendor from the reasoning repo after that repo's synthesis
is regenerated. The checksum changes; commit the new snapshot deliberately so the
knowledge version used by any past run stays reconstructable.
