# Review guide

This handoff publishes the active source packages and packaging metadata that
were previously incomplete in GitHub. It also includes the current discovery
edits, tests, contracts, static knowledge assets and examples. The source inventory
is in [review_file_manifest.json](review_file_manifest.json).

## Suggested review order

| Review question | Files |
| --- | --- |
| What does the controller execute, resume and stop? | `enzyme_discovery_agent/agent.py`, `production.py`, `steering.py` |
| Which scientific instructions and defaults reach the model? | `enzyme_discovery_agent/prompts/`, `config.py`, `bootstrap.py` |
| How much context and tool material is repeated? | `enzyme_discovery_agent/context.py`, `llm_client.py`, `claude_subscription.py` |
| How are identity, native function and target fit distinguished? | `enzyme_discovery_agent/models/claim_validation.py`, `native_annotation.py`, `reaction_roles.py`, `target_review.py` |
| What qualifies as a primary proposal? | `enzyme_discovery_agent/tools/discovery_proposals.py`, `models/modeling_portfolio.py` |
| Are evidence interpretations and source fingerprints retained? | `enzyme_discovery_agent/models/research_decisions.py`, `reasoning_trace.py`, `candidate_registry.py` |
| How do retrieval, family coverage and batch scheduling work? | `enzyme_discovery_agent/gates/`, `tools/literature.py`, `tools/database.py`, `models/rolling.py` |
| What reaches the CSVs, handoff and reports? | `enzyme_discovery_agent/output/`, `run_layout.py`, `models/handoff_contract.py` |
| How are structural setups and prediction reuse checked? | `structural_modeling_agent/agent.py`, `validation.py`, `cofold_execution.py`, `reuse.py`, `campaign_catalog/lookup.py` |
| Which metric is frozen? | `structural_modeling_agent/CANDIDATE_RANKING_POLICY.md`, `knowledge/attack_angle_metric.json`, `proxies/attack_angle.py` |
| What can submit computation? | Both agents' `production.py`, `structural_modeling_agent/lambda_boltz_backend.py`, `enzyme_design_agent/tools/` |

The canonical modeling package is `structural_modeling_agent`. Older tracked
paths under `structural_modeling` contain historical material and are not the
package selected by `pyproject.toml`. Some documentation links to cluster-held
campaigns or archived studies; those are external data dependencies.

## Current scientific status

The earlier 48-entry panel is an exploration panel, not 48 established catalysts.
The most recent bounded discovery continuation retrieved additional sequence
records and literature hits but saved zero primary candidate proposals before
its conservative request-admission checkpoint. This is a workflow shortfall,
not a finding that no suitable enzymes exist.

A compact, source-derived status record is provided in
[review_checkpoint.json](review_checkpoint.json). It contains the actual recent
actions and budget accounting, rather than a reconstructed successful run.

The local MSA producer was cancelled at the user's request. No GPU modeling was
submitted for this panel. The server-side MSA batch had already been accepted by
ColabFold; cancellation on that remote service was not confirmed.

## Review response and subsequent repairs

See [REVIEW_RESPONSE_20261104.md](REVIEW_RESPONSE_20261104.md) for the audited
workload breakdown, corrections to the assessment and lead counts, implemented
repairs and the queued targeting follow-up. The historical checkpoint above is
retained as a record of that earlier campaign.

## High-priority issues

- The adaptive controller spent substantial effort inspecting and interpreting
  existing records before expanding the candidate set. Review whether the
  planning protocol and tool permissions match the lighter prompt guidance.
- Primary proposals require source passages, every target component's proposed
  role and an immediate or coupled reaction hypothesis. These checks establish
  traceability and declared scope; they do not automatically establish biochemical
  entailment. Unannotated proteins can qualify through specific sourced catalytic
  architecture; a family label alone is insufficient.
- Native activity annotations, substrate-scope experiments, target hypotheses and
  structural confidence must remain distinct. Related sequences do not supply
  independent mechanistic evidence.
- The Argo proxy reported implausibly small input-token counts in the recent
  continuation. Provider-reported usage alone did not establish complete physical
  token consumption; that launcher used separate conservative admission estimates.
- Context and effort settings affect cost and behavior. Their presence in the
  current source is not evidence that all combinations have been tested live.
- Persistent molecular-property cache entries are bound to the descriptor source
  and RDKit version. These target-independent descriptors are not biological
  evidence or a transfer of another campaign's discoveries.

## Reproducing the software checks

The checked results and their scope are recorded in
[review_validation.json](review_validation.json).

Install the project and test extras using the root README, then run:

```bash
python -m pytest -q
```

For focused review:

```bash
python -m pytest -q enzyme_discovery_agent/tests
python -m pytest -q campaign_catalog/tests
python -m pytest -q enzyme_design_agent/tests
python -m pytest -q structural_modeling_agent/tests
```

The global pytest configuration uses importlib collection so components may have
tests with the same basename. No model weights or prediction ensembles are
included in the package. Actual folding, design and external database services
need the separately documented environments, credentials and resource policies.

## Included and external material

Included: the four maintained source packages; their tests and small static
assets; dependency and packaging files; examples; developer documentation; and
selected repository utilities.

External: credentials and local environments; model weights and external engine
installations; large sequence and structure databases; full prediction ensembles,
MSAs and trajectories; runtime caches and claims; historical source snapshots;
and the separate Overleaf repository. Existing historical files already tracked
in Git were preserved. This handoff does not delete local research outputs or
convert them into evidence supporting the primary panel.
