---
name: cofold-setup
description: Resolve molecular state, assembly, cofactors, reactive atoms, and scientific hypotheses for Boltz-2 cofolding and proxy evaluation.
domains: [cofold, reasoning, proxies]
---

# Cofold Setup & Scientific Reasoning Playbook

1. **Investigate before proposing compute**:
   - Use `inspect_candidate`, `read_candidate_section`, `read_evidence`, `retrieve_reference`, and `assess_assembly`.
   - Distinguish exact-sequence primary evidence from homolog transfers; never infer cofactor counts from chain counts or assume a default metal.
   - Distinguish biological oligomer (UniProt SUBUNIT) from the minimal functional unit required to enclose the complete ligand site.
   - If shared-pool predictions already exist (`existing_cofolds`), call `review_existing_predictions` before requesting any new cofold.

2. **Propose & validate**:
   - Call `propose_model` (prefer `modeling_input_from="candidate"|"current"` + `modeling_input_changes` instead of copying large unchanged JSON).
   - Declare competing hypotheses, falsifiable predictions, and explicit `modeled_step.reaction_center` (`PA`/`O3A` for ATP alpha attack; `PG`/`O3B` for gamma transfer; carbonyl `C`/`O` for ~107° carbonyl attack).
   - Run `validate_setup` and `inspect_reasoning`; respond via `respond_to_reasoning_feedback` before `render_inputs` and `submit_cofold`.

3. **Post-cofold inspection & proxies**:
   - Call `validate_predictions` with the producer manifest to bind input/execution hashes.
   - Never equate prediction confidence with physical validity or experimental activity; retain all failed poses.
   - For structural proxy panels, call `get_structural_proxy_guidance` -> `validate_structural_proxy_plan` -> `declare_proxy_plan` -> `evaluate_proxy_plan`.
