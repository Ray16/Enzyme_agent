---
name: reaction-barriers
description: Plan, submit, reconcile, and fail-closed evaluate QM/MM (DFT/MM, AM1/D) and ML/MM reaction-barrier studies.
domains: [barrier, mlmm]
---

# Reaction Barriers & ML/MM Playbook

1. **Method policy**:
   - Main enzyme energy assessment: **embedded DFT/MM** when setup and resources support it.
   - **Cluster/UMA** is for explicitly justified exploratory mechanism/path screening only; **UMA-xTB** is an embedded reference potential, not DFT.
   - Never silently fall back from embedded QM/MM to a vacuum cluster when parameters or compute fail.

2. **Barrier tool lifecycle**:
   - `inspect_barrier_contract` -> `prepare_barrier_inputs` (`embedded` or `cluster`) -> `plan_barrier_calculation` -> `submit_barrier_calculation` -> `check_barrier_calculation` -> `inspect_barrier_result` -> `record_barrier_assessment`.
   - For queued/running jobs, finish inconclusively to checkpoint `waiting_for_backend`; never resubmit uncertain jobs or mutate an active plan.

3. **Fail-closed ML/MM & reporting gates**:
   - Use `inspect_mlmm_engine_options` and `assess_mlmm_profile` (strict fail-closed gate: single imaginary mode > 50 cm⁻¹, geometry-based reactant/product labeling, mode-count consistency).
   - A constrained scan maximum is never a certified saddle, activation free energy, or proof of experimental activity.
