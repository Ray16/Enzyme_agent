---
name: dynamics-and-design
description: Plan and register purpose-gated BioEmu2 conformational-dynamics or RFD4-Proteina de novo design follow-ups after primary cofold inspection.
domains: [followups]
---

# Dynamics & De Novo Design Follow-Ups Playbook

1. **Prerequisites**:
   - Complete and inspect the primary cofold ensemble first; check `inspect_workflow_policy`.
   - BioEmu2 and RFD4-Proteina are never automatic fallbacks when a cofold provider is unavailable.
2. **Planning (`plan_followup`)**:
   - State the scientific question, why cofolding cannot answer it, expected information, and consequences under supporting vs. refuting outcomes.
3. **Execution handoffs**:
   - **BioEmu2 (`prepare_bioemu2`, `inspect_dynamics`)**: equilibrium conformational sampling only (never interpret equilibrium frames as kinetic trajectories); retain all frames including engine-rejected ones; never overwrite primary cofold evidence.
   - **RFD4-Proteina (`register_design_run`)**: produces a reviewable design handoff with a shared-sequence multistate evaluation plan; linking a design run does not validate generated enzymes.
