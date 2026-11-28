---
name: dynamics-and-design
description: Plan and register purpose-gated BioEmu2 conformational-dynamics or strategy-open enzyme design follow-ups, promote family-cofold winners, and build design validation tasks after primary cofold inspection.
domains: [followups]
---

# Dynamics & Enzyme Design Follow-Ups Playbook

1. **Prerequisites**:
   - Complete and inspect the primary cofold ensemble first (`plan_followup` enforces this baseline gate).
   - BioEmu2 and enzyme design are never automatic fallbacks when a cofold provider is unavailable.
2. **Planning (`plan_followup`)**:
   - Use `purpose="enzyme_design"` for strategy-open handoffs (letting `enzyme_design_agent` choose between saturation redesign, RFD4-Proteina, AlphaProtein Novo, DISCO, or GENzyme) or `purpose="de_novo_design"` when specifically requesting de novo generation.
   - State the scientific question, why cofolding cannot answer it, expected information, and consequences under supporting vs. refuting outcomes.
3. **Execution handoffs & promotion bridges**:
   - **BioEmu2 (`prepare_bioemu2`, `inspect_dynamics`)**: equilibrium conformational sampling only (never interpret equilibrium frames as kinetic trajectories); retain all frames including engine-rejected ones; never overwrite primary cofold evidence.
   - **Enzyme Design (`register_design_run`, `build_design_validation_task`)**: produces a reviewable `--handoff` request; linking a design run does not validate generated enzymes. Use `build_design_validation_task` to route designed sequences back into the frozen Boltz-2 / geometry contract.
   - **Family-Cofold Promotion (`promote_family_cofold`)**: promotes top-K completed family-cofold candidates into verified MD preparation manifests (with cofactor assets, GAFF2 polarizability overrides, prep/MD env split, and PROPKA audit policy) and strategy-open design handoffs.
