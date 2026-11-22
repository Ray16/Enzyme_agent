---
name: catalytic-triage
description: Multi-axis non-native scaffold triage funnel (developability, pocket envelope, access, MD persistence, preorganization, side reactions, and barriers) separating hard rejects from evolvable defects.
domains: [core, analysis, developability]
---

# Multi-Axis Catalytic Triage Playbook

Evaluate non-native scaffolds across independent physical axes without collapsing
them into a single arbitrary score. Separate **hard catalytic/steric blocks**
from **soft, evolvable recognition defects** (fixable with 2-6 mutations).

## Tiered Escalation Funnel

1. **Tier 0 — Hard Gates (seconds, CPU)**:
   - Verify mechanism & activation compatibility (`validate_setup`).
   - Run `assess_developability`: a curated veto (membrane/insoluble/obligate hetero-complex) triggers a soluble-homolog search within the same family rather than wasting GPU compute.

2. **Tier 1 — Static Structural Screening (minutes, GPU cofold)**:
   - Cofold the requested target substrate(s) and, when informative, a native-substrate control.
   - Run `analyze_pocket_envelope` (native vs. target):
     - Both fail -> `modeling_artifact` (fix assembly/metal/protonation before judging the scaffold).
     - Target larger than native + buried clash -> `hard_reject` (`buried_steric_exclusion`).
     - Target smaller than native / loose pocket -> `soft_evolvable_defect` (cavity-shrinking design target).
     - Distal tail solvent-exposed -> permissive scaffold!
   - Run `analyze_substrate_contacts` (ensemble frequencies) and `analyze_substrate_channels(backend="fast", skip_open=True)` on the full biological assembly.

3. **Tier 2 — Dynamic Persistence, Preorganization & Side Reactions (hours, MD)**:
   - For Tier-1 survivors or evolvable drifters, evaluate MD reactive-geometry persistence alongside:
     - `analyze_side_reactions`: declare reaction-specific competing pathways (solvent attack at the appropriate geometry, intramolecular cyclization, futile cofactor hydrolysis, or off-target C-H abstraction — never assume one fixed angle across enzyme classes).
     - `analyze_preorganization`: apo-vs-holo catalytic-constellation RMSD, TS/intermediate charge-center stabilization, and MD-averaged projected electric field along the forming/breaking bond axis.

4. **Tier 3 — Chemical Barrier (days, QM/MM or ML/MM)**:
   - Escalate only MD-persistent, low-side-reaction candidates to embedded DFT/MM (`reaction-barriers` skill) and fail-closed ML/MM verdicts.

5. **Synthesis & Feedback**:
   - Call `build_catalytic_profile` to record all evaluated and missing axes, `failure_modes`, `evolvability_class`, and specific `design_targets` for discovery/design feedback.
