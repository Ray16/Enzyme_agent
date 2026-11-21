---
name: md-preparation
description: Prepare and validate protonation, metal hydration, ligand parameters, solvent, and MD/enhanced-sampling protocols.
domains: [sampling]
---

# MD & Sampling Preparation Playbook

1. **Declare the protocol early**:
   - Inspect `inspect_sampling_setup_contract`, `inspect_preparation_contract`, and `inspect_processivity_toolchain`.
   - Call `declare_sampling_setup` and `declare_preparation_study` early with unresolved choices explicit; update after each numerical check.
   - Use `record_preparation_findings` to persist working notes with source hashes and concrete next actions.

2. **Chemical & microstate discipline**:
   - Review ligand chemical graphs/atom mapping, catalytic and nearby histidines (`HID`/`HIE`/`HIP` — never leave generic `HIS`), metal coordination/hydration, force-field parameters, and solvent together.
   - Apo pKa predictions alone do not establish bound active-site protonation.
   - Keep distinct proton inventories as separate setups/variants; never compare raw total energies across different proton counts.

3. **Operators & validation**:
   - Use `run_preparation_operator` for bounded (<=60 s, 1 CPU) diagnostic/prep scripts opening hash-verified `inputs/<name>`.
   - A zero exit code is not scientific validation; inspect numerical outputs (`inspect_preparation_structure`, `inspect_preparation_study`) and never impose artificial restraints just to preserve a favorable substrate pose.
