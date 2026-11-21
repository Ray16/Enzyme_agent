---
name: processivity-kinetics
description: Build and evaluate source-grounded transport, catalysis, and committed-escape kinetic networks for processivity assessment.
domains: [sampling]
---

# Processivity Kinetics Playbook

1. **Inspect contract & toolchain**:
   - Call `inspect_processivity_contract` and `inspect_processivity_toolchain` before declaring any kinetic network.
2. **State-graph rules (`assess_processivity_network`)**:
   - Keep chain-extension cycles distinct from auxiliary bond changes and keep committed escape distinct from transient active-pocket loss.
   - First committed release is absorbing for same-association processivity; release followed by rebinding starts a new association and cannot inflate continuation probability.
   - Preserve unknown physical rates and mismatched conditions as explicit gaps; synthetic solver examples, retention alone, or biased trajectory times do not establish processivity.
