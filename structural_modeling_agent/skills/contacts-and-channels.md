---
name: contacts-and-channels
description: Quantify substrate-accommodation contact frequencies and descriptive full-assembly transport channels across cofold ensembles or MD trajectories.
domains: [analysis]
---

# Substrate Contacts & Transport Channels Playbook

1. **Substrate contacts (`analyze_substrate_contacts`)**:
   - Prefer ensemble (`mode="ensemble"`) or MD trajectory (`mode="traj"`) frequency aggregates over any single static pose.
   - Salt bridges require complementary charge (`ARG`/`LYS`/`HIP` <-> carboxylate; `ASP`/`GLU` <-> ammonium). Neutral/unassigned `HIS` in raw Boltz PDBs is scored as an H-bond donor, never a cationic salt bridge.
   - Distinguish residue-specific side-chain anchors (`salt_bridge`, `h_bond`) from non-specific `backbone_hbond` or `hydrophobic` packing.
   - The detector reports facts only; pair with structural visualization before making hold/drift conclusions.

2. **Transport channels (`analyze_substrate_channels`)**:
   - Default to `backend="fast"` (parallel native Voronoi/Delaunay + Dijkstra, ~10-20x faster than CAVER 3.0); use `backend="caver"` only when exact CAVER 3.0 provenance is required.
   - Use `skip_open=True` when screening panels to skip tunnel search on solvent-open sites (`burial_fraction < 0.55`).
   - **Always pass the full biological assembly**: dropping a partner subunit can fabricate wide open "tunnels" at an interfacial active site.
   - **Descriptive only**: static bottleneck radii are lower bounds (channels breathe under dynamics) and tunnel topology is not an activity ranker.
