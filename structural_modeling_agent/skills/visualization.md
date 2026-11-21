---
name: visualization
description: Render house-style PyMOL active-site contact figures, channel figures, and fast interactive 3Dmol.js channel viewers.
domains: [visualization]
---

# Structural Visualization Playbook

1. **Choose the right medium**:
   - For fast interactive inspection of channels/pockets (no ray tracing, ~1-2 s), call `render_channel_visualization` with `format="html"` (self-contained 3Dmol.js viewer).
   - For publication 2D figures, call `render_channel_visualization` with `format="png"` (`publication=False` draft ~12 s, `publication=True` ~30 s) or `render_contact_figure` for paired active-site + contact-frequency panels.
2. **House plot style**:
   - White background, light translucent protein cartoon, gray40 carbons for substrate and ATP with element-colored heteroatoms, forest-green Mg spheres, no baked-in PyMOL text, and bold bottom-right white measurement boxes in matplotlib.
3. **Concurrency safety**:
   - Avoid launching many concurrent headless PyMOL `ray` processes on shared nodes; keep PyMOL renders bounded (batch runners cap `render_workers=2`).
