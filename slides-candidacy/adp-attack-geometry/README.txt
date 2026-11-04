ADP-FORMING ACTIVATION GEOMETRY — PuuA (P78061)

Main figure: PuuA_ADP_attack_geometry.png
Editable annotations: PuuA_ADP_attack_geometry.svg
Vector-document export: PuuA_ADP_attack_geometry.pdf
PyMOL scene: PuuA_ADP_attack_geometry.pse (local generated file)

This uses saved model 0 from the existing discovery cofold campaign. It has
one PuuA chain, ATP, adipate, and two Mg ions. The reactive coordinate is the
closest saved adipate oxygen to ATP PG, with O3B defining the leaving-group
axis. Element colors and protein transparency reuse the AS9 style.

The molecular coordinates are unchanged. A face-on camera preserves the 3D
attack angle in the 2D annotation plane. Independently rendered fiducials
calibrate pixel coordinates. validation.json records the source hash,
atom assignments, distance, visible angle and projection error.

The pictured pose is a prediction, not a transition state or validated active
assembly. It is model 0 rather than a pose selected for optimized geometry.
Native ADP-forming chemistry supports the PG/O3B coordinate but does not prove
activity on the non-native adipate substrate or establish metal stoichiometry.

Rebuild from this directory using the existing environments:
  /homes/rzhu/miniforge3/envs/pymol/bin/python render_site.py
  /homes/rzhu/miniforge3/envs/palm/bin/python compose_figure.py

The source structure is resolved by render_site.py. No inference or new fleet
computation is required. Rendering is CPU-only and capped at two threads.
