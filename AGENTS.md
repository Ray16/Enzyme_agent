# Production discovery and structural follow-up

Read `AGENT.md` for the user's resource, modeling and synchronization rules.
For production enzyme-discovery requests, use the standard CLI's automatic
rolling structural-modeling coordinator, or call
`enzyme_discovery_agent.rolling_modeling.attach` when using the Python API.
Do not launch discovery alone unless the user requests that scope explicitly.

Each new ranked candidate enters agent-led scientific setup investigation.
Use shared family/mechanism handoffs and the structural modeling agent's
`family-cofold` execution mode. Resolve a protocol once, list its member enzymes,
and generate their inputs in code. Do not repeat a research-agent session for
each member. Use parallel workers with disjoint GPU/CPU allocations when capacity
permits; preserve resource-contention stops and partial outputs.
The modeling agent gathers independent sources and decides whether a justified,
informative target-substrate cofold merits compute. Missing native annotation
alone neither authorizes compute nor excludes a hypothesis. Preserve pending
decisions and reasons. Reuse requires the exact sequence and molecular setup;
native-only models cannot substitute for the requested non-native substrates.
Keep source gathering, provisional hypotheses, structural observations and
experimentally demonstrated activity distinct. Never seed discovery with
historical candidate answers.

## Default protein-structure plot style
Every protein/active-site structure figure uses the house scheme from
`structural_modeling_agent/analysis/render_contacts.py::render_structure` unless the
user asks otherwise: white background; gray90 cartoon at `cartoon_transparency 0.9`
(no `transparency_mode`, `backface_cull`, `ray_interior_color` or `ray_trace_mode 1`);
ligand (substrate and ATP) carbons gray40 with element-coloured heteroatoms; forest Mg
spheres (`sphere_scale 0.45`); thin dashes (radius 0.058, gap 0.35, length 0.25), blue
`[0.12,0.31,0.85]` for the attack axis and gray40 otherwise; orient/zoom on the site; no
baked-in PyMOL text. Put measurements in a bold, bottom-right white rounded box and labels
in matplotlib. References: `structural_modeling_agent/runs/glutarate_cadaverine_correct_protonation_panel/analysis/contact_frequency/acsA_Q5CD72/acsA_Q5CD72_contact_frequency.png`
and `structural_modeling_agent/runs/glutarate_cadaverine_glna2_dimer_20261117/analysis_email/render_failure_mode.py`.
Exception: anchor-grid design figures keep their orange-substrate/teal-ATP scheme.

## Git attribution and remote sync
- Always commit and push under the repository's configured user identity (never under Gemini, Claude, Antigravity, Codex, or any other AI name/email). Never pass `--author`, override `GIT_AUTHOR_*`/`GIT_COMMITTER_*`, or add `Co-Authored-By` / AI-attribution trailers to commits or PRs.
- Commit and push to `origin` (`github.com/Ray16/Enzyme_agent`) after every substantial code change (more than ~30 lines of code), staging only the relevant files and never force-pushing.
