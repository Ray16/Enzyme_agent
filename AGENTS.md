# Production discovery and structural follow-up

Read `AGENT.md` for the user's resource, modeling and synchronization rules.
For production enzyme-discovery requests, use the standard CLI's automatic
rolling structural-modeling coordinator, or call
`enzyme_discovery_agent.rolling_modeling.attach` when using the Python API.
Do not launch discovery alone unless the user requests that scope explicitly.

Each new ranked candidate enters agent-led scientific setup investigation.
The modeling agent gathers independent sources and decides whether a justified,
informative target-substrate cofold merits compute. Missing native annotation
alone neither authorizes compute nor excludes a hypothesis. Preserve pending
decisions and reasons. Reuse requires the exact sequence and molecular setup;
native-only models cannot substitute for the requested non-native substrates.
Keep source gathering, provisional hypotheses, structural observations and
experimentally demonstrated activity distinct. Never seed discovery with
historical candidate answers.
