# Enzyme discovery agent

The active agent starts from a target reaction, searches literature and biological
databases, evaluates candidate evidence, and produces a shortlist with a
structural-modeling handoff. Runtime code and packaged resources live in this
directory. Previous-version code is not imported from the archive.

The downstream consumer is the independent structural-modeling agent:
`python -m structural_modeling_agent --task RUN/structural_modeling_handoff.json --output-dir MODELING_RUN`.
It owns modeling decisions, evidence-backed setup validation, cofold input
generation and post-cofold pose checks. Discovery's modeling fields are proposals
for that agent to inspect. See `structural_modeling_agent/README.md` for its standalone
task format, tools, resume behavior and provider-adapter interface.

## Running and dependencies

New CLI campaigns default to 200,000 accounted tokens and one hour; both limits
can be overridden explicitly. Screened first-pass work is evaluated in batches
of 20 before additional automatic breadth work. Unresolved identities and retry
cooldowns do not block that queue indefinitely. Evaluation batches prioritize
under-assessed resolved families, then annotation groups within a family;
pending and rejected assessments count towards effort already spent. Equal
coverage preserves the evidence-priority order. Unresolved annotation groups
remain scheduling hints, not evidence of independent function or target activity.
Survey promotion and screening commit registry updates in batches under the same
process-safe lock and lifecycle rules, avoiding one full-file rewrite per member.
Failed commits propagate so unpersisted promotions remain retryable.
Screening selects a rolling tranche of roughly 100 representatives, with at most
20 per resolved fold before protected leads and coverage-floor exceptions. This
is a work-allocation limit: other accessions remain recorded and eligible when
new evidence or changed priorities trigger the next screening pass. Within an
evidence tier, saved member fit precedes sequence length. Allocation reuses
recorded member facts or recent same-run surveys instead of repeating them.
Initial fold surveys interleave all resolved mechanistic hypotheses before
deepening any one hypothesis; shared folds retain all hypothesis links.

## Adaptive research decisions

Literature and native-reaction databases provide complementary discovery routes.
`search_modelseed_reactions` accepts chemical terms, participant IDs, reaction IDs
or evidence-derived EC prefixes. It returns complete participant lists, EC and
reaction cross-references, source URLs and pagination. Chemical text matching
handles capitalization differences in the database. Complete EC numbers match
exactly; partial numbers search descendants.

After assessing reacting roles, direction and substrate context, the agent can
map a small relevant set of complete EC annotations to UniProt sequences and
screen them through the existing rolling workflow. This does not require that a
paper first name the enzyme. Raw reaction and compound results are saved under
`artifacts/database/`; a native reaction match does not establish target activity.

The agent records testable hypotheses and competing explanations, predicts what
would distinguish them, and registers those predictions before executing the
chosen tool. A completed test supplies a new, separately recorded observation.
The next decision must cite that outcome and explain whether the hypothesis is
retained, revised or suspended. Brief scientific justifications are recorded;
provider-internal thinking is not required by this protocol.
Priority updates are tracked separately from evidence actions. Changing a work
allocation does not itself test a biochemical hypothesis. Reinterpretation of
existing observations is allowed and labelled separately from new evidence.
Read-only campaign inspection remains available during deliberation. Recorded
action-change counts exclude rewritten rationales.
After a completed inspection round, a strategic checkpoint presents the planning
interface with every permitted action contract. Further information requests are
still possible, with an explicit question and expected information gain, rather
than restarting the same unstructured inspection cycle.
Each review allows two read-only inspections and four research actions by
default. These counters survive resume. At the action limit only
`conclude_research_review` remains available; it records a provisional conclusion,
unresolved alternatives and a reopening condition, then releases pending batch
evaluation. It never marks discovery complete or requires a positive activity
claim. Qualitatively changed observations reopen the review with a new allowance.
Identical candidate assessments and deduplicated requests are disclosed as
noninformative actions, not new evidence.
Rejected proposals return the current citation and revision IDs for repair.
If repeated proposals cannot be grounded while screened candidates are ready,
the controller defers the unresolved review and resumes their ordinary evaluation.
This changes work allocation only; it does not change scientific claims or family
priorities. Without independent ready work, the campaign saves an incomplete checkpoint.

New substrate-role contradictions or imported modeling observations can interrupt
automatic evaluation for a model decision. The model chooses the next test; the
controller checks provenance, tool permissions and execution. It does not decide
which scientific interpretation is correct. A successful lookup does not imply
biological support, and an unavailable service cannot establish inactivity.

`plan_research_step` executes one permitted evidence or priority action after
registering its prediction. `get_decision_context` retrieves current observations
and earlier tests. `assess_next_batch` collects a bounded native-evidence batch.
`interpret_research_outcome` records an interpretation of the current saved
outcome without executing another test or resetting the review allowance.
It verifies the result fingerprint and outcome citation; the interpretation
remains the agent's scientific judgment. Adaptive tool menus offer direct
controls and permitted inspections, with consequential actions exposed through
the planner. A bounded protocol-repair opportunity precedes an incomplete
checkpoint when repeated proposals cannot be grounded and no independent
evaluation work is ready.
`set_family_focus(defer=[...])` reversibly pauses pending work and automatic
expansion. Reported `annotation:` scopes allow finer scheduling within a broad
fold; these labels do not establish biological independence or equivalence.
Batch assessments require explicit accessions/scopes or a deliberate
`scope_mode="ready_queue"`. Selected new accessions are screened before assessment;
the result reports the actual sample, so an untested mechanism stays unresolved.
`read_primary_paper` exposes the existing open-access retrieval path as a bounded,
paginated reader with a source hash. Paper identifiers and relevant passages
remain visible after later actions. Protein identifiers found in the text are
leads requiring paper-to-sequence verification; they are not activity claims.

Pending paper-scoped identities stay in the actionable frontier. Retrieved
primary-title matches are interleaved with leads from other source papers;
these groups diversify work allocation and do not establish mechanistic
independence. `resolve_actionable_literature_leads` performs a bounded batch of
explicit name/taxon or deposited-ID resolutions, retaining ambiguity and service
failures. It does not require every paper's full text to finish first.

`research_portfolio.csv` exposes source-cited investigation hypotheses,
including proteins without native-function admission. `unresolved_literature_leads.csv`
keeps named leads visible with their identity status. Both are distinct from the
native-evidence-ranked table and from completed structural results; none is a
table of experimentally demonstrated target activities.

Predictions, raw results, interpretations and content-addressed evidence are kept
under `agent_state/research_decisions/`. Historical records remain citable during
revision when their current version is also cited; superseded evidence cannot
silently stand in for current data. The monitor displays this decision trail.

The append-only journal `reasoning/trace.jsonl` records each prediction before
its action, then its observation and eventual review conclusion. Events have
sequential IDs and a chained checksum. Referenced source files and evidence
snapshots carry checksums; scientific interpretations remain explicitly unverified.
The journal is written during execution, independently of model context trimming.
It contains the explicit scientific decision fields, not provider-private thinking.
The first retrieved sources receive citable evidence IDs and trigger a bounded
strategy review before broad automation. Participant-role and mapped-reaction
interpretations are journaled with their immutable source artifacts as well.

Checkpoints and finalization also produce `reasoning/trace.json` for programmatic
analysis and `reasoning/trace.md` for human review. Each example links a hypothesis,
alternative, prediction, actual action/result, evidence, revision condition and
follow-up decisions. The export never invents a retrospective explanation. Existing
campaign artifacts can be recovered, labelled `reconstructed_from_saved_artifact`;
they are not claimed as prospective captures. Export or refresh any saved run with:

```bash
python -m enzyme_discovery_agent.export_reasoning_trace runs/your_run
```

The offline viewer is `reasoning/trace.html`; it shows execution and revision
links, searchable records, and the underlying sources. All visualization and
diagnostic files stay in `reasoning/`, leaving the run's main folder focused on
ranking and handoff outputs. During active execution, export with `--live` to
read only already-journaled records without reconstructing in-flight artifacts.

`get_reaction_step_context` exposes source-supported molecular occurrences and
atom maps. `record_reaction_step` checks atom conservation, structures and local
bond changes, including whether reacting ends belong to the same molecule or
different molecules. Whole-molecule group counts need not match. Activation,
enzyme acceptance and further extension remain separate hypotheses.
`get_research_memory(topic="reasoning")` retrieves saved decisions and current or
superseded interpretations. When context is tight, a compact working set retains
source IDs, the active hypothesis/outcome and recoverable inspection snapshots;
complete records remain on disk. Title-only citations remain eligible for
metadata and abstract resolution rather than being treated as irrelevant.

Repeated exports preserve journal events. A damaged journal raises an error;
changed source files are reported by the exporter. Checksums establish record
integrity, not biological correctness or an independent timestamp guarantee.

Survey selection and final ranking share required-substrate accommodation logic.
An unmatched input cannot disappear from a survey average. Descriptive family
profile means remain available, with their separate scope. Full survey results
are retained under `artifacts/family_surveys/` for inspection and replay.

The suggested list is refreshed atomically after evaluation batches, individual
evaluations, role/step assessments and `set_candidate_focus` decisions, without
extra model or literature calls. Per-candidate assessments separate strengths,
observed weaknesses and unknowns. Missing evidence alone cannot justify exclusion
or deprioritization. Serious known conflicts take precedence over favorable
unrelated metrics; other measured tradeoffs are retained rather than averaged
into a single winner. Highest-priority labels require a recorded hypothesis,
not a fixed fraction of the pool. Candidates removed from the suggested list
remain in the registry and can be reopened with a new source-cited decision.
Bounded review packets include leading and held candidates, so changes in
criteria or native evidence can prompt reconsideration of overlooked records.
These assessments guide the next investigation and expansion seed, not only a
display label. The leading expansion window counts distinct families so several
members of one family cannot hide an eligible unresolved alternative. The displayed
order also interleaves families among equally prioritized, incomparable candidates;
known conflicts and observed dominance retain precedence. Candidate
inspection preserves the native reaction, reacting occurrences and per-target
substrate comparisons ahead of administrative history during context compaction.

For the local project-home status page, run
`python -m enzyme_discovery_agent.output.run_index --project /path/to/project --watch`.
It updates one `index.html` at the project home and keeps monitor state in
`analysis/status_monitor/`. It reuses saved-state snapshots and cannot invoke or
steer discovery; individual run folders retain their ranking and reasoning artifacts.

For an ablation, set `ENZYME_AGENT_ADAPTIVE_REASONING=0` before starting a fresh
campaign with the same model, target and allowance. Compare cold campaigns or
matched retrieval replays; resuming prior adaptive decisions is not an independent
baseline. The opt-in live development check consumes subscription tokens:

```bash
python -m enzyme_discovery_agent.eval.adaptive_evals --output analysis/my_new_check
```

This controlled check tests evidence-sensitive priority changes and subsequent
revision. Demonstrating better enzyme discovery requires separate budget-matched
baselines, independent targets and prospective activity evidence.

The controller uses a compact system prompt and compact tool descriptions.
Ordinary turns retain the last two assistant transactions. Adaptive decisions
instead prioritize a mandatory evidence packet, the previous prediction/outcome,
and the latest informative read-only inspection. Context pressure cannot silently
discard the citations needed to choose a test. Saved previews are bounded;
complete transcripts and tool artifacts remain on disk. Mandatory
instructions that exceed the context allowance cause an explicit error rather
than silent truncation. Reflection and supplemental research respond to stalled
evidence progress instead of imposing an unconditional three-turn cadence.

A stopped campaign writes `report.md` and `agent_state/report_delivery.json` as
well as its resumable checkpoint. The report explicitly identifies incomplete
coverage; its existence does not mean discovery finished successfully.

Lookup results distinguish confirmed results, confirmed empty responses, and
unavailable services. InterPro queries may broaden after a confirmed empty
response; the output records the requested and used queries. An unavailable
service does not authorize broadening. ChEBI transport failures are retryable,
and incomplete expansion attempts remain retryable after bounded backoff.

Mechanism extraction preserves its own source-specific evidence. It does not
fill missing reaction steps from a family-indexed cache. The historical cache
utility requires an explicit store path and is retained for offline inspection.

Install the project from the repository root. For structural scoring and plots,
also install the `structural-modeling` optional dependency extra. Configure the
model and service credentials through the existing project configuration.

```bash
python -m enzyme_discovery_agent --help
python -m pytest enzyme_discovery_agent/tests
```

Discovery artifacts are written to `runs/`. The imported discovery snapshots in
`runs/imported_amide_discovery` and `runs/imported_hydrolase_discovery` are historical
campaign data made available to the active downstream workflow, not examples
injected into discovery prompts. Their original locations and fingerprints are
recorded in `structural_modeling_agent/campaigns/active_artifact_provenance.json`.

## Claude Code subscription backend

The default for new runs is Claude Sonnet 5.5 through the signed-in Claude Code
account. You can also select the backend explicitly:

```bash
python -m enzyme_discovery_agent --model claude-code:claude-sonnet-5-5 \
  --token-budget 100000 --runtime-budget-seconds 7200 "your target reaction"
```

The backend launches the official `claude` executable with the personal settings
file at `~/.claude/personal-settings.json`. Set `ENZYME_CLAUDE_SETTINGS` or
`ENZYME_CLAUDE_EXECUTABLE` if your installation uses different paths. It does
not require a subscription token in the project configuration. The selected
model is pinned, and no other provider is used as fallback. The backend verifies
the reported model and first-party provider, and stops with a checkpoint if a
request cannot be verified or its usage is unavailable.

Claude chooses from controller-provided tools using native MCP calls. Those MCP
calls only record choices; the discovery controller applies its usual gates and
executes the tools. Each request starts a fresh Claude Code session using the
controller's saved conversation; it does not load project hooks, built-in tools,
or other MCP servers. Text and base64 PNG, JPEG, GIF and WebP image inputs are
supported; remote image URLs are rejected. Cached input tokens count towards the
agent's token allowance. Claude Code requests high effort and uses the provider's
temperature behavior. Budget checks occur between calls, so a request already in
flight can exceed the remaining allowance. Provider failures with unknown usage
are marked as unmetered and stop the run instead of silently retrying. A resume
restores the backend from the run's usage record.

The live integration test consumes subscription usage and is disabled by default:

```bash
RUN_CLAUDE_SUBSCRIPTION_LIVE=1 python -m pytest -q enzyme_discovery_agent/tests/test_claude_live.py
```

## Google AI Studio configuration

Create a key at <https://aistudio.google.com/apikey> and save it in the repository's
gitignored `.env` or export it in the shell. Existing shell variables take
precedence over `.env`; `GEMINI_API_KEY` takes precedence over the optional
`GOOGLE_API_KEY` alias in this agent.

```dotenv
GEMINI_API_KEY=your_google_ai_studio_key
ENZYME_AGENT_FALLBACK_PROVIDER=gemini
GEMINI_CASCADE_MODELS=gemini-3.8-flash,gemini-3.6-flash,gemini-3-flash-preview,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash,gemini-2.5-flash-lite
```

The optional Argo ladder starts with Argo Sonnet, followed by the configured Google
models in capability priority: 3.8 Flash, 3.6 Flash, 3 Flash, 3.5 Flash-Lite,
3.1 Flash-Lite, 2.5 Flash, optional Groq GPT-OSS 120B, then 2.5 Flash-Lite. This policy follows published
general capability comparisons; it is not a measured discovery-task ranking.
Models unavailable to the project are skipped. A Google daily
quota failure advances to the next model and retries the same request with its
saved evidence. Only exhaustion of the full ladder produces a resource checkpoint.
OpenRouter remains opt-in through `ENZYME_AGENT_FALLBACK_PROVIDER=openrouter`;
the default ladder does not enable billing or add a paid provider.

```bash
python -m enzyme_discovery_agent --model argo:claude-5-sonnet "your target reaction"
python -m enzyme_discovery_agent --model gemini:auto "your target reaction"
python -m enzyme_discovery_agent --model gemini:gemini-3.5-flash-lite "your target reaction"
python -m enzyme_discovery_agent --resume-run runs/your_run
```

`gemini:auto` skips Argo and uses the free-model ladder, including Groq when
configured. Naming a particular Google model pins
it, useful for controlled model comparisons. Resume restores the requested ladder
rather than accidentally pinning the last serving model. `ENZYME_AGENT_MODEL`
can override the default. `GEMINI_CASCADE_MODELS` selects a subset of the seven
free-tier models above; duplicates are removed and capability order is retained,
even if the setting lists the models in a different order. Models outside this
list are rejected for automatic routing. A pinned model remains an explicit
single-model selection. Free-tier model selection does not override a project's
billing tier; use an API project with billing disabled for free-only operation.
The older `GEMINI_MODEL` setting does not set the cascade order.

Each Google model has a separate RPM/TPM reservation window. Unknown project
quotas start at 2 RPM, configurable through `GEMINI_CASCADE_REQUESTS_PER_MINUTE`.
Numeric Google QuotaFailure responses teach the appropriate model's RPM, input
TPM, and daily request limit. A zero quota skips the unavailable model. Short
minute limits wait and retry; daily limits advance the ladder. Learned daily
exhaustion lasts until midnight Pacific time and survives resume.

At startup, the Google models API supplies input-context and maximum-output
limits. Contexts that exceed a model's input limit skip that model; requested
outputs respect its output limit. All seven supported Gemini text models request
`reasoning_effort="high"` on every call, including pinned runs, internal extraction,
candidate evaluation, reports and resumed campaigns. Provider defaults are not
used for these models. Gemini 2.5 maps high effort to a 24,576-token thinking
budget; the Gemini 3 models map it to a high thinking level.

To leave room for an answer, the wire output allowance is the caller's allowance
(8,192 by default) plus 24,576 tokens of thinking headroom, within the model's
output limit. This headroom is not a fixed thinking-budget request for Gemini 3.
An insufficient output limit skips the model instead of reducing reasoning.
HTTP timeouts are at least 300 seconds for these requests. Truncated or empty
responses are counted but discarded before falling through at the same reasoning
effort. The seven known model limits are 1,048,576 input and 65,536 output tokens;
retrieved metadata and explicit limit overrides remain authoritative. Unknown
models retain conservative 32,768 input and 8,192 output fallbacks and do not
receive unverified reasoning parameters. The bounded agent context remains in
effect independently of the provider's larger context window.

Argo Sonnet 5 also requests `reasoning_effort="high"`, adds 24,576 tokens of
output headroom and uses an HTTP timeout of at least 300 seconds. The installed
Argo proxy's conversion to native Anthropic was verified to emit
`thinking: {type: "adaptive"}` and `output_config: {effort: "high"}` for this
request. Unsupported settings are not retried at a lower effort. Other Argo
models retain their existing settings until their controls are verified.

Model-specific overrides can be supplied as a JSON object in `GEMINI_MODEL_LIMITS`,
with model IDs as keys and `requests_per_minute`, `input_tokens_per_minute`,
`requests_per_day`, `input_token_limit` and `output_token_limit` as fields. They
apply before the first request, including after a restart. Provider-observed
quotas take precedence over configured quotas on resume. Requests exceeding a
model's per-minute input allowance are routed to the next model before sending.
Daily allowance is learned/enforced by the provider rather than inferred from
cumulative campaign counts or historical usage peaks. Use the limits shown for
that project in AI Studio, not another model's or another account's limits.
Explicit pinned runs retain `GEMINI_REQUESTS_PER_MINUTE` and
`GEMINI_INPUT_TOKENS_PER_MINUTE` settings. Other processes using the project can
still consume quota; provider responses remain authoritative.

Profiles, exhausted models, reset times, and the ladder are saved in
`agent_state/llm_cascade.json`. Every model switch is logged, and token/request
accounting identifies the actual serving model. Tool-call history retains its
source model and original signatures; cross-model replay uses Google's documented
migration sentinel without altering the stored audit history.

## Optional Groq fallback

Set `GROQ_API_KEY` in the process environment or the repository's gitignored `.env`,
outside this package. The agent never writes it into run manifests, quota records,
or source snapshots. With the key present, `openai/gpt-oss-120b` is inserted after
Gemini 2.5 Flash and before Gemini 2.5 Flash-Lite. Without it, the Google ladder
continues directly to the final tier. `ENZYME_AGENT_GROQ_FALLBACK=0` disables
automatic Groq routing. A direct `--model groq:openai/gpt-oss-120b` pins Groq for
controlled checks. Other Groq models are not selected automatically.

The default free-plan limits are 30 RPM, 8,000 total TPM, 1,000 requests/day and
200,000 tokens/day. They can be lowered with `GROQ_REQUESTS_PER_MINUTE`,
`GROQ_TOKENS_PER_MINUTE`, `GROQ_REQUESTS_PER_DAY`, and `GROQ_TOKENS_PER_DAY`.
These are conservative local allowances; account-level usage by other processes
can exhaust the provider's quota sooner. Use a Groq Free plan; selecting this
model does not disable billing on an account upgraded to a paid plan.

Before sending, the client estimates the complete input, tool schemas, and the
requested output allowance. If these cannot fit the TPM allowance, the request
passes intact to the next model; evidence is never truncated to fit Groq. The
default output allowance is 4,096 tokens (`GROQ_MAX_COMPLETION_TOKENS`), including
reasoning tokens; an explicit caller allowance is preserved. High reasoning is
enabled, reasoning is kept out of the answer text, and parallel tool calls are
disabled for this model. Text-only input and complete output are required.

Concurrent requests reserve quota before sending. The minute limiter includes
input and output; daily reservations use a conservative rolling 24-hour window
and survive resume in `agent_state/llm_cascade.json`. Provider usage settles token
reservations when available. Failed requests or missing usage retain their
reservation. Short 429 minute limits retry with pacing; daily exhaustion records
the provider's reset delay and falls through to Gemini 2.5 Flash-Lite. Truncated
Groq output is counted but discarded and the original request is retried on the
next model. Original tool-call signatures remain in the stored audit history;
provider-specific envelopes are translated only when sending requests.

References: [Groq rate limits](https://console.groq.com/docs/rate-limits),
[OpenAI compatibility](https://console.groq.com/docs/openai),
[GPT-OSS reasoning options](https://console.groq.com/docs/reasoning).

## Evidence and target scope

A background run accepts user steering without restarting. Send instructions or
controls to its run directory:

```bash
python -m enzyme_discovery_agent.steering runs/your_run instruction --message "Prioritize resolving missing evidence before further expansion."
python -m enzyme_discovery_agent.steering runs/your_run pause
python -m enzyme_discovery_agent.steering runs/your_run status
python -m enzyme_discovery_agent.steering runs/your_run resume
python -m enzyme_discovery_agent.steering runs/your_run checkpoint
python -m enzyme_discovery_agent.steering runs/your_run clear_instructions
```

Instructions persist until cleared, survive context compaction and explicit
resume, and force a model decision before automatic discovery gates run again.
Supplied identities remain user leads and require normal evidence validation.
Change the target itself through a new campaign rather than silently reusing
evaluations under a different chemistry contract.

Commands take effect between work batches. In-flight provider calls or searches
finish first; pending controls defer further tool dispatch. Pause saves the
transcript and interim report, then waits without discovery or provider calls.
Resume releases that live wait; a process that has exited still needs
`--resume-run`. Checkpoint saves an interim report while discovery continues.

Commands and receipts live in `agent_state/steering/`; controller acceptance,
control boundaries and affected model decisions are recorded in the event log.
Status distinguishes queued commands, accepted instructions, instructions
delivered to the model, and applied deterministic controls. Delivery alone does
not establish that a requested scientific action succeeded; inspect subsequent
decisions and evidence. This CLI is the control interface; a background worker
does not automatically read messages from a separate chat application.

Family derivation and focused ranking use the declared reaction direction and
input topology. Carrier-dependent families are valid leads for carrier-bound
targets; formation, cleavage, transfer and redox queries do not share a fixed
formation-only rubric. Native-substrate comparisons remain evidence about scope,
not proof of activity or inactivity on a new substrate.

An absent UniProt reaction leaves a candidate in `evidence_pending` rather than
making absence a contradiction. A characterized enzyme resolved from a paper can
supply the missing reaction evidence when the extraction retains a candidate-linked
verbatim passage, paper identifier and fingerprinted local source text. Cited
analogues alone cannot supply that support. Literature evidence does not override
a conflicting recorded native reaction. Missing or changed sources remain
unresolved, and the handoff retains the literature provenance for review.

Reports, CSVs and modeling handoffs reconcile historical evaluations with current
registry decisions. Rejected or functionally unsupported candidates are excluded.
Regenerating a handoff after all candidates become ineligible writes an explicit
empty selection, preventing a previous shortlist from remaining actionable.

## Recovery and shared state

Use `--resume-run RUN_DIR` for a run whose manifest is `running`,
`budget_exhausted`, or `failed` with a valid transcript and target-context checkpoint.
New transcript checkpoints preserve gate state and family-survey
progress; literature inputs are restored from saved artifacts. Tool artifacts
receive process-safe, increasing identifiers shared by direct and automatic
evaluation, so resumed work cannot replace earlier evidence. Small iteration
budgets retain at least one discovery turn before completion.

Registry and notebook updates use advisory process locks and atomic JSON
replacement. Structural evidence import uses the same registry transaction, so
separate discovery and modeling processes can attach metadata without losing one
another's updates. An exclusive `controller.lock` also rejects a second discovery
orchestrator before it reads a resume checkpoint or dispatches work. Separate
modeling processes may still attach evidence through the registry transactions.

Source snapshots include all sibling runtime packages, packaged JSON resources
and dependency declarations. Changed code on resume creates a separately hashed
snapshot; earlier snapshots remain intact. Mechanism-family cache schema 5 rejects
older entries and does not cache a derivation under a single-model identity when
the serving provider changes during its calls.

The structural importer accepts the independent modeling agent's `report.json`
or its schema-3.0 `structural_modeling_evidence.json`. Import rechecks source and
artifact fingerprints, preserves raw failed/incomplete poses and geometry units,
and does not change discovery ranks or reopen rejected candidates.

The shared LLM client records provider-reported usage for internal extraction,
discovery and report requests, checkpoints it after each response and prevents
legacy callbacks from counting the same response twice. `unmetered_calls` records
responses without provider usage; `accounting_complete` remains false for those
responses or a resumed legacy ledger whose earlier accounting was incomplete.
Older campaign token totals cannot be reconstructed from missing telemetry.

## Rolling discovery and resource checkpoints

The default research state now includes saved modeling/proxy observations and
read-only exploration diagnostics. New structural facts affect research progress;
re-exported wrapper hashes, timestamps and rewritten interpretation prose do not.
Scores retain their original units and transfer scope. Saved physical checks are
reported as saved observations, not rerun by the context builder.

Use `get_exploration_state` to inspect `overview`, `families`, `hypotheses`,
`methods`, `modeling`, `questions` or `searches`, with `offset`, `limit` and
optional `query`. This reads only the explicit run and makes no network calls.
Bounded context examples are not a complete frontier; pagination exposes omitted
records. Annotation groups are heuristic, sequence clustering remains unassessed,
and failed/partial requests do not establish biological absence or exhaustion.
These diagnostics do not change ranking, family decisions, provider routing,
scientific score definitions or compute permissions.

The staged controller program and its compatibility gates are documented in
[`docs/discovery_loop_design.md`](../docs/discovery_loop_design.md). The offline
`python -m enzyme_discovery_agent.eval.decision_evals` command checks synthetic
decision rubrics; it makes no model calls and does not measure model capability.

Discovery, evaluation and expansion revisit new work throughout a campaign.
Automatic Foldseek, ESM and literature expansion select seeds from evaluated
candidate priorities rather than registry order. Unevaluated seeds are deferred,
including explicit structural and embedding tool requests. Completed automatic
searches are recorded under `agent_state/expansion_searches.json`; later top
candidates can trigger searches without repeating earlier ones.

Complementary research also runs after every three model decisions while queued
routes remain, even when the model keeps producing changing extraction output.
The queue covers additional sources, broader target-derived searches and citation
follow-up. Its scheduling counter survives resume, including migration from older
event logs. Pending user steering and required model reflection take precedence.
Weak or failed mechanism extractions remain available in the evidence audit, but
rewriting their facts does not reset stall detection.

Every model turn rebuilds working memory from this campaign's saved evidence.
The context includes recent search outcomes, unresolved literature identities and
the latest notebook passages. `get_research_memory` retrieves older searches,
unresolved leads, family decisions and notebook passages using `topic`, optional
`query`, `offset` and `limit`. Results retain source locators and distinguish
unverified leads and agent notes from primary evidence. Full records remain on
disk; the context and recall pages contain bounded previews. No provider or
model-specific summary is required to restore this memory after switching or
restarting, and recall never reads another campaign's files.

For a live ranking and candidate lifecycle plot, run the optional CPU-only reader:

```bash
python -m enzyme_discovery_agent.output.live_progress RUN_DIR --watch
```

The plotting environment needs Plotly. Open `RUN_DIR/candidate_progress.html` for
the current ranking, pending and rejected candidates, and recorded lifecycle
transitions. `candidate_ranking.csv` provides the same current candidate table;
the existing `ranked_candidates.csv` remains the checkpoint/final-report export.
The reader checks for changes every 15 seconds and the page reloads every 30
seconds. It uses saved records without provider, database, embedding or modeling
requests, and does not update the candidate registry. Historical ranks are not
reconstructed from lifecycle transitions.

The private hosted Discovery Observatory can display the same read-only summary
remotely. `python -m enzyme_discovery_agent.output.site_publisher` reads a single
JSON configuration line from standard input with `run_dir`, `site_url`, and the
Site service-access `token`; the credential stays in process memory and must not
be saved in source, logs, shell arguments, or run artifacts. It publishes bounded
summaries while the worker is alive and stops after the final checkpoint. The Site
holds one current run snapshot at a time; start a publisher for a later run when
you deliberately switch the view. The local `live_progress` watcher is optional.

Mechanism-family derivation can start from the declared target chemistry;
literature harvesting uses the available paper pool rather than requiring three
search rounds. Progress tracking distinguishes new evidence and novel searches
from notebook activity or repeated completion notices. After repeated idle
turns, notebook tools are withheld; a continuing stall writes a resumable
`failed` checkpoint and an interim report instead of consuming the remaining quota.

The legacy `--max-iterations` value does not terminate discovery. Google AI
Studio daily quota exhaustion ends the invocation with a resumable resource
checkpoint. Daily quota failures are distinguished from short-term RPM/TPM
limits, which retain pacing and retries. A daily quota failure does not trigger
another provider or further model requests for that invocation. Resume after
quota becomes available again.

Optional additional limits are available: configure
`--token-budget` and/or `--runtime-budget-seconds` for each invocation. Token
allowances include provider-reported usage from internal extraction calls. Limits
are checked before requests and between batches; already running requests or
remote searches can finish beyond a limit. Without either allowance, discovery
continues until its evidence-led search is exhausted.

At a resource limit, the agent writes `report_interim.md` from saved evaluations,
updates candidate outputs, and records `budget_exhausted` in the manifest.
`--resume-run RUN_DIR` accepts this status and restores the transcript, pending
evaluation state and expansion history. A resumed invocation receives the newly
specified allowance; cumulative token accounting is retained. When `--model` is
omitted, resumption restores the saved Google model if it served the run. The checkpoint
report does not make additional LLM, literature, structure or GPU requests.

## Structural-proxy feedback

After candidates have been selected for modeling, four registered tools support
the choice and review of descriptors:

- `get_structural_proxy_guidance`: inspect the reaction-independent catalog,
  input requirements, hypothesis questions and method limitations.
- `validate_structural_proxy_plan`: check declared hypotheses, expected
  directions, evidence identifiers and explicit atom/chain assignments.
- `inspect_experimental_proxy_bundle`: review a supplied frozen definition,
  assay endpoint, calibration scope and provenance before applying it.
- `get_structural_evaluation`: retrieve either legacy analysis tables or a
  current `candidate_proxy_evidence.json` containing frozen experimental-model
  applications and their calibration/transfer scope.

The packaged catalog covers confidence, geometry, accommodation, catalytic
environment and consistency. It contains no benchmark answer identities and
does not select descriptors using experimental labels. Mechanistic relevance
must be established for the particular modeled reaction step and assay endpoint.

`structural_modeling_agent` computes the descriptors, links them to measurements in
`exp_data`, and exports frozen calibration bundles. These bundles retain score
definitions, training standardization, sampling requirements and source
fingerprints. Applying a bundle does not refit it or change the discovery rank.
Evidence is attached to an active discovery run with:

```bash
python -m enzyme_discovery_agent.import_structural_evidence \
  enzyme_discovery_agent/runs/RUN_ID structural_modeling_agent/integrated_campaigns/CAMPAIGN/candidate_proxy_evidence.json
```

Run this command from the repository root. Incompatible mechanisms, molecular states and assemblies retain an
explicit unavailable/inapplicable result. New experiments can motivate revised
proxy hypotheses or a new version of a calibrated bundle.

See `structural_modeling_agent/README.md` for scoring, calibration and figure ownership.

## Evidence, evaluation, and cost review

Source fingerprints and exact quotation checks establish where a passage came
from, not that it entails every extracted claim. Automated promotion from
literature-only reaction evidence additionally requires a narrow, affirmative,
candidate-specific statement with matching direction and bond evidence. This
lexical check is conservative and is not expert verification; unsupported or
ambiguous passages remain pending. Review the source before relying on a claim.

A search is automatically deduplicated only against an identical successful
request in the same run. Failed or partial searches can be retried, including
after resume. Shared names are tentative grouping hints; they do not by
themselves establish sequence identity or justify dropping a search hit.
Evaluation budgets may still defer candidates, which remain in the registry.

Analyze existing event telemetry without launching discovery or provider calls:

```bash
python -m enzyme_discovery_agent.eval.cost_analysis runs/your_run
```

The report separates recorded model tokens and tool costs, includes search
novelty and deduplication counts, and states where recorded data is incomplete.
Tool durations may overlap and the event span can include gaps between resumed
invocations. Current token and time allowances are soft boundaries checked
between calls and batches; an in-flight call can finish beyond them. The legacy
`--max-iterations` is a planning hint, not a hard stop.

For independent evaluation, maintain a separate adjudicated case JSON list and
record a matching output list outside runtime discovery inputs, then run:

```bash
python -m enzyme_discovery_agent.eval.adjudicated cases.json outputs.json
```

Each case requires a unique `id` and an `adjudications` mapping of exact
candidate identifiers to `positive` or `negative`; each output has the same
`id` and lists for `retrieved`, `resolved`, `evaluated`, `eligible`, and
`ranked` (ordered). The scorer reports stage recall, precision among judged
candidates, unjudged counts, rank, and supplied abstention and claim-review
counts. Claim reviews are a list of objects with a `verdict` of `supported`,
`unsupported`, or `uncertain`. It does not certify adjudications, case independence, or evidence
entailment. Development cases are not held out; independent cases need separate
curation and expert review before their scores support generalization claims.
The legacy run scorer counts a candidate as currently eligible only when its
current registry state and recorded claim validation support eligibility;
older records missing validation remain unverified for that metric.

## Structural-modeling handoff

Discovery writes `structural_modeling_handoff.json` (schema 3.0) and a companion
`structural_modeling_handoff.md` in the run directory. The JSON is authoritative;
the Markdown is a readable projection of the same snapshot. The envelope has
separate `candidates`, `unresolved_leads`, and `excluded_candidates` lists.
Admission requires a current evaluation, explicit positive claim validation and
a candidate-linked supported identity claim; absent or conflicting evidence
remains a lead. A neighbor is never admitted solely on similarity. Incomplete
checkpoints refresh the handoff and label individually admitted candidates
provisional. An empty handoff is a valid no-work result.

Candidate evidence retains current claim validation, conflicting fields,
selection rationale, rank when present, and unresolved modeling proposals. A
saved ranking is explicitly marked unverified because its freshness is not
established. Supplied or cached sequence values are fingerprinted but remain
unverified for modeling until the receiver establishes their identity. Source
snapshots must be fingerprinted and located under the run artifacts directory
to be copied into `handoff_bundle/sources`; unavailable or changed sources are
reported as such. Copy the JSON and `handoff_bundle` together when relocating.
The modeling agent verifies bundle bytes and path boundaries before copying an
available source into its own run. No discovery claim or proposal proves the
scientific correctness of a modeling setup.

The 3.0 `integrity.hash_profile` is `python-json-sortkeys-utf8-v1`: SHA-256 over
UTF-8 Python `json.dumps` with `sort_keys=True`, `ensure_ascii=False`,
`allow_nan=False`, and `separators=(',', ':')`. It is deterministic for the
supported Python implementation but is **not RFC 8785**; another implementation
must reproduce that profile exactly or use the supplied validation library.
The target fingerprint covers the target object; the snapshot fingerprint covers
every top-level field except `integrity`. Unknown top-level fields, including
execution-objective overrides, are rejected. A change to saved
registry, target, ranking or manifest during construction aborts publication.

The main structural-modeling agent validates 3.0 before use. It adapts a
well-formed 2.0 handoff as investigation-only leads where admission evidence is
unknown; standalone schema-1.0 modeling tasks remain a separate supported input.
The older `modeling_reasoning.resolve_handoff` utility is still a schema-2.0
workflow; use the main modeling agent for schema 3.0. The standalone legacy
input builder still accepts 2.0, while schema 3.0 passes through the new
contract validator. These compatibility paths do not make older records
scientifically validated.

Schema-3.0 admission requires a successful evaluation; rows with missing,
failed, or otherwise unknown evaluation status remain investigation leads.
The snapshot hash detects changed content, but it is not a digital signature or
proof of who created the handoff. The main modeling agent independently resolves
sequence identity before execution and currently supports only exact active
UniProt primary accessions for that check. Its offline `--inspect-handoff`
command can inspect leads without starting the agent.
