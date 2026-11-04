# Review response: discovery throughput and assessment publication

The review correctly identifies expensive model calls, poor prompt reuse,
unresolved literature names and a default-mode assessment gap. The cited campaign
also exposes a different dominant workload than the review's synthesis suggests.
The read-only audit is [review_run_audit_20261104.json](review_run_audit_20261104.json).

Reproduce it locally without provider calls:

```bash
python -m enzyme_discovery_agent.audit_run /path/to/saved/run
```

## What the cited campaign shows

Run `20261103_154105` stopped after 6,828.34 seconds with 3,009,558 recorded tokens.
Its recorded configuration was direct mode (`ADAPTIVE_REASONING=false`) with
`AUTO_TARGET_REVIEW=true`.

| Workload | Calls | Recorded tokens | Fraction of total | Cache reads / prompt tokens |
| --- | ---: | ---: | ---: | ---: |
| Automatic target assessments | 51 | 2,460,289 | 81.75% | 0.10% |
| Discovery controller | 20 | 490,638 | 16.30% | 72.64% |
| Other internal requests | 9 | 58,631 | 1.95% | 25.74% |

The registry contains 305 saved target assessments: 131 investigate, 172 defer
and 2 exclude. All 131 CSV rows omit their saved assessment decision. Those blank
export cells do not establish that the agent never made a target assessment.
No primary proposals were recorded; an investigate assessment is not a strong
primary recommendation or evidence of target activity.

There are 12 registered paper-scoped leads and 59 pending-resolution events.
The 59 events are attempts, not 59 independently registered leads. The failure
to resolve the paper-scoped identities remains a real problem.

The shared substrate-resemblance value appears in 83 rows. It is not a measured
per-protein substrate-fit signal. Changing that value without additional
candidate-specific evidence would not repair targeting. Paper counts are exposed
as evidence metadata; the current deterministic ranking does not use them as a
numerical ranking key, so the proposed fame-based explanation is not established
by this audit.

## Repairs

- Assessment tool schemas no longer embed the current accessions or participant
  and source-ID lists. The common target contract precedes changing candidate
  data. Unknown accessions, wrong source references and incomplete target-role
  mappings still fail admission against the actual saved packet.
- Automatic assessment is enabled in both controller modes. One review batch
  yields to a controller decision by default. The scheduling counter survives
  resume, and the existing bounded correction behavior remains configurable.
- CSV publication reads the current source-validated assessment from the registry
  even when its native-evidence projection omits that field.
- Direct-mode history retains a fixed boundary between periodic compactions;
  the latest state remains last. Rewritten source history invalidates the window,
  complete tool transactions stay paired, and the context-size bound still applies.
  Adaptive evidence projection retains its existing source-preservation behavior.
- Successful subscription-auth checks are cached for at most five minutes and
  invalidated by settings or credential-file changes and transport failure.
  Per-response provider validation and environment isolation remain in place.
  Scientific reasoning effort is not automatically reduced.
- Identity lookup accepts source-derived species-prefixed aliases and explicit
  protein descriptions, with taxon-scoped queries. Multiple distinct sequences
  and conflicting direct source associations remain unresolved. An unsourced
  registry hint cannot override an explicit paper clause. New identity evidence
  or resolver code reopens follow-up; transient batch failures use a shorter retry
  interval.
- Live ranking applies the existing exact-sequence collapse and preserves its
  alternatives. A shared protein name alone is not treated as verified sequence
  equivalence. The unused history-pruning method was removed.

## Validation and limits

Synthetic regressions cover stable schemas and common prefixes, source-bound
admission, assessment publication, review scheduling, history invalidation,
source-derived identity aliases, ambiguity and authentication caching. The run
audit is read-only. Historical CSVs and decisions have not been rewritten.

These checks do not demonstrate a live cache-hit ratio, a twofold speedup,
recovery of every literature lead or a strong experimental panel. No new paid
production run, MSA submission or GPU prediction is part of this repair pass.

## Queued follow-up: targeted retrieval

After this repair pass, review how candidate-specific native reactions, reacting
roles, substrate structures and small-molecule versus carrier/macromolecular
topology influence member retrieval and prioritization. Verify that resolved
literature identities influence the next search and that evidence shared by
related sequences is not counted as independent mechanistic support. Keep
unresolved substrate acceptance explicit, preserve alternative justified catalytic
routes, and do not encode preferred families or benchmark identities as defaults.
