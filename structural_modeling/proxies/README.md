# Structural proxies

A system-free catalogue of structural proxies to try when turning a predicted
structure ensemble into an activity prediction, plus the statistics that decide
whether any of them earned belief.

This package knows nothing about any particular enzyme, cofactor or reaction.

## Division of labour

| | who writes it | what it is |
|---|---|---|
| **resolver** | the **agent**, per system | short code reading one pose and returning a `ReactiveSite`: which atom attacks, which is attacked, what leaves, the ideal attack angle for that chemistry |
| **proxies** | this package | every quantity computable from `ReactiveSite` + the pose, over the ensemble |
| **validation** | this package | resolver sanity checks, direction audit, confound control, selection null |

Role assignment is reaction knowledge. A general module that hard-coded it would
be wrong for most systems, so the agent writes a resolver for the system in front
of it, validates it, and plugs it in. Everything downstream is shared.

```python
from structural_modeling.proxies import (
    parse_pose, check_resolver, build_ensemble, evaluate, screen, LIBRARY,
)

def my_resolver(pose):                 # agent-written, this system only
    ...                                # -> ReactiveSite | None

poses  = [parse_pose(p) for p in sorted(glob(f"{pred}/*_model_*.pdb"))]

report = check_resolver(my_resolver, poses)     # ALWAYS before scoring
assert report.ok, report

scores = evaluate(build_ensemble(poses, [my_resolver(p) for p in poses]))
```

`build_ensemble` superposes on protein CA, so positional statistics are measured
in the pocket frame rather than in whatever frame each prediction landed in.

## The catalogue

Each proxy declares the determinant it probes (ids from
`knowledge/structural_proxy_prior.yaml`) and its expected direction, so a proxy
that only works once its sign is flipped after seeing labels can be failed.

| proxy | determinant | dir | needs |
|---|---|---|---|
| `approach_distance` | ground_state_binding | − | nucleophile, electrophile |
| `nucleophile_burial` | ground_state_binding | + | nucleophile |
| `near_attack_fraction` | reactive_conformation | + | + leaving, gate |
| `trajectory_angle_deviation` | reactive_conformation | − | + leaving, gate |
| `trajectory_alignment` | reactive_conformation | + | + frame, gate |
| `ts_charge_stabilisation` | transition_state_electrostatics | + | charge_centre |
| `base_proximity` | nucleophile_base_activation | − | nucleophile |
| `leaving_group_stabilisation` | leaving_group_metal_stabilization | + | leaving |
| `reactive_preorganisation` | dynamic_preorganization | − | nucleophile |
| `approach_consistency` | dynamic_preorganization | − | nucleophile, electrophile |
| `distance_stability` | dynamic_preorganization | − | nucleophile, electrophile |

`integrated_barrier` has **no** entry. No ensemble geometry estimates a reaction
barrier; it needs QM or MD. Leaving the determinant visibly unfilled is the
honest state, and `UNFILLED_DETERMINANTS` records it.

Two design notes worth keeping:

- **Angles are stored as deviation from the mechanism's ideal**, not as raw
  angles, so one direction of merit holds across mechanism classes whose ideal
  angles differ. Raw angles are only comparable within a single class.
- **Preorganisation proxies are gate-free on purpose.** Gated proxies are
  undefined when no pose reaches near-attack — which is exactly what a true
  negative looks like — so a gated-only panel silently discards its most
  informative cases. Gate-free proxies score them.

## What is excluded, and why

- **metal / cofactor coordination geometry** — undefined for a system with a
  different cofactor or none
- **burial or contacts of a specific cofactor moiety** — cofactor vocabulary
- **anything addressed by atom name** — atom identity is mechanism data and
  reaches proxies only through `ReactiveSite`
- **residue-specific pKa, named catalytic residues** — need per-enzyme
  annotation, so they are a targeted follow-up the agent can write, not a
  default try
- **composite / fused scores** — forbidden by the prior's `no_composite`;
  report proxies decomposed

## Validation is not optional

`screen(scores, activity, LIBRARY, confound=...)` applies three gates and returns
a verdict per proxy:

1. **direction audit** — the sign must match the pre-declared direction.
2. **confound control** — partial correlation against a nuisance covariate.
   Pass model confidence (e.g. mean complex pLDDT) whenever available: fold or
   complex confidence alone can be a serviceable activity predictor without
   reflecting catalysis, and a proxy correlated with it inherits that
   predictiveness for free.
3. **selection null** — permute the labels, score every proxy, keep the maximum,
   repeat. This reproduces the actual selection procedure. The maximum over a set
   of uninformative features is biased upward; the bias grows with the size of the
   set and shrinks with the number of actives, and on a panel with few actives it
   is large enough to produce respectable-looking AUROCs from noise. Comparing a
   winner against 0.5 instead of against this null will promote noise.

A proxy is promoted only if it passes the direction audit **and** clears the
selection null. Scores are reported with their verdict attached, never bare.

### Panels with few actives

When a panel has only a handful of actives, the selection null is wide and almost
nothing will clear it. That is the correct answer, not a failure of the method:
such a panel can rank candidates but cannot adjudicate between proxies. Report
the null alongside the scores and say so.
