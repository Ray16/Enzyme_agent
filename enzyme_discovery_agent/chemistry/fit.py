"""Shared substrate-fit assignment.

A reaction's target may have several substrates (e.g. two co-substrates).  Both the
family-survey PROMOTION path and the candidate-EVALUATION path must decide, for a
member enzyme, how well its NATIVE substrates accommodate the target substrates.

Each native substrate occurrence serves at most one target occurrence. Otherwise
several target participants could all match the same favorable native partner,
hiding a poor or unresolved counterpart. Assignment maximizes the weakest pair's
descriptor resemblance. It proposes analogue pairings; it does not identify
reacting atoms or establish that the participants have equivalent chemical roles.

This module is the single place that assignment lives; both paths call it so they
judge substrates identically.
"""

from __future__ import annotations

from itertools import permutations
from collections import Counter


def comparison_coverage(similarity):
    """Account for every required target occurrence, retaining missing evidence."""
    accommodation=similarity.get('accommodation') or {}
    expected=similarity.get('target_smiles') or [p.get('target_smiles') for p in accommodation.get('per_target') or []]
    expected=[s for s in expected if s]
    required=similarity.get('n_target_molecules') or accommodation.get('n_target_substrates') or len(expected)
    comparisons=similarity.get('property_comparisons') or []
    from enzyme_discovery_agent.chemistry_logic import whole_package_pair_fit
    observed=Counter(p.get('target_smiles') for p in comparisons if p.get('target_smiles')
                     and p.get('native_smiles') and whole_package_pair_fit(p) is not None)
    missing=list((Counter(expected)-observed).elements())
    resolved=sum((Counter(expected)&observed).values()) if expected else sum(whole_package_pair_fit(p) is not None for p in comparisons)
    # Older single-participant records attach their sole comparison to the
    # sole explicit role rather than repeating its identity in that comparison.
    roles=accommodation.get('per_target') or []
    if (required==1 and len(expected)==1 and len(comparisons)==1
            and not comparisons[0].get('target_smiles') and len(roles)==1 and roles[0].get('native_smiles')
            and whole_package_pair_fit(comparisons[0]) is not None):
        resolved=1;missing=[]
    return {'required':required or None,'resolved':resolved,'missing_target_smiles':missing,
            'complete':not missing and resolved>=required if required else None}


def accommodation_score(similarity: dict) -> float | None:
    """Shared role-aware ordering for survey selection and final ranking.

    Every required target substrate contributes its own distinct-native match.
    Explicitly unmatched roles remain a weak analogue (zero); unresolved
    comparisons remain unknown. No similarity establishes target activity.
    """
    from enzyme_discovery_agent.chemistry_logic import whole_package_pair_fit
    accommodation = similarity.get("accommodation") or {}
    if accommodation.get('role_transfer_status')=='unresolved':
        return None
    coverage=comparison_coverage(similarity)
    if coverage['complete'] is False:
        return None
    if accommodation.get("unaccommodated_targets"):
        return 0.0
    comparisons = similarity.get("property_comparisons") or []
    fits = [whole_package_pair_fit(c) for c in comparisons]
    return min(fits) if fits and all(value is not None for value in fits) else None


def assign_one_to_one(targets, natives, score):
    """Assign each target substrate a DISTINCT native, maximising the WORST pair.

    ``score(target, native) -> float`` (higher = better fit) is supplied by the
    caller, so this works for any per-pair metric (functional-group overlap,
    fingerprint similarity, …).  Returns a list ``[(target, native|None, score)]``
    in target order.

    Selection is MAXIMIN: among all assignments that give each target a distinct
    native, pick the one whose *weakest* pair is strongest (tie-broken by the total),
    because the enzyme is limited by the target substrate it accommodates worst.
    When there are fewer natives than targets, the surplus targets get ``None`` at
    score 0.0 (the enzyme cannot host every target substrate on a distinct native).

    Enumerates assignments exactly for the small substrate counts that occur in
    practice; falls back to a greedy maximin for larger sets so it never blows up.
    Pure and deterministic (stable tie-breaks), so it is unit-testable offline.
    """
    targets = list(targets)
    natives = list(natives)
    if not targets:
        return []
    if not natives:
        return [(t, None, 0.0) for t in targets]

    S = [[float(score(t, n)) for n in natives] for t in targets]
    n_t, n_n = len(targets), len(natives)

    def _package(assign):
        # assign[ti] = native index or None
        out = []
        for ti in range(n_t):
            ni = assign[ti]
            out.append((targets[ti], natives[ni] if ni is not None else None,
                        S[ti][ni] if ni is not None else 0.0))
        return out

    def _key(assign):
        scores = [S[ti][assign[ti]] if assign[ti] is not None else 0.0
                  for ti in range(n_t)]
        return (min(scores), sum(scores))

    # Exact enumeration for small sets (the common case: 1-3 substrates a side).
    if n_t <= 5 and n_n <= 7:
        best_assign = None
        best_key = None
        k = min(n_t, n_n)
        # Choose which k targets get a native (when targets outnumber natives),
        # and which distinct native each gets.
        for tsel in permutations(range(n_t), k):
            for nperm in permutations(range(n_n), k):
                assign = [None] * n_t
                for j in range(k):
                    assign[tsel[j]] = nperm[j]
                key = _key(assign)
                if best_key is None or key > best_key:
                    best_key, best_assign = key, assign
        return _package(best_assign)

    # Greedy maximin fallback: repeatedly take the highest-scoring (target, native)
    # pair among the still-unassigned, until targets or natives run out.
    assign = [None] * n_t
    used_t, used_n = set(), set()
    pairs = sorted(
        ((S[ti][ni], ti, ni) for ti in range(n_t) for ni in range(n_n)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    for sc, ti, ni in pairs:
        if ti in used_t or ni in used_n:
            continue
        assign[ti] = ni
        used_t.add(ti)
        used_n.add(ni)
        if len(used_t) == n_t or len(used_n) == n_n:
            break
    return _package(assign)
