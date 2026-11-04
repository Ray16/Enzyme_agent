"""Mechanism-general structural proxies the agent tries for ANY enzyme.

This package is system-free. It supplies the proxy catalogue, the ensemble
machinery and the statistics; the AGENT supplies a short, system-specific
*resolver* that says which atom plays which catalytic role, and validates it.

    from structural_modeling_agent.proxies import (
        parse_pose, check_resolver, build_ensemble, evaluate, LIBRARY, screen,
    )

    def my_resolver(pose):            # written per system, by the agent
        ...                           # -> ReactiveSite(nucleophile=..., ...)

    poses = [parse_pose(p) for p in sorted(glob(f"{pred}/*_model_*.pdb"))]

    report = check_resolver(my_resolver, poses)      # validate BEFORE scoring
    assert report.ok, report

    scores = evaluate(build_ensemble(poses, [my_resolver(p) for p in poses]))

Then, before believing any score, run ``screen`` with the activity labels and a
confidence confound. See README.md for what is in, what is out, and why.
"""
from .calibration import (  # noqa: F401
    ProxyVerdict, auroc, partial_spearman, robust_auroc, screen, selection_null,
)
from .library import (  # noqa: F401
    BY_ID, DEFAULT_DISTANCE_GATE, LIBRARY, MIN_GATED_POSES, UNFILLED_DETERMINANTS,
    Ensemble, Proxy, build_ensemble, evaluate,
)
from .roles import (  # noqa: F401
    Atom, Pose, ReactiveSite, Resolver, ResolverReport, check_resolver, parse_pose,
)

__all__ = [
    "parse_pose", "Atom", "Pose", "ReactiveSite", "Resolver",
    "check_resolver", "ResolverReport",
    "build_ensemble", "evaluate", "Ensemble", "Proxy", "LIBRARY", "BY_ID",
    "DEFAULT_DISTANCE_GATE", "MIN_GATED_POSES", "UNFILLED_DETERMINANTS",
    "screen", "selection_null", "robust_auroc", "auroc", "partial_spearman",
    "ProxyVerdict",
]
