"""Mechanism-general structural proxies the agent tries for ANY enzyme.

    from structural_modeling.proxies import (
        parse_pose, resolve, build_ensemble, evaluate, LIBRARY, screen,
    )

    poses = [parse_pose(p) for p in sorted(glob(f"{pred}/*_model_*.pdb"))]
    sites = [resolve(p, "acyl_addition_elimination") for p in poses]
    scores = evaluate(build_ensemble(poses, sites))      # proxy id -> value

Then, before believing any of it, run ``screen`` with the activity labels and a
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
    RESOLVERS, Pose, ReactiveSite, parse_pose, resolve,
)

__all__ = [
    "parse_pose", "resolve", "RESOLVERS", "Pose", "ReactiveSite",
    "build_ensemble", "evaluate", "Ensemble", "Proxy", "LIBRARY", "BY_ID",
    "DEFAULT_DISTANCE_GATE", "MIN_GATED_POSES", "UNFILLED_DETERMINANTS",
    "screen", "selection_null", "robust_auroc", "auroc", "partial_spearman",
    "ProxyVerdict",
]
