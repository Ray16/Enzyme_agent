"""The structural-proxy library: what the agent TRIES on any enzyme.

Each proxy is mechanism-general -- written over the roles in ``roles.ReactiveSite``
and over ensemble statistics, never over named atoms or a particular cofactor. A
proxy earns a place here only if it can be computed for an arbitrary enzyme once
its mechanism class is known.

Deliberately EXCLUDED (see README for the full argument):
  * metal- and cofactor-coordination geometry -- undefined for systems that use a
    different cofactor, or none
  * burial or contact features of a specific cofactor moiety -- cofactor vocabulary
  * features addressed by atom name -- atom identity is mechanism DATA and reaches
    proxies only through ``ReactiveSite``
  * residue-specific pKa or named catalytic residues -- these need per-enzyme
    annotation, so they are a targeted follow-up, not a default try
  * any composite / fused score -- forbidden by the prior's ``no_composite``

Each entry declares ``expected_direction`` so the direction audit in
``calibration.py`` can fail a proxy that only "works" with its sign flipped after
seeing the labels, and ``determinant`` so results roll up to
``knowledge/structural_proxy_prior.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .roles import Pose, ReactiveSite

DEFAULT_DISTANCE_GATE = 4.0     # A; a GATE on which poses are interpretable, never a score
MIN_GATED_POSES = 5             # below this the ensemble cannot support a claim


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _angle(a, b, c) -> float:
    ba, bc = a - b, c - b
    v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(v, -1.0, 1.0))))


def _kabsch(P: np.ndarray, Q: np.ndarray):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _s, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def _rmsf(X: np.ndarray) -> float:
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))


@dataclass(frozen=True)
class Ensemble:
    """One complex: the per-pose reactive sites plus the poses they came from.

    ``sites`` and ``poses`` are parallel and already superposed on protein CA
    (reference = first pose), so positional spread is measured in the pocket frame
    rather than in whatever arbitrary frame each prediction landed in.
    """

    sites: Sequence[ReactiveSite]
    poses: Sequence[Pose]

    @property
    def distances(self) -> np.ndarray:
        return np.array([s.attack_distance for s in self.sites])

    def gate(self, cut: float = DEFAULT_DISTANCE_GATE) -> np.ndarray:
        return self.distances < cut


def build_ensemble(poses: Sequence[Pose], sites: Sequence[ReactiveSite]) -> Ensemble:
    """Superpose on protein CA so positional statistics are frame-invariant."""
    if not poses:
        return Ensemble(sites=[], poses=[])
    ref = poses[0].ca
    moved: list[ReactiveSite] = []
    keep: list[Pose] = []
    for pose, site in zip(poses, sites):
        if site is None:
            continue
        if pose.ca.shape == ref.shape and len(ref):
            R, t = _kabsch(pose.ca, ref)
        else:
            R, t = np.eye(3), np.zeros(3)

        def mv(v):
            return None if v is None else R @ v + t

        moved.append(ReactiveSite(
            nucleophile=mv(site.nucleophile), electrophile=mv(site.electrophile),
            leaving=mv(site.leaving), charge_centre=mv(site.charge_centre),
            frame=tuple(mv(f) for f in site.frame), ideal_angle=site.ideal_angle))
        keep.append(pose)
    return Ensemble(sites=moved, poses=keep)


# --------------------------------------------------------------------------- #
# proxies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proxy:
    id: str
    determinant: str          # an id in knowledge/structural_proxy_prior.yaml
    expected_direction: str   # "+" higher => more active, "-" lower => more active
    gated: bool               # does it require distance-gated poses?
    doc: str
    fn: Callable[[Ensemble], float | None]

    def __call__(self, ens: Ensemble) -> float | None:
        return self.fn(ens)


def _approach_distance(ens: Ensemble):
    d = ens.distances
    return float(np.percentile(d, 10)) if len(d) else None


def _near_attack_fraction(ens: Ensemble, cut=DEFAULT_DISTANCE_GATE, tol=30.0):
    if not ens.sites:
        return None
    ok = 0
    for s in ens.sites:
        if s.attack_distance >= cut or s.leaving is None:
            continue
        ang = _angle(s.nucleophile, s.electrophile, s.leaving)
        if abs(ang - s.ideal_angle) <= tol:
            ok += 1
    return ok / len(ens.sites)


def _trajectory_angle(ens: Ensemble, cut=DEFAULT_DISTANCE_GATE):
    """THE INCUMBENT, generalised: mean deviation from the mechanism's ideal attack
    angle among distance-gated poses. Returned as deviation so that, unlike the
    adenylation-only 'mean in-line angle', the direction is the same for an SN2
    (ideal 180) and a Burgi-Dunitz addition (ideal 107)."""
    g = ens.gate(cut)
    if g.sum() < MIN_GATED_POSES:
        return None
    dev = [abs(_angle(s.nucleophile, s.electrophile, s.leaving) - s.ideal_angle)
           for s, keep in zip(ens.sites, g) if keep and s.leaving is not None]
    return float(np.mean(dev)) if dev else None


def _trajectory_alignment(ens: Ensemble, cut=DEFAULT_DISTANCE_GATE):
    """How well the attack vector lines up with the electrophile's local frame
    normal: 90 deg = perpendicular (productive for an sp2 centre), 0 = in-plane.
    Distinguishes a genuine out-of-plane approach from an in-plane contact that
    happens to satisfy the scalar angle window."""
    g = ens.gate(cut)
    if g.sum() < MIN_GATED_POSES:
        return None
    vals = []
    for s, keep in zip(ens.sites, g):
        if not keep or len(s.frame) < 2:
            continue
        nrm = np.cross(s.frame[0] - s.electrophile, s.frame[1] - s.electrophile)
        n = np.linalg.norm(nrm)
        if n < 1e-6:
            continue
        v = s.nucleophile - s.electrophile
        vals.append(float(np.degrees(np.arcsin(
            np.clip(abs(np.dot(v / (np.linalg.norm(v) + 1e-9), nrm / n)), 0, 1)))))
    return float(np.mean(vals)) if vals else None


def _preorganisation(ens: Ensemble):
    """Positional RMSF of the nucleophile in the pocket frame across the ensemble.

    GATE-FREE on purpose: it is defined even when no pose reaches near-attack, so
    substrates that never approach -- the informative negatives, which the gated
    angle proxy must score as 'inconclusive' -- still receive a value."""
    if len(ens.sites) < MIN_GATED_POSES:
        return None
    return _rmsf(np.array([s.nucleophile for s in ens.sites]))


def _approach_consistency(ens: Ensemble):
    """Angular spread (deg) of the nucleophile->electrophile vector."""
    if len(ens.sites) < MIN_GATED_POSES:
        return None
    v = np.array([s.electrophile - s.nucleophile for s in ens.sites])
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    m = v.mean(0)
    m /= (np.linalg.norm(m) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(v @ m, -1, 1))).std())


def _distance_stability(ens: Ensemble):
    d = ens.distances
    return float(d.std()) if len(d) >= MIN_GATED_POSES else None


def _nucleophile_burial(ens: Ensemble, r=4.0):
    if not ens.sites:
        return None
    vals = [int((np.linalg.norm(p.protein - s.nucleophile, axis=1) < r).sum())
            for p, s in zip(ens.poses, ens.sites) if len(p.protein)]
    return float(np.mean(vals)) if vals else None


def _ts_charge_stabilisation(ens: Ensemble, r=3.5):
    """Protein H-bond donors around the atom that accumulates charge in the TS --
    the generic oxyanion hole, applicable to any polar transition state."""
    vals = []
    for p, s in zip(ens.poses, ens.sites):
        if s.charge_centre is None or not len(p.donors):
            continue
        vals.append(int((np.linalg.norm(p.donors - s.charge_centre, axis=1) < r).sum()))
    return float(np.mean(vals)) if vals else None


def _base_proximity(ens: Ensemble):
    """Distance from the nucleophile to the nearest protein general base/acid.
    Generic stand-in for nucleophile activation; no catalytic-residue annotation
    required, so it can be tried before the catalytic machinery is known."""
    vals = []
    for p, s in zip(ens.poses, ens.sites):
        if not len(p.bases):
            continue
        vals.append(float(np.linalg.norm(p.bases - s.nucleophile, axis=1).min()))
    return float(np.mean(vals)) if vals else None


def _leaving_group_stabilisation(ens: Ensemble, r=3.5):
    vals = []
    for p, s in zip(ens.poses, ens.sites):
        if s.leaving is None or not len(p.donors):
            continue
        vals.append(int((np.linalg.norm(p.donors - s.leaving, axis=1) < r).sum()))
    return float(np.mean(vals)) if vals else None


LIBRARY: tuple[Proxy, ...] = (
    Proxy("approach_distance", "ground_state_binding", "-", False,
          "10th-percentile nucleophile-electrophile distance over the ensemble.",
          _approach_distance),
    Proxy("nucleophile_burial", "ground_state_binding", "+", False,
          "Mean protein heavy atoms within 4 A of the nucleophile.",
          _nucleophile_burial),
    Proxy("near_attack_fraction", "reactive_conformation", "+", True,
          "Fraction of poses inside both the distance gate and the mechanism's "
          "angular window.", _near_attack_fraction),
    Proxy("trajectory_angle_deviation", "reactive_conformation", "-", True,
          "Mean |attack angle - mechanism ideal| among distance-gated poses. "
          "Generalisation of the validated adenylation in-line-angle proxy.",
          _trajectory_angle),
    Proxy("trajectory_alignment", "reactive_conformation", "+", True,
          "Mean out-of-plane angle of the attack vector against the electrophile's "
          "local frame; separates true perpendicular approach from in-plane contact.",
          _trajectory_alignment),
    Proxy("ts_charge_stabilisation", "transition_state_electrostatics", "+", False,
          "Mean protein H-bond donors within 3.5 A of the TS charge centre "
          "(generic oxyanion hole).", _ts_charge_stabilisation),
    Proxy("base_proximity", "nucleophile_base_activation", "-", False,
          "Mean distance from the nucleophile to the nearest Arg/Lys/His nitrogen.",
          _base_proximity),
    Proxy("leaving_group_stabilisation", "leaving_group_metal_stabilization", "+", False,
          "Mean protein H-bond donors within 3.5 A of the leaving atom.",
          _leaving_group_stabilisation),
    Proxy("reactive_preorganisation", "dynamic_preorganization", "-", False,
          "Positional RMSF of the nucleophile across the superposed ensemble. "
          "Gate-free, so it is defined for non-binders too.", _preorganisation),
    Proxy("approach_consistency", "dynamic_preorganization", "-", False,
          "Angular spread of the attack vector across the ensemble.",
          _approach_consistency),
    Proxy("distance_stability", "dynamic_preorganization", "-", False,
          "Standard deviation of the attack distance across the ensemble.",
          _distance_stability),
)

#: ``integrated_barrier`` has no entry: no ensemble geometry estimates a barrier.
#: It needs QM/MD, and leaving the determinant visibly unfilled is the honest state.
UNFILLED_DETERMINANTS = ("integrated_barrier",)

BY_ID = {p.id: p for p in LIBRARY}


def evaluate(ens: Ensemble, proxies: Sequence[Proxy] | None = None) -> dict[str, float | None]:
    """Compute every proxy for one complex. ``None`` means 'not supported by this
    ensemble' (too few poses, or the role it needs was not resolvable) and must be
    propagated as missing, never silently coerced to zero."""
    return {p.id: p(ens) for p in (proxies or LIBRARY)}
