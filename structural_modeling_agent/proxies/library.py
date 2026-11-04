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
  * composite / fused scores -- outside this generic core; separately declared
    physical combinations require component evidence and held-out validation

Each entry declares ``expected_direction`` so the direction audit in
``calibration.py`` can fail a proxy that only "works" with its sign flipped after
seeing the labels, and ``determinant`` so results roll up to
``knowledge/structural_proxy_prior.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    alignment_ok: bool = True

    @property
    def distances(self) -> np.ndarray:
        return np.array([s.attack_distance for s in self.sites])

    def gate(self, cut: float = DEFAULT_DISTANCE_GATE) -> np.ndarray:
        return self.distances < cut


def build_ensemble(poses: Sequence[Pose], sites: Sequence[ReactiveSite]) -> Ensemble:
    """Superpose on protein CA so positional statistics are frame-invariant."""
    if len(poses) != len(sites):
        raise ValueError("poses and reactive sites must have identical lengths")
    if not poses:
        return Ensemble(sites=[], poses=[])
    resolved = [(p, s) for p, s in zip(poses, sites) if s is not None]
    if not resolved:
        return Ensemble(sites=[], poses=[], alignment_ok=False)
    reference = resolved[0][0]
    ref = reference.ca
    alignment_ok = True
    moved: list[ReactiveSite] = []
    keep: list[Pose] = []
    for pose, site in zip(poses, sites):
        if site is None:
            continue
        matched = (pose.ca.shape == ref.shape and len(ref) >= 3
                   and np.linalg.matrix_rank(ref - ref.mean(0)) >= 2
                   and np.linalg.matrix_rank(pose.ca - pose.ca.mean(0)) >= 2
                   and (not reference.ca_keys or pose.ca_keys == reference.ca_keys))
        if matched:
            R, t = _kabsch(pose.ca, ref)
        else:
            R, t = np.eye(3), np.zeros(3)
            alignment_ok = False

        def mv(v):
            return None if v is None else R @ v + t

        moved.append(ReactiveSite(
            nucleophile=mv(site.nucleophile), electrophile=mv(site.electrophile),
            leaving=mv(site.leaving), charge_centre=mv(site.charge_centre),
            frame=tuple(mv(f) for f in site.frame), ideal_angle=site.ideal_angle,
            angle_reference=mv(site.angle_reference)))
        # Protein, donor, ligand and role coordinates MUST share the same frame.
        # Transforming roles alone changes otherwise invariant contact distances.
        keep.append(replace(pose, ca=pose.ca @ R.T + t,
                            protein=pose.protein @ R.T + t,
                            donors=pose.donors @ R.T + t,
                            bases=pose.bases @ R.T + t,
                            ligands={ch: [(n, e, mv(x)) for n, e, x in atoms]
                                     for ch, atoms in pose.ligands.items()},
                            atoms=tuple(replace(a, xyz=mv(a.xyz)) for a in pose.atoms)))
    return Ensemble(sites=moved, poses=keep, alignment_ok=alignment_ok)


def _angle_reference(site: ReactiveSite):
    return site.angle_reference if site.angle_reference is not None else site.leaving


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
    version: str = "1.0"

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
        reference = _angle_reference(s)
        if reference is None:
            return None
        if s.attack_distance >= cut:
            continue
        ang = _angle(s.nucleophile, s.electrophile, reference)
        if abs(ang - s.ideal_angle) <= tol:
            ok += 1
    return ok / len(ens.sites)


def _trajectory_angle(ens: Ensemble, cut=DEFAULT_DISTANCE_GATE):
    """Mean deviation from the mechanism's ideal attack angle among distance-gated
    poses. Expressed as a DEVIATION rather than a raw angle so one direction of
    merit holds across mechanism classes with different ideal angles; a raw angle
    is only comparable within a single class."""
    g = ens.gate(cut)
    if g.sum() < MIN_GATED_POSES:
        return None
    if any(_angle_reference(s) is None for s, keep in zip(ens.sites, g) if keep):
        return None
    dev = [abs(_angle(s.nucleophile, s.electrophile, _angle_reference(s)) - s.ideal_angle)
           for s, keep in zip(ens.sites, g) if keep]
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
    if len(ens.sites) < MIN_GATED_POSES or not ens.alignment_ok:
        return None
    return _rmsf(np.array([s.nucleophile for s in ens.sites]))


def _approach_consistency(ens: Ensemble):
    """v2: acos(mean resultant length), degrees in [0,90]; zero means aligned."""
    if len(ens.sites) < MIN_GATED_POSES or not ens.alignment_ok:
        return None
    v = np.array([s.electrophile - s.nucleophile for s in ens.sites])
    lengths = np.linalg.norm(v, axis=1, keepdims=True)
    if not np.isfinite(v).all() or np.any(lengths <= 1e-12):
        return None
    resultant = float(np.linalg.norm((v / lengths).mean(0)))
    return float(np.degrees(np.arccos(np.clip(resultant, 0, 1))))


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
    """Distance to a nearby ionizable sidechain N/O; an exploratory proximity
    measure. This does not identify a catalytic acid/base or its protonation."""
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
          "Mean |attack angle - mechanism ideal| among distance-gated poses.",
          _trajectory_angle),
    Proxy("trajectory_alignment", "reactive_conformation", "+", True,
          "Mean out-of-plane angle of the attack vector against the electrophile's "
          "local frame; separates true perpendicular approach from in-plane contact.",
          _trajectory_alignment),
    Proxy("ts_charge_stabilisation", "transition_state_electrostatics", "+", False,
          "Mean protein H-bond donors within 3.5 A of the TS charge centre "
          "(generic oxyanion hole).", _ts_charge_stabilisation),
    Proxy("base_proximity", "nucleophile_base_activation", "-", False,
          "Mean distance to the nearest ionizable sidechain N/O (unannotated).",
          _base_proximity),
    Proxy("leaving_group_stabilisation", "leaving_group_metal_stabilization", "+", False,
          "Mean protein H-bond donors within 3.5 A of the leaving atom.",
          _leaving_group_stabilisation),
    Proxy("reactive_preorganisation", "dynamic_preorganization", "-", False,
          "Positional RMSF of the nucleophile across the superposed ensemble. "
          "Gate-free, so it is defined for non-binders too.", _preorganisation),
    Proxy("approach_consistency", "dynamic_preorganization", "-", False,
          "v2 mean-resultant angular dispersion: degrees(acos(norm(mean(unit attack vectors)))); zero aligned, 90 zero resultant. Historical v1 calibrations require review.",
          _approach_consistency, version="2.0"),
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
