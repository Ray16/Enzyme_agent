"""Tests for the generic structural-proxy package.

Everything is built on synthetic geometry, deliberately: the package must not
depend on any particular enzyme, and a test that needs a real campaign's output
would smuggle a system back in.
"""
from __future__ import annotations

import numpy as np
import pytest

from structural_modeling.proxies import (
    LIBRARY, UNFILLED_DETERMINANTS, Pose, ReactiveSite, auroc, build_ensemble,
    check_resolver, evaluate, partial_spearman, robust_auroc, screen,
    selection_null,
)


def make_pose(nuc, elec, n_prot=30, seed=0):
    rng = np.random.default_rng(seed)
    prot = rng.normal(0, 8, (n_prot, 3))
    ca = rng.normal(0, 8, (12, 3))
    return Pose(ca=ca, protein=prot, donors=prot[:6], bases=prot[6:10],
                ligands={"B": [("C1", "C", elec)], "C": [("N1", "N", nuc)]})


def site_at(nuc, elec, ideal=180.0, leaving=None, charge=None, frame=()):
    return ReactiveSite(nucleophile=np.asarray(nuc, float),
                        electrophile=np.asarray(elec, float),
                        leaving=None if leaving is None else np.asarray(leaving, float),
                        charge_centre=None if charge is None else np.asarray(charge, float),
                        frame=tuple(np.asarray(f, float) for f in frame),
                        ideal_angle=ideal)


def ensemble(offsets, ideal=180.0, leaving=(-2.0, 0, 0), frame=()):
    poses, sites = [], []
    for i, off in enumerate(offsets):
        nuc = np.array([off, 0.0, 0.0])
        elec = np.zeros(3)
        poses.append(make_pose(nuc, elec, seed=i))
        sites.append(site_at(nuc, elec, ideal, leaving, charge=(0, 1.2, 0), frame=frame))
    return build_ensemble(poses, sites)


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #
def test_attack_distance_is_euclidean():
    assert site_at((3, 4, 0), (0, 0, 0)).attack_distance == pytest.approx(5.0)


def test_every_proxy_declares_a_determinant_and_direction():
    for p in LIBRARY:
        assert p.determinant, p.id
        assert p.expected_direction in ("+", "-"), p.id


def test_proxy_ids_unique():
    ids = [p.id for p in LIBRARY]
    assert len(ids) == len(set(ids))


def test_integrated_barrier_is_declared_unfilled():
    assert "integrated_barrier" in UNFILLED_DETERMINANTS
    assert not any(p.determinant == "integrated_barrier" for p in LIBRARY)


# --------------------------------------------------------------------------- #
# proxies
# --------------------------------------------------------------------------- #
def test_preorganisation_rewards_a_tight_ensemble():
    tight = evaluate(ensemble([3.0, 3.01, 2.99, 3.0, 3.02, 3.0]))
    loose = evaluate(ensemble([3.0, 7.0, 1.6, 9.0, 2.0, 11.0]))
    assert tight["reactive_preorganisation"] < loose["reactive_preorganisation"]
    assert tight["distance_stability"] < loose["distance_stability"]


def test_preorganisation_defined_when_nothing_reaches_near_attack():
    """The point of a gate-free proxy: a non-binder still gets a value, where the
    gated proxies must return None."""
    far = ensemble([11.0, 11.1, 10.9, 11.2, 11.0, 10.8])
    out = evaluate(far)
    assert out["reactive_preorganisation"] is not None
    assert out["trajectory_angle_deviation"] is None
    assert out["near_attack_fraction"] == 0.0


# Superposition applies a Kabsch rotation to every role, and the attack angle is
# invariant under it -- but arccos is ill-conditioned near 180 deg, so the ~1e-7
# error in an SVD-derived rotation shows up as ~1e-3 deg here. Tolerances below
# are set accordingly; anything larger would be a real geometry bug.
ANGLE_TOL = 0.01


def test_trajectory_deviation_is_zero_for_perfect_geometry():
    """Nucleophile opposite the leaving group => 180 deg => zero deviation."""
    ens = ensemble([3.0] * 6, ideal=180.0, leaving=(-2.0, 0, 0))
    assert evaluate(ens)["trajectory_angle_deviation"] == pytest.approx(0.0, abs=ANGLE_TOL)


def test_trajectory_deviation_is_mechanism_relative():
    """Same geometry, different declared ideal => different deviation. This is why
    deviation is stored rather than a raw angle."""
    geom = [3.0] * 6
    inline = evaluate(ensemble(geom, ideal=180.0))["trajectory_angle_deviation"]
    burgi = evaluate(ensemble(geom, ideal=107.0))["trajectory_angle_deviation"]
    assert inline == pytest.approx(0.0, abs=ANGLE_TOL)
    assert burgi == pytest.approx(73.0, abs=ANGLE_TOL)


def test_gated_proxies_need_minimum_evidence():
    assert evaluate(ensemble([3.0, 3.0]))["trajectory_angle_deviation"] is None


def test_trajectory_alignment_separates_in_plane_from_perpendicular():
    frame = [(0.0, 1.2, 0.0), (1.1, -0.6, 0.0)]          # frame spans the xy plane
    perp, inplane = [], []
    for i in range(6):
        elec = np.zeros(3)
        perp.append((make_pose(np.array([0, 0, 3.0]), elec, seed=i),
                     site_at((0, 0, 3.0), elec, 107.0, (-2, 0, 0), (0, 1.2, 0), frame)))
        inplane.append((make_pose(np.array([3.0, 0, 0]), elec, seed=i),
                        site_at((3.0, 0, 0), elec, 107.0, (-2, 0, 0), (0, 1.2, 0), frame)))
    a = evaluate(build_ensemble(*map(list, zip(*perp))))["trajectory_alignment"]
    b = evaluate(build_ensemble(*map(list, zip(*inplane))))["trajectory_alignment"]
    assert a == pytest.approx(90.0, abs=1.0)
    assert b == pytest.approx(0.0, abs=1.0)


def test_missing_roles_give_none_not_zero():
    poses = [make_pose(np.array([3.0, 0, 0]), np.zeros(3), seed=i) for i in range(6)]
    sites = [site_at((3, 0, 0), (0, 0, 0), leaving=None, charge=None) for _ in range(6)]
    out = evaluate(build_ensemble(poses, sites))
    assert out["trajectory_angle_deviation"] is None
    assert out["ts_charge_stabilisation"] is None
    assert out["reactive_preorganisation"] is not None


# --------------------------------------------------------------------------- #
# resolver validation
# --------------------------------------------------------------------------- #
def test_check_resolver_accepts_a_sane_resolver():
    poses = [make_pose(np.array([3.0, 0, 0]), np.zeros(3), seed=i) for i in range(10)]
    rep = check_resolver(lambda p: site_at(p.ligands["C"][0][2], p.ligands["B"][0][2]), poses)
    assert rep.ok and rep.coverage == 1.0


def test_check_resolver_flags_self_match():
    poses = [make_pose(np.array([3.0, 0, 0]), np.zeros(3), seed=i) for i in range(10)]
    rep = check_resolver(lambda p: site_at(p.ligands["B"][0][2], p.ligands["B"][0][2]), poses)
    assert not rep.ok
    assert any("nucleophile == electrophile" in p for p in rep.problems)


def test_check_resolver_flags_low_coverage():
    poses = [make_pose(np.array([3.0, 0, 0]), np.zeros(3), seed=i) for i in range(10)]
    rep = check_resolver(
        lambda p: None if p.ca[0][0] > 0 else site_at((3, 0, 0), (0, 0, 0)), poses)
    if rep.coverage < 0.9:
        assert any("resolved only" in x for x in rep.problems)


def test_check_resolver_flags_inconsistent_ideal_angle():
    poses = [make_pose(np.array([3.0, 0, 0]), np.zeros(3), seed=i) for i in range(10)]
    angles = iter([180.0] * 5 + [107.0] * 5)
    rep = check_resolver(lambda p: site_at((3, 0, 0), (0, 0, 0), next(angles)), poses)
    assert any("ideal_angle is not constant" in x for x in rep.problems)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_auroc_matches_hand_computed_value():
    assert auroc(np.array([3.0, 2.0, 1.0]), np.array([1, 0, 0])) == pytest.approx(1.0)
    assert auroc(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 0])) == pytest.approx(0.0)
    assert auroc(np.array([1.0, 1.0]), np.array([1, 0])) == pytest.approx(0.5)


def test_robust_auroc_orients_itself():
    act = np.arange(30.0)
    assert robust_auroc(-act, act) == pytest.approx(robust_auroc(act, act))


def test_partial_spearman_removes_a_pure_confound():
    rng = np.random.default_rng(0)
    z = rng.normal(size=60)
    x = z + 0.01 * rng.normal(size=60)
    y = z + 0.01 * rng.normal(size=60)
    assert abs(partial_spearman(x, y, z)) < 0.5      # shared cause removed


def test_selection_null_is_above_half_and_grows_with_library_size():
    """The core statistical point: max-over-proxies of noise is biased upward."""
    rng = np.random.default_rng(0)
    act = rng.normal(size=40)
    small = selection_null(rng.normal(size=(3, 40)), act, n_perm=60, seed=1)
    large = selection_null(rng.normal(size=(60, 40)), act, n_perm=60, seed=1)
    assert np.median(small) > 0.5
    assert np.median(large) > np.median(small)


def test_screen_fails_a_direction_violating_proxy():
    rng = np.random.default_rng(0)
    act = rng.normal(size=40)
    # approach_distance declares "-"; hand it a perfectly POSITIVE correlate
    verdicts = screen({"approach_distance": act.copy()}, act, LIBRARY, n_perm=40)
    v = {x.proxy_id: x for x in verdicts}["approach_distance"]
    assert not v.direction_ok and not v.promoted


def test_screen_does_not_promote_pure_noise():
    rng = np.random.default_rng(3)
    act = rng.normal(size=45)
    scores = {p.id: rng.normal(size=45) for p in LIBRARY}
    assert not any(v.promoted for v in screen(scores, act, LIBRARY, n_perm=60))
