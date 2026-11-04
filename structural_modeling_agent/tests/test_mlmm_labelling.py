"""Endpoint labelling must be MEASURED from geometry, not trusted.

The bug these cover happened live. A chorismate-mutase run reproduced the
published transition state exactly (same saddle, potential-energy surface
agreeing to 0.01 kcal/mol with an independent run) but assigned its endpoints
backwards: prephenate sits ~31.6 kcal/mol below chorismate, so the lower minimum
was called "reactant". The run therefore reported 45.5 kcal/mol -- the REVERSE
barrier -- for a reaction whose forward barrier is 13.87.

Nothing about the chemistry was wrong. Only the label. So the gate must catch it
and the profile must be recoverable, not discarded.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from structural_modeling_agent.mlmm import (  # noqa: E402
    AtomRef, BondChange, NotReportable, ReactionProfile, ReactionStep, State,
    classify_endpoints, measure_driven_coordinates, read_pdb,
)
from structural_modeling_agent.mlmm.structure import StructureError  # noqa: E402

# The real CM Claisen, declared in the CHEMICAL FORWARD direction
# (chorismate -> prephenate): C1-C8 forms, C3-O1P breaks.
R = lambda n: AtomRef(n, resname="PRE", resseq=352)  # noqa: E731
CM_STEP = ReactionStep(
    step_id="claisen", nucleophile=R("C8"), electrophile=R("C1"),
    leaving_group=R("O1P"), proton_coupled=False,
    bond_changes=(BondChange("form", R("C1"), R("C8"), 1.56),
                  BondChange("break", R("C3"), R("O1P"), 3.81)),
)
# Measured from the real structures.
CHORISMATE = {"C1-C8": 3.24, "C3-O1P": 1.48}   # reactant
PREPHENATE = {"C1-C8": 1.56, "C3-O1P": 3.81}   # product


def _state(label, e, **kw):
    base = dict(kind="free_energy", n_imag=0, imaginary_cm=(), n_vib=9261,
                max_force_ev_per_a=0.004, n_atoms=208, net_charge=-1)
    base.update(kw)
    return State(label=label, energy_kcal=e, **base)


def cm_profile(*, swapped: bool, step=CM_STEP):
    """Real CM energies; ``swapped`` reproduces the engine's mistake."""
    chor = _state("chorismate", 0.0, driven_coordinates=CHORISMATE)
    prep = _state("prephenate", -31.59, driven_coordinates=PREPHENATE)
    ts = _state("TS", 13.87, n_imag=1, imaginary_cm=(269.4,), n_vib=9260,
                is_saddle=True, driven_coordinates={})
    r, p = (prep, chor) if swapped else (chor, prep)
    return ReactionProfile(
        system_id="CM_1COM", step_id="claisen",
        reactant=r, transition_state=ts, product=p, step=step,
        reaction_mode_overlap=0.85, endpoint_mechanism_ok=True,
    )


def _labelling(profile):
    return [c for c in profile.verdict().checks
            if c.name == "endpoint_labelling"][0]


# --------------------------------------------------------------------------- #
# the classifier
# --------------------------------------------------------------------------- #

def test_forming_bond_means_reactant_is_longer():
    v = classify_endpoints(CM_STEP, CHORISMATE, PREPHENATE)
    assert v.reactant_is == "first" and v.unanimous
    assert v.votes == {"C1-C8": "first", "C3-O1P": "first"}


def test_classifier_is_symmetric():
    v = classify_endpoints(CM_STEP, PREPHENATE, CHORISMATE)
    assert v.reactant_is == "second" and v.unanimous


def test_classifier_ignores_energy_entirely():
    """The product is the LOWER minimum here, so energy order says the opposite.

    classify_endpoints never receives an energy, which is the point.
    """
    v = classify_endpoints(CM_STEP, CHORISMATE, PREPHENATE)
    assert v.reactant_is == "first"   # chorismate, despite being 31.6 kcal HIGHER


def test_unchanged_coordinates_decide_nothing():
    v = classify_endpoints(CM_STEP, CHORISMATE, dict(CHORISMATE))
    assert v.reactant_is is None
    assert all(x is None for x in v.votes.values())
    assert "no coordinate changed" in v.detail


def test_disagreeing_coordinates_are_not_resolved_by_majority():
    """Both coordinates informative but pointing opposite ways: not a clean pair."""
    # C1-C8 says "first" (forms: reactant is longer). C3-O1P says "second"
    # (breaks: reactant is shorter, and 1.48 < 3.81).
    conflicting = {"C1-C8": 3.24, "C3-O1P": 3.81}
    other = {"C1-C8": 1.56, "C3-O1P": 1.48}
    v = classify_endpoints(CM_STEP, conflicting, other)
    assert v.reactant_is is None
    assert "DISAGREE" in v.detail
    assert set(v.votes.values()) == {"first", "second"}


def test_one_informative_coordinate_still_decides_the_label():
    """An unchanged coordinate abstains; a single clear one is enough to label."""
    partial = {"C1-C8": 1.56, "C3-O1P": 1.48}   # C-C formed, C-O unchanged
    v = classify_endpoints(CM_STEP, CHORISMATE, partial)
    assert v.reactant_is == "first"
    assert v.votes == {"C1-C8": "first", "C3-O1P": None}


def test_partial_concerted_reaction_is_caught_separately():
    """Labelling can still succeed while the reaction itself is incomplete."""
    import dataclasses
    prof = cm_profile(swapped=False)
    partial = dataclasses.replace(
        prof, product=dataclasses.replace(
            prof.product, driven_coordinates={"C1-C8": 1.56, "C3-O1P": 1.48}))
    checks = {c.name: c for c in partial.verdict().checks}
    assert checks["endpoint_labelling"].status == "PASS"
    assert checks["all_coordinates_changed"].status == "FAIL"
    assert "C3-O1P" in checks["all_coordinates_changed"].detail


def test_all_coordinates_changed_passes_on_a_clean_pair():
    checks = {c.name: c for c in cm_profile(swapped=False).verdict().checks}
    assert checks["all_coordinates_changed"].status == "PASS"
    assert checks["all_coordinates_changed"].value == 2


def test_missing_measurement_does_not_vote():
    v = classify_endpoints(CM_STEP, {"C1-C8": 3.24}, {"C1-C8": 1.56})
    assert v.reactant_is == "first"
    assert v.votes["C3-O1P"] is None


# --------------------------------------------------------------------------- #
# the gate check
# --------------------------------------------------------------------------- #

def test_correct_labelling_passes_and_barrier_is_forward():
    prof = cm_profile(swapped=False)
    c = _labelling(prof)
    assert c.status == "PASS" and c.value == "driven_coordinate"
    assert prof.transition_state.energy_kcal - prof.reactant.energy_kcal == \
        pytest.approx(13.87)


def test_swapped_labelling_fails_with_the_corrected_number():
    prof = cm_profile(swapped=True)
    c = _labelling(prof)
    assert c.status == "FAIL" and c.value == "swapped"
    assert "REVERSE barrier" in c.detail
    assert "13.87" in c.detail          # the corrected value is handed over
    assert "45.4" in c.detail or "45.5" in c.detail


def test_swapped_profile_refuses_to_report_a_barrier():
    with pytest.raises(NotReportable):
        cm_profile(swapped=True).barrier(strict=False)


def test_relabel_recovers_the_forward_barrier():
    fixed = cm_profile(swapped=True).relabel_from_geometry()
    assert fixed.transition_state.energy_kcal - fixed.reactant.energy_kcal == \
        pytest.approx(13.87)
    assert fixed.product.energy_kcal - fixed.reactant.energy_kcal == \
        pytest.approx(-31.59)
    assert _labelling(fixed).status == "PASS"
    assert fixed.provenance["relabelled"] == "endpoints swapped by geometry"


def test_relabel_is_a_noop_when_already_correct():
    prof = cm_profile(swapped=False)
    assert prof.relabel_from_geometry() is prof


def test_relabel_refuses_when_geometry_cannot_decide():
    prof = cm_profile(swapped=False)
    import dataclasses
    stuck = dataclasses.replace(
        prof, product=dataclasses.replace(prof.product,
                                          driven_coordinates=dict(CHORISMATE)))
    with pytest.raises(NotReportable, match="does not decide"):
        stuck.relabel_from_geometry()


def test_undecidable_labelling_is_a_failure_not_an_abstention():
    """A profile whose endpoints cannot be told apart must not be reportable."""
    import dataclasses
    prof = cm_profile(swapped=False)
    stuck = dataclasses.replace(
        prof, product=dataclasses.replace(prof.product,
                                          driven_coordinates=dict(CHORISMATE)))
    c = _labelling(stuck)
    assert c.status == "FAIL" and c.value == "undecidable"


def test_without_a_step_it_falls_back_to_the_recorded_flag():
    """No ReactionStep -> cannot measure -> abstain rather than assume."""
    prof = cm_profile(swapped=True, step=None)
    c = _labelling(prof)
    assert c.status == "ABSTAIN"


def test_scan_direction_trap_is_detectable():
    """Declaring the step in the SCAN direction inverts the answer.

    The CM protocol starts from the product crystal and scans backwards, so a
    spec written to mirror the scan declares 'break C1-C8 / form C3-O1P' and the
    classifier then calls prephenate the reactant -- self-consistently, and
    wrongly. The spec must state the CHEMICAL forward direction.
    """
    backwards = ReactionStep(
        step_id="claisen-reversed", nucleophile=R("C8"), electrophile=R("C1"),
        leaving_group=R("O1P"), proton_coupled=False,
        bond_changes=(BondChange("break", R("C1"), R("C8"), 3.0),
                      BondChange("form", R("C3"), R("O1P"), 1.43)),
    )
    fwd = classify_endpoints(CM_STEP, CHORISMATE, PREPHENATE)
    rev = classify_endpoints(backwards, CHORISMATE, PREPHENATE)
    assert fwd.reactant_is == "first"
    assert rev.reactant_is == "second"      # inverted, as declared
    assert fwd.unanimous and rev.unanimous  # both internally consistent


# --------------------------------------------------------------------------- #
# measuring coordinates off a structure
# --------------------------------------------------------------------------- #

def _pdb(tmp_path, coords, name="s.pdb"):
    """Minimal PRE 352 with C1, C8, C3, O1P at given positions."""
    p = tmp_path / name
    with open(p, "w") as fh:
        for i, (nm, xyz) in enumerate(coords.items(), 1):
            x, y, z = xyz
            fh.write(f"HETATM{i:>5d} {nm:<4s} PRE A 352    "
                     f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
                     f"{'':10s}{nm[0]:>2s}\n")
        fh.write("END\n")
    return str(p)


def test_measure_driven_coordinates_reads_the_structure(tmp_path):
    path = _pdb(tmp_path, {"C1": (0, 0, 0), "C8": (3.24, 0, 0),
                           "C3": (0, 5, 0), "O1P": (0, 6.48, 0)})
    got = measure_driven_coordinates(read_pdb(path), CM_STEP)
    assert got["C1-C8"] == pytest.approx(3.24)
    assert got["C3-O1P"] == pytest.approx(1.48)


def test_measure_raises_on_ambiguous_atom(tmp_path):
    """A trimer has three ligand copies; an unqualified ref must not pick one."""
    path = tmp_path / "trimer.pdb"
    with open(path, "w") as fh:
        for i, rs in enumerate((352, 353, 354), 1):
            fh.write(f"HETATM{i:>5d} C1   PRE A{rs:>4d}    "
                     f"{0.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
                     f"{'':10s} C\n")
        fh.write("END\n")
    unqualified = ReactionStep(
        step_id="x", nucleophile=AtomRef("C1"), electrophile=AtomRef("C1"),
        bond_changes=(BondChange("form", AtomRef("C1", resname="PRE"),
                                 AtomRef("C1", resname="PRE"), 1.5),),
        proton_coupled=False,
    )
    with pytest.raises(StructureError, match="ambiguous"):
        measure_driven_coordinates(read_pdb(str(path)), unqualified)


def test_bond_change_key_and_direction():
    form, brk = CM_STEP.bond_changes
    assert form.key == "C1-C8" and brk.key == "C3-O1P"
    assert form.reactant_is_longer is True
    assert brk.reactant_is_longer is False
