"""The output gate must reject the exact numbers that were previously reported.

Every case below is built from a real recorded run, with its real values. A gate
that would have passed any of these is not worth having, so each test names the
file it came from.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from structural_modeling_agent.mlmm import (  # noqa: E402
    NotReportable, ReactionProfile, State, diagram_values,
    imaginary_from_frequencies, kcal, mode_overlap,
)


def _state(label, e, **kw):
    """A clean, fully-recorded state; tests override only what they probe."""
    base = dict(
        kind="free_energy", n_imag=0, imaginary_cm=(), n_vib=162,
        max_force_ev_per_a=0.005, n_atoms=204, net_charge=-2,
        driven_coordinates={"C-O": 1.43},
    )
    base.update(kw)
    return State(label=label, energy_kcal=e, **base)


def clean_profile(**kw):
    """A profile that SHOULD be reportable, so the negative tests mean something."""
    r = _state("reactant", 0.0, driven_coordinates={"C-O": 1.43})
    ts = _state("TS", 15.0, n_imag=1, imaginary_cm=(286.0,), n_vib=161,
                is_saddle=True, driven_coordinates={"C-O": 1.95})
    p = _state("product", -32.0, driven_coordinates={"C-O": 2.80})
    defaults = dict(
        system_id="CM", step_id="claisen", reactant=r, transition_state=ts,
        product=p, reaction_mode_overlap=0.82, labelled_by="driven_coordinate",
        endpoint_mechanism_ok=True,
    )
    defaults.update(kw)
    return ReactionProfile(**defaults)


def _check(report, name):
    hits = [c for c in report.checks if c.name == name]
    assert hits, f"no check named {name!r}; have {[c.name for c in report.checks]}"
    return hits[0]


# --------------------------------------------------------------------------- #
# the clean case must pass, or every rejection below is vacuous
# --------------------------------------------------------------------------- #

def test_clean_profile_is_reportable():
    prof = clean_profile()
    report = prof.verdict()
    assert report.ok, report.render()
    assert prof.barrier() == pytest.approx(15.0)
    assert prof.reaction_energy() == pytest.approx(-32.0)


# --------------------------------------------------------------------------- #
# the recorded "agreement with experiment" that was a minimum
# CM_2CHT_eff/uma-s-1p2/result_k1.json: dG_act 15.27, exp 15.4, n_imag_ts 0
# --------------------------------------------------------------------------- #

def test_zero_imaginary_modes_is_not_a_transition_state():
    ts = _state("TS", 15.27, n_imag=0, imaginary_cm=(), n_vib=162,
                driven_coordinates={"C-O": 1.426})
    prof = clean_profile(transition_state=ts)
    report = prof.verdict()
    assert not report.ok
    c = _check(report, "ts_single_imaginary")
    assert c.status == "FAIL"
    assert "minimum, not a" in c.detail
    with pytest.raises(NotReportable):
        prof.barrier()


def test_equal_mode_counts_prove_no_saddle():
    """The recorded 162/162/162/162 across reactant, TS and product."""
    prof = clean_profile(
        reactant=_state("reactant", 0.0, n_vib=162),
        transition_state=_state("TS", 15.27, n_imag=1, imaginary_cm=(286.0,),
                                n_vib=162, driven_coordinates={"C-O": 1.95}),
        product=_state("product", 11.6, n_vib=162,
                       driven_coordinates={"C-O": 2.8}),
    )
    report = prof.verdict()
    assert not report.ok
    c = _check(report, "mode_count_consistency")
    assert c.status == "FAIL"
    assert "one fewer real mode" in c.detail


def test_product_identical_to_reactant_is_not_a_reaction():
    """Recorded: breaking bond 1.4250 -> 1.4260 A, i.e. 0.001 A of 'reaction'."""
    prof = clean_profile(
        reactant=_state("reactant", 0.0, driven_coordinates={"Cbr-Ob": 1.4250}),
        product=_state("product", 0.03, driven_coordinates={"Cbr-Ob": 1.4260}),
    )
    report = prof.verdict()
    assert not report.ok
    c = _check(report, "endpoints_distinct")
    assert c.status == "FAIL"
    assert c.value == pytest.approx(0.001, abs=1e-6)


def test_reseeded_reactant_reference_is_rejected():
    """10.6 of the reported 15.3 kcal/mol came from re-referencing the reactant."""
    prof = clean_profile(reactant=_state("reactant", 0.0, reseeded=True))
    report = prof.verdict()
    assert not report.ok
    c = _check(report, "stationary:reactant")
    assert c.status == "FAIL"
    assert "re-seeded" in c.detail


def test_spurious_off_pathway_mode_is_rejected():
    """Recorded overlap 0.04 -- a clean single imaginary mode, wrong coordinate."""
    prof = clean_profile(reaction_mode_overlap=0.04)
    report = prof.verdict()
    assert not report.ok
    c = _check(report, "reaction_mode_overlap")
    assert c.status == "FAIL"
    assert "off-pathway" in c.detail


# --------------------------------------------------------------------------- #
# energy-based endpoint labelling -> the reverse barrier
# common/ts_sella.py:294
# --------------------------------------------------------------------------- #

def test_energy_based_labelling_is_rejected():
    prof = clean_profile(labelled_by="energy")
    report = prof.verdict()
    assert not report.ok
    c = _check(report, "endpoint_labelling")
    assert c.status == "FAIL"
    assert "REVERSE barrier" in c.detail


def test_unrecorded_labelling_abstains():
    report = clean_profile(labelled_by=None).verdict()
    assert not report.ok
    assert _check(report, "endpoint_labelling").status == "ABSTAIN"


# --------------------------------------------------------------------------- #
# non-stationary geometries (optimizer hit its cap)
# cm_eff_final.log:413 -- Sella stopped at step 400, fmax 0.0442 vs target 0.03
# --------------------------------------------------------------------------- #

def test_unconverged_geometry_blocks_thermochemistry():
    ts = _state("TS", 15.0, n_imag=1, imaginary_cm=(286.0,), n_vib=161,
                max_force_ev_per_a=0.0442, driven_coordinates={"C-O": 1.95})
    report = clean_profile(transition_state=ts).verdict()
    assert not report.ok
    c = _check(report, "stationary:TS")
    assert c.status == "FAIL"
    assert c.value == pytest.approx(0.0442)


def test_missing_force_record_abstains_not_passes():
    ts = _state("TS", 15.0, n_imag=1, imaginary_cm=(286.0,), n_vib=161,
                max_force_ev_per_a=None, driven_coordinates={"C-O": 1.95})
    report = clean_profile(transition_state=ts).verdict()
    assert not report.ok
    assert _check(report, "stationary:TS").status == "ABSTAIN"
    # non-strict triage still lets it through, explicitly
    assert clean_profile(transition_state=ts).verdict(strict=False).ok


# --------------------------------------------------------------------------- #
# one threshold, one count
# cm_dg_k2.json carried ts_valid true AND n_imag_ts 2 in the same file
# --------------------------------------------------------------------------- #

def test_two_imaginary_modes_cannot_be_valid():
    ts = _state("TS", 15.0, n_imag=2, imaginary_cm=(286.0, 32.0), n_vib=161,
                driven_coordinates={"C-O": 1.95})
    report = clean_profile(transition_state=ts).verdict()
    assert not report.ok
    assert _check(report, "ts_single_imaginary").status == "FAIL"


def test_soft_mode_below_threshold_does_not_count():
    """A 32 cm-1 libration is noise, not a second reaction coordinate.

    Measured: a pure rigid-body rotation moved a near-zero mode from 0.2i to
    27.3i, so counting at ~0 is orientation-dependent.
    """
    n, mags = imaginary_from_frequencies([-286.0, -32.0, 100.0, 500.0])
    assert n == 1 and mags == (286.0,)
    n2, _ = imaginary_from_frequencies([complex(0, 286.0), complex(0, 32.0)])
    assert n2 == 1


def test_imaginary_counting_handles_both_conventions():
    neg_real = imaginary_from_frequencies([-286.0, 768.0])
    complexes = imaginary_from_frequencies([complex(0, 286.0), complex(768.0, 0)])
    assert neg_real == complexes == (1, (286.0,))


def test_higher_order_saddle_is_rejected():
    """MPD Hvol nimg21: n_imag 4 = [839i, 88i, 87i, 75i], written out anyway."""
    ts = _state("TS", 16.377, n_imag=4,
                imaginary_cm=(839.1, 87.8, 86.9, 74.6), n_vib=161,
                driven_coordinates={"O-PG": 1.9})
    report = clean_profile(transition_state=ts).verdict()
    assert not report.ok
    c = _check(report, "ts_single_imaginary")
    assert c.status == "FAIL"
    assert "higher-order" in c.detail


def test_endpoint_with_imaginary_mode_is_rejected():
    """A reactant that is not a minimum is not a valid reference."""
    prof = clean_profile(
        reactant=_state("reactant", 0.0, n_imag=1, imaginary_cm=(15.9,)),
    )
    # 15.9 cm-1 is below threshold -> noise, must NOT fail
    assert _check(prof.verdict(), "minimum_clean:reactant").status == "PASS"
    prof2 = clean_profile(
        reactant=_state("reactant", 0.0, n_imag=1, imaginary_cm=(295.0,)),
    )
    assert _check(prof2.verdict(), "minimum_clean:reactant").status == "FAIL"


# --------------------------------------------------------------------------- #
# reference-state comparability
# --------------------------------------------------------------------------- #

def test_atom_count_change_breaks_comparability():
    prof = clean_profile(product=_state("product", -32.0, n_atoms=205))
    report = prof.verdict()
    assert not report.ok
    assert _check(report, "composition_conserved").status == "FAIL"


def test_charge_change_breaks_comparability():
    prof = clean_profile(product=_state("product", -32.0, net_charge=-1))
    report = prof.verdict()
    assert not report.ok
    assert _check(report, "charge_conserved").status == "FAIL"


def test_mixed_energy_kinds_rejected():
    prof = clean_profile(product=_state("product", -32.0, kind="potential"))
    report = prof.verdict()
    assert not report.ok
    assert _check(report, "energy_kind").status == "FAIL"


# --------------------------------------------------------------------------- #
# what the number may be compared to
# --------------------------------------------------------------------------- #

def test_bare_potential_energy_is_not_comparable_to_kcat():
    prof = clean_profile(
        reactant=_state("reactant", 0.0, kind="potential"),
        transition_state=_state("TS", 15.0, kind="potential", n_imag=1,
                                imaginary_cm=(286.0,), n_vib=161,
                                driven_coordinates={"C-O": 1.95}),
        product=_state("product", -32.0, kind="potential",
                       driven_coordinates={"C-O": 2.8}),
    )
    assert prof.verdict().ok, prof.verdict().render()
    ok, why = prof.comparable_to_experiment()
    assert ok is False and "kcat" in why
    assert prof.label_for_plot() == "DeltaE (kcal/mol)"


def test_free_energy_is_comparable():
    prof = clean_profile()
    ok, _ = prof.comparable_to_experiment()
    assert ok is True
    assert prof.label_for_plot() == "DeltaG (kcal/mol)"


# --------------------------------------------------------------------------- #
# the plot path cannot bypass the verdict
# --------------------------------------------------------------------------- #

def test_diagram_refuses_an_unreportable_profile():
    bad = clean_profile(transition_state=_state(
        "TS", 15.27, n_imag=0, n_vib=162, driven_coordinates={"C-O": 1.95}))
    with pytest.raises(NotReportable):
        diagram_values(bad)


def test_diagram_values_are_relative_to_the_reactant():
    vals = diagram_values(clean_profile())
    assert vals == [0.0, 15.0, -32.0]


def test_to_dict_records_the_verdict_and_is_serialisable():
    import json
    d = clean_profile().to_dict()
    assert d["reportable"] is True
    assert d["barrier_kcal"] == pytest.approx(15.0)
    assert d["energy_kind"] == "free_energy"
    json.dumps(d)  # must not raise -- a numpy bool here crashed a real run


def test_to_dict_on_a_bad_profile_says_so_without_raising():
    d = clean_profile(labelled_by="energy").to_dict()
    assert d["reportable"] is False
    assert any(c["status"] == "FAIL" for c in d["verdict"]["checks"])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def test_unit_conversions_are_explicit():
    assert kcal(1.0, "hartree") == pytest.approx(627.5094740631)
    assert kcal(1.0, "eV") == pytest.approx(23.0605478, abs=1e-5)
    assert kcal(4.184, "kJ/mol") == pytest.approx(1.0)
    assert kcal(2.5, "kcal/mol") == 2.5
    with pytest.raises(ValueError, match="unknown energy unit"):
        kcal(1.0, "rydberg")


def test_mode_overlap_is_a_cosine_magnitude():
    assert mode_overlap([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert mode_overlap([1, 0, 0], [-1, 0, 0]) == pytest.approx(1.0)  # sign-free
    assert mode_overlap([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="length mismatch"):
        mode_overlap([1, 0], [1, 0, 0])
    with pytest.raises(ValueError, match="zero-length"):
        mode_overlap([0, 0, 0], [1, 0, 0])


def test_missing_transition_state_is_not_a_barrier():
    prof = ReactionProfile(
        system_id="x", step_id="y",
        reactant=_state("reactant", 0.0),
        product=_state("product", -32.0, driven_coordinates={"C-O": 2.8}),
        labelled_by="driven_coordinate",
    )
    report = prof.verdict()
    assert not report.ok
    assert _check(report, "ts_present").status == "FAIL"
    with pytest.raises(NotReportable):
        prof.barrier()
