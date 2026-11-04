"""Reading real engine output must not inherit the defects in it.

Fixtures reproduce the real file formats exactly: ``thermoanalysis.yaml`` flat
hartree keys, ``frequencies_cm-1.txt`` signed wavenumbers, and the cluster-route
JSON schema.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from structural_modeling_agent.mlmm import (  # noqa: E402
    IngestError, profile_from_cluster_json, profile_from_freq_dirs,
    read_frequencies, read_thermoanalysis, recorded_vs_recounted,
    state_from_freq_dir,
)
from structural_modeling_agent.mlmm.ingest import _mode_counts  # noqa: E402

HA = 627.5094740631


def write_freq_state(root, name, *, e_ha, freqs, g_offset_ha=0.0):
    """One mlmm freq output directory, in the real format."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    n_imag = sum(1 for f in freqs if f < 0)
    with open(os.path.join(d, "thermoanalysis.yaml"), "w") as fh:
        fh.write(f"temperature_K: 298.15\npressure_atm: 1.0\n")
        fh.write(f"num_imag_freq: {n_imag}\n")
        fh.write(f"electronic_energy_ha: {e_ha!r}\n")
        fh.write(f"zpe_correction_ha: 0.3475584007323158\n")
        fh.write(f"sum_EE_and_ZPE_ha: {e_ha + 0.3475584!r}\n")
        fh.write(f"sum_EE_and_thermal_enthalpy_ha: {e_ha + 0.3764405!r}\n")
        fh.write(f"sum_EE_and_thermal_free_energy_ha: "
                 f"{e_ha + 0.2768781 + g_offset_ha!r}\n")
        fh.write("S_cal_per_mol_K: 209.54664641283438\n")
    with open(os.path.join(d, "frequencies_cm-1.txt"), "w") as fh:
        for i, f in enumerate(freqs, 1):
            fh.write(f"{i:>4d}   {f:>12.4f}\n")
    return d


# --------------------------------------------------------------------------- #
# low-level readers
# --------------------------------------------------------------------------- #

def test_thermoanalysis_reads_flat_hartree_keys(tmp_path):
    d = write_freq_state(str(tmp_path), "ts_freq", e_ha=-38462.402095151425,
                         freqs=[-286.3, 78.1, 87.9])
    t = read_thermoanalysis(os.path.join(d, "thermoanalysis.yaml"))
    assert t["electronic_energy_ha"] == pytest.approx(-38462.402095151425)
    assert t["num_imag_freq"] == 1
    assert "temperature_K" in t


def test_thermoanalysis_missing_file_raises(tmp_path):
    with pytest.raises(IngestError, match="no thermoanalysis.yaml"):
        read_thermoanalysis(str(tmp_path / "nope.yaml"))


def test_frequencies_parse_signed(tmp_path):
    d = write_freq_state(str(tmp_path), "f", e_ha=-1.0,
                         freqs=[-286.3005, 78.1314, 3456.0])
    assert read_frequencies(os.path.join(d, "frequencies_cm-1.txt")) == \
        pytest.approx([-286.3005, 78.1314, 3456.0])


# --------------------------------------------------------------------------- #
# the mode-count bookkeeping bug this module had, as a regression test
# --------------------------------------------------------------------------- #

def test_n_vib_counts_every_non_imaginary_mode():
    """Regression: n_vib must be len(freqs) - n_imag.

    Counting only modes above +threshold dropped legitimate low-frequency
    vibrations and manufactured an apparent two-mode gap on real data whose
    Hessian subspaces were identical.
    """
    freqs = [-286.0, -22.8, 15.0, 30.0, 400.0]   # one real imaginary, one soft
    n_imag, mags, n_vib = _mode_counts(freqs, threshold_cm=50.0)
    assert n_imag == 1 and mags == (286.0,)
    assert n_vib == 4                             # 5 modes - 1 imaginary
    assert n_imag + n_vib == len(freqs)           # nothing may vanish


def test_sub_threshold_negative_joins_the_real_set():
    n_imag, _, n_vib = _mode_counts([-22.8, 100.0], threshold_cm=50.0)
    assert n_imag == 0 and n_vib == 2


def test_ts_and_minimum_differ_by_exactly_one_mode(tmp_path):
    """Real pattern: equal totals, TS has one imaginary, reactant none."""
    root = str(tmp_path)
    write_freq_state(root, "rx_freq", e_ha=-38462.41,
                     freqs=[-15.9] + [100.0 + i for i in range(169)])
    write_freq_state(root, "ts_freq", e_ha=-38462.40,
                     freqs=[-296.0, -22.8] + [100.0 + i for i in range(168)])
    prof = profile_from_freq_dirs(root, system_id="DesD", step_id="adenylation",
                                  kind="free_energy", net_charge=-3)
    ts, rx = prof.transition_state, prof.reactant
    assert (ts.n_imag, rx.n_imag) == (1, 0)
    assert ts.n_vib == rx.n_vib - 1
    mc = [c for c in prof.verdict().checks
          if c.name == "mode_count_consistency"][0]
    # The inferred product has no frequency calculation. Correct rx/TS counts
    # cannot silently supply the missing minimum's required evidence.
    assert mc.status == "ABSTAIN"


# --------------------------------------------------------------------------- #
# recorded counts are NOT inherited
# --------------------------------------------------------------------------- #

def test_recorded_imaginary_count_is_not_trusted(tmp_path):
    """The file says num_imag_freq: 3; two of them are sub-threshold noise."""
    d = write_freq_state(str(tmp_path), "ts_freq", e_ha=-100.0,
                         freqs=[-296.0, -22.8, -15.9, 100.0, 200.0])
    recorded = read_thermoanalysis(os.path.join(d, "thermoanalysis.yaml"))
    assert recorded["num_imag_freq"] == 3
    state = state_from_freq_dir(d, "TS")
    assert state.n_imag == 1          # recounted on one threshold
    assert state.imaginary_cm == (296.0,)


def test_absent_spectrum_leaves_count_unknown_not_assumed(tmp_path):
    """No frequencies file -> abstain, because the recorded count's threshold
    is unknown."""
    d = write_freq_state(str(tmp_path), "ts_freq", e_ha=-100.0, freqs=[-296.0])
    os.remove(os.path.join(d, "frequencies_cm-1.txt"))
    state = state_from_freq_dir(d, "TS")
    assert state.n_imag is None and state.n_vib is None


# --------------------------------------------------------------------------- #
# energy kind and units
# --------------------------------------------------------------------------- #

def test_energy_kind_selects_the_right_field_and_converts(tmp_path):
    d = write_freq_state(str(tmp_path), "ts_freq", e_ha=-100.0, freqs=[-296.0])
    pot = state_from_freq_dir(d, "TS", kind="potential")
    gib = state_from_freq_dir(d, "TS", kind="free_energy")
    assert pot.energy_kcal == pytest.approx(-100.0 * HA)
    assert gib.energy_kcal == pytest.approx((-100.0 + 0.2768781) * HA)
    assert pot.kind == "potential" and gib.kind == "free_energy"


def test_unknown_energy_kind_raises(tmp_path):
    d = write_freq_state(str(tmp_path), "ts_freq", e_ha=-100.0, freqs=[-296.0])
    with pytest.raises(IngestError, match="unknown energy kind"):
        state_from_freq_dir(d, "TS", kind="gibbs_free")


def test_barrier_matches_the_repo_formula(tmp_path):
    """dG = (G_ts - G_rx) * 627.5094740631, the engine's own extraction."""
    root = str(tmp_path)
    write_freq_state(root, "rx_freq", e_ha=-38462.40, freqs=[100.0, 200.0])
    write_freq_state(root, "ts_freq", e_ha=-38462.40, freqs=[-296.0, 200.0],
                     g_offset_ha=0.0145)
    prof = profile_from_freq_dirs(root, system_id="x", step_id="y")
    dG = prof.transition_state.energy_kcal - prof.reactant.energy_kcal
    assert dG == pytest.approx(0.0145 * HA, rel=1e-9)


# --------------------------------------------------------------------------- #
# a missing product must not be papered over
# --------------------------------------------------------------------------- #

def test_absent_product_blocks_the_profile(tmp_path):
    root = str(tmp_path)
    write_freq_state(root, "rx_freq", e_ha=-100.0, freqs=[100.0, 200.0])
    write_freq_state(root, "ts_freq", e_ha=-99.98, freqs=[-296.0, 200.0])
    prof = profile_from_freq_dirs(root, system_id="x", step_id="y")
    assert prof.product.label == "product(absent)"
    report = prof.verdict()
    assert not report.ok
    names = {c.name: c.status for c in report.checks}
    assert names["endpoints_distinct"] == "ABSTAIN"


# --------------------------------------------------------------------------- #
# cluster-route JSON
# --------------------------------------------------------------------------- #

def _cluster_json(tmp_path, **over):
    d = dict(
        pose="Hvol_WT", charge=-4, n_total=117, n_free=46,
        e_reactant_eV=-100.0, e_ts_eV=-99.0, e_product_eV=-100.5,
        n_imag=1, n_imag_raw=2, imag_cm=[162.4], imag_cm_all=[162.4, 28.1],
        ts_valid=True,
        reactant_geom={"nuc_p": 3.47}, product_geom={"nuc_p": 1.65},
        reactant_mechanism_ok=True, product_mechanism_ok=True,
        identity_swapped=False,
    )
    d.update(over)
    p = tmp_path / "barrier_result_neb_k1.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_cluster_json_uses_the_full_spectrum(tmp_path):
    """imag_cm_all is preferred over the filtered imag_cm."""
    prof = profile_from_cluster_json(_cluster_json(tmp_path), kind="potential")
    assert prof.provenance["spectrum_field"] == "imag_cm_all"
    assert prof.transition_state.n_imag == 1      # 28.1 is sub-threshold
    assert prof.transition_state.imaginary_cm == (162.4,)


def test_cluster_json_reconstructs_the_reference(tmp_path):
    prof = profile_from_cluster_json(_cluster_json(tmp_path), kind="potential")
    assert prof.reaction_energy(strict=False) == pytest.approx(
        -0.5 * 23.060547830619026)


def test_cluster_json_reads_driven_coordinates(tmp_path):
    prof = profile_from_cluster_json(_cluster_json(tmp_path), kind="potential")
    c = [x for x in prof.verdict().checks if x.name == "endpoints_distinct"][0]
    assert c.status == "PASS"
    assert c.value == pytest.approx(1.82, abs=0.01)


def test_cluster_json_detects_unreacted_endpoints(tmp_path):
    """The recorded 1.4159 -> 1.4158 A 'reaction'."""
    p = _cluster_json(tmp_path, reactant_geom={"N2-O1": 1.4159},
                      product_geom={"N2-O1": 1.4158})
    prof = profile_from_cluster_json(p, kind="potential")
    c = [x for x in prof.verdict().checks if x.name == "endpoints_distinct"][0]
    assert c.status == "FAIL"


def test_missing_identity_swap_is_reported_as_energy_labelling(tmp_path):
    """No evidence of geometry-based labelling must not be read as fine."""
    d = json.loads(open(_cluster_json(tmp_path)).read())
    d.pop("identity_swapped")
    p = tmp_path / "no_swap.json"
    p.write_text(json.dumps(d))
    prof = profile_from_cluster_json(str(p), kind="potential")
    assert prof.labelled_by == "energy"
    c = [x for x in prof.verdict().checks if x.name == "endpoint_labelling"][0]
    assert c.status == "FAIL"


def test_mechanism_flags_are_carried(tmp_path):
    p = _cluster_json(tmp_path, product_mechanism_ok=False)
    prof = profile_from_cluster_json(p, kind="potential")
    assert prof.endpoint_mechanism_ok is False
    c = [x for x in prof.verdict().checks if x.name == "endpoint_mechanism"][0]
    assert c.status == "FAIL"


def test_sella_fallback_guess_marks_non_stationary(tmp_path):
    """Keeping the unrefined guess means the 'TS' is not a stationary point."""
    p = _cluster_json(tmp_path, sella_fallback="guess")
    prof = profile_from_cluster_json(p, kind="potential")
    assert prof.transition_state.reseeded is True
    c = [x for x in prof.verdict().checks if x.name == "stationary:TS"][0]
    assert c.status == "FAIL"


def test_charge_and_atom_count_are_carried(tmp_path):
    prof = profile_from_cluster_json(_cluster_json(tmp_path), kind="potential")
    for s in prof.states:
        assert s.net_charge == -4 and s.n_atoms == 117


def test_requesting_free_energy_without_gibbs_fields_raises(tmp_path):
    with pytest.raises(IngestError, match="no Gibbs fields"):
        profile_from_cluster_json(_cluster_json(tmp_path), kind="free_energy")


def test_gibbs_fields_are_read_as_relative(tmp_path):
    p = _cluster_json(tmp_path, dG_act=9.11, dG_rxn=-33.2)
    prof = profile_from_cluster_json(p, kind="free_energy")
    assert prof.reactant.energy_kcal == 0.0
    assert prof.transition_state.energy_kcal == pytest.approx(9.11)
    assert prof.product.energy_kcal == pytest.approx(-33.2)


def test_recorded_vs_recounted_flags_self_inconsistency(tmp_path):
    """n_imag 1 vs n_imag_raw 2 with ts_valid true: the file disagrees with itself."""
    rv = recorded_vs_recounted(_cluster_json(tmp_path))
    assert rv["recorded_n_imag"] == 1 and rv["recorded_n_imag_raw"] == 2
    assert rv["recounted_n_imag"] == 1
    assert rv["self_consistent"] is False


def test_recorded_vs_recounted_agrees_when_consistent(tmp_path):
    p = _cluster_json(tmp_path, n_imag=1, n_imag_raw=1, imag_cm_all=[162.4])
    rv = recorded_vs_recounted(p)
    assert rv["self_consistent"] is True


def test_string_frequencies_are_tolerated(tmp_path):
    p = _cluster_json(tmp_path, imag_cm_all=["839.05i", "28.09i"])
    prof = profile_from_cluster_json(p, kind="potential")
    assert prof.transition_state.imaginary_cm == (839.05,)
