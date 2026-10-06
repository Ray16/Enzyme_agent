"""Read real engine output into a `ReactionProfile` so the verdict can judge it.

Two output shapes exist in practice and both are supported:

* **ONIOM ML/MM route** -- one directory per state, each holding
  ``thermoanalysis.yaml`` (energies in hartree) and ``frequencies_cm-1.txt``
  (signed wavenumbers, negative = imaginary). Produced by ``mlmm freq``.
* **frozen-boundary cluster route** -- a single ``barrier_result_*.json`` /
  ``*_dg_k*.json`` carrying energies, the imaginary spectrum, endpoint geometries
  and the mechanism flags.

Two deliberate choices, both about not inheriting a defect:

**Imaginary modes are recounted from the raw frequencies, never read from the
recorded count.** The engines disagreed with themselves: one path counted modes
above 50 cm-1 and another counted anything above 1e-6, so a single run could
report ``ts_valid: true`` next to ``n_imag_ts: 2``. Recounting on one threshold
makes that impossible. Where a file preserves the full spectrum
(``imag_cm_all``) that is used in preference to the filtered list.

**Endpoint labelling is read, not assumed.** A file that shows no evidence of
geometry-based labelling is reported as ``labelled_by="energy"``, which the
verdict fails -- because labelling the lower minimum "reactant" reports the
reverse barrier for every exothermic step. Guessing "probably fine" here is how
that bug survived.

Nothing in this module computes chemistry. It parses, converts units explicitly,
and refuses to invent a field it cannot find.
"""
from __future__ import annotations

import json
import os
import re

from .verdict import (
    IMAGINARY_THRESHOLD_CM, ReactionProfile, State, imaginary_from_frequencies,
    kcal,
)

# thermoanalysis.yaml keys, per energy kind. Picking one kind for the whole
# profile is enforced by the verdict; this maps the choice onto the file.
THERMO_ENERGY_KEYS = {
    "potential": "electronic_energy_ha",
    "zpe_corrected": "sum_EE_and_ZPE_ha",
    "enthalpy": "sum_EE_and_thermal_enthalpy_ha",
    "free_energy": "sum_EE_and_thermal_free_energy_ha",
}


class IngestError(ValueError):
    """Raised when an output file cannot be read into a judgeable state."""


# --------------------------------------------------------------------------- #
# low-level readers
# --------------------------------------------------------------------------- #

def read_thermoanalysis(path: str) -> dict[str, float]:
    """Parse ``thermoanalysis.yaml`` -- flat ``key: float`` pairs.

    Intentionally not a YAML dependency: the file is flat, and a parser that
    silently coerces types is a liability on a file whose numbers decide a
    barrier. Non-numeric values are skipped, matching the engine's own reader.
    """
    if not os.path.isfile(path):
        raise IngestError(f"no thermoanalysis.yaml at {path}")
    out: dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, _, val = line.partition(":")
            try:
                out[key.strip()] = float(val.strip())
            except ValueError:
                continue
    if not out:
        raise IngestError(f"{path}: no numeric entries")
    return out


def read_frequencies(path: str) -> list[float]:
    """Parse ``frequencies_cm-1.txt`` -> signed wavenumbers, file order.

    Format is ``<index> <signed wavenumber>`` per line; negative means
    imaginary.
    """
    if not os.path.isfile(path):
        raise IngestError(f"no frequencies file at {path}")
    freqs: list[float] = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                freqs.append(float(parts[1]))
            except ValueError:
                continue
    if not freqs:
        raise IngestError(f"{path}: no frequencies parsed")
    return freqs


def _mode_counts(freqs: list[float], *, threshold_cm: float
                 ) -> tuple[int, tuple[float, ...], int]:
    """(n_imag, imaginary magnitudes, n_real_vibrational) on ONE threshold.

    A mode whose magnitude is below the threshold is counted as neither
    imaginary nor a real vibration: it is noise. A measured rigid-body rotation
    moved such a mode from 0.2i to 27.3i cm-1, so assigning it either way makes
    the result orientation-dependent.
    """
    n_imag, mags = imaginary_from_frequencies(freqs, threshold_cm=threshold_cm)
    n_vib = sum(1 for f in freqs if f >= threshold_cm)
    return n_imag, mags, n_vib


# --------------------------------------------------------------------------- #
# ONIOM ML/MM route: one directory per state
# --------------------------------------------------------------------------- #

def state_from_freq_dir(directory: str, label: str, *,
                        kind: str = "free_energy",
                        n_atoms: int | None = None,
                        net_charge: int | None = None,
                        max_force_ev_per_a: float | None = None,
                        driven_coordinates: dict[str, float] | None = None,
                        reseeded: bool = False,
                        threshold_cm: float = IMAGINARY_THRESHOLD_CM,
                        ) -> State:
    """One `State` from an ``mlmm freq`` output directory."""
    if kind not in THERMO_ENERGY_KEYS:
        raise IngestError(
            f"unknown energy kind {kind!r}; expected one of "
            f"{sorted(THERMO_ENERGY_KEYS)}"
        )
    thermo = read_thermoanalysis(os.path.join(directory, "thermoanalysis.yaml"))
    key = THERMO_ENERGY_KEYS[kind]
    if key not in thermo:
        raise IngestError(
            f"{directory}: thermoanalysis.yaml has no {key!r}; cannot report "
            f"{kind}. Available: {sorted(thermo)}"
        )
    energy = kcal(thermo[key], "hartree")

    freq_path = os.path.join(directory, "frequencies_cm-1.txt")
    if os.path.isfile(freq_path):
        freqs = read_frequencies(freq_path)
        n_imag, mags, n_vib = _mode_counts(freqs, threshold_cm=threshold_cm)
    else:
        # Without the spectrum we cannot recount, and the recorded count was
        # produced on an unknown threshold. Leave it unknown so the verdict
        # abstains rather than trusting it.
        n_imag, mags, n_vib = None, (), None

    return State(
        label=label, energy_kcal=energy, kind=kind,
        n_imag=n_imag, imaginary_cm=mags, n_vib=n_vib,
        max_force_ev_per_a=max_force_ev_per_a,
        n_atoms=n_atoms, net_charge=net_charge, reseeded=reseeded,
        is_saddle=(n_imag == 1),
        driven_coordinates=dict(driven_coordinates or {}),
    )


def profile_from_freq_dirs(results_dir: str, *, system_id: str, step_id: str,
                           ts_dir: str = "ts_freq", reactant_dir: str = "rx_freq",
                           product_dir: str | None = None,
                           kind: str = "free_energy",
                           net_charge: int | None = None,
                           n_atoms: int | None = None,
                           reaction_mode_overlap: float | None = None,
                           labelled_by: str | None = None,
                           endpoint_mechanism_ok: bool | None = None,
                           driven_coordinates: dict[str, dict[str, float]] | None = None,
                           ) -> ReactionProfile:
    """A `ReactionProfile` from an ``mlmm`` results directory.

    ``product_dir`` is optional because the recorded runs of this shape
    frequently computed only the TS and reactant -- no product, hence no reaction
    energy and no way to confirm the endpoints differ. That absence is carried
    into the profile honestly: the product falls back to the reactant and the
    verdict's endpoint check then fails, which is the correct outcome for a
    "barrier" with no product state.
    """
    coords = driven_coordinates or {}
    reactant = state_from_freq_dir(
        os.path.join(results_dir, reactant_dir), "reactant", kind=kind,
        net_charge=net_charge, n_atoms=n_atoms,
        driven_coordinates=coords.get("reactant"),
    )
    ts = state_from_freq_dir(
        os.path.join(results_dir, ts_dir), "TS", kind=kind,
        net_charge=net_charge, n_atoms=n_atoms,
        driven_coordinates=coords.get("TS"),
    )
    if product_dir and os.path.isdir(os.path.join(results_dir, product_dir)):
        product = state_from_freq_dir(
            os.path.join(results_dir, product_dir), "product", kind=kind,
            net_charge=net_charge, n_atoms=n_atoms,
            driven_coordinates=coords.get("product"),
        )
    else:
        product = State(
            label="product(absent)", energy_kcal=reactant.energy_kcal, kind=kind,
            n_imag=None, n_vib=None, n_atoms=n_atoms, net_charge=net_charge,
            driven_coordinates=dict(reactant.driven_coordinates),
        )
    return ReactionProfile(
        system_id=system_id, step_id=step_id,
        reactant=reactant, transition_state=ts, product=product,
        reaction_mode_overlap=reaction_mode_overlap,
        labelled_by=labelled_by,
        endpoint_mechanism_ok=endpoint_mechanism_ok,
        provenance={"route": "oniom_mlmm", "results_dir": os.path.abspath(results_dir),
                    "energy_kind": kind},
    )


# --------------------------------------------------------------------------- #
# frozen-boundary cluster route: one JSON
# --------------------------------------------------------------------------- #

def _spectrum(d: dict) -> tuple[list[float], str]:
    """Prefer the unfiltered spectrum; say which field was used."""
    for field in ("imag_cm_all", "imag_cm"):
        vals = d.get(field)
        if isinstance(vals, list) and vals:
            return [-abs(_as_float(v)) for v in vals], field
    return [], "none"


def _as_float(v) -> float:
    """Tolerate '839.05i' style strings alongside numbers."""
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"\s*([-+]?[\d.eE+-]+)\s*i?\s*$", str(v))
    if not m:
        raise IngestError(f"cannot read frequency {v!r}")
    return float(m.group(1))


def profile_from_cluster_json(path: str, *, system_id: str | None = None,
                              step_id: str | None = None,
                              kind: str = "free_energy",
                              threshold_cm: float = IMAGINARY_THRESHOLD_CM,
                              ) -> ReactionProfile:
    """A `ReactionProfile` from a cluster-route result JSON.

    Energies are taken from the absolute ``e_*_eV`` fields where present, rather
    than the pre-differenced ``ts``/``product`` values, so the reference is
    reconstructed here instead of inherited.
    """
    with open(path) as fh:
        d = json.load(fh)

    base = os.path.basename(path)
    system_id = system_id or d.get("pose") or os.path.basename(
        os.path.dirname(path)) or base
    step_id = step_id or os.path.splitext(base)[0]

    have_abs = all(k in d for k in ("e_reactant_eV", "e_ts_eV", "e_product_eV"))
    if have_abs and kind == "potential":
        e_r = kcal(d["e_reactant_eV"], "eV")
        e_t = kcal(d["e_ts_eV"], "eV")
        e_p = kcal(d["e_product_eV"], "eV")
    elif kind == "free_energy" and "dG_act" in d:
        # Gibbs values in these files are already relative, in kcal/mol.
        e_r = 0.0
        e_t = float(d["dG_act"])
        e_p = float(d.get("dG_rxn", 0.0))
    elif have_abs:
        raise IngestError(
            f"{path}: holds absolute electronic energies but {kind!r} was "
            "requested and no Gibbs fields are present. Pass "
            "kind='potential', and remember a bare potential-energy barrier is "
            "not comparable to a kcat-derived free energy."
        )
    else:
        raise IngestError(
            f"{path}: no usable energies for kind={kind!r} "
            f"(keys: {sorted(d)[:12]}...)"
        )

    freqs, src = _spectrum(d)
    if freqs:
        n_imag, mags = imaginary_from_frequencies(freqs, threshold_cm=threshold_cm)
    else:
        n_imag, mags = None, ()

    n_atoms = d.get("n_total")
    charge = d.get("charge")
    charge = int(charge) if isinstance(charge, (int, float)) else None

    # A kept, unrefined guess is not a stationary point.
    fallback = str(d.get("sella_fallback") or "")
    non_stationary = fallback == "guess"

    # Geometry-based labelling leaves a trace; its absence means energy order
    # decided, which is the bug.
    if "identity_swapped" in d:
        labelled_by = "driven_coordinate"
    else:
        labelled_by = "energy"

    mech_flags = [d.get("reactant_mechanism_ok"), d.get("product_mechanism_ok")]
    if any(f is None for f in mech_flags):
        mechanism_ok = None
    else:
        mechanism_ok = all(bool(f) for f in mech_flags)

    r_geom = {k: float(v) for k, v in (d.get("reactant_geom") or {}).items()}
    p_geom = {k: float(v) for k, v in (d.get("product_geom") or {}).items()}
    ts_geom = {
        k[3:]: float(d[k]) for k in ("ts_nuc_p", "ts_p_lg", "ts_h_acc") if k in d
    }

    common = dict(kind=kind, n_atoms=n_atoms, net_charge=charge)
    reactant = State(label="reactant", energy_kcal=e_r, n_imag=0,
                     imaginary_cm=(), driven_coordinates=r_geom,
                     reseeded=non_stationary, **common)
    ts = State(label="TS", energy_kcal=e_t, n_imag=n_imag, imaginary_cm=mags,
               is_saddle=(n_imag == 1), driven_coordinates=ts_geom,
               reseeded=non_stationary, **common)
    product = State(label="product", energy_kcal=e_p, n_imag=0, imaginary_cm=(),
                    driven_coordinates=p_geom, reseeded=non_stationary, **common)

    return ReactionProfile(
        system_id=str(system_id), step_id=str(step_id),
        reactant=reactant, transition_state=ts, product=product,
        labelled_by=labelled_by,
        endpoint_mechanism_ok=mechanism_ok,
        provenance={
            "route": "frozen_cluster", "file": os.path.abspath(path),
            "spectrum_field": src, "energy_kind": kind,
            "recorded_n_imag": str(d.get("n_imag")),
            "recorded_n_imag_raw": str(d.get("n_imag_raw")),
            "recorded_ts_valid": str(d.get("ts_valid")),
            "sella_fallback": fallback or "none",
        },
    )


def recorded_vs_recounted(path: str, *, threshold_cm: float = IMAGINARY_THRESHOLD_CM
                          ) -> dict[str, object]:
    """Compare a file's own imaginary-mode count against a single-threshold recount.

    Diagnostic for the threshold inconsistency: a file reporting
    ``ts_valid: true`` with ``n_imag_raw: 2`` disagrees with itself, and whether
    the barrier it carries was computed with the same mode set as its reactant
    depends on which count the consumer happened to read.
    """
    with open(path) as fh:
        d = json.load(fh)
    freqs, src = _spectrum(d)
    n_recount, mags = (imaginary_from_frequencies(freqs, threshold_cm=threshold_cm)
                       if freqs else (None, ()))
    return {
        "file": os.path.abspath(path),
        "spectrum_field": src,
        "spectrum": [abs(f) for f in freqs],
        "recorded_n_imag": d.get("n_imag"),
        "recorded_n_imag_raw": d.get("n_imag_raw"),
        "recorded_n_imag_ts": d.get("n_imag_ts"),
        "recorded_ts_valid": d.get("ts_valid"),
        "recounted_n_imag": n_recount,
        "recounted_imaginary_cm": list(mags),
        "threshold_cm": threshold_cm,
        "self_consistent": (
            None if n_recount is None else
            (d.get("n_imag") == d.get("n_imag_raw") == n_recount)
        ),
    }
