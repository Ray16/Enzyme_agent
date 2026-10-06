"""The output gate: decide whether a computed profile may be reported at all.

The input gate (``gate.py``) asks whether a structure deserves GPU time. This
module asks the opposite-end question: given a finished run, is the number it
produced a reaction barrier, or an artifact wearing one?

That question is not rhetorical. In the previous campaign roughly twenty of
thirty barrier-style outputs failed the single-imaginary-mode criterion and
every one of them still wrote a number -- including the one result that
"agreed with experiment", which was a minimum, not a saddle: zero imaginary
modes, a product geometrically identical to its reactant (the breaking bond
1.425 -> 1.426 A), and two thirds of its 15.3 kcal/mol contributed by
re-referencing the reactant to a constrained scan point. The engine's own
guards fired -- reaction-mode overlap 0.04, "driven bond NON-REACTIVE
(product == reactant)" -- and the reporting layer printed the number anyway,
next to the experimental value.

So the rule here is that a verdict is *fatal*, not advisory. ``ReactionProfile``
refuses to hand out a barrier unless the profile earned it; reach the number
through ``barrier()``, which raises, rather than reading the field.

Four classes of defect this is built to catch:

* **not a saddle** -- n_imag != 1, or a vibrational-mode count that proves the
  "TS" has as many real modes as the minima flanking it;
* **not a reaction** -- endpoints that never moved along the driven coordinate,
  or that mode-descend into the same basin;
* **not stationary** -- thermochemistry built on geometries whose gradient is
  nowhere near zero, where harmonic ZPE and entropy are undefined;
* **not comparable** -- states differing in composition or charge, a reactant
  reference silently re-seeded, or a bare potential-energy difference being
  compared to a kcat-derived free energy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .gate import GateReport, Status

# One imaginary-frequency threshold, used everywhere.
#
# A measured noise floor justifies it: on a relaxed 3-atom molecule, a pure
# rigid-body ROTATION of the input moved a near-zero normal mode from 0.2i to
# 27.3i cm^-1 -- so any count taken at ~0 is a function of how the molecule
# happened to be oriented in the file, and a 200-atom cluster is worse. Counting
# at 1e-6 cm^-1 (as one path in the previous engine did) is therefore not
# physical, and it disagreed with the 50 cm^-1 used elsewhere in the same engine,
# which is how one output came to carry `ts_valid: true` next to `n_imag_ts: 2`.
IMAGINARY_THRESHOLD_CM = 50.0

# Gradient tolerance for calling a geometry stationary enough to build a
# harmonic Hessian on. The previous engine optimized to 0.10 eV/A and then ran
# thermochemistry at that point.
MAX_FORCE_TOL_EV_PER_A = 0.02

# Minimum overlap between the imaginary mode and the intended reaction
# coordinate. Below this the saddle is off-pathway, however clean its spectrum.
MIN_REACTION_MODE_OVERLAP = 0.30

# Minimum change in a driven coordinate between endpoints, in A, for the step to
# have happened at all.
MIN_DRIVEN_COORDINATE_CHANGE = 0.50

HARTREE_TO_KCAL = 627.5094740631
EV_TO_KCAL = 23.060547830619026
# E_h/a_0 -> eV/A. Optimizer logs report forces in atomic units while the
# stationarity tolerance above is in eV/A, so the conversion has to be explicit;
# comparing 0.000039 E_h/a_0 against a 0.02 eV/A threshold without it would read
# as converged by a factor of 50 for the wrong reason.
HARTREE_PER_BOHR_TO_EV_PER_A = 51.42208619083232

EnergyKind = str  # "potential" | "zpe_corrected" | "enthalpy" | "free_energy"
_ENERGY_KINDS = ("potential", "zpe_corrected", "enthalpy", "free_energy")


class NotReportable(RuntimeError):
    """Raised when a barrier is requested from a profile that did not earn one."""


@dataclass(frozen=True)
class EndpointVerdict:
    """Which endpoint the GEOMETRY says is the reactant.

    ``reactant_is`` is "first", "second" or None (undecidable). ``votes`` records
    the per-coordinate decision so a disagreement is visible rather than averaged
    away.
    """

    reactant_is: str | None
    votes: dict[str, str | None]
    detail: str

    @property
    def unanimous(self) -> bool:
        decided = [v for v in self.votes.values() if v is not None]
        return bool(decided) and len(set(decided)) == 1


def classify_endpoints(step, first: dict[str, float], second: dict[str, float],
                       *, min_change: float = MIN_DRIVEN_COORDINATE_CHANGE
                       ) -> EndpointVerdict:
    """Decide which of two endpoint geometries is the reactant.

    Uses only the declared bond changes: a forming bond is long in the reactant
    and short in the product; a breaking bond is the reverse. Energy is never
    consulted, which is the entire point -- for an exothermic reaction the
    product is the LOWER minimum, so energy order says the opposite of the truth.

    Each coordinate votes independently. Unanimity is required; a split vote
    means the two structures are not a clean reactant/product pair (one endpoint
    has moved along one coordinate but not another) and is reported as
    undecidable rather than resolved by majority.
    """
    votes: dict[str, str | None] = {}
    notes: list[str] = []
    for bc in step.bond_changes:
        k = bc.key
        if k not in first or k not in second:
            votes[k] = None
            notes.append(f"{k}: not measured on both endpoints")
            continue
        d1, d2 = first[k], second[k]
        if abs(d1 - d2) < min_change:
            votes[k] = None
            notes.append(f"{k}: {d1:.2f} vs {d2:.2f} A -- unchanged "
                         f"(<{min_change} A), no information")
            continue
        if bc.reactant_is_longer:
            winner = "first" if d1 > d2 else "second"
            why = "forms, so the reactant is the longer one"
        else:
            winner = "first" if d1 < d2 else "second"
            why = "breaks, so the reactant is the shorter one"
        votes[k] = winner
        notes.append(f"{k}: {d1:.2f} vs {d2:.2f} A -- {why} => {winner}")

    decided = [v for v in votes.values() if v is not None]
    if not decided:
        return EndpointVerdict(None, votes,
                               "no coordinate changed enough to decide; "
                               + "; ".join(notes))
    if len(set(decided)) > 1:
        return EndpointVerdict(None, votes,
                               "coordinates DISAGREE on which endpoint is the "
                               "reactant, so these two structures are not a "
                               "clean reactant/product pair; " + "; ".join(notes))
    return EndpointVerdict(decided[0], votes, "; ".join(notes))


@dataclass(frozen=True)
class State:
    """One point on the profile, with the evidence needed to judge it.

    ``n_vib`` is the count of real vibrational modes. It is the cheapest
    decisive check available: a true saddle has exactly one fewer real mode than
    the minima around it, so reactant/TS/product coming back 162/162/162 proves
    none of them is a saddle -- no energies needed.

    ``None`` anywhere means "not recorded", which the gate treats as unchecked
    and therefore blocking, never as satisfied.
    """

    label: str
    energy_kcal: float
    kind: EnergyKind = "potential"
    n_imag: int | None = None
    imaginary_cm: tuple[float, ...] = ()
    n_vib: int | None = None
    max_force_ev_per_a: float | None = None
    n_atoms: int | None = None
    net_charge: int | None = None
    # True when this state's geometry came from a constrained scan point or was
    # otherwise re-seeded rather than freely optimized. Such a point has a
    # nonzero gradient along the constraint, so its harmonic thermochemistry is
    # undefined -- and a re-seeded reactant silently moves the whole reference.
    reseeded: bool = False
    is_saddle: bool = False
    driven_coordinates: dict[str, float] = field(default_factory=dict)

    @property
    def significant_imaginary(self) -> tuple[float, ...]:
        return tuple(f for f in self.imaginary_cm
                     if abs(f) >= IMAGINARY_THRESHOLD_CM)


@dataclass
class ReactionProfile:
    """A reaction energy profile plus the verdict on whether it is reportable."""

    system_id: str
    step_id: str
    reactant: State
    product: State
    transition_state: State | None = None
    intermediates: tuple[State, ...] = ()
    # Overlap of the TS imaginary mode with the intended reaction coordinate,
    # and how the endpoints were labelled.
    reaction_mode_overlap: float | None = None
    labelled_by: str | None = None          # "driven_coordinate" | "energy" | None
    # Did the endpoints, relaxed along the imaginary mode, actually arrive at the
    # assumed reactant and product? This is the check that correctly caught every
    # untrustworthy decarboxylase result in the previous campaign, so it is
    # carried through rather than recomputed.
    endpoint_mechanism_ok: bool | None = None
    # The declared ReactionStep. Supplying it upgrades the endpoint-labelling
    # check from "trust the recorded flag" to "measure it", which is the
    # difference between abstaining on a swapped profile and catching it.
    step: object | None = None
    # Provenance: where the numbers came from, so a figure can be traced back.
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def states(self) -> tuple[State, ...]:
        mid = (self.transition_state,) if self.transition_state else ()
        return (self.reactant,) + self.intermediates + mid + (self.product,)

    def verdict(self, *, strict: bool = True) -> GateReport:
        report = GateReport(system_id=self.system_id,
                            step_id=f"{self.step_id}:profile", strict=strict)
        self._check_saddle(report)
        self._check_mode_counts(report)
        self._check_endpoints_distinct(report)
        self._check_reaction_mode(report)
        self._check_endpoint_mechanism(report)
        self._check_labelling(report)
        self._check_stationarity(report)
        self._check_comparability(report)
        self._check_energy_kind(report)
        return report

    def barrier(self, *, strict: bool = True) -> float:
        """The forward barrier in kcal/mol, or raise.

        Deliberately the only way to get the number. Reading an attribute cannot
        be made to fail, and a reporting layer that *can* print an unvalidated
        barrier eventually does.
        """
        report = self.verdict(strict=strict)
        if not report.ok:
            raise NotReportable(
                "this profile is not reportable as a barrier:\n" + report.render()
            )
        assert self.transition_state is not None  # guaranteed by _check_saddle
        return self.transition_state.energy_kcal - self.reactant.energy_kcal

    def reaction_energy(self, *, strict: bool = True) -> float:
        report = self.verdict(strict=strict)
        if not report.ok:
            raise NotReportable(
                "this profile is not reportable:\n" + report.render()
            )
        return self.product.energy_kcal - self.reactant.energy_kcal

    # ------------------------------------------------------------------ #

    def _check_saddle(self, report: GateReport) -> None:
        ts = self.transition_state
        if ts is None:
            report.add("ts_present", "FAIL",
                       "no transition state on this profile; endpoint energies "
                       "alone are not a barrier")
            return
        if ts.n_imag is None:
            report.add("ts_single_imaginary", "ABSTAIN",
                       f"no imaginary-mode count recorded for {ts.label}; a "
                       "saddle cannot be asserted without one")
            return
        sig = ts.significant_imaginary
        spectrum = ", ".join(f"{abs(f):.0f}i" for f in sig) or "none"
        if ts.n_imag != 1 or len(sig) != 1:
            report.add(
                "ts_single_imaginary", "FAIL",
                f"{ts.label} has {ts.n_imag} imaginary mode(s) "
                f"({len(sig)} above {IMAGINARY_THRESHOLD_CM:.0f} cm-1: {spectrum}). "
                + ("Zero imaginary modes means this is a minimum, not a "
                   "transition state -- the reported difference is between two "
                   "nearby minima."
                   if ts.n_imag == 0 else
                   "Several imaginary modes means a higher-order stationary "
                   "point, typically a floppy group competing with the reaction "
                   "coordinate."),
                value=ts.n_imag,
            )
        else:
            report.add("ts_single_imaginary", "PASS",
                       f"{ts.label}: exactly one imaginary mode ({spectrum})",
                       value=ts.n_imag)
        for state in (self.reactant, self.product) + self.intermediates:
            if state.n_imag is None:
                report.add(f"minimum_clean:{state.label}", "ABSTAIN",
                           "no imaginary-mode count recorded; cannot confirm "
                           "this endpoint is a real minimum")
            elif state.significant_imaginary:
                spec = ", ".join(f"{abs(f):.0f}i"
                                 for f in state.significant_imaginary)
                report.add(f"minimum_clean:{state.label}", "FAIL",
                           f"endpoint has imaginary mode(s) {spec}; it is not a "
                           "minimum, so the barrier measured from it is not "
                           "referenced to a real state", value=state.n_imag)
            else:
                report.add(f"minimum_clean:{state.label}", "PASS",
                           "no significant imaginary modes", value=state.n_imag)

    def _check_mode_counts(self, report: GateReport) -> None:
        """A saddle has exactly one fewer real vibrational mode than its minima."""
        ts = self.transition_state
        if ts is None or ts.n_vib is None:
            report.add("mode_count_consistency", "ABSTAIN",
                       "real-mode counts not recorded; this is the cheapest "
                       "decisive saddle check -- record n_vib per state")
            return
        minima = [s for s in (self.reactant, self.product) + self.intermediates
                  if s.n_vib is not None]
        if not minima:
            report.add("mode_count_consistency", "ABSTAIN",
                       "no endpoint mode counts to compare the TS against")
            return
        bad = [s for s in minima if s.n_vib != ts.n_vib + 1]
        counts = ", ".join(f"{s.label}={s.n_vib}" for s in minima)
        if bad:
            report.add(
                "mode_count_consistency", "FAIL",
                f"TS has {ts.n_vib} real modes but {counts}. A saddle must have "
                "exactly one fewer real mode than each adjacent minimum; equal "
                "counts prove the 'TS' is another minimum.",
                value=ts.n_vib,
            )
        else:
            report.add("mode_count_consistency", "PASS",
                       f"TS {ts.n_vib} real modes vs minima ({counts})",
                       value=ts.n_vib)

    def _check_endpoints_distinct(self, report: GateReport) -> None:
        """Did the reaction happen? Compare the driven coordinates, not energies."""
        shared = set(self.reactant.driven_coordinates) & \
            set(self.product.driven_coordinates)
        if not shared:
            report.add("endpoints_distinct", "ABSTAIN",
                       "no driven coordinates recorded on the endpoints; "
                       "whether the reaction occurred cannot be verified, and "
                       "an energy difference alone cannot distinguish a reaction "
                       "from two copies of the same basin")
            return
        moved = {
            k: abs(self.product.driven_coordinates[k]
                   - self.reactant.driven_coordinates[k])
            for k in sorted(shared)
        }
        best = max(moved.values())
        detail = ", ".join(f"{k} moved {v:.3f} A" for k, v in moved.items())
        if best < MIN_DRIVEN_COORDINATE_CHANGE:
            report.add(
                "endpoints_distinct", "FAIL",
                f"{detail} -- all below {MIN_DRIVEN_COORDINATE_CHANGE} A. The "
                "product is geometrically the reactant: no bond was made or "
                "broken, so the reported difference is not a reaction energy.",
                value=round(best, 3),
            )
        else:
            report.add("endpoints_distinct", "PASS", detail, value=round(best, 3))

        # A concerted step must move EVERY coordinate it declares. Checking only
        # the largest change lets a partial reaction through: one bond made while
        # another never breaks is a different mechanism, and in the recorded
        # corpus exactly that produced a "reactant" that relaxed back to product.
        declared = {bc.key for bc in getattr(self.step, "bond_changes", ())} \
            if self.step is not None else set()
        if len(declared) > 1:
            unmoved = sorted(
                k for k in declared
                if k in moved and moved[k] < MIN_DRIVEN_COORDINATE_CHANGE
            )
            unmeasured = sorted(declared - set(moved))
            if unmoved:
                report.add(
                    "all_coordinates_changed", "FAIL",
                    f"declared coordinate(s) {', '.join(unmoved)} did not change "
                    f"between the endpoints (<{MIN_DRIVEN_COORDINATE_CHANGE} A) "
                    "while others did. A concerted step must move all of them; "
                    "this is a partial reaction, i.e. a different mechanism.",
                    value=len(unmoved),
                )
            elif unmeasured:
                report.add("all_coordinates_changed", "ABSTAIN",
                           f"declared coordinate(s) {', '.join(unmeasured)} were "
                           "not measured on the endpoints")
            else:
                report.add("all_coordinates_changed", "PASS",
                           f"all {len(declared)} declared coordinates changed",
                           value=len(declared))

    def _check_reaction_mode(self, report: GateReport) -> None:
        if self.transition_state is None:
            return
        if self.reaction_mode_overlap is None:
            report.add("reaction_mode_overlap", "ABSTAIN",
                       "overlap between the imaginary mode and the intended "
                       "reaction coordinate not recorded; a clean single-mode "
                       "saddle can still be off-pathway")
            return
        ov = self.reaction_mode_overlap
        if ov < MIN_REACTION_MODE_OVERLAP:
            report.add(
                "reaction_mode_overlap", "FAIL",
                f"imaginary mode overlaps the intended reaction coordinate by "
                f"only {ov:.2f} (need >= {MIN_REACTION_MODE_OVERLAP:.2f}). This "
                "is a spurious off-pathway mode -- usually a floppy sidechain or "
                "water, not the chemistry.",
                value=round(ov, 2),
            )
        else:
            report.add("reaction_mode_overlap", "PASS",
                       f"overlap {ov:.2f} with the reaction coordinate",
                       value=round(ov, 2))

    def _check_endpoint_mechanism(self, report: GateReport) -> None:
        if self.endpoint_mechanism_ok is None:
            report.add("endpoint_mechanism", "ABSTAIN",
                       "no mechanism-consistency result recorded. Descending the "
                       "imaginary mode from the saddle must land on the assumed "
                       "reactant and product; without it a clean saddle may "
                       "belong to a different reaction.")
        elif self.endpoint_mechanism_ok is False:
            report.add("endpoint_mechanism", "FAIL",
                       "mode-descended endpoints do NOT match the assumed "
                       "mechanism. The saddle connects something else -- this "
                       "barrier is not the barrier for the declared step.",
                       value=False)
        else:
            report.add("endpoint_mechanism", "PASS",
                       "mode-descended endpoints match the assumed mechanism",
                       value=True)

    def geometric_endpoints(self) -> EndpointVerdict | None:
        """Which endpoint the geometry says is the reactant, or None if uncheckable."""
        if self.step is None or not getattr(self.step, "bond_changes", ()):
            return None
        return classify_endpoints(self.step,
                                  self.reactant.driven_coordinates,
                                  self.product.driven_coordinates)

    def relabel_from_geometry(self) -> "ReactionProfile":
        """Return this profile with the endpoints assigned by geometry.

        Use when the engine swapped them: the structures and the saddle are
        fine, only the assignment is wrong, so swapping recovers the correct
        forward barrier rather than discarding the run. Raises if geometry
        cannot decide -- a profile that is merely confusing must not be
        "fixed" by guessing.
        """
        v = self.geometric_endpoints()
        if v is None or v.reactant_is is None:
            raise NotReportable(
                "cannot relabel: geometry does not decide which endpoint is the "
                "reactant. " + (v.detail if v else "no ReactionStep supplied")
            )
        if v.reactant_is == "first":
            return self
        import dataclasses
        swapped_r = dataclasses.replace(self.product, label=self.product.label
                                        + "(relabelled reactant)")
        swapped_p = dataclasses.replace(self.reactant, label=self.reactant.label
                                        + "(relabelled product)")
        return dataclasses.replace(
            self, reactant=swapped_r, product=swapped_p,
            labelled_by="driven_coordinate",
            provenance={**self.provenance,
                        "relabelled": "endpoints swapped by geometry",
                        "relabel_evidence": v.detail},
        )

    def _check_labelling(self, report: GateReport) -> None:
        """Endpoints must be identified by geometry, never by which is lower.

        Labelling the lower minimum "reactant" makes the reported barrier the
        REVERSE barrier for every exothermic step, and forces ADeltaE_rxn >= 0 by
        construction -- so every reaction comes out endothermic.

        When a ReactionStep is available this is MEASURED rather than trusted: a
        swapped profile is caught and the corrected barrier is reported in the
        failure, instead of the check abstaining and leaving a human to notice.
        """
        v = self.geometric_endpoints()
        if v is not None:
            if v.reactant_is == "first":
                report.add("endpoint_labelling", "PASS",
                           f"geometry confirms the labelling ({v.detail})",
                           value="driven_coordinate")
                return
            if v.reactant_is == "second":
                if self.transition_state is not None:
                    fwd = self.transition_state.energy_kcal - self.product.energy_kcal
                    corrected = (f" The forward barrier is {fwd:.2f} kcal/mol, "
                                 f"not the "
                                 f"{self.transition_state.energy_kcal - self.reactant.energy_kcal:.2f} "
                                 "this profile reports.")
                else:
                    corrected = ""
                report.add(
                    "endpoint_labelling", "FAIL",
                    "ENDPOINTS ARE SWAPPED: the state labelled 'reactant' is "
                    "geometrically the product. The reported barrier is "
                    "therefore the REVERSE barrier." + corrected
                    + " The structures and the saddle are fine -- call "
                    "relabel_from_geometry() to recover the forward barrier. "
                    f"Evidence: {v.detail}",
                    value="swapped",
                )
                return
            report.add("endpoint_labelling", "FAIL",
                       f"geometry cannot identify the reactant: {v.detail}",
                       value="undecidable")
            return
        if self.labelled_by is None:
            report.add("endpoint_labelling", "ABSTAIN",
                       "how reactant and product were identified is not "
                       "recorded; if it was by energy, the reported barrier is "
                       "the reverse barrier for any exothermic step")
            return
        if self.labelled_by == "energy":
            report.add(
                "endpoint_labelling", "FAIL",
                "endpoints were labelled by energy (lower = reactant). For an "
                "exothermic step this reports the REVERSE barrier and forces the "
                "reaction energy positive. Label by the driven coordinate / "
                "chemical identity instead.",
                value=self.labelled_by,
            )
        else:
            report.add("endpoint_labelling", "PASS",
                       f"endpoints labelled by {self.labelled_by}",
                       value=self.labelled_by)

    def _check_stationarity(self, report: GateReport) -> None:
        for state in self.states:
            if state.reseeded:
                report.add(
                    f"stationary:{state.label}", "FAIL",
                    "geometry was re-seeded or taken from a constrained scan "
                    "point rather than freely optimized. Its gradient is nonzero "
                    "along the constraint, so harmonic ZPE and entropy are "
                    "undefined -- and a re-seeded reactant shifts the reference "
                    "for the whole profile.",
                    value="reseeded",
                )
                continue
            f = state.max_force_ev_per_a
            if f is None:
                report.add(f"stationary:{state.label}", "ABSTAIN",
                           "max|F| not recorded; thermochemistry on a "
                           "non-stationary geometry is undefined")
            elif f > MAX_FORCE_TOL_EV_PER_A:
                report.add(
                    f"stationary:{state.label}", "FAIL",
                    f"max|F| = {f:.4f} eV/A exceeds "
                    f"{MAX_FORCE_TOL_EV_PER_A} eV/A; this geometry is not a "
                    "stationary point and a harmonic Hessian on it is not "
                    "meaningful. Optimization likely hit its step cap.",
                    value=round(f, 4),
                )
            else:
                report.add(f"stationary:{state.label}", "PASS",
                           f"max|F| = {f:.4f} eV/A", value=round(f, 4))

    def _check_comparability(self, report: GateReport) -> None:
        """Every point must be the same atoms at the same charge."""
        atoms = {s.label: s.n_atoms for s in self.states}
        charges = {s.label: s.net_charge for s in self.states}
        if any(v is None for v in atoms.values()):
            report.add("composition_conserved", "ABSTAIN",
                       "atom counts not recorded per state; a profile whose "
                       "points differ in composition is not a profile")
        else:
            uniq = set(atoms.values())
            if len(uniq) > 1:
                report.add("composition_conserved", "FAIL",
                           "states differ in atom count ("
                           + ", ".join(f"{k}={v}" for k, v in atoms.items())
                           + "); a proton or water moved in or out of the region, "
                           "so these energies are not comparable",
                           value=len(uniq))
            else:
                report.add("composition_conserved", "PASS",
                           f"all states have {next(iter(uniq))} atoms",
                           value=next(iter(uniq)))
        if any(v is None for v in charges.values()):
            report.add("charge_conserved", "ABSTAIN",
                       "net charge not recorded per state")
        else:
            uniq_q = set(charges.values())
            if len(uniq_q) > 1:
                report.add("charge_conserved", "FAIL",
                           "states differ in net charge ("
                           + ", ".join(f"{k}={v:+d}" for k, v in charges.items())
                           + "); energies at different charges are not comparable",
                           value=len(uniq_q))
            else:
                report.add("charge_conserved", "PASS",
                           f"all states at q = {next(iter(uniq_q)):+d}",
                           value=next(iter(uniq_q)))

    def _check_energy_kind(self, report: GateReport) -> None:
        kinds = {s.kind for s in self.states}
        unknown = kinds - set(_ENERGY_KINDS)
        if unknown:
            report.add("energy_kind", "FAIL",
                       f"unrecognised energy kind(s) {sorted(unknown)}; "
                       f"expected one of {_ENERGY_KINDS}")
            return
        if len(kinds) > 1:
            report.add("energy_kind", "FAIL",
                       f"profile mixes energy kinds {sorted(kinds)}; a barrier "
                       "built from a potential energy at one point and a free "
                       "energy at another is meaningless", value=len(kinds))
        else:
            kind = next(iter(kinds))
            report.add("energy_kind", "PASS",
                       f"all states reported as {kind}", value=kind)

    # ------------------------------------------------------------------ #

    def label_for_plot(self) -> str:
        """The y-axis label this profile is actually entitled to.

        A bare potential-energy difference must not be drawn, or compared, as a
        free energy -- a kcat-derived experimental number is a DeltaG.
        """
        kind = self.reactant.kind
        return {
            "potential": "DeltaE (kcal/mol)",
            "zpe_corrected": "DeltaE + ZPE (kcal/mol)",
            "enthalpy": "DeltaH (kcal/mol)",
            "free_energy": "DeltaG (kcal/mol)",
        }.get(kind, "energy (kcal/mol)")

    def comparable_to_experiment(self) -> tuple[bool, str]:
        """May this be set beside a kcat-derived activation free energy?"""
        kind = self.reactant.kind
        if kind == "free_energy":
            return True, "free energies are comparable to a kcat-derived DeltaG"
        return False, (
            f"this profile reports {kind}, not a free energy. An experimental "
            "barrier from kcat is a DeltaG; comparing a bare DeltaE to it omits "
            "ZPE, thermal and entropic terms. Compute thermochemistry first, or "
            "state plainly that the comparison is indicative only."
        )

    def to_dict(self) -> dict:
        report = self.verdict()
        return {
            "system_id": self.system_id,
            "step_id": self.step_id,
            "reportable": report.ok,
            "energy_kind": self.reactant.kind,
            "barrier_kcal": (
                self.transition_state.energy_kcal - self.reactant.energy_kcal
                if self.transition_state else None
            ),
            "reaction_energy_kcal": (
                self.product.energy_kcal - self.reactant.energy_kcal
            ),
            "states": [
                {"label": s.label, "energy_kcal": s.energy_kcal, "kind": s.kind,
                 "n_imag": s.n_imag, "n_vib": s.n_vib,
                 "max_force_ev_per_a": s.max_force_ev_per_a,
                 "n_atoms": s.n_atoms, "net_charge": s.net_charge,
                 "reseeded": s.reseeded}
                for s in self.states
            ],
            "reaction_mode_overlap": self.reaction_mode_overlap,
            "labelled_by": self.labelled_by,
            "verdict": report.to_dict(),
            "provenance": dict(self.provenance),
        }


def diagram_values(profile: ReactionProfile, *, strict: bool = True) -> list[float]:
    """Energies for ``mlmm energy-diagram``, relative to the reactant.

    Goes through the verdict, so an unreportable profile cannot be plotted. The
    plotter itself takes bare numbers and will happily draw anything.
    """
    report = profile.verdict(strict=strict)
    if not report.ok:
        raise NotReportable(
            "refusing to build a diagram from an unreportable profile:\n"
            + report.render()
        )
    ref = profile.reactant.energy_kcal
    return [round(s.energy_kcal - ref, 3) for s in profile.states]


def kcal(value: float, unit: str) -> float:
    """Convert to kcal/mol from hartree, eV or kcal/mol, or raise.

    Explicit because a silent unit assumption is how a 0.0006 kcal/mol "flat
    saddle" and a 1097 kcal/mol clash spike end up in the same table.
    """
    u = unit.strip().lower()
    if u in ("kcal", "kcal/mol", "kcalpermol"):
        return float(value)
    if u in ("hartree", "ha", "au", "a.u."):
        return float(value) * HARTREE_TO_KCAL
    if u in ("ev", "electronvolt"):
        return float(value) * EV_TO_KCAL
    if u in ("kj", "kj/mol"):
        return float(value) / 4.184
    raise ValueError(f"unknown energy unit {unit!r}")


def imaginary_from_frequencies(freqs_cm: list[complex] | list[float],
                               *, threshold_cm: float = IMAGINARY_THRESHOLD_CM
                               ) -> tuple[int, tuple[float, ...]]:
    """Count imaginary modes on ONE threshold, from real or complex frequencies.

    Accepts the two conventions in circulation: complex values whose imaginary
    part carries the mode, and real values where a negative number means
    imaginary. Returns (count, magnitudes) using ``threshold_cm`` for both, so a
    single run cannot report two different counts.
    """
    mags: list[float] = []
    for f in freqs_cm:
        if isinstance(f, complex):
            mag = abs(f.imag) if abs(f.imag) > abs(f.real) else 0.0
        else:
            mag = abs(float(f)) if float(f) < 0 else 0.0
        if mag >= threshold_cm:
            mags.append(mag)
    mags.sort(reverse=True)
    return len(mags), tuple(mags)


def mode_overlap(imaginary_mode: list[float],
                 reaction_vector: list[float]) -> float:
    """|cos| between the imaginary mode and the intended reaction coordinate.

    Both are mass-weighted Cartesian displacement vectors of the same length.
    """
    if len(imaginary_mode) != len(reaction_vector):
        raise ValueError(
            f"length mismatch: mode has {len(imaginary_mode)} components, "
            f"reaction vector {len(reaction_vector)}"
        )
    dot = sum(a * b for a, b in zip(imaginary_mode, reaction_vector))
    na = math.sqrt(sum(a * a for a in imaginary_mode))
    nb = math.sqrt(sum(b * b for b in reaction_vector))
    if na == 0 or nb == 0:
        raise ValueError("zero-length vector: overlap undefined")
    return abs(dot) / (na * nb)
