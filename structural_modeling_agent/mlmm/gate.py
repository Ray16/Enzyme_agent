"""The structure-quality gate: run before any ML/MM compute, fail closed.

Why this exists. In the previous campaign the solver was not what failed -- the
*input structures* were. Every recorded failure traces back to something checkable
in advance:

* a cofold pose that was not a near-attack conformation at all (nucleophile 5.4 A
  from the electrophile, unaligned), driven uphill by a modelling assumption that
  then dominated the result;
* a metal whose first shell was 5-coordinate in one variant and 6 in another,
  producing an 11 kcal/mol difference that was a pose artifact;
* a declared ligand charge inconsistent with the species actually built, so the
  ML region carried the wrong electron count;
* MM virtual sites counted as atoms, putting a net +380 where -4 belonged;
* a concerted mechanism missing one of its coordinates, giving a +225 kcal/mol
  endpoint.

None of those needed a GPU to catch. All of them cost one. So the gate is a hard
precondition: it reports PASS/FAIL/WARN/ABSTAIN per check, and an unknown is an
ABSTAIN that blocks by default rather than a pass. A barrier computed from a
structure that did not clear this gate is not reportable.

    from structural_modeling_agent.mlmm import check_structure, read_pdb

    report = check_structure(read_pdb("system.pdb"), spec)
    if not report.ok:
        raise SystemExit(report.render())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .spec import MLMMSpec, MetalSite
from .structure import (
    Atom, Structure, StructureError, WATER_RESNAMES, angle_deg, neighbours,
)

Status = Literal["PASS", "FAIL", "WARN", "ABSTAIN"]

# Below this, two heavy atoms in different residues are clashing badly enough
# that minimisation will either blow up or distort the active site to escape.
CLASH_DISTANCE = 1.9          # A
# A hydrogen bond donor/acceptor pair this close is fine; only flag real overlap.
CLASH_DISTANCE_WITH_H = 1.2   # A


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    value: float | int | str | None = None

    @property
    def blocking(self) -> bool:
        return self.status in ("FAIL", "ABSTAIN")

    def __str__(self) -> str:
        v = "" if self.value is None else f" [{self.value}]"
        return f"{self.status:<8} {self.name}{v}: {self.detail}"


@dataclass
class GateReport:
    system_id: str
    step_id: str
    checks: list[Check] = field(default_factory=list)
    strict: bool = True

    def add(self, name: str, status: Status, detail: str,
            value: float | int | str | None = None) -> None:
        self.checks.append(Check(name, status, detail, value))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def abstentions(self) -> list[Check]:
        return [c for c in self.checks if c.status == "ABSTAIN"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "WARN"]

    @property
    def ok(self) -> bool:
        """Clear to compute. In strict mode an ABSTAIN blocks, same as a FAIL.

        An unchecked precondition is not a satisfied one. Pass ``strict=False``
        only to triage a structure you already know is incomplete.
        """
        if self.failures:
            return False
        return not (self.strict and self.abstentions)

    def render(self) -> str:
        head = f"ML/MM structure gate -- {self.system_id} / {self.step_id}"
        verdict = "CLEAR" if self.ok else "BLOCKED"
        body = "\n".join(f"  {c}" for c in self.checks)
        tail = (
            f"  => {verdict}: {len(self.failures)} failed, "
            f"{len(self.abstentions)} unchecked, {len(self.warnings)} warned"
        )
        return f"{head}\n{body}\n{tail}"

    def to_dict(self) -> dict:
        return {
            "system_id": self.system_id,
            "step_id": self.step_id,
            "ok": self.ok,
            "strict": self.strict,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "value": c.value}
                for c in self.checks
            ],
        }


def measure_driven_coordinates(structure: Structure, step) -> dict[str, float]:
    """Measure every declared bond-change distance on one structure.

    The output feeds ``State.driven_coordinates``, which is what lets the output
    gate decide which endpoint is the reactant from GEOMETRY instead of trusting
    a label. Raises rather than guessing: an unresolvable or ambiguous atom
    reference here would silently produce a coordinate for the wrong copy of a
    multi-copy ligand, and a trimer has three.
    """
    out: dict[str, float] = {}
    for bc in step.bond_changes:
        a = structure.select(bc.first.name, resseq=bc.first.resseq,
                             resname=bc.first.resname, chain=bc.first.chain)
        b = structure.select(bc.second.name, resseq=bc.second.resseq,
                             resname=bc.second.resname, chain=bc.second.chain)
        if bc.key in out:
            raise StructureError(
                f"two bond changes share the key {bc.key!r}; qualify their "
                "AtomRefs so each coordinate is distinguishable"
            )
        out[bc.key] = float(np.linalg.norm(a.xyz - b.xyz))
    return out


def check_structure(structure: Structure, spec: MLMMSpec, *,
                    strict: bool = True) -> GateReport:
    """Run the whole battery. Never raises on a bad structure -- it reports."""
    report = GateReport(system_id=spec.system_id, step_id=spec.step.step_id, strict=strict)
    _check_hydrogens(structure, report)
    _check_electron_parity(structure, spec, report)
    _check_ligands(structure, spec, report)
    _check_attack_geometry(structure, spec, report)
    _check_bond_changes(structure, spec, report)
    _check_proton_coordinate(structure, spec, report)
    for metal in spec.metals:
        _check_metal(structure, spec, metal, report)
    _check_active_site_waters(structure, spec, report)
    _check_clashes(structure, spec, report)
    return report


# --------------------------------------------------------------------------- #
# individual checks
# --------------------------------------------------------------------------- #

def _check_hydrogens(structure: Structure, report: GateReport) -> None:
    n_h = sum(1 for a in structure.real if a.is_hydrogen)
    n_heavy = len(structure.heavy)
    if n_h == 0:
        report.add(
            "hydrogens_present", "FAIL",
            "no hydrogens: this is a heavy-atom-only structure (e.g. raw Boltz "
            "output). Protonate before anything else -- charge, parity and the "
            "proton coordinate are all undefined without them.",
            value=0,
        )
        return
    ratio = n_h / max(1, n_heavy)
    if ratio < 0.4:
        report.add(
            "hydrogens_present", "WARN",
            f"only {n_h} H for {n_heavy} heavy atoms (ratio {ratio:.2f}); a fully "
            "protonated protein is usually near 1.0. Some residue or ligand is "
            "probably missing its hydrogens.",
            value=round(ratio, 2),
        )
    else:
        report.add("hydrogens_present", "PASS",
                   f"{n_h} H for {n_heavy} heavy atoms", value=n_h)


def _check_electron_parity(structure: Structure, spec: MLMMSpec,
                           report: GateReport) -> None:
    """A closed-shell singlet needs an even electron count. Cheap, and decisive.

    This is the check that catches a wrong declared charge, a missing proton, or
    an element misread -- all of which otherwise run to completion and produce a
    number.
    """
    n_e = structure.electron_count(spec.net_charge)
    unpaired = spec.unpaired_electrons
    expected_parity = unpaired % 2
    if n_e < 0:
        report.add("electron_parity", "FAIL",
                   f"negative electron count ({n_e}) -- net charge {spec.net_charge:+d} "
                   "cannot be right for this atom list", value=n_e)
        return
    if n_e % 2 != expected_parity:
        report.add(
            "electron_parity", "FAIL",
            f"{n_e} electrons at q={spec.net_charge:+d} is inconsistent with "
            f"multiplicity {spec.multiplicity} ({unpaired} unpaired): parity "
            f"{n_e % 2} != {expected_parity}. The declared charge, the "
            "protonation, or an element assignment is wrong -- off-by-one on a "
            "proton is the usual cause.",
            value=n_e,
        )
    else:
        report.add("electron_parity", "PASS",
                   f"{n_e} electrons at q={spec.net_charge:+d}, multiplicity "
                   f"{spec.multiplicity}", value=n_e)
    n_virtual = len(structure.atoms) - len(structure.real)
    if n_virtual:
        report.add("virtual_sites_excluded", "PASS",
                   f"{n_virtual} MM virtual site(s) present and excluded from the "
                   "electron count (keep them for an MM point-charge field, never "
                   "for the ML region)", value=n_virtual)


def _check_ligands(structure: Structure, spec: MLMMSpec, report: GateReport) -> None:
    if not spec.ligands:
        report.add("ligands_declared", "ABSTAIN",
                   "no ligands declared; if the step involves a substrate or "
                   "cofactor, declare it with its formal charge")
        return
    for lig in spec.ligands:
        atoms = structure.by_resname(lig.resname)
        if not atoms:
            report.add(f"ligand:{lig.resname}", "FAIL",
                       f"declared ligand {lig.resname!r} is absent from the structure")
            continue
        residues = {a.residue_key for a in atoms}
        heavy = [a for a in atoms if not a.is_hydrogen and not a.is_virtual]
        n_h = sum(1 for a in atoms if a.is_hydrogen)
        detail = (f"{len(residues)} cop{'y' if len(residues) == 1 else 'ies'}, "
                  f"{len(heavy)} heavy atoms, {n_h} H, declared q={lig.formal_charge:+d}")
        status: Status = "PASS"
        problems = []
        if lig.expected_heavy_atoms is not None:
            per_copy = len(heavy) / len(residues)
            if abs(per_copy - lig.expected_heavy_atoms) > 1e-6:
                status = "FAIL"
                problems.append(
                    f"expected {lig.expected_heavy_atoms} heavy atoms per copy, "
                    f"found {per_copy:g} -- the species built is not the species declared"
                )
        if lig.expect_hydrogens and n_h == 0:
            status = "FAIL"
            problems.append("no hydrogens on this ligand")
        if len(residues) > 1:
            problems.append(
                f"{len(residues)} copies present -- make sure the reaction "
                "coordinate is anchored to the intended one (qualify AtomRef by resseq)"
            )
            status = "WARN" if status == "PASS" else status
        report.add(f"ligand:{lig.resname}", status,
                   detail + ("; " + "; ".join(problems) if problems else ""),
                   value=len(heavy))


def _resolve(structure: Structure, ref, report: GateReport, label: str) -> Atom | None:
    try:
        return structure.select(ref.name, resseq=ref.resseq,
                                resname=ref.resname, chain=ref.chain)
    except StructureError as exc:
        report.add(label, "FAIL", str(exc))
        return None


def _check_attack_geometry(structure: Structure, spec: MLMMSpec,
                           report: GateReport) -> None:
    """Is this a near-attack conformation at all?

    The most consequential check. A pose that is not near-attack cannot be fixed
    by the TS search; driving it there makes the result an artifact of the driving
    protocol, which is precisely what happened to the decarboxylase orthologs.
    """
    step = spec.step
    nuc = _resolve(structure, step.nucleophile, report, "attack:nucleophile")
    ele = _resolve(structure, step.electrophile, report, "attack:electrophile")
    if nuc is None or ele is None:
        return
    d = float(np.linalg.norm(nuc.xyz - ele.xyz))
    if d > step.max_attack_distance:
        report.add(
            "attack_distance", "FAIL",
            f"{step.nucleophile} to {step.electrophile} is {d:.2f} A, beyond the "
            f"{step.max_attack_distance:.2f} A near-attack window. This pose is not "
            "poised to react; select a better one rather than driving this one.",
            value=round(d, 2),
        )
    else:
        report.add("attack_distance", "PASS",
                   f"{step.nucleophile} to {step.electrophile} = {d:.2f} A "
                   f"(window <= {step.max_attack_distance:.2f} A)", value=round(d, 2))

    if step.leaving_group is None:
        report.add("attack_angle", "ABSTAIN",
                   "no leaving group declared, so the attack angle is undefined; "
                   "declare one for any displacement mechanism")
        return
    lg = _resolve(structure, step.leaving_group, report, "attack:leaving_group")
    if lg is None:
        return
    try:
        theta = angle_deg(nuc.xyz, ele.xyz, lg.xyz)
    except StructureError as exc:
        report.add("attack_angle", "FAIL", str(exc))
        return
    lo, hi = step.min_attack_angle, step.max_attack_angle
    if not (lo <= theta <= hi):
        report.add(
            "attack_angle", "FAIL",
            f"nucleophile-electrophile-leaving angle is {theta:.1f} deg, outside "
            f"[{lo:.0f}, {hi:.0f}] for an ideal {step.ideal_attack_angle:.0f} deg "
            "mechanism. The trajectory is not aligned for this chemistry.",
            value=round(theta, 1),
        )
    else:
        report.add("attack_angle", "PASS",
                   f"attack angle {theta:.1f} deg (ideal "
                   f"{step.ideal_attack_angle:.0f}, window [{lo:.0f}, {hi:.0f}])",
                   value=round(theta, 1))


def _check_bond_changes(structure: Structure, spec: MLMMSpec,
                        report: GateReport) -> None:
    step = spec.step
    if not step.bond_changes:
        report.add("bond_changes", "ABSTAIN",
                   "no bond changes declared; mlmm scan needs at least one "
                   "--scan-lists entry and the gate cannot verify a coordinate "
                   "that was not declared")
        return
    for i, bc in enumerate(step.bond_changes, 1):
        a = _resolve(structure, bc.first, report, f"bond{i}:first")
        b = _resolve(structure, bc.second, report, f"bond{i}:second")
        if a is None or b is None:
            continue
        d = float(np.linalg.norm(a.xyz - b.xyz))
        moving = "toward" if (bc.kind == "form") == (bc.target < d) else "away from"
        sane = (bc.kind == "form" and bc.target < d) or (bc.kind == "break" and bc.target > d)
        status: Status = "PASS" if sane else "FAIL"
        report.add(
            f"bond{i}:{bc.kind}", status,
            f"{bc.first} to {bc.second}: now {d:.2f} A, target {bc.target:.2f} A "
            f"({moving} the target)"
            + ("" if sane else
               f"; a '{bc.kind}' coordinate must drive {'shorter' if bc.kind == 'form' else 'longer'}"
               " -- the target is on the wrong side of the current distance"),
            value=round(d, 2),
        )
    if step.n_coordinates > 2:
        report.add(
            "coordinate_count", "WARN",
            f"{step.n_coordinates} coordinates change in one step. Concerted "
            "multi-coordinate transition states are hard: expect coarse path "
            "interpolation to produce clash spikes and spurious imaginary modes. "
            "Use a finer path and verify the saddle carefully.",
            value=step.n_coordinates,
        )


def _check_proton_coordinate(structure: Structure, spec: MLMMSpec,
                             report: GateReport) -> None:
    step = spec.step
    if step.proton_coupled is None:
        report.add("proton_coordinate", "ABSTAIN",
                   "whether this step is proton-coupled was never stated. Set "
                   "ReactionStep.proton_coupled explicitly: declaring only the "
                   "heavy-atom coordinates of a proton-coupled step yields a "
                   "nonsense endpoint, and silence is how that slips through.")
        return
    if step.proton_coupled is False:
        if step.proton is not None:
            report.add("proton_coordinate", "FAIL",
                       f"proton_coupled=False but a proton {step.proton} is "
                       "declared; the claim and the coordinates disagree")
        else:
            report.add("proton_coordinate", "PASS",
                       "declared not proton-coupled; no proton coordinate expected")
        return
    if step.proton is None:
        report.add("proton_coordinate", "FAIL",
                   "proton_coupled=True but no proton declared; the coordinate "
                   "that changes is unspecified")
        return
    h = _resolve(structure, step.proton, report, "proton")
    if h is None:
        return
    if not h.is_hydrogen:
        report.add("proton_coordinate", "FAIL",
                   f"{step.proton} is {h.element}, not a hydrogen")
        return
    if step.proton_acceptor is None:
        report.add("proton_coordinate", "ABSTAIN",
                   "proton declared without an acceptor; the coordinate is undefined")
        return
    acc = _resolve(structure, step.proton_acceptor, report, "proton_acceptor")
    if acc is None:
        return
    d = float(np.linalg.norm(h.xyz - acc.xyz))
    donor = neighbours(structure, h, 1.3, heavy_only=True)
    donor_s = (f"{donor[0][0].resname}{donor[0][0].resseq}/{donor[0][0].name} "
               f"at {donor[0][1]:.2f} A" if donor else "NONE found within 1.3 A")
    status: Status = "PASS" if d <= 3.0 else "WARN"
    report.add(
        "proton_coordinate", status,
        f"{step.proton} (donor {donor_s}) to acceptor {step.proton_acceptor} "
        f"= {d:.2f} A" + ("" if status == "PASS" else
                          "; too far for the transfer to complete in this step"),
        value=round(d, 2),
    )
    if not donor:
        report.add("proton_donor", "FAIL",
                   f"{step.proton} has no heavy atom within 1.3 A -- it is not "
                   "bonded to anything")


def _check_metal(structure: Structure, spec: MLMMSpec, metal: MetalSite,
                 report: GateReport) -> None:
    name = f"metal:{metal.ref}"
    m = _resolve(structure, metal.ref, report, name)
    if m is None:
        return
    if metal.expected_coordination is None:
        report.add(name, "ABSTAIN",
                   "no expected coordination declared; metal shell completeness "
                   "is the dominant artifact source, so declare it")
        return
    shell = neighbours(structure, m, metal.shell_cutoff, heavy_only=True,
                       exclude_same_residue=True)
    # Only donor atoms coordinate; carbons at 2.6 A are not ligands.
    donors = [(a, d) for a, d in shell if a.element in ("O", "N", "S", "CL")]
    n = len(donors)
    want = metal.expected_coordination
    ligand_desc = ", ".join(
        f"{a.resname}{a.resseq}/{a.name} {d:.2f}" for a, d in donors[:8]
    ) or "none"
    if n != want:
        report.add(
            name, "FAIL",
            f"{n}/{want} coordination within {metal.shell_cutoff} A "
            f"(donors: {ligand_desc}). An incomplete metal shell inflates the "
            "barrier, and an inconsistent one between variants makes their "
            "difference an artifact. Complete it with explicit waters before running.",
            value=n,
        )
    else:
        report.add(name, "PASS",
                   f"{n}/{want} coordination (donors: {ligand_desc})", value=n)

    if metal.nucleophile_must_be_free:
        nuc = _resolve(structure, spec.step.nucleophile, report, f"{name}:nuc")
        if nuc is not None:
            d_nuc = float(np.linalg.norm(nuc.xyz - m.xyz))
            if d_nuc <= metal.shell_cutoff:
                report.add(
                    f"{name}:nucleophile_free", "FAIL",
                    f"the nucleophile {spec.step.nucleophile} coordinates this "
                    f"metal at {d_nuc:.2f} A. For this mechanism the metal "
                    "sequesters the nucleophile rather than activating it, which "
                    "raises the barrier sharply. (Not universal -- carbonic "
                    "anhydrase and two-metal polymerases have metal-activated "
                    "nucleophiles; this check is opt-in per system.)",
                    value=round(d_nuc, 2),
                )
            else:
                report.add(f"{name}:nucleophile_free", "PASS",
                           f"nucleophile is {d_nuc:.2f} A from the metal, not "
                           "in its first shell", value=round(d_nuc, 2))


def _check_active_site_waters(structure: Structure, spec: MLMMSpec,
                              report: GateReport) -> None:
    if spec.min_active_site_waters <= 0:
        report.add("active_site_waters", "PASS",
                   "no active-site water requirement declared", value=0)
        return
    try:
        centre = structure.select(
            spec.step.electrophile.name,
            resseq=spec.step.electrophile.resseq,
            resname=spec.step.electrophile.resname,
            chain=spec.step.electrophile.chain,
        )
    except StructureError:
        report.add("active_site_waters", "ABSTAIN",
                   "cannot locate the electrophile to centre the active site")
        return
    water_o = [
        a for a in structure.waters()
        if a.element == "O" and
        float(np.linalg.norm(a.xyz - centre.xyz)) <= spec.active_site_radius
    ]
    n = len(water_o)
    if n < spec.min_active_site_waters:
        report.add(
            "active_site_waters", "FAIL",
            f"{n} water(s) within {spec.active_site_radius:.1f} A of the "
            f"electrophile, need >= {spec.min_active_site_waters}. Cofold models "
            "place no waters at all, so these must be added during prep -- but "
            "over-provisioning is its own artifact, crowding the nucleophile onto "
            "the metal. Add what the shell needs, not more.",
            value=n,
        )
    else:
        report.add("active_site_waters", "PASS",
                   f"{n} water(s) within {spec.active_site_radius:.1f} A", value=n)


def _check_clashes(structure: Structure, spec: MLMMSpec, report: GateReport) -> None:
    """Heavy-atom overlap between different residues.

    A grid search would be better for a big system; for the active-site radius
    this stays well under a second, and the gate runs once per structure.
    """
    try:
        centre = structure.select(
            spec.step.electrophile.name,
            resseq=spec.step.electrophile.resseq,
            resname=spec.step.electrophile.resname,
            chain=spec.step.electrophile.chain,
        )
    except StructureError:
        report.add("steric_clashes", "ABSTAIN",
                   "cannot locate the electrophile to bound the clash search")
        return
    local = [
        a for a in structure.heavy
        if float(np.linalg.norm(a.xyz - centre.xyz)) <= spec.active_site_radius + 4.0
    ]
    worst: tuple[float, str] | None = None
    for i, a in enumerate(local):
        for b in local[i + 1:]:
            if a.residue_key == b.residue_key:
                continue
            d = float(np.linalg.norm(a.xyz - b.xyz))
            if d < CLASH_DISTANCE and (worst is None or d < worst[0]):
                worst = (d, f"{a.resname}{a.resseq}/{a.name} - "
                            f"{b.resname}{b.resseq}/{b.name}")
    if worst is not None:
        d, pair = worst
        # A metal-donor contact is a bond, not a clash; those are checked above.
        report.add(
            "steric_clashes", "FAIL",
            f"heavy-atom overlap {d:.2f} A between {pair} (threshold "
            f"{CLASH_DISTANCE} A). Minimisation will either blow up or distort "
            "the active site to escape it.",
            value=round(d, 2),
        )
    else:
        report.add("steric_clashes", "PASS",
                   f"no inter-residue heavy-atom contact below {CLASH_DISTANCE} A "
                   f"in the active site ({len(local)} atoms searched)")


# --------------------------------------------------------------------------- #
# cross-variant consistency
# --------------------------------------------------------------------------- #

def check_freeze_consistency(
    variants: dict[str, tuple[Structure, MLMMSpec, set[int]]],
) -> GateReport:
    """Verify that variants about to be *compared* were constrained comparably.

    This is the check that would have caught the largest artifact in the previous
    campaign. Two variants were compared where one's metal-coordinating sidechain
    was free to relax and the other's was frozen; the frozen one's metal shell
    under-relaxed and its barrier came out inflated, and the resulting difference
    was an artifact of the freeze list rather than chemistry.

    The rule: for every variant, each metal's *actual* coordinating sidechains must
    be free. Matching the literal residue numbers across variants is not enough and
    not even right -- different variants coordinate through different residues. What
    must match is the *principle*.

    ``variants`` maps a label to (structure, spec, frozen atom serials, 1-based).
    """
    report = GateReport(system_id="|".join(sorted(variants)), step_id="freeze_scheme")
    if len(variants) < 2:
        report.add("freeze_consistency", "ABSTAIN",
                   "fewer than two variants given; nothing to compare")
        return report

    for label, (structure, spec, frozen) in sorted(variants.items()):
        if not spec.metals:
            report.add(f"{label}:metal_sidechains", "ABSTAIN",
                       "no metals declared, so there is no shell to keep relaxable")
            continue
        for metal in spec.metals:
            try:
                m = structure.select(metal.ref.name, resseq=metal.ref.resseq,
                                     resname=metal.ref.resname, chain=metal.ref.chain)
            except StructureError as exc:
                report.add(f"{label}:metal_sidechains", "FAIL", str(exc))
                continue
            shell = neighbours(structure, m, metal.shell_cutoff, heavy_only=True,
                               exclude_same_residue=True)
            protein_donors = [
                a for a, _ in shell
                if a.element in ("O", "N", "S")
                and a.resname.upper() not in WATER_RESNAMES
                and not any(a.resname.upper() == l.resname.upper() for l in spec.ligands)
            ]
            pinned = [a for a in protein_donors if a.serial in frozen]
            if pinned:
                who = ", ".join(f"{a.resname}{a.resseq}/{a.name}" for a in pinned)
                report.add(
                    f"{label}:metal_sidechains", "FAIL",
                    f"metal {metal.ref} is coordinated by frozen protein atom(s) "
                    f"{who}. Its shell cannot relax, so this variant's barrier is "
                    "not comparable to one whose coordinating sidechain was free. "
                    "Free each variant's ACTUAL coordinating sidechains.",
                    value=len(pinned),
                )
            else:
                who = ", ".join(
                    f"{a.resname}{a.resseq}/{a.name}" for a in protein_donors
                ) or "none"
                report.add(f"{label}:metal_sidechains", "PASS",
                           f"metal {metal.ref} protein donors free to relax ({who})",
                           value=len(protein_donors))

    counts = {label: len(frozen) for label, (_, _, frozen) in variants.items()}
    spread = max(counts.values()) - min(counts.values())
    totals = {label: len(s.atoms) for label, (s, _, _) in variants.items()}
    free = {label: totals[label] - counts[label] for label in counts}
    if len(set(free.values())) > 1:
        report.add(
            "free_atom_count", "WARN",
            "variants have different numbers of free atoms "
            + ", ".join(f"{k}={v}" for k, v in sorted(free.items()))
            + f" (frozen-count spread {spread}). Not automatically wrong -- a "
            "different pocket needs a different free set -- but a barrier "
            "difference tracking the free-atom count is a freeze artifact, not "
            "chemistry. Confirm the difference is chemically motivated.",
            value=spread,
        )
    else:
        report.add("free_atom_count", "PASS",
                   f"all variants free {next(iter(free.values()))} atoms")
    return report
