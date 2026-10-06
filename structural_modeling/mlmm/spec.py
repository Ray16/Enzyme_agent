"""What the agent declares about one ML/MM reaction step.

Division of labour follows ``proxies``: the agent supplies the system-specific
declaration, this package supplies everything computable from it. Nothing here
knows about any particular enzyme.

A spec is a *claim* about the chemistry -- which atom attacks what, what leaves,
what the net charge is, which metals must be fully coordinated. ``gate.py`` tests
the claim against an actual structure before any GPU time is spent. The claim is
written down so that a barrier can never be reported without the mechanism it
assumed being recoverable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Near-attack windows. Deliberately generous: the gate's job is to reject
# structures that cannot react at all, not to pre-judge a barrier. The
# mechanism-specific ideal angles live in
# modeling_reasoning/knowledge/reaction_coordinates.json (180 in-line
# displacement, ~107 Burgi-Dunitz addition) and are passed in per step.
DEFAULT_MAX_ATTACK_DISTANCE = 4.0     # A, nucleophile -> electrophile
DEFAULT_MIN_ATTACK_ANGLE = 150.0      # deg, for an in-line (180) mechanism
DEFAULT_ANGLE_TOLERANCE = 30.0        # deg around a non-180 ideal angle

# Octahedral Mg-O first shell. 2.6 A is the cutoff a previous campaign used for
# its coordination check; it is loose enough to count a slightly long water and
# tight enough to exclude second-shell oxygens.
DEFAULT_METAL_SHELL_CUTOFF = 2.6      # A


@dataclass(frozen=True)
class AtomRef:
    """A pointer to one atom, resolved by name plus enough qualifiers to be unique.

    Never an index. Boltz names ligand atoms by RDKit canonical rank, which moves
    between RDKit versions, and mlmm freeze lists are 1-based into a topology that
    changes whenever a water is added -- so an index that was right once is a
    silent landmine. ``Structure.select`` raises on ambiguity.
    """

    name: str
    resseq: int | None = None
    resname: str | None = None
    chain: str | None = None

    def __str__(self) -> str:
        parts = [p for p in (self.chain, self.resname, self.resseq) if p is not None]
        return f"{'/'.join(str(p) for p in parts)}/{self.name}" if parts else self.name


@dataclass(frozen=True)
class BondChange:
    """One bond formed or broken along the reaction coordinate.

    ``target`` is the distance in A that the scan drives this bond to, matching
    the ``mlmm scan --scan-lists`` contract.
    """

    kind: Literal["form", "break"]
    first: AtomRef
    second: AtomRef
    target: float

    def __str__(self) -> str:
        arrow = "->" if self.kind == "form" else "x"
        return f"{self.first} {arrow} {self.second} @ {self.target:.2f} A"


@dataclass(frozen=True)
class MetalSite:
    """A catalytic metal whose first shell must be complete before reacting.

    Incomplete or inconsistent metal coordination was the single largest source
    of spurious barriers in the previous campaign: a 5-coordinate vs 6-coordinate
    Mg between two variants produced an 11 kcal/mol difference that was an
    artifact. ``expected_coordination=None`` means "do not check", which the gate
    reports as an explicit abstention rather than a pass.
    """

    ref: AtomRef
    expected_coordination: int | None = 6
    shell_cutoff: float = DEFAULT_METAL_SHELL_CUTOFF
    formal_charge: int = 2
    # If the nucleophile coordinates the metal it is sequestered, not activated --
    # true for phosphoryl transfer, false for carbonic anhydrase and two-metal
    # polymerases where the metal activates the nucleophile. Opt-in per system.
    nucleophile_must_be_free: bool = False


@dataclass(frozen=True)
class LigandSpec:
    """A non-protein species and the formal charge it is modelled with.

    The charge is declared, not inferred. ``mlmm mm-parm -l`` needs it, the ML
    region net charge is built from it, and an acyl-adenylate written as -1 when
    it is -2 breaks the electron count without any stage failing.
    """

    resname: str
    formal_charge: int
    expected_heavy_atoms: int | None = None
    expect_hydrogens: bool = True


@dataclass(frozen=True)
class ReactionStep:
    """One elementary step: the Michaelis state plus the bonds that change."""

    step_id: str
    nucleophile: AtomRef
    electrophile: AtomRef
    leaving_group: AtomRef | None = None
    bond_changes: tuple[BondChange, ...] = ()
    ideal_attack_angle: float = 180.0
    max_attack_distance: float = DEFAULT_MAX_ATTACK_DISTANCE
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE
    # A proton that moves as part of the same step. Omitting a real one produced a
    # +225 kcal/mol endpoint in a previous campaign; a concerted mechanism must
    # declare every coordinate that changes.
    proton: AtomRef | None = None
    proton_acceptor: AtomRef | None = None

    @property
    def min_attack_angle(self) -> float:
        return max(0.0, self.ideal_attack_angle - self.angle_tolerance)

    @property
    def max_attack_angle(self) -> float:
        return min(180.0, self.ideal_attack_angle + self.angle_tolerance)

    @property
    def n_coordinates(self) -> int:
        """How many coordinates change. >2 is a hard TS search; worth knowing up front."""
        n = len(self.bond_changes)
        return n + 1 if self.proton is not None else n


@dataclass(frozen=True)
class MLMMSpec:
    """The complete, checkable claim about one ML/MM reaction step.

    ``net_charge`` is the ML region's total charge and ``multiplicity`` its spin
    multiplicity (2S+1), exactly as ``mlmm -q/-m`` take them. Both are declared
    here so the gate can verify electron parity against the actual atom list
    instead of trusting a number typed into a shell script.
    """

    system_id: str
    step: ReactionStep
    net_charge: int
    multiplicity: int = 1
    ligands: tuple[LigandSpec, ...] = ()
    metals: tuple[MetalSite, ...] = ()
    # Waters completing a metal shell or bridging the active site. Boltz does not
    # place waters at all, so these are added during prep and their presence is a
    # precondition, not a detail.
    min_active_site_waters: int = 0
    active_site_radius: float = 6.0
    # Residues whose sidechains must be free to relax. The freeze scheme must be
    # matched across variants that will be compared: freeing one variant's
    # metal-coordinating sidechain while freezing another's makes the resulting
    # difference an artifact of the freeze list.
    free_sidechain_resseqs: tuple[int, ...] = ()
    notes: str = ""
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def expected_ligand_charge(self) -> int:
        return sum(l.formal_charge for l in self.ligands)

    @property
    def unpaired_electrons(self) -> int:
        return self.multiplicity - 1

    def describe(self) -> str:
        lines = [
            f"{self.system_id} / step {self.step.step_id}",
            f"  q = {self.net_charge:+d}, multiplicity = {self.multiplicity}",
            f"  attack: {self.step.nucleophile} -> {self.step.electrophile}"
            + (f", leaving {self.step.leaving_group}" if self.step.leaving_group else ""),
            f"  ideal angle {self.step.ideal_attack_angle:.0f} deg,"
            f" {self.step.n_coordinates} changing coordinate(s)",
        ]
        for bc in self.step.bond_changes:
            lines.append(f"    {bc}")
        for m in self.metals:
            want = m.expected_coordination
            lines.append(
                f"  metal {m.ref}: coordination "
                + (f"{want}" if want is not None else "UNCHECKED")
                + f" within {m.shell_cutoff} A"
            )
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return "\n".join(lines)
