"""The contract between a system-specific resolver and the generic proxy library.

DIVISION OF LABOUR
------------------
This package is deliberately system-free. It does not know what enzyme is being
studied, what cofactor is present, or which atom attacks which. Those are
properties of one reaction, and a general module that tried to encode them would
either be wrong for most systems or quietly become a catalogue of special cases.

So the split is:

    the AGENT writes, per system:   a resolver -- the code that reads one predicted
                                    pose and returns a ReactiveSite, i.e. which atom
                                    is the nucleophile, which is the electrophile,
                                    what leaves, and what the ideal attack angle is
                                    for that chemistry. It is short, specific, and
                                    the agent VALIDATES it (see ``check_resolver``).

    this package provides:          the ReactiveSite contract, a pose parser, the
                                    ensemble machinery, the proxy library computed
                                    from ReactiveSite alone, and the statistics that
                                    decide whether any proxy's score means anything.

A resolver is a callable ``(Pose) -> ReactiveSite | None``, returning None for a
pose where the roles cannot be found. Nothing else about it is constrained.

WHY ``leaving`` AND ``charge_centre`` ARE SEPARATE
--------------------------------------------------
Most substitutions at a trigonal/tetrahedral centre share one regularity that is
worth knowing when writing a resolver: the leaving group is a BRIDGING substituent
(bonded onward, so it departs as part of a fragment) while the atom that develops
negative charge is a TERMINAL substituent. That is a useful heuristic for the
agent's resolver, not a rule this module enforces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

#: A system-specific resolver, written by the agent for one reaction.
Resolver = Callable[["Pose"], "ReactiveSite | None"]


@dataclass(frozen=True)
class ReactiveSite:
    """One pose's catalytic geometry, in role terms. The agent's resolver fills this.

    nucleophile   attacking atom                                      (required)
    electrophile  atom under attack                                   (required)
    leaving       atom departing along the breaking bond; None if the mechanism
                  has no leaving group (e.g. a simple addition)
    charge_centre atom accumulating negative charge in the transition state --
                  the target of oxyanion-hole-style stabilisation; None if the
                  transition state is not polarised this way
    frame         other substituents on the electrophile, defining its local
                  plane; needed only by the trajectory-alignment proxy
    ideal_angle   ideal nucleophile-electrophile-leaving angle in degrees for this
                  chemistry (~180 in-line displacement, ~107 Burgi-Dunitz addition)

    Proxies that need a role you leave as None are skipped for that complex and
    reported as missing, never as zero.
    """

    nucleophile: np.ndarray
    electrophile: np.ndarray
    leaving: np.ndarray | None = None
    charge_centre: np.ndarray | None = None
    frame: tuple[np.ndarray, ...] = field(default_factory=tuple)
    ideal_angle: float = 180.0

    @property
    def attack_distance(self) -> float:
        return float(np.linalg.norm(self.nucleophile - self.electrophile))


@dataclass(frozen=True)
class Pose:
    """A parsed predicted structure, with no interpretation applied.

    ``ligands`` maps chain id -> [(atom_name, element, xyz)] and is kept keyed by
    composition rather than by position, because chain ORDER differs between
    campaigns modelling the same chemistry; a resolver that indexes chains by
    number reads the wrong entity without erroring.
    """

    ca: np.ndarray                       # protein CA, for ensemble superposition
    protein: np.ndarray                  # protein heavy atoms
    donors: np.ndarray                   # protein H-bond donor N/O
    bases: np.ndarray                    # Arg/Lys/His sidechain N
    ligands: dict[str, list[tuple[str, str, np.ndarray]]]

    def chain_with(self, element: str) -> str | None:
        """First ligand chain containing ``element``. Composition, not position."""
        for ch, atoms in self.ligands.items():
            if any(e == element for _n, e, _x in atoms):
                return ch
        return None

    def lone_atom_chain(self, element: str) -> str | None:
        """Chain that is a single atom of ``element`` (e.g. an isolated metal)."""
        for ch, atoms in self.ligands.items():
            if len(atoms) == 1 and atoms[0][1] == element:
                return ch
        return None

    def bonded(self, centre: np.ndarray, chain: str, element: str | None = None,
               cut: float = 1.75) -> list[tuple[str, str, np.ndarray]]:
        """Heavy atoms of ``chain`` within ``cut`` of ``centre``, excluding itself."""
        out = []
        for name, elem, xyz in self.ligands.get(chain, []):
            if element is not None and elem != element:
                continue
            d = float(np.linalg.norm(xyz - centre))
            if 1e-6 < d < cut:
                out.append((name, elem, xyz))
        return out


_DONOR_SC = {
    "ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"},
    "TYR": {"OH"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "HIS": {"ND1", "NE2"},
    "TRP": {"NE1"}, "CYS": {"SG"},
}
_BASE_SC = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"}}


def parse_pose(path: str) -> Pose:
    """Read a PDB into a Pose. Stdlib + numpy only, no Biopython."""
    ca, prot, don, bas = [], [], [], []
    lig: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            name = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21]
            elem = (line[76:78].strip() or name[0]).upper()
            if elem == "H":
                continue
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if rec == "HETATM":
                lig.setdefault(chain, []).append((name, elem, xyz))
                continue
            prot.append(xyz)
            if name == "CA":
                ca.append(xyz)
            if name == "N" or (resn in _DONOR_SC and name in _DONOR_SC[resn]):
                don.append(xyz)
            if resn in _BASE_SC and name in _BASE_SC[resn]:
                bas.append(xyz)

    def arr(v):
        return np.array(v) if v else np.zeros((0, 3))

    return Pose(ca=arr(ca), protein=arr(prot), donors=arr(don), bases=arr(bas), ligands=lig)


# --------------------------------------------------------------------------- #
# validating a system-specific resolver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolverReport:
    n_poses: int
    n_resolved: int
    coverage: float
    distance_range: tuple[float, float] | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        head = (f"resolver: {self.n_resolved}/{self.n_poses} poses resolved "
                f"({self.coverage:.0%})")
        if self.distance_range:
            head += f", attack distance {self.distance_range[0]:.2f}-{self.distance_range[1]:.2f} A"
        return head + ("\n  OK" if self.ok else
                       "".join(f"\n  PROBLEM: {p}" for p in self.problems))


def check_resolver(resolver: Resolver, poses: Sequence[Pose],
                   min_coverage: float = 0.9,
                   plausible_distance: tuple[float, float] = (1.5, 25.0)) -> ResolverReport:
    """Sanity-check a system-specific resolver before trusting any proxy built on it.

    Catches the failure modes that silently produce confident nonsense: the
    resolver finding nothing, picking the same atom for nucleophile and
    electrophile, returning physically impossible separations, or selecting a
    nucleophile that is actually part of the electrophile's own molecule (an
    intramolecular self-match, the most common bug when a resolver forgets to
    exclude the donor chain).

    This validates MECHANICS, not biochemistry. It cannot tell you the resolver
    picked the chemically correct atom -- only the agent, reasoning about the
    reaction, can do that.
    """
    problems: list[str] = []
    sites, dists, selfmatch, clash = [], [], 0, 0
    for pose in poses:
        site = resolver(pose)
        if site is None:
            continue
        sites.append(site)
        d = site.attack_distance
        dists.append(d)
        if d < 0.5:
            selfmatch += 1
        if site.leaving is not None and np.allclose(site.leaving, site.nucleophile):
            clash += 1

    n, k = len(poses), len(sites)
    coverage = (k / n) if n else 0.0
    if n == 0:
        problems.append("no poses supplied")
    elif coverage < min_coverage:
        problems.append(f"resolved only {coverage:.0%} of poses (want >={min_coverage:.0%}); "
                        "the selection rule is probably too strict or chain-order dependent")
    if selfmatch:
        problems.append(f"{selfmatch} poses have nucleophile == electrophile "
                        "(forgot to exclude the electrophile's own molecule?)")
    if clash:
        problems.append(f"{clash} poses have leaving group == nucleophile")
    rng = None
    if dists:
        lo, hi = float(min(dists)), float(max(dists))
        rng = (lo, hi)
        if lo < plausible_distance[0]:
            problems.append(f"minimum attack distance {lo:.2f} A is below a bonded "
                            "contact; roles are probably mis-assigned")
        if lo > plausible_distance[1]:
            problems.append(f"closest approach over the whole ensemble is {lo:.1f} A; "
                            "the two roles may be on unrelated entities")
    if sites:
        ideals = {round(s.ideal_angle, 1) for s in sites}
        if len(ideals) > 1:
            problems.append(f"ideal_angle is not constant across poses: {sorted(ideals)}")
    return ResolverReport(n_poses=n, n_resolved=k, coverage=coverage,
                          distance_range=rng, problems=tuple(problems))
