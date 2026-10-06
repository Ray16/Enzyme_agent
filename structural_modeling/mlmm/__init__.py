"""ML/MM reaction energetics for the structural-modeling pipeline.

This package does NOT implement an ML/MM engine. It drives `mlmm-toolkit`
(Ohmura, Inoue & Terada, ChemRxiv 2025, doi:10.26434/chemrxiv-2025-jft1k,
GPL-3.0), which already provides the 3-layer ONIOM machinery, link-atom
coupling, microiteration, path search, saddle optimization, IRC and
thermochemistry. Reimplementing any of that here would be strictly worse.

What this package owns is the part the engine cannot know:

    spec.py        what the agent CLAIMS about one reaction step -- roles,
                   charges, metals, the coordinates that change
    gate.py        whether a given structure is fit to be computed at all,
                   decided BEFORE any GPU time and failing closed
    verdict.py     whether a FINISHED profile may be reported as a barrier --
                   the output gate, fatal rather than advisory
    structure.py   a reader that keeps the things an ML/MM setup lives on
                   (hydrogens, elements, virtual sites) and guesses nothing

Division of labour mirrors ``proxies``: the agent supplies the system-specific
declaration, this package supplies everything derivable from it, and nothing
here knows about any particular enzyme.

Why the gate comes first. In the previous campaign the solver was not what
failed -- the input structures were, and every failure was checkable in advance:
a pose that was never near-attack, a metal shell complete in one variant and not
another, a declared charge inconsistent with the species built, MM virtual sites
counted as atoms, a concerted mechanism missing one coordinate. A barrier
computed from a structure that did not clear this gate is not reportable.

    from structural_modeling.mlmm import (
        AtomRef, BondChange, LigandSpec, MLMMSpec, MetalSite, ReactionStep,
        check_structure, read_pdb,
    )

    spec = MLMMSpec(
        system_id="...", step=ReactionStep(...), net_charge=-3,
        ligands=(LigandSpec("ATP", -4),), metals=(MetalSite(AtomRef("MG")),),
    )
    report = check_structure(read_pdb("system.pdb"), spec)
    if not report.ok:
        raise SystemExit(report.render())
"""
from .gate import (  # noqa: F401
    CLASH_DISTANCE, Check, GateReport, Status, check_freeze_consistency,
    check_structure,
)
from .spec import (  # noqa: F401
    DEFAULT_ANGLE_TOLERANCE, DEFAULT_MAX_ATTACK_DISTANCE,
    DEFAULT_METAL_SHELL_CUTOFF, DEFAULT_MIN_ATTACK_ANGLE, AtomRef, BondChange,
    LigandSpec, MLMMSpec, MetalSite, ReactionStep,
)
from .structure import (  # noqa: F401
    Atom, Structure, StructureError, VIRTUAL_SITE_NAMES, WATER_RESNAMES, Z,
    angle_deg, neighbours, read_pdb,
)

__all__ = [
    # spec
    "AtomRef", "BondChange", "LigandSpec", "MLMMSpec", "MetalSite", "ReactionStep",
    "DEFAULT_MAX_ATTACK_DISTANCE", "DEFAULT_MIN_ATTACK_ANGLE",
    "DEFAULT_ANGLE_TOLERANCE", "DEFAULT_METAL_SHELL_CUTOFF",
    # gate
    "check_structure", "check_freeze_consistency", "GateReport", "Check",
    "Status", "CLASH_DISTANCE",
    # structure
    "read_pdb", "Structure", "Atom", "StructureError", "neighbours", "angle_deg",
    "Z", "WATER_RESNAMES", "VIRTUAL_SITE_NAMES",
]
