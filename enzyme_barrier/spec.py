"""Declarative REACTION spec for the enzyme reaction-barrier pipeline.

A spec declares only the CHEMISTRY (substrate/cofactor, metal, the atoms that react,
the bond changes) by RESIDUE+ATOM NAMES -- never hard-coded global indices, never a
specific enzyme. The enzyme itself is a separate argument (an Amber topology + a holo
coordinate file) passed to carve(). So one spec (e.g. "adenylation_adipate") drives the
barrier for WT and every design that runs that reaction; swapping the substrate is a new
spec ("adenylation_glutarate"), and swapping the enzyme is just a different prmtop/coords.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class AtomRef:
    """One atom by residue name + atom name (optional residue index to disambiguate a
    resname that occurs more than once). Resolved against the topology at carve time."""
    resname: str
    atom: str
    resid: int | None = None            # optional 0-based residue index disambiguation

    def __str__(self) -> str:
        return f"{self.resname}:{self.atom}" + (f"@{self.resid}" if self.resid is not None else "")


@dataclass(frozen=True)
class BondChange:
    a: AtomRef
    b: AtomRef
    kind: str                            # "forming" | "breaking"


@dataclass(frozen=True)
class ReactionSpec:
    """Chemistry only -- no enzyme, no structure paths. The enzyme (prmtop+coords) and an
    enzyme_id label are arguments to carve()."""
    reaction_id: str                           # e.g. "adenylation_adipate" (the REACTION, not the enzyme)
    ligands: tuple[tuple[str, int], ...]       # (resname, formal_charge), e.g. (("ATP",-4),("DP2",-2))
    metals: tuple[str, ...]                     # metal resnames, e.g. ("MG",)
    nucleophile: AtomRef                         # attacking atom (e.g. adipate/glutarate O4)
    electrophile: AtomRef                        # attacked center (e.g. ATP PA)
    leaving: AtomRef                             # atom across the breaking bond (ATP O3A -> PPi)
    multiplicity: int = 1
    r_lig: float = 6.0                           # frozen-shell radius from ANY ligand atom
    free_r: float = 6.0                          # free-region radius from the reaction atoms
    ion_shell: float = 5.0                       # only neutralize with ions within this of a kept atom
    # Near-attack conditioning. nac_target None => auto = vdW(nuc)+vdW(elec) (general: the
    # nucleophile at van der Waals contact with the electrophile, poised but not bonded).
    # nac_angle None => no angle restraint (scan+gate handle it); else the desired attack
    # angle (deg) for nucleophile-electrophile-leaving, e.g. ~180 in-line SN2@P.
    # A Burgi-Dunitz carbonyl angle requires the carbonyl O, not this leaving atom.
    nac_target: float | None = None              # near-attack d(nuc-elec) to condition to (A); None=vdW sum
    nac_angle: float | None = None               # desired attack angle nuc-elec-leaving (deg); None=unrestrained
    nac_k: float = 50.0                          # NAC distance-restraint force const (kcal/mol/A^2)
    nac_k_angle: float = 50.0                    # NAC angle-restraint force const (kcal/mol/rad^2)
    ion_charges: tuple[tuple[str, int], ...] = (("Na+", 1), ("Cl-", -1))
    water_resnames: tuple[str, ...] = ("WAT", "HOH")
    carbonyl_oxygen: AtomRef | None = None

    @property
    def bond_changes(self) -> tuple[BondChange, ...]:
        return (BondChange(self.nucleophile, self.electrophile, "forming"),
                BondChange(self.electrophile, self.leaving, "breaking"))

    @property
    def reaction_atoms(self) -> tuple[AtomRef, ...]:
        return (self.nucleophile, self.electrophile, self.leaving)

    @classmethod
    def load(cls, path: str | Path) -> "ReactionSpec":
        d = json.loads(Path(path).read_text())
        for k in ("nucleophile", "electrophile", "leaving"):
            d[k] = AtomRef(**d[k])
        if d.get('carbonyl_oxygen') is not None:d['carbonyl_oxygen']=AtomRef(**d['carbonyl_oxygen'])
        d["ligands"] = tuple(tuple(x) for x in d["ligands"])
        d["metals"] = tuple(d["metals"])
        if "ion_charges" in d:
            d["ion_charges"] = tuple(tuple(x) for x in d["ion_charges"])
        if "water_resnames" in d:
            d["water_resnames"] = tuple(d["water_resnames"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def describe(self) -> str:
        bc = "; ".join(f"{b.kind} {b.a}-{b.b}" for b in self.bond_changes)
        lig = ", ".join(f"{r}({q:+d})" for r, q in self.ligands)
        return (f"[{self.reaction_id}] nuc {self.nucleophile} -> {self.electrophile}; {bc}\n"
                f"  ligands {lig}; metals {self.metals}; mult {self.multiplicity}; "
                f"r_lig={self.r_lig} free_r={self.free_r}")
