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
    metal_charges: tuple[tuple[str, int], ...] = ()
    metal_shell_A: float = 2.7
    forming_distance_A: float | None = None
    breaking_distance_A: float | None = None
    state_classifier: str = 'bond_exchange'

    def __post_init__(self):
        import math
        if type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError('Multiplicity must be a positive integer')
        if self.state_classifier not in {'bond_exchange','phosphate_transfer'}:
            raise ValueError('Unknown chemical-state classifier')
        for name in ('r_lig','free_r','ion_shell','metal_shell_A'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('forming_distance_A','breaking_distance_A'):
            value=getattr(self,name)
            if value is not None and (not math.isfinite(value) or value<=0):
                raise ValueError(f'{name} must be positive and finite')
        for entries in (self.ligands,self.metal_charges):
            names=[name for name,_ in entries]
            if len(set(names))!=len(names):raise ValueError('Duplicate residue charge declarations')
            if any(type(q) is not int for _,q in entries):
                raise ValueError('Molecular and metal charges must be integers')
        if len(set(self.metals))!=len(self.metals):raise ValueError('Duplicate metal declarations')

    def qm_residue_charges(self) -> dict[str, int]:
        """Explicit molecular/metal charges; only historical Mg has a default."""
        charges = dict(self.ligands)
        declared = dict(self.metal_charges)
        if set(declared) - set(self.metals):
            raise ValueError('metal_charges names must be declared in metals')
        for name in self.metals:
            if name in declared:
                charge = declared[name]
            elif name in charges:
                charge = charges[name]
            elif name.upper() == 'MG':
                charge = 2
            else:
                raise ValueError(f'Explicit metal charge required for {name}')
            if name in charges and charges[name] != charge:
                raise ValueError(f'Conflicting charge declarations for {name}')
            charges[name] = charge
        if any(int(q) != q for q in charges.values()):
            raise ValueError('QM molecular charges must be integers')
        return charges

    def path_definition(self, numbers, positions, reaction_atoms) -> dict:
        """Bond-exchange path targets, independent of enzyme or atom names.

        Missing targets are geometric hypotheses based on covalent radii and the
        initial breaking distance; they are recorded for chemical review.
        """
        import numpy as np
        from ase.data import covalent_radii
        i,j,k = reaction_atoms
        if len({i,j,k})!=3:raise ValueError('Reaction atoms must be distinct')
        x = np.asarray(positions)
        df = float(np.linalg.norm(x[i]-x[j])); db = float(np.linalg.norm(x[j]-x[k]))
        form = self.forming_distance_A
        breaking = self.breaking_distance_A
        if form is None:
            form = float(covalent_radii[numbers[i]] + covalent_radii[numbers[j]])
        if breaking is None:
            breaking = db + 1.5
        if not np.isfinite([df,db,form,breaking]).all() or not 0 < form < df or breaking <= db:
            raise ValueError('Bond-exchange targets must shorten the forming bond and extend the breaking bond')
        return dict(forming_distance_A=float(form),breaking_distance_A=float(breaking),
                    reactant_forming_A=df,reactant_breaking_A=db,
                    targets_inferred=self.forming_distance_A is None or self.breaking_distance_A is None,
                    scope='geometric bond-exchange hypothesis; chemistry and connectivity require review')

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
        if 'metal_charges' in d:
            d['metal_charges'] = tuple(tuple(x) for x in d['metal_charges'])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def describe(self) -> str:
        bc = "; ".join(f"{b.kind} {b.a}-{b.b}" for b in self.bond_changes)
        lig = ", ".join(f"{r}({q:+d})" for r, q in self.ligands)
        return (f"[{self.reaction_id}] nuc {self.nucleophile} -> {self.electrophile}; {bc}\n"
                f"  ligands {lig}; metals {self.metals}; mult {self.multiplicity}; "
                f"r_lig={self.r_lig} free_r={self.free_r}")
