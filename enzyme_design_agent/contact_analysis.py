"""Versioned, hydrogen-free protein-donor / ligand-acceptor contact analysis.

This deliberately identifies contacts, not hydrogen bonds or interaction energies.
Target groups must come from a chemical graph, never from spatial clustering alone.
No coordinates, protonation states, or hydrogens are generated or optimized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ContactPolicy:
    version: str = "heavy_atom_donor_acceptor_v2"
    minimum_contact_A: float = 2.4
    polar_max_A: float = 3.5
    ionic_max_A: float = 4.0
    audit_max_A: float = 4.5
    scope: str = "protein_sidechain_to_ligand_acceptor"

    def __post_init__(self):
        if self.scope != "protein_sidechain_to_ligand_acceptor":
            raise ValueError("Unsupported scope; backbone and other interaction classes need explicit profiles")
        if not 0 < self.polar_max_A <= self.audit_max_A or not 0 < self.ionic_max_A <= self.audit_max_A:
            raise ValueError("Invalid distance thresholds")
        if not 0 < self.minimum_contact_A < min(self.polar_max_A, self.ionic_max_A):
            raise ValueError('Invalid minimum contact distance')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Atom:
    chain: str
    residue: str
    resi: str
    name: str
    element: str
    xyz: tuple[float, float, float]
    confidence: float | None = None

    def __post_init__(self):
        if len(self.xyz) != 3 or not all(math.isfinite(v) for v in self.xyz):
            raise ValueError('Coordinates must be three finite numbers')
        if not self.chain or not self.resi or not self.name:
            raise ValueError('Complete atom identity is required')


@dataclass(frozen=True)
class TargetGroup:
    """Named acceptor atoms and charge of their resonance-delocalized group.

    A negative group charge enables potential ionic classification at all its
    specified acceptor atoms. Unsupported chemistry must not be guessed from PDB.
    """
    group_id: str
    chain: str
    resi: str
    acceptors: tuple[str, ...]
    formal_charge: int
    provenance: str


DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"},
    "ASN": {"ND2"}, "GLN": {"NE2"}, "SER": {"OG"},
    "THR": {"OG1"}, "TYR": {"OH"}, "TRP": {"NE1"},
    "HIS": {"ND1", "NE2"}, "CYS": {"SG"},
}
STANDARD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
CATIONIC = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}}
COVALENT_RADII = {"N": .71, "O": .66, "S": 1.05}


def evaluate_contacts(atoms: Iterable[Atom], groups: Iterable[TargetGroup],
                      policy: ContactPolicy = ContactPolicy(), protein_chain="A") -> list[dict]:
    """Audit all nearby sidechain N/O/S pairs, including explicit rejection reasons.

    His tautomer and Cys thiol state are unresolved: these are recorded separately
    and excluded from the primary count. Lys/Arg ionic classification is explicitly
    potential, under their conventional cationic states, not an inferred microstate.
    Only overlap below the sum of covalent radii is rejected as a severe clash;
    ordinary van der Waals overlap is not evidence against a hydrogen bond.
    """
    atoms, groups = list(atoms), list(groups)
    amap = {(a.chain, a.resi, a.name): a for a in atoms}
    if len(amap) != len(atoms):
        raise ValueError("Duplicate atom identities / unresolved alternate conformations")
    prot = [a for a in atoms if a.chain == protein_chain and a.element in {"N", "O", "S"}
            and a.name not in {"N", "O", "OXT"}]
    out = []
    for g in groups:
        if not g.acceptors or not g.provenance:
            raise ValueError("Acceptor identity and chemical provenance are required")
        if len(set(g.acceptors)) != len(g.acceptors):
            raise ValueError('Duplicate acceptor identities')
        for name in g.acceptors:
            target = amap[(g.chain, g.resi, name)]  # missing atoms fail closed
            if target.element not in {'N', 'O', 'S'}:
                raise ValueError('Unsupported acceptor element for this profile')
            for a in prot:
                d = float(np.linalg.norm(np.subtract(a.xyz, target.xyz)))
                if d > policy.audit_max_A:
                    continue
                donor = a.name in DONORS.get(a.residue, set())
                ionic = a.name in CATIONIC.get(a.residue, set()) and g.formal_charge < 0
                kind = "potential_ionic" if ionic else "donor_acceptor_polar"
                limit = policy.ionic_max_A if ionic else policy.polar_max_A
                ambiguous = a.residue in {"HIS", "CYS"}
                severe_clash = d < COVALENT_RADII.get(a.element, .7) + COVALENT_RADII.get(target.element, .7)
                if a.residue not in STANDARD:
                    reason = "untyped_residue"
                elif not donor:
                    reason = "incompatible_donor_acceptor"
                elif severe_clash:
                    reason = "severe_clash"
                elif d < policy.minimum_contact_A:
                    reason = 'anomalously_short_contact'
                elif d > limit:
                    reason = "beyond_class_distance"
                elif ambiguous:
                    reason = "conditional_protonation_or_tautomer"
                else:
                    reason = "compatible_heavy_atom_contact"
                out.append(dict(chain=a.chain, res=a.residue, ri=a.resi, atom=a.name,
                                target_chain=g.chain, target_resi=g.resi, O=name,
                                group=g.group_id, d=d, kind=kind, cutoff_A=limit,
                                accepted=reason == "compatible_heavy_atom_contact",
                                conditional=reason in {"conditional_protonation_or_tautomer", 'anomalously_short_contact'},
                                reason=reason))
    # Quarantine the whole residue/group if ANY chemically compatible pair is
    # compressed. A longer alternative to its other oxygen must not hide it.
    flagged = {group_key(c) for c in out if c['reason'] in {'severe_clash', 'anomalously_short_contact'}}
    for c in out:
        if group_key(c) in flagged and (c['accepted'] or c['conditional']):
            c['accepted'] = False
            c['conditional'] = True
            if c['reason'] not in {'severe_clash', 'anomalously_short_contact'}:
                c['reason'] = 'group_contains_anomalously_short_contact'
    return out


def group_key(c):
    return c["chain"], c["ri"], c["res"], c["target_chain"], c["target_resi"], c["group"]


def collapse_for_display(contacts: list[dict]) -> list[dict]:
    """One displayed line per residue/target group; retain two-point geometry.

    Maximum atom-disjoint matching, then minimum total distance. Two donors to two
    acceptors are reported as a possible two-point grip in the audit. The main
    figure draws only the shortest pair. This is a display reduction only.
    """
    grouped = {}
    for c in contacts:
        if c["accepted"] or c["conditional"]:
            grouped.setdefault(group_key(c), []).append(c)
    result = []
    for key, pairs in sorted(grouped.items()):
        pairs = sorted(pairs, key=lambda c: (c["d"], c["atom"], c["O"]))
        best = [pairs[0]]
        for size in range(2, min(len({p['atom'] for p in pairs}), len({p['O'] for p in pairs})) + 1):
            choices = [s for s in combinations(pairs, size)
                       if len({p["atom"] for p in s}) == size and len({p["O"] for p in s}) == size]
            if choices:
                best = list(min(choices, key=lambda s: sum(p["d"] for p in s)))
        result.append(dict(**{k: pairs[0][k] for k in ("chain", "res", "ri", "target_chain", "target_resi", "group", "kind", "conditional")},
                           pairs=[pairs[0]], independent_pairs=best,
                           uncertainty_reasons=sorted({p['reason'] for p in pairs if p['conditional']}),
                           atom_pair_count=len(pairs), display_line_count=1,
                           collapsed_atom_pairs=len(pairs) - 1))
    return result


def contact_keys(contacts):
    return {group_key(c) for c in contacts if c["accepted"]}


def distinct_residue_count(contacts):
    return len({(c["chain"], c["ri"]) for c in contacts if c["accepted"]})
