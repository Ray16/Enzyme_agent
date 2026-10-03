#!/usr/bin/env python3
"""Resolve WHICH OLIGOMERIC UNIT to cofold, with provenance.

This is a modeling determinant of the same kind as catalytic-metal count: a thing a
human used to decide by eye, which silently decides whether the predicted pocket is
the real one. Getting it wrong is expensive in both directions - modelling a monomer
whose site is interfacial leaves a third of the pocket open to solvent, while
modelling an annotated hexamer whose six sites are independent multiplies cost ~36x
(pair representation is ~O(N^2)) and buys nothing.

The module encodes one rule, from ``knowledge/oligomeric_state.json``:

    model the assembly  <=>  site_to_nearest_other_chain < ligand_reach

and refuses to treat UniProt's SUBUNIT comment as an answer to it. SUBUNIT says what
the protein *is*; the rule asks what the *ligand* touches.

Two failure modes this module exists to prevent, both of which actually happened:

* **Circular justification.** "The tetramer is needed because 24-41% of the pocket
  comes from other subunits" - where that fraction was measured on poses docked into
  the tetramer. Chains can only line a ligand if you put them there. ``assess`` marks
  any such evidence ``circular`` and refuses to score it.
* **Chain-count misreading.** PDB ``oligomeric_count`` counts chains. For a
  self-processing Ntn hydrolase deposited as one chain per protomer,
  ``tetrameric(4)`` is four protomers, i.e. the tetramer of heterodimers - not two.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE = os.path.join(HERE, "knowledge", "oligomeric_state.json")

# Buffer / cryoprotectant components. They bind crystal-packing interfaces and will
# manufacture a false interfacial signal if treated as substrate analogues.
NON_SUBSTRATE_HETS = {
    "MES", "GOL", "EDO", "PEG", "PG4", "ACT", "DMS", "TRS", "EPE", "FMT", "IMD",
    "SO4", "PO4", "CL", "NA", "K", "CA", "MG", "ZN", "MN", "FE", "CO", "NI", "CU",
    "HOH",
}


def knowledge() -> dict:
    with open(KNOWLEDGE) as fh:
        return json.load(fh)


def ligand_reach_angstrom(n_heavy_atoms: int, topology: str = "linear") -> float:
    """Half the extended end-to-end length of the ligand, from the reacting atom.

    The distal end of the substrate is what reaches an interface, so reach is measured
    from the scissile bond outward, not across the whole molecule. ~1.27 A of extension
    per heavy atom along an sp3 chain; branched/cyclic ligands extend far less per atom.
    """
    per_atom = 1.27 if topology == "linear" else 0.6
    return 0.5 * n_heavy_atoms * per_atom


@dataclass
class UnitEvidence:
    """Evidence for one candidate. Fields left None are simply absent, not negative."""
    accession: str
    # rank-1: residues within 5 A of a bound SUBSTRATE-LIKE ligand, split by chain
    liganded_other_chain_fraction: float | None = None
    liganded_pdb: str | None = None
    liganded_comp_id: str | None = None
    # rank-3: nucleophile -> nearest other-chain atom, in the LARGEST assembly
    site_to_interface_A: float | None = None
    interface_pdb: str | None = None
    # rank-4 (Q1 only)
    uniprot_subunit: str | None = None
    # rank-5
    transferred_from: str | None = None
    # provenance hygiene
    derived_from_docked_poses: bool = False
    notes: list[str] = field(default_factory=list)


def assess(evidence: UnitEvidence, ligand_heavy_atoms: int,
           ligand_topology: str = "linear") -> dict:
    """Decide the modeling unit and return a resolution record.

    Returns ``unit`` in {"assembly", "functional_unit", "undetermined"} together with
    the basis actually used, so a downstream reader can see whether the choice rested
    on a bound ligand or merely on an annotation.
    """
    reach = ligand_reach_angstrom(ligand_heavy_atoms, ligand_topology)
    rec = {
        "accession": evidence.accession,
        "ligand_reach_A": round(reach, 1),
        "origin": "inference",
        "evidence_rank_used": None,
        "unit": "undetermined",
        "rationale": None,
        "warnings": list(evidence.notes),
    }

    if evidence.derived_from_docked_poses:
        rec["warnings"].append(
            "CIRCULAR: evidence derived from poses docked into the very assembly it "
            "would justify; not scored. Need a deposited assembly, solution "
            "stoichiometry, or a liganded structure.")

    # rank 1 - a bound substrate-like ligand is decisive
    if evidence.liganded_other_chain_fraction is not None:
        frac = evidence.liganded_other_chain_fraction
        interfacial = frac >= 0.15
        rec.update(
            origin="database",
            evidence_rank_used=1,
            unit="assembly" if interfacial else "functional_unit",
            rationale=(
                f"{evidence.liganded_comp_id or 'ligand'} bound in "
                f"{evidence.liganded_pdb}: {frac:.0%} of 5 A contact residues come "
                f"from another chain -> site is "
                f"{'interfacial' if interfacial else 'intra-subunit'}"),
        )
        return rec

    # rank 3 - distance falsifier
    if evidence.site_to_interface_A is not None:
        d = evidence.site_to_interface_A
        if d > reach:
            rec.update(
                origin="database", evidence_rank_used=3, unit="functional_unit",
                rationale=(f"catalytic site is {d:.1f} A from the nearest other chain "
                           f"in {evidence.interface_pdb}, beyond the ligand's "
                           f"{reach:.1f} A reach - the neighbour cannot contact the "
                           f"substrate"))
        else:
            rec.update(
                origin="database", evidence_rank_used=3, unit="assembly",
                rationale=(f"catalytic site is {d:.1f} A from the nearest other chain "
                           f"in {evidence.interface_pdb}, within the ligand's "
                           f"{reach:.1f} A reach - the neighbour can line the "
                           f"specificity channel"))
            rec["warnings"].append(
                "weak positive: proximity shows the assembly MAY matter, not that it "
                "does; a liganded structure or a functional-unit/assembly A-B would "
                "settle it")
        return rec

    # rank 5 - family transfer
    if evidence.transferred_from:
        rec.update(origin="experience", evidence_rank_used=5, unit="functional_unit",
                   rationale=(f"transferred from {evidence.transferred_from}, a "
                              f"liganded relative in the same family with an "
                              f"intra-subunit site"))
        return rec

    # rank 4 - annotation only. Deliberately NOT decisive.
    if evidence.uniprot_subunit:
        rec.update(
            origin="database", evidence_rank_used=4, unit="assembly",
            rationale=(f"no structural evidence for the site; falling back to the "
                       f"annotated state ({evidence.uniprot_subunit}) as the "
                       f"conservative choice"))
        rec["warnings"].append(
            "SUBUNIT answers 'what oligomer exists', not 'does the ligand site need "
            "it'; this is a default, not a finding")
        return rec

    rec["rationale"] = "no evidence available"
    return rec


def measure_site_interface(pdb_id: str, assembly_id: str, uniprot_sequence: str,
                           catalytic_position: int, cache_dir: str | None = None):
    """Min distance from a catalytic residue to any other chain, in a real assembly.

    Maps ``catalytic_position`` (UniProt numbering) onto the deposited chains by
    alignment, because PDB numbering routinely differs. Returns a dict per chain.
    Requires gemmi + biopython; imported lazily so the decision logic above stays
    dependency-free and unit-testable.
    """
    import urllib.request
    import gemmi
    from Bio import Align

    cache_dir = cache_dir or os.path.expanduser("~/.cache/panel_assemblies")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id}-assembly{assembly_id}.cif")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id}-assembly{assembly_id}.cif", path)

    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_waters()
    model = st[0]

    def chain_seq(chain):
        residues, letters = [], []
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            if info and info.is_amino_acid():
                residues.append(res)
                letters.append(info.one_letter_code.upper())
        return "".join(letters), residues

    aligner = Align.PairwiseAligner(mode="global", open_gap_score=-10,
                                    extend_gap_score=-0.5, match_score=2,
                                    mismatch_score=-1)

    out = []
    for chain in model:
        seq, residues = chain_seq(chain)
        if len(seq) < 50:
            continue
        aln = aligner.align(uniprot_sequence, seq)[0]
        target = None
        for (u0, u1), (p0, p1) in zip(aln.aligned[0], aln.aligned[1]):
            if u0 <= catalytic_position - 1 < u1:
                target = residues[p0 + (catalytic_position - 1 - u0)]
                break
        if target is None:
            continue
        best = float("inf")
        near12 = set()
        for other in model:
            if other.name == chain.name:
                continue
            for res in other:
                info = gemmi.find_tabulated_residue(res.name)
                if not (info and info.is_amino_acid()):
                    continue
                for atom in res:
                    for t_atom in target:
                        d = t_atom.pos.dist(atom.pos)
                        best = min(best, d)
                        if d <= 12.0:
                            near12.add((other.name, res.seqid.num))
        out.append({"chain": chain.name, "residue": target.name,
                    "seqid": target.seqid.num,
                    "min_distance_other_chain_A": None if best == float("inf") else round(best, 1),
                    "other_chain_residues_12A": len(near12)})
    return out
