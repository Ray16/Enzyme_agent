#!/usr/bin/env python3
"""Resolve WHICH OLIGOMERIC UNIT to cofold, for any enzyme, with provenance.

A modeling determinant of the same kind as catalytic-metal count: a thing a human used
to decide by eye, which silently decides whether the predicted pocket is the real one.
Wrong in either direction is costly - modelling a single subunit whose site is
interfacial leaves part of the pocket open to solvent, while modelling an annotated
oligomer whose sites are independent multiplies cost superlinearly (the pair
representation is ~O(N^2)) and buys nothing.

One rule, read from ``knowledge/oligomeric_state.json``:

    model the assembly  <=>  site_to_nearest_other_chain < ligand_reach

and a refusal to treat UniProt's SUBUNIT comment as an answer to it. SUBUNIT says what
the protein *is*; the rule asks what the *ligand* touches.

Nothing here is specific to a target, family or campaign: thresholds, radii and the
buffer-component exclusion list all live in the knowledge file, and evidence is
discovered from UniProt + RCSB at call time. Per-target measurements belong with the
campaign that made them.

Two failure modes this module exists to prevent, both of which actually happened:

* **Circular justification.** "The oligomer is needed because X% of the pocket comes
  from other subunits" - where X was measured on poses docked into that oligomer.
  Chains can only line a ligand if you put them there. ``assess`` marks such evidence
  circular and refuses to score it.
* **Chain-count misreading.** PDB ``oligomeric_count`` counts chains. For a
  self-processing precursor deposited as one chain per protomer, ``tetrameric(4)`` is
  four protomers, not two.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE = os.path.join(HERE, "knowledge", "oligomeric_state.json")
_RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{}"
_RCSB_ASSEMBLY = "https://data.rcsb.org/rest/v1/core/assembly/{}/{}"
_RCSB_FILE = "https://files.rcsb.org/download/{}-assembly{}.cif"
_UNIPROT = "https://rest.uniprot.org/uniprotkb/{}.json"


def _require_gemmi():
    """Import gemmi or fail LOUDLY -- never let a missing dependency degrade silently.

    Both structural tiers (ligand-contact and site-interface measurement) need
    gemmi.  Historically the ``import gemmi`` lived deep inside those functions, so
    in an env without gemmi the ImportError was swallowed by the per-candidate
    ``except Exception`` handlers and every candidate quietly fell back to the
    annotation-only rank-4 default -- the oligomer reasoning *looked* like it ran
    when it had not.  Resolving the unit without this measurement is not a
    degraded answer, it is no answer, so a missing gemmi must abort with an
    actionable error rather than be recorded as a note.
    """
    try:
        import gemmi  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise RuntimeError(
            "modeling_unit requires the 'gemmi' package for structural measurement "
            "(oligomeric-unit resolution). Install it into the active environment, "
            "e.g. `python -m pip install gemmi`."
        ) from exc
    return gemmi


def knowledge() -> dict:
    with open(KNOWLEDGE) as fh:
        return json.load(fh)


def _param(name, *path, kb=None):
    """Read a parameter from the knowledge file; fail loudly if absent.

    Deliberately has no in-code default: a silent fallback is how a tuned threshold
    becomes an invisible hard-coded constant.
    """
    kb = kb or knowledge()
    node = kb["parameters"][name]
    for key in path:
        node = node[key]
    return node["value"] if isinstance(node, dict) and "value" in node else node


def _get_json(url):
    with urllib.request.urlopen(url) as fh:
        return json.load(fh)


def ligand_reach_angstrom(n_heavy_atoms: int, topology: str = "linear", kb=None) -> float:
    """Half the extended end-to-end ligand length, measured from the reacting atom."""
    kb = kb or knowledge()
    reach = kb["parameters"]["ligand_reach"]
    per_atom = reach["extension_per_heavy_atom_A"][topology]
    return reach["fraction_of_length_from_reacting_atom"] * n_heavy_atoms * per_atom


@dataclass
class UnitEvidence:
    """Evidence for one candidate. Fields left None are absent, not negative."""
    accession: str
    liganded_other_chain_fraction: float | None = None
    liganded_pdb: str | None = None
    liganded_comp_id: str | None = None
    site_to_interface_A: float | None = None
    interface_pdb: str | None = None
    uniprot_subunit: str | None = None
    transferred_from: str | None = None
    derived_from_docked_poses: bool = False
    notes: list[str] = field(default_factory=list)


def assess(evidence: UnitEvidence, ligand_heavy_atoms: int,
           ligand_topology: str = "linear", kb=None) -> dict:
    """Decide the modeling unit and return a resolution record.

    ``unit`` is one of {"assembly", "functional_unit", "undetermined"}; the record also
    reports which evidence rank was used, so a reader can tell whether the choice
    rested on a bound ligand or merely on an annotation.
    """
    kb = kb or knowledge()
    reach = ligand_reach_angstrom(ligand_heavy_atoms, ligand_topology, kb=kb)
    threshold = _param("interfacial_contact_fraction_threshold", kb=kb)
    rec = {
        "accession": evidence.accession,
        "ligand_reach_A": round(reach, 1),
        "origin": "inference",
        "evidence_rank_used": None,
        "unit": "undetermined",
        "rationale": None,
        # informational trace vs things the reader must not ignore - kept apart, since
        # folding notes into warnings makes real warnings invisible
        "notes": list(evidence.notes),
        "warnings": [],
    }

    if evidence.derived_from_docked_poses:
        rec["warnings"].append(
            "CIRCULAR: evidence derived from poses docked into the very assembly it "
            "would justify; not scored. Need a deposited assembly, solution "
            "stoichiometry, or a liganded structure.")

    if evidence.liganded_other_chain_fraction is not None:
        frac = evidence.liganded_other_chain_fraction
        interfacial = frac >= threshold
        rec.update(
            origin="database", evidence_rank_used=1,
            unit="assembly" if interfacial else "functional_unit",
            rationale=(f"{evidence.liganded_comp_id or 'ligand'} bound in "
                       f"{evidence.liganded_pdb}: {frac:.0%} of contact residues come "
                       f"from another chain (threshold {threshold:.0%}) -> site is "
                       f"{'interfacial' if interfacial else 'intra-subunit'}"))
        return rec

    if evidence.site_to_interface_A is not None:
        d = evidence.site_to_interface_A
        beyond = d > reach
        rec.update(
            origin="database", evidence_rank_used=3,
            unit="functional_unit" if beyond else "assembly",
            rationale=(f"catalytic site is {d:.1f} A from the nearest other chain in "
                       f"{evidence.interface_pdb}, "
                       f"{'beyond' if beyond else 'within'} the ligand's {reach:.1f} A "
                       f"reach - the neighbour "
                       f"{'cannot contact' if beyond else 'can line the channel of'} "
                       f"the substrate"))
        if not beyond:
            rec["warnings"].append(
                "weak positive: proximity shows the assembly MAY matter, not that it "
                "does; a liganded structure or a functional-unit/assembly A-B settles it")
        return rec

    if evidence.transferred_from:
        rec.update(origin="experience", evidence_rank_used=5, unit="functional_unit",
                   rationale=(f"transferred from {evidence.transferred_from}, a "
                              f"liganded relative in the same family with an "
                              f"intra-subunit site"))
        return rec

    if evidence.uniprot_subunit:
        rec.update(origin="database", evidence_rank_used=4, unit="assembly",
                   rationale=(f"no structural evidence about the site; falling back to "
                              f"the annotated state ({evidence.uniprot_subunit}) as "
                              f"the conservative choice"))
        rec["warnings"].append(
            "SUBUNIT answers 'what oligomer exists', not 'does the ligand site need "
            "it'; this is a default, not a finding")
        return rec

    rec["rationale"] = "no evidence available"
    return rec


# --------------------------------------------------------------------------- #
# evidence discovery - generic, driven by accession
# --------------------------------------------------------------------------- #

def uniprot_facts(accession: str) -> dict:
    """Sequence, SUBUNIT text, PDB ids and annotated active-site positions."""
    doc = _get_json(_UNIPROT.format(accession))
    subunit = None
    for comment in doc.get("comments", []):
        if comment.get("commentType") == "SUBUNIT":
            subunit = " ".join(t["value"] for t in comment.get("texts", []))
            break
    # Keep the description: an entry often annotates several active-site residues
    # (nucleophile, proton donor, charge relay) and the distance test is only
    # meaningful from the NUCLEOPHILE, where the scissile bond must sit.
    act_sites = [(f["location"]["start"]["value"], (f.get("description") or ""))
                 for f in doc.get("features", []) if f["type"] == "Active site"]
    pdbs = [x["id"] for x in doc.get("uniProtKBCrossReferences", [])
            if x["database"] == "PDB"]
    interpro = [x["id"] for x in doc.get("uniProtKBCrossReferences", [])
                if x["database"] == "InterPro"]
    return {"accession": accession,
            "sequence": doc["sequence"]["value"],
            "subunit": subunit,
            "active_sites": act_sites,
            "pdb_ids": pdbs,
            "interpro": interpro}


_INTERPRO = "https://www.ebi.ac.uk/interpro/api/entry/interpro/{}"
_IPR_TYPE_CACHE: dict[str, str | None] = {}


def interpro_entry_type(ipr_id: str) -> str | None:
    """InterPro entry type ('family', 'domain', 'homologous_superfamily', ...).

    Needed because entry *specificity* decides whether a verdict may be transferred:
    sharing a superfamily domain says nothing about oligomeric behaviour, while
    sharing a family does. Cached per process; returns None if unavailable, which
    callers must treat as 'not transferable' rather than as permission.
    """
    if ipr_id in _IPR_TYPE_CACHE:
        return _IPR_TYPE_CACHE[ipr_id]
    try:
        doc = _get_json(_INTERPRO.format(ipr_id))
        kind = (doc.get("metadata", {}).get("type") or "").lower() or None
    except Exception:
        kind = None
    _IPR_TYPE_CACHE[ipr_id] = kind
    return kind


def nucleophile_position(active_sites, keywords=("nucleophile", "acyl-ester",
                                                 "acyl-enzyme")):
    """Pick the active-site position that marks the nucleophile, if annotated.

    ``active_sites`` is a list of (position, description). Falls back to None rather
    than guessing: measuring from a proton donor instead of the nucleophile silently
    moves the reference point several angstroms and can flip the verdict.
    """
    for pos, desc in active_sites:
        low = desc.lower()
        if any(k in low for k in keywords):
            return pos
    return None


def largest_assembly(pdb_id: str):
    """(assembly_id, oligomeric_count) for the biggest deposited assembly.

    Assembly 1 is often the crystallographic monomer, so never assume it.
    """
    entry = _get_json(_RCSB_ENTRY.format(pdb_id))
    ids = entry.get("rcsb_entry_container_identifiers", {}).get("assembly_ids", [])
    best = (None, -1)
    for aid in ids:
        try:
            info = _get_json(_RCSB_ASSEMBLY.format(pdb_id, aid))
        except Exception:
            continue
        count = info.get("pdbx_struct_assembly", {}).get("oligomeric_count")
        count = int(count) if count is not None else -1
        if count > best[1]:
            best = (aid, count)
    return best


def _load_assembly(pdb_id, assembly_id, cache_dir=None):
    gemmi = _require_gemmi()
    cache_dir = cache_dir or os.path.expanduser("~/.cache/modeling_unit_assemblies")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id}-assembly{assembly_id}.cif")
    if not os.path.exists(path):
        urllib.request.urlretrieve(_RCSB_FILE.format(pdb_id, assembly_id), path)
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_waters()
    return st


def _amino_acid(res):
    gemmi = _require_gemmi()
    info = gemmi.find_tabulated_residue(res.name)
    return info is not None and info.is_amino_acid()


def _chain_sequence(chain):
    gemmi = _require_gemmi()
    residues, letters = [], []
    for res in chain:
        info = gemmi.find_tabulated_residue(res.name)
        if info and info.is_amino_acid():
            residues.append(res)
            letters.append(info.one_letter_code.upper())
    return "".join(letters), residues


def _map_position(uniprot_sequence, chain_sequence, residues, position):
    from Bio import Align
    aligner = Align.PairwiseAligner(mode="global", open_gap_score=-10,
                                    extend_gap_score=-0.5, match_score=2,
                                    mismatch_score=-1)
    aln = aligner.align(uniprot_sequence, chain_sequence)[0]
    for (u0, u1), (p0, p1) in zip(aln.aligned[0], aln.aligned[1]):
        if u0 <= position - 1 < u1:
            return residues[p0 + (position - 1 - u0)]
    return None


def measure_ligand_contacts(pdb_id, assembly_id, cache_dir=None, kb=None):
    """Per bound substrate-like ligand, the other-chain fraction of its contacts."""
    gemmi = _require_gemmi()
    kb = kb or knowledge()
    radius = _param("contact_radius_A", kb=kb)
    min_heavy = _param("min_ligand_heavy_atoms", kb=kb)
    excluded = set(kb["non_substrate_het_codes"]["codes"])

    st = _load_assembly(pdb_id, assembly_id, cache_dir)
    model = st[0]
    search = gemmi.NeighborSearch(st, radius, 0).populate()

    results = []
    for chain in model:
        for res in chain:
            if _amino_acid(res) or res.name in excluded or len(res) < min_heavy:
                continue
            by_chain = {}
            for atom in res:
                for mark in search.find_atoms(atom.pos, "\0", radius=radius):
                    cra = mark.to_cra(model)
                    if not _amino_acid(cra.residue):
                        continue
                    by_chain.setdefault(cra.chain.name, set()).add(cra.residue.seqid.num)
            total = sum(len(v) for v in by_chain.values())
            if not total:
                continue
            own = max(by_chain.items(), key=lambda kv: len(kv[1]))[0]
            other = total - len(by_chain[own])
            results.append({"comp_id": res.name, "chain": chain.name,
                            "contact_residues": total,
                            "other_chain_residues": other,
                            "other_chain_fraction": other / total,
                            "by_chain": {k: len(v) for k, v in by_chain.items()}})
    return results


def measure_site_interface(pdb_id, assembly_id, uniprot_sequence, catalytic_position,
                           cache_dir=None, kb=None):
    """Min distance from a catalytic residue to any other chain, per chain."""
    kb = kb or knowledge()
    near_radius = _param("neighbourhood_radius_A", kb=kb)
    min_len = _param("min_polymer_length", kb=kb)

    st = _load_assembly(pdb_id, assembly_id, cache_dir)
    model = st[0]
    out = []
    for chain in model:
        seq, residues = _chain_sequence(chain)
        if len(seq) < min_len:
            continue
        target = _map_position(uniprot_sequence, seq, residues, catalytic_position)
        if target is None:
            continue
        best = float("inf")
        near = set()
        for other in model:
            if other.name == chain.name:
                continue
            for res in other:
                if not _amino_acid(res):
                    continue
                for atom in res:
                    for t_atom in target:
                        d = t_atom.pos.dist(atom.pos)
                        if d < best:
                            best = d
                        if d <= near_radius:
                            near.add((other.name, res.seqid.num))
        out.append({"chain": chain.name, "residue": target.name,
                    "seqid": target.seqid.num,
                    "min_distance_other_chain_A": (None if best == float("inf")
                                                   else round(best, 1)),
                    f"other_chain_residues_{int(near_radius)}A": len(near)})
    return out


def gather_evidence(accession, catalytic_position=None, cache_dir=None, kb=None):
    """Discover the best available evidence for one accession, generically.

    Walks the hierarchy: prefers a bound substrate-like ligand in the largest deposited
    assembly; falls back to the catalytic-site-to-interface distance; then to SUBUNIT.
    """
    kb = kb or knowledge()
    _require_gemmi()  # fail loud now, not inside a swallowed per-PDB handler below
    facts = uniprot_facts(accession)
    ev = UnitEvidence(accession=accession, uniprot_subunit=facts["subunit"])
    if catalytic_position is None:
        catalytic_position = nucleophile_position(facts["active_sites"])
        if catalytic_position is None and facts["active_sites"]:
            ev.notes.append(
                "no active site is annotated as the nucleophile; distance measured "
                "from an unspecified catalytic residue and may not mark the scissile "
                "position")

    # Rank entries by assembly size first. Walking PDB ids in UniProt order can land
    # on an entry whose largest assembly is a monomer and conclude from it, while a
    # sibling entry of the same protein deposits the full oligomer.
    candidates = []
    for pdb_id in facts["pdb_ids"]:
        try:
            assembly_id, count = largest_assembly(pdb_id)
        except Exception as exc:
            ev.notes.append(f"{pdb_id}: {type(exc).__name__}: {exc}")
            continue
        if assembly_id is not None:
            candidates.append((count, pdb_id, assembly_id))
    candidates.sort(reverse=True)

    for count, pdb_id, assembly_id in candidates:
        try:
            hits = measure_ligand_contacts(pdb_id, assembly_id, cache_dir, kb)
            if hits:
                # Choose the MOST-ENGAGED ligand (most protein contacts), not simply
                # the first qualifying het in chain order: the first one is often an
                # adventitious binder, while the substrate-like ligand makes the most
                # contacts.  Record when the choice was contested so it can be audited.
                best = max(hits, key=lambda h: h["contact_residues"])
                ev.liganded_other_chain_fraction = best["other_chain_fraction"]
                ev.liganded_pdb = pdb_id
                ev.liganded_comp_id = best["comp_id"]
                ev.notes.append(
                    f"{pdb_id} assembly {assembly_id} ({count} chains): "
                    f"{best['comp_id']} contacts {best['by_chain']}")
                if len(hits) > 1:
                    others = ", ".join(sorted({h["comp_id"] for h in hits if h is not best}))
                    ev.notes.append(
                        f"{pdb_id}: chose {best['comp_id']} as most-engaged ligand "
                        f"({best['contact_residues']} contacts) over {others}")
                return ev, facts
            if ev.site_to_interface_A is None and catalytic_position:
                rows = measure_site_interface(pdb_id, assembly_id, facts["sequence"],
                                              catalytic_position, cache_dir, kb)
                distances = [r["min_distance_other_chain_A"] for r in rows
                             if r["min_distance_other_chain_A"] is not None]
                if distances:
                    ev.site_to_interface_A = min(distances)
                    ev.interface_pdb = pdb_id
                    ev.notes.append(
                        f"{pdb_id} assembly {assembly_id} ({count} chains): "
                        f"catalytic position {catalytic_position} is "
                        f"{ev.site_to_interface_A} A from another chain")
        except Exception as exc:  # a missing assembly file must not abort the walk
            ev.notes.append(f"{pdb_id}: {type(exc).__name__}: {exc}")
    return ev, facts


def resolve_panel(accessions, ligand_heavy_atoms, ligand_topology="linear",
                  cache_dir=None, kb=None):
    """Resolve a set of accessions together, then fill gaps by family transfer.

    Resolving as a set is strictly better than one at a time: an accession with no
    structure and no SUBUNIT is undetermined alone, but if a relative sharing an
    InterPro family has rank-1 (liganded) evidence, that verdict transfers. Donors are
    restricted to rank-1 so a weak annotation-only default never propagates, and the
    donor accession plus the shared family are recorded on the recipient.
    """
    kb = kb or knowledge()
    _require_gemmi()  # loud dependency check before the per-accession error swallow
    transferable = set(_param("transferable_interpro_types", kb=kb))
    records, families, sequences = {}, {}, {}
    for acc in accessions:
        try:
            ev, facts = gather_evidence(acc, cache_dir=cache_dir, kb=kb)
            rec = assess(ev, ligand_heavy_atoms, ligand_topology, kb=kb)
            rec["uniprot_subunit"] = facts["subunit"]
            families[acc] = set(facts.get("interpro") or [])
            sequences[acc] = facts["sequence"]
        except Exception as exc:
            rec = {"accession": acc, "unit": "undetermined",
                   "error": f"{type(exc).__name__}: {exc}",
                   "notes": [], "warnings": []}
            families[acc] = set()
        records[acc] = rec

    donors = [(a, r) for a, r in records.items() if r.get("evidence_rank_used") == 1]
    for acc, rec in records.items():
        # Transfer into the undetermined, AND over a rank-4 SUBUNIT default: the
        # hierarchy holds that SUBUNIT answers "what oligomer exists", never "does the
        # ligand site need it", so liganded evidence from a relative is the stronger
        # claim about the site. Without this, orthologs resolve inconsistently -
        # whichever happens to carry a SUBUNIT comment gets the assembly and its
        # siblings get the functional unit.
        # A rank-3 *positive* is a weak positive by construction (reach assumes a fully
        # extended ligand), so liganded evidence from a relative outranks it. A rank-3
        # *negative* ("beyond reach") is definitive geometry and is never overridden.
        # A rank-4 record rests on the accession's OWN annotation, which is direct
        # evidence about that protein; a transfer from a relative must not displace it.
        overridable = (
            rec.get("unit") == "undetermined"
            or (rec.get("evidence_rank_used") == 3 and rec.get("unit") == "assembly")
        )
        if not overridable:
            continue
        best = None
        for donor_acc, donor in donors:
            shared = {ipr for ipr in families[acc] & families[donor_acc]
                      if interpro_entry_type(ipr) in transferable}
            if shared and (best is None or len(shared) > len(best[2])):
                best = (donor_acc, donor, shared)
        if best is None:
            continue
        donor_acc, donor, shared = best
        rec["unit"] = donor["unit"]
        rec["origin"] = "experience"
        rec["evidence_rank_used"] = 5
        rec["rationale"] = (f"transferred from {donor_acc} (shared InterPro "
                            f"{sorted(shared)[0]}), which has liganded evidence: "
                            f"{donor['rationale']}")
        rec.setdefault("warnings", []).append(
            f"family transfer from {donor_acc}; no structure or SUBUNIT for this "
            f"accession itself")

    # Ortholog tier: a still-undetermined accession inherits from a close homolog that
    # has any direct evidence (its own annotation included). Tighter than a shared
    # domain - above the identity threshold the subunit interface is itself conserved -
    # and it stops orthologs of the same protein being modelled inconsistently just
    # because only one of them carries a SUBUNIT comment.
    identity_cut = _param("ortholog_identity_threshold", kb=kb)
    resolved = [(a, r) for a, r in records.items()
                if r.get("unit") != "undetermined"
                and r.get("evidence_rank_used") in (1, 3, 4)]
    for acc, rec in records.items():
        if rec.get("unit") != "undetermined" or acc not in sequences:
            continue
        best = None
        for donor_acc, donor in resolved:
            if donor_acc not in sequences:
                continue
            ident = sequence_identity(sequences[acc], sequences[donor_acc])
            if ident >= identity_cut and (best is None or ident > best[2]):
                best = (donor_acc, donor, ident)
        if best is None:
            rec["rationale"] = (
                "no structure, no SUBUNIT, no family-level or ortholog donor; "
                "modelling the minimal unit as the only defined option")
            rec["unit"] = "functional_unit"
            rec.setdefault("warnings", []).append(
                "NO EVIDENCE for the oligomeric unit; minimal unit chosen by default")
            continue
        donor_acc, donor, ident = best
        rec["unit"] = donor["unit"]
        rec["origin"] = "experience"
        rec["evidence_rank_used"] = 4.5
        rec["rationale"] = (f"ortholog transfer from {donor_acc} at {ident:.0%} "
                            f"sequence identity: {donor['rationale']}")
        rec.setdefault("warnings", []).append(
            f"ortholog transfer from {donor_acc} ({ident:.0%} identity)")
    return records


def sequence_identity(a: str, b: str) -> float:
    """Global pairwise identity, as a fraction of the shorter sequence."""
    from Bio import Align
    aligner = Align.PairwiseAligner(mode="global", open_gap_score=-10,
                                    extend_gap_score=-0.5, match_score=1,
                                    mismatch_score=0)
    aln = aligner.align(a, b)[0]
    matches = sum(1 for x, y in zip(aln[0], aln[1]) if x == y and x != "-")
    return matches / min(len(a), len(b))
