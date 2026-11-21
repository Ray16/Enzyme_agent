"""Substrate-accommodation contact reasoning for the structural modeling agent.

Given a holo structure (a Boltz cofold pose PDB, or an MD trajectory), this reasons about
*why* a substrate is (or is not) well accommodated in an enzyme active site: it locates each
substrate moiety geometrically, finds the protein residues that contact it, classifies each
contact by chemistry (salt bridge / H-bond / hydrophobic), and scores whether the substrate is
anchored at both ends (the "bidentate clamp" that distinguishes step-1 holders from drifters).

Works for step 1 (a di-carboxylate acid: reactive COO- locked to the catalytic Mg/phosphate,
distal COO- clamped across the pocket) and step 2 (an amine nucleophile: does an acidic/H-bond
pocket hold the attacking N at the activated carbonyl). No RDKit dependency -- moieties are found
by covalent geometry so it is robust on raw cofold atom names.

Boltz cofold convention (resname LIG, split by chain):
  protein = chain A (standard residues) ; ATP = chain B (PA/PB/PG, O3A ...) ;
  Mg = chain C ; carboxylate/amine substrate = chain D.

This module only DETECTS interactions (facts) and aggregates their FREQUENCY. It renders no verdict
and no hold/drift call -- the agent reasons over the frequency table (and pairs it with a structural
render) to validate anchors, weigh ambiguous ones (e.g. His salt bridges, protonation-dependent),
and flag issues. Prefer a frequency aggregate (ensemble / traj) over any single static pose.

CLI:
  # one structure -> interaction facts
  python -m structural_modeling_agent.analysis.substrate_contacts pose --pose model.pdb
  # contact FREQUENCY across the 50 Boltz poses of one enzyme
  python -m structural_modeling_agent.analysis.substrate_contacts ensemble --poses-json poses.json
  # contact FREQUENCY across MD frames
  python -m structural_modeling_agent.analysis.substrate_contacts traj --top c.pdb --traj prod.dcd
  # per-candidate ensemble across a manifest (each entry may carry a "poses" list)
  python -m structural_modeling_agent.analysis.substrate_contacts panel --manifest m.json --json out.json
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import json, sys, argparse
import numpy as np
from scipy.spatial import cKDTree

from ..io_utils import bounded_workers, init_single_thread_worker

try:
    import MDAnalysis as mda
except Exception as e:  # pragma: no cover
    mda = None
    _MDA_ERR = e

# ---- chemistry tables -------------------------------------------------------
CATION_ATOMS = {  # side-chain cationic nitrogens that make a salt bridge to COO-
    # His is cationic ONLY when doubly protonated: AMBER names that HIP (CHARMM HSP). Neutral His
    # (HIE/HID/HSE/HSD) and the protonation-UNASSIGNED generic "HIS" from a Boltz cofold PDB are NOT
    # cations and must not be scored as salt bridges -- they are H-bond donors (below). Keying salt
    # bridges on a bare "HIS" silently treated every (even neutral) histidine as a cation.
    "ARG": ["NH1", "NH2", "NE"], "LYS": ["NZ"], "HIP": ["ND1", "NE2"], "HSP": ["ND1", "NE2"],
}
HBOND_DONOR_RES = {  # neutral polar side chains / backbone that can H-bond a carboxylate O
    "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"], "ASN": ["ND2"], "GLN": ["NE2"],
    # every neutral His tautomer + protonation-unassigned generic "HIS" -> neutral H-bond donor
    "HIS": ["ND1", "NE2"], "HIE": ["ND1", "NE2"], "HID": ["ND1", "NE2"],
    "HSE": ["ND1", "NE2"], "HSD": ["ND1", "NE2"], "TRP": ["NE1"], "CYS": ["SG"],
}
ACIDIC_ATOMS = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}  # anchor an ammonium (step 2)

SALT_CUT = 4.0     # N(+)...O(-) salt bridge
HBOND_CUT = 3.6    # donor...acceptor H-bond (heavy-atom)
HPHOB_CUT = 4.5    # C...C hydrophobic packing
MG_COORD_CUT = 2.7


def _sel(u, chain):
    # cofold chains arrive as segid OR chainID depending on reader; try both
    for key in (f"segid {chain}", f"chainID {chain}"):
        try:
            g = u.select_atoms(key)
            if len(g):
                return g
        except Exception:
            continue
    return u.select_atoms("index -1")  # empty


def _bonded(a, group, cut=1.9):
    d = np.linalg.norm(group.positions - a.position, axis=1)
    return [g for g, dist in zip(group, d) if 0.1 < dist < cut]


def detect_functional_groups(sub):
    """Generic functional-group detection on ANY substrate by covalent geometry (no RDKit, no
    chemistry-family assumption). Returns a list of moieties, each a dict with a formal-charge SIGN
    used to decide which protein contacts count as salt bridges. Covers:
      carboxylate/phosphate O-clusters (anionic -1), amine N not bonded to a carbonyl (cationic +1),
      amide/neutral N (0), hydroxyl O (0), thiol S (0). Extend here for new chemistries."""
    C = sub.select_atoms("name C*"); O = sub.select_atoms("name O*")
    N = sub.select_atoms("name N*"); S = sub.select_atoms("name S*")
    P = sub.select_atoms("name P*")
    groups = []
    # carboxylate / carbonic: a carbon bonded to >=2 oxygens (anionic when deprotonated)
    for c in C:
        bo = _bonded(c, O)
        if len(bo) >= 2:
            groups.append({"kind": "carboxylate", "charge": -1,
                           "atoms": [c] + bo, "center": c})
    # phosphate groups on the substrate itself (rare; anionic)
    for p in P:
        bo = _bonded(p, O)
        if len(bo) >= 3:
            groups.append({"kind": "phosphate", "charge": -1, "atoms": [p] + bo, "center": p})
    # nitrogens: amine (cationic) vs amide/aromatic (neutral) by whether bonded to a carbonyl C
    carbonyl_C = {c.index for c in C if len(_bonded(c, O)) >= 1 and len(_bonded(c, O)) < 2
                  and any(len(_bonded(o, O)) == 0 for o in _bonded(c, O))}
    for n in N:
        bc = _bonded(n, C)
        is_amide = any(c.index in carbonyl_C for c in bc)
        groups.append({"kind": "amide_N" if is_amide else "amine",
                       "charge": 0 if is_amide else 1, "atoms": [n], "center": n})
    # hydroxyls: O bonded to exactly one C and not part of a carboxylate (neutral)
    carbox_O = {o.index for g in groups if g["kind"] == "carboxylate" for o in g["atoms"][1:]}
    for o in O:
        if o.index in carbox_O:
            continue
        if len(_bonded(o, C)) == 1 and not _bonded(o, P):
            groups.append({"kind": "hydroxyl", "charge": 0, "atoms": [o], "center": o})
    # thiols
    for s in S:
        groups.append({"kind": "thiol", "charge": 0, "atoms": [s], "center": s})
    return groups


# metal ions commonly seen as catalytic cofactors (element symbols). Generic, not Mg-specific.
METAL_ELEMENTS = {"MG", "MN", "ZN", "CA", "FE", "NI", "CO", "CU", "NA", "K",
                  "CD", "MO", "W", "V", "SE"}


def _element(atom):
    """Best-effort element symbol from MDAnalysis (element attr, else type, else name)."""
    for attr in ("element", "type"):
        v = getattr(atom, attr, "") or ""
        if v:
            return v.strip().upper()
    return "".join(ch for ch in atom.name if ch.isalpha()).upper()


def split_ligands(u, hetero_sel=None, substrate_resnames=None, cofactor_resnames=None):
    """Classify NON-protein ligand atoms by COMPOSITION, for ANY enzyme (no fixed chains, no Mg
    assumption). Per ligand chain/residue:
      - a lone metal ion (or all-metal chain) -> metal pool (any element, any count)
      - a chain containing P bonded to >=3 O (nucleotide / phospho cofactor) -> cofactor pool
      - everything else organic -> substrate pool
    Optional overrides (from a reactive-site spec) pin substrate/cofactor by residue name.
    Returns (cofactor, metals, substrate) AtomGroups."""
    het = u.select_atoms(hetero_sel) if hetero_sel else \
        u.select_atoms("not protein and not resname HOH WAT SOL TIP3 T3P")
    if not len(het):
        raise ValueError("no non-protein ligand atoms found")
    cof_i, metal_i, sub_i = [], [], []
    # group by (segid, resid) so multi-residue ligands split correctly
    for seg in sorted(set(het.segids)):
        segatoms = het.select_atoms(f"segid {seg}")
        for res in segatoms.residues:
            g = res.atoms
            rn = res.resname.upper()
            if substrate_resnames and rn in substrate_resnames:
                sub_i += list(g.indices); continue
            if cofactor_resnames and rn in cofactor_resnames:
                cof_i += list(g.indices); continue
            elems = {_element(a) for a in g}
            is_metal_only = elems <= METAL_ELEMENTS or (len(g) == 1 and _element(g[0]) in METAL_ELEMENTS)
            if is_metal_only:
                metal_i += list(g.indices); continue
            # phospho-cofactor? phosphorus bonded to >=3 O
            P = g.select_atoms("name P*"); O = g.select_atoms("name O*")
            is_cofactor = any(len(_bonded(p, O)) >= 3 for p in P) and len(g) > 6
            (cof_i if is_cofactor else sub_i).extend(list(g.indices))
    return u.atoms[cof_i], u.atoms[metal_i], u.atoms[sub_i]


def _reaction_center(u, cof, metals, reaction_center_sel=None):
    """Points defining the electrophilic reaction center, used to rank which substrate group is
    'reactive'. Priority: explicit selection (from a spec) > cofactor phosphorus > cofactor atoms
    > metal(s). Returns (points array|None, description)."""
    if reaction_center_sel:
        g = u.select_atoms(reaction_center_sel)
        if len(g):
            return g.positions, reaction_center_sel
    if len(cof):
        P = cof.select_atoms("name P*")
        if len(P):
            return P.positions, "cofactor_phosphorus"
        return cof.positions, "cofactor_atoms"
    if len(metals):
        return metals.positions, "metal"
    return None, None


def identify_groups(u, mech=None, reaction_center_sel=None,
                    substrate_resnames=None, cofactor_resnames=None, hetero_sel=None):
    """Generic, enzyme-agnostic: detect every substrate functional group and rank them by proximity
    to the reaction center (the group closest to it is 'reactive'; the rest are secondary anchoring
    sites). Returns a dict with functional groups, metals, cofactor, and reaction-center info."""
    cof, metals, sub = split_ligands(u, hetero_sel=hetero_sel, substrate_resnames=substrate_resnames,
                                     cofactor_resnames=cofactor_resnames)
    if not len(sub):
        raise ValueError("no substrate atoms resolved")
    fgs = detect_functional_groups(sub)
    rc_pts, rc_desc = _reaction_center(u, cof, metals, reaction_center_sel)
    for fg in fgs:
        pos = np.array([a.position for a in fg["atoms"]])
        if rc_pts is not None and len(rc_pts):
            fg["d_to_center"] = float(np.min(np.linalg.norm(pos[:, None, :] - rc_pts[None, :, :], axis=2)))
        else:
            fg["d_to_center"] = None
    ranked = sorted([g for g in fgs if g["d_to_center"] is not None], key=lambda g: g["d_to_center"])
    reactive = ranked[0] if ranked else (fgs[0] if fgs else None)
    for fg in fgs:
        fg["role"] = "reactive" if fg is reactive else "secondary"
    return {"functional_groups": fgs, "metals": metals, "cofactor": cof,
            "reaction_center": rc_desc, "reactive_group": reactive,
            "d_reactive_to_center_A": round(reactive["d_to_center"], 2)
            if reactive and reactive["d_to_center"] is not None else None}


def _classify_residue(res_atoms, target_pos, resname, target_charge, mg_pos=None):
    """Best contact category between one protein residue and a target moiety of known charge.
    A salt bridge requires COMPLEMENTARY charge (cation<->carboxylate anion; acid<->ammonium);
    a like-charged acidic residue near a carboxylate is NOT a salt bridge -- if a metal sits
    between them it is flagged metal_bridge, else ignored (would be repulsive)."""
    def mind(names):
        a = res_atoms.select_atoms("name " + " ".join(names)) if names else res_atoms.select_atoms("index -1")
        if not len(a):
            return 1e9, None
        dmat = np.linalg.norm(a.positions[:, None, :] - target_pos[None, :, :], axis=2)
        i = np.unravel_index(np.argmin(dmat), dmat.shape)
        return float(dmat[i]), a.positions[i[0]]
    # complementary salt bridge
    if target_charge < 0 and resname in CATION_ATOMS:      # carboxylate <- cation
        d, _ = mind(CATION_ATOMS[resname])
        if d <= SALT_CUT:
            return ("salt_bridge", d)
    if target_charge > 0 and resname in ACIDIC_ATOMS:      # ammonium <- acid
        d, _ = mind(ACIDIC_ATOMS[resname])
        if d <= SALT_CUT:
            return ("salt_bridge", d)
    # A like-charged acidic residue near a carboxylate is NOT a substrate anchor: it is the
    # metal-binding catalytic scaffold (coordinates the Mg, not the substrate), and it is
    # electrostatically repulsive to the carboxylate. Do not report it as a contact at all.
    if target_charge < 0 and resname in ACIDIC_ATOMS:
        return (None, 1e9)
    # SIDE-CHAIN H-bond (residue-identity-specific -> a real specificity anchor)
    if resname in HBOND_DONOR_RES:
        d, _ = mind(HBOND_DONOR_RES[resname])
        if d <= HBOND_CUT:
            return ("h_bond", d)
    # BACKBONE amide H-bond: reported as a distinct category because it is residue-identity-
    # INDEPENDENT (any residue's backbone N there would do it) -- e.g. Ala/Gly have no polar side
    # chain, so their only "contact" is backbone and is NOT a sequence-specific anchor.
    dN, _ = mind(["N"])
    if dN <= HBOND_CUT:
        return ("backbone_hbond", dN)
    # hydrophobic packing (any side-chain carbon)
    dC, _ = mind(["CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE", "CZ"])
    if dC <= HPHOB_CUT:
        return ("hydrophobic", dC)
    return (None, 1e9)


def _contacts_to(u, target_atoms, target_charge, mg_pos=None):
    """Protein residues contacting the target; one best-category row per residue.
    target_charge: -1 for a carboxylate, +1 for an amine (sets which salt bridges are valid)."""
    if target_atoms is None:
        return []
    tpos = np.array([a.position for a in target_atoms]) if not hasattr(target_atoms, "positions") \
        else target_atoms.positions
    near = u.select_atoms(f"protein and around {HPHOB_CUT + 0.5} group tgt",
                          tgt=_as_group(u, target_atoms))
    out = []
    for res in near.residues:
        cat, d = _classify_residue(res.atoms, tpos, res.resname, target_charge, mg_pos)
        if cat:
            out.append({"res": f"{res.resname}{res.resid}", "category": cat, "dist_A": round(d, 2)})
    out.sort(key=lambda r: r["dist_A"])
    return out


def _as_group(u, atoms):
    if hasattr(atoms, "indices"):
        return atoms
    idx = [a.index for a in atoms]
    return u.atoms[idx]


def _fg_atomgroup(u, fg):
    return u.atoms[[a.index for a in fg["atoms"]]]


def _metal_coordinated(fg, metals):
    if not len(metals):
        return False
    pos = np.array([a.position for a in fg["atoms"]])
    return bool(np.min(np.linalg.norm(pos[:, None, :] - metals.positions[None, :, :], axis=2)) <= MG_COORD_CUT)


def _resname_of(contact):
    return "".join(ch for ch in contact["res"] if ch.isalpha()).upper()


def analyze_structure(u, mech=None, reaction_center_sel=None,
                      substrate_resnames=None, cofactor_resnames=None, hetero_sel=None):
    """DETECT (one structure, any enzyme) every substrate functional group and its interactions with
    every protein residue -- salt bridge (complementary charge only), H-bond, hydrophobic, metal
    coordination -- with distances. PURE DETECTION: returns facts, no hold/drift verdict. The agent
    reasons over these facts, ideally over a frequency AGGREGATE across many Boltz poses or MD frames
    (see analyze_ensemble / analyze_trajectory), never one static pose. hetero_sel restricts which
    non-protein atoms are considered ligands (use it for solvated MD to exclude water/counter-ions)."""
    g = identify_groups(u, mech, reaction_center_sel, substrate_resnames, cofactor_resnames, hetero_sel)
    metals = g["metals"]
    results = []
    for fg in g["functional_groups"]:
        ag = _fg_atomgroup(u, fg)
        contacts = _contacts_to(u, ag, fg["charge"], metals.positions if len(metals) else None)
        results.append({
            "kind": fg["kind"], "role": fg["role"], "charge": fg["charge"],
            "d_to_center_A": round(fg["d_to_center"], 2) if fg["d_to_center"] is not None else None,
            "metal_coordinated": _metal_coordinated(fg, metals),
            "contacts": contacts,
        })
    return {
        "reaction_center": g["reaction_center"],
        "d_reactive_to_center_A": g["d_reactive_to_center_A"],
        "n_functional_groups": len(results),
        "metals_present": int(len(metals)),
        "groups": results,
    }


def _aggregate(frame_dicts):
    """Aggregate per-structure DETECTION into contact FREQUENCY statistics -- the robust signal to
    reason over instead of a single pose. frame_dicts = list of analyze_structure() outputs (from a
    Boltz-pose ensemble or MD frames). Per substrate group (keyed role:kind): the fraction of
    structures each (residue, interaction-category) is present, the metal-coordination frequency, and
    the mean distance of the group to the reaction center."""
    from collections import defaultdict, Counter
    n = len(frame_dicts)
    if not n:
        return {"n_structures": 0, "groups": {}}
    gc = defaultdict(Counter); gmc = Counter(); gd = defaultdict(list)
    for fr in frame_dicts:
        for grp in fr.get("groups", []):
            gk = f"{grp['role']}:{grp['kind']}"
            for key in {(c["res"], c["category"]) for c in grp["contacts"]}:  # once per structure
                gc[gk][key] += 1
            if grp["metal_coordinated"]:
                gmc[gk] += 1
            if grp["d_to_center_A"] is not None:
                gd[gk].append(grp["d_to_center_A"])
    out = {}
    for gk, c in gc.items():
        out[gk] = {
            "contacts": [{"res": k[0], "category": k[1], "frequency": round(v / n, 3)}
                         for k, v in sorted(c.items(), key=lambda kv: -kv[1])],
            "metal_coordination_freq": round(gmc[gk] / n, 3),
            "d_to_center_A_mean": round(float(np.mean(gd[gk])), 2) if gd[gk] else None,
            "d_to_center_A_std": round(float(np.std(gd[gk])), 2) if gd[gk] else None,
        }
    return {"n_structures": n, "groups": out}


_HPHOB_ATOMS = ("CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE", "CZ")


def _prepare_trajectory_plan(u, reaction_center_sel=None, substrate_resnames=None,
                             cofactor_resnames=None, hetero_sel=None):
    """Precompute topology-invariant atom indices once on frame 0 for fast MD traversal."""
    cof, metals, sub = split_ligands(
        u, hetero_sel=hetero_sel, substrate_resnames=substrate_resnames,
        cofactor_resnames=cofactor_resnames,
    )
    if not len(sub):
        raise ValueError("no substrate atoms resolved")
    fgs = detect_functional_groups(sub)
    fg_specs = [
        {"kind": fg["kind"], "charge": fg["charge"], "idx": np.asarray([a.index for a in fg["atoms"]], dtype=int)}
        for fg in fgs
    ]
    # Reaction center atom indices (same priority as _reaction_center)
    rc_idx = None
    rc_desc = None
    if reaction_center_sel:
        g = u.select_atoms(reaction_center_sel)
        if len(g):
            rc_idx, rc_desc = np.asarray(g.indices, dtype=int), reaction_center_sel
    if rc_idx is None and len(cof):
        p_atoms = cof.select_atoms("name P*")
        if len(p_atoms):
            rc_idx, rc_desc = np.asarray(p_atoms.indices, dtype=int), "cofactor_phosphorus"
        else:
            rc_idx, rc_desc = np.asarray(cof.indices, dtype=int), "cofactor_atoms"
    if rc_idx is None and len(metals):
        rc_idx, rc_desc = np.asarray(metals.indices, dtype=int), "metal"
    metal_idx = np.asarray(metals.indices, dtype=int)

    # Pre-index protein atoms and per-residue chemistry atom indices once
    prot = u.select_atoms("protein")
    prot_idx = np.asarray(prot.indices, dtype=int)
    res_specs = []
    atom_to_res = np.empty(len(prot_idx), dtype=int)
    offset = 0
    for r_i, res in enumerate(prot.residues):
        atoms = res.atoms
        n_a = len(atoms)
        atom_to_res[offset : offset + n_a] = r_i
        offset += n_a
        rn = res.resname
        name_to_idx = {}
        for a in atoms:
            name_to_idx.setdefault(a.name, []).append(a.index)

        def _pick(names):
            out = []
            for nm in names:
                out.extend(name_to_idx.get(nm, ()))
            return np.asarray(out, dtype=int) if out else None

        res_specs.append({
            "label": f"{rn}{res.resid}",
            "resname": rn,
            "cation": _pick(CATION_ATOMS[rn]) if rn in CATION_ATOMS else None,
            "acid": _pick(ACIDIC_ATOMS[rn]) if rn in ACIDIC_ATOMS else None,
            "hbond": _pick(HBOND_DONOR_RES[rn]) if rn in HBOND_DONOR_RES else None,
            "backbone_n": _pick(("N",)),
            "hphob": _pick(_HPHOB_ATOMS),
        })
    return {
        "fg_specs": fg_specs,
        "rc_idx": rc_idx,
        "rc_desc": rc_desc,
        "metal_idx": metal_idx,
        "prot_idx": prot_idx,
        "atom_to_res": atom_to_res,
        "res_specs": res_specs,
    }


def _min_dist(coords, idx_a, target_pos):
    if idx_a is None or not len(idx_a):
        return 1e9
    pos_a = coords[idx_a]
    return float(np.min(np.linalg.norm(pos_a[:, None, :] - target_pos[None, :, :], axis=2)))


def _classify_residue_fast(coords, spec, target_pos, target_charge):
    rn = spec["resname"]
    if target_charge < 0 and rn in CATION_ATOMS:
        d = _min_dist(coords, spec["cation"], target_pos)
        if d <= SALT_CUT:
            return ("salt_bridge", d)
    if target_charge > 0 and rn in ACIDIC_ATOMS:
        d = _min_dist(coords, spec["acid"], target_pos)
        if d <= SALT_CUT:
            return ("salt_bridge", d)
    if target_charge < 0 and rn in ACIDIC_ATOMS:
        return (None, 1e9)
    if rn in HBOND_DONOR_RES:
        d = _min_dist(coords, spec["hbond"], target_pos)
        if d <= HBOND_CUT:
            return ("h_bond", d)
    d_n = _min_dist(coords, spec["backbone_n"], target_pos)
    if d_n <= HBOND_CUT:
        return ("backbone_hbond", d_n)
    d_c = _min_dist(coords, spec["hphob"], target_pos)
    if d_c <= HPHOB_CUT:
        return ("hydrophobic", d_c)
    return (None, 1e9)


def _analyze_frame_fast(coords, plan):
    """Evaluate one MD frame using precomputed topology indices and a single cKDTree."""
    fg_specs = plan["fg_specs"]
    rc_idx = plan["rc_idx"]
    metal_idx = plan["metal_idx"]
    prot_idx = plan["prot_idx"]
    atom_to_res = plan["atom_to_res"]
    res_specs = plan["res_specs"]

    rc_pts = coords[rc_idx] if rc_idx is not None and len(rc_idx) else None
    metal_pts = coords[metal_idx] if len(metal_idx) else None
    prot_tree = cKDTree(coords[prot_idx]) if len(prot_idx) else None

    fg_positions = [coords[fg["idx"]] for fg in fg_specs]
    d_centers = []
    for pos in fg_positions:
        if rc_pts is not None:
            d_centers.append(float(np.min(np.linalg.norm(pos[:, None, :] - rc_pts[None, :, :], axis=2))))
        else:
            d_centers.append(None)

    ranked_indices = [i for i, d in enumerate(d_centers) if d is not None]
    if ranked_indices:
        reactive_i = min(ranked_indices, key=lambda i: d_centers[i])
    else:
        reactive_i = 0 if fg_specs else None

    results = []
    near_cut = HPHOB_CUT + 0.5
    for i, (fg, tpos, d_center) in enumerate(zip(fg_specs, fg_positions, d_centers)):
        contacts = []
        if prot_tree is not None and len(tpos):
            hit_lists = prot_tree.query_ball_point(tpos, r=near_cut)
            near_res = set()
            for hits in hit_lists:
                for h in hits:
                    near_res.add(int(atom_to_res[h]))
            for r_i in near_res:
                spec = res_specs[r_i]
                cat, d = _classify_residue_fast(coords, spec, tpos, fg["charge"])
                if cat:
                    contacts.append({"res": spec["label"], "category": cat, "dist_A": round(d, 2)})
            contacts.sort(key=lambda r: r["dist_A"])
        metal_coord = (
            bool(np.min(np.linalg.norm(tpos[:, None, :] - metal_pts[None, :, :], axis=2)) <= MG_COORD_CUT)
            if metal_pts is not None else False
        )
        results.append({
            "kind": fg["kind"],
            "role": "reactive" if i == reactive_i else "secondary",
            "charge": fg["charge"],
            "d_to_center_A": round(d_center, 2) if d_center is not None else None,
            "metal_coordinated": metal_coord,
            "contacts": contacts,
        })
    d_react = d_centers[reactive_i] if reactive_i is not None else None
    return {
        "reaction_center": plan["rc_desc"],
        "d_reactive_to_center_A": round(d_react, 2) if d_react is not None else None,
        "n_functional_groups": len(results),
        "metals_present": int(len(metal_idx)),
        "groups": results,
    }


def _analyze_pose_worker(args):
    p, mech, reaction_center_sel, substrate_resnames, cofactor_resnames = args
    try:
        return analyze_structure(
            mda.Universe(p), mech, reaction_center_sel, substrate_resnames, cofactor_resnames
        ), False
    except Exception:
        return None, True


def analyze_ensemble(pose_paths, mech=None, reaction_center_sel=None,
                     substrate_resnames=None, cofactor_resnames=None, workers=None):
    """Contact-frequency statistics across an ENSEMBLE of static poses (e.g. the 50 Boltz cofold
    models for one enzyme). Pairs detection with statistics so the agent reasons over how OFTEN an
    interaction occurs, not whether it happens to occur in one pose."""
    pose_paths = list(pose_paths)
    n_workers = bounded_workers(workers, n_items=len(pose_paths), min_items_for_pool=6, max_cap=8)
    frames, errs = [], 0
    if n_workers > 1:
        tasks = [(p, mech, reaction_center_sel, substrate_resnames, cofactor_resnames) for p in pose_paths]
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_single_thread_worker) as pool:
            for res, failed in pool.map(_analyze_pose_worker, tasks):
                if failed:
                    errs += 1
                else:
                    frames.append(res)
    else:
        for p in pose_paths:
            try:
                frames.append(analyze_structure(mda.Universe(p), mech, reaction_center_sel,
                                                substrate_resnames, cofactor_resnames))
            except Exception:
                errs += 1
    agg = _aggregate(frames)
    agg["source"] = "boltz_pose_ensemble"
    agg["n_poses_input"] = len(pose_paths)
    agg["n_failed"] = errs
    return agg


def analyze_trajectory(top, traj, mech=None, reaction_center_sel=None,
                       substrate_resnames=None, cofactor_resnames=None, stride=1, hetero_sel=None):
    """Contact-frequency statistics across MD frames (same aggregate as the pose ensemble). `traj` may
    be a single DCD or a LIST of DCDs (e.g. replicate seeds sharing one topology)."""
    u = mda.Universe(top, traj)
    plan = _prepare_trajectory_plan(
        u, reaction_center_sel=reaction_center_sel,
        substrate_resnames=substrate_resnames, cofactor_resnames=cofactor_resnames,
        hetero_sel=hetero_sel,
    )
    frames = [_analyze_frame_fast(u.atoms.positions, plan) for _ in u.trajectory[::stride]]
    agg = _aggregate(frames)
    agg["source"] = "md_trajectory"
    agg["n_frames"] = len(frames)
    return agg


# ---- CLI --------------------------------------------------------------------
def _cli():
    if mda is None:
        sys.exit(f"MDAnalysis required: {_MDA_ERR}")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pose", help="detect interactions in ONE structure (facts, no verdict)")
    p1.add_argument("--pose", required=True)
    pe = sub.add_parser("ensemble", help="contact FREQUENCY across many Boltz poses")
    pe.add_argument("--poses-json", help="JSON list of pose PDB paths")
    pe.add_argument("--pose-dir", help="directory of *.pdb poses")
    p2 = sub.add_parser("traj", help="contact FREQUENCY across MD frames")
    p2.add_argument("--top", required=True); p2.add_argument("--traj", required=True, nargs="+")
    p2.add_argument("--stride", type=int, default=1)
    p3 = sub.add_parser("panel", help="per-candidate ensemble frequency across a manifest")
    p3.add_argument("--manifest", required=True)
    for p in (p1, pe, p2, p3):
        p.add_argument("--mech", default=None, help="optional hint; analysis is generic")
        p.add_argument("--reaction-center-sel", default=None,
                       help="MDAnalysis selection for the electrophile/reaction center (from a spec)")
        p.add_argument("--substrate-resnames", default=None, help="comma-list to pin substrate residues")
        p.add_argument("--cofactor-resnames", default=None, help="comma-list to pin cofactor residues")
        p.add_argument("--hetero-sel", default=None,
                       help="MDAnalysis selection for ligand atoms (MD: exclude water/counter-ions)")
        p.add_argument("--json", default=None)
    a = ap.parse_args()
    subr = set(a.substrate_resnames.upper().split(",")) if a.substrate_resnames else None
    cofr = set(a.cofactor_resnames.upper().split(",")) if a.cofactor_resnames else None
    hsel = getattr(a, "hetero_sel", None)
    if a.cmd == "pose":
        res = analyze_structure(mda.Universe(a.pose), a.mech, a.reaction_center_sel, subr, cofr, hsel)
    elif a.cmd == "ensemble":
        import glob as _glob
        paths = json.load(open(a.poses_json)) if a.poses_json else sorted(_glob.glob(f"{a.pose_dir}/*.pdb"))
        res = analyze_ensemble(paths, a.mech, a.reaction_center_sel, subr, cofr)
    elif a.cmd == "traj":
        res = analyze_trajectory(a.top, a.traj, a.mech, a.reaction_center_sel, subr, cofr, a.stride, hsel)
    else:
        man = json.load(open(a.manifest))
        res = {}
        for o in man:
            try:
                poses = o.get("poses") or [o["pose"]]   # ensemble if provided, else single pose
                res[o["label"]] = analyze_ensemble(poses, o.get("mech"), a.reaction_center_sel, subr, cofr)
                res[o["label"]]["name"] = o.get("name"); res[o["label"]]["acc"] = o.get("acc")
            except Exception as e:
                res[o["label"]] = {"error": str(e)}
    out = json.dumps(res, indent=1)
    if a.json:
        open(a.json, "w").write(out)
        print(f"wrote {a.json}")
    else:
        print(out)


if __name__ == "__main__":
    _cli()
