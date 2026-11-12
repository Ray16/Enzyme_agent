"""Spec-driven active-site cluster builder (Stage 2), generalized from the validated
AS9 builder. Consumes a ReactionSpec (chemistry, by residue+atom name) + an enzyme
(Amber prmtop + holo coords) + an enzyme_id label. No hard-coded indices, no enzyme
baked in. Writes cluster_<enzyme_id>.{xyz,meta.json,map.json}.

Carve rules (fail-closed gates in gate.py): whole residues within r_lig of ANY ligand
atom are kept (frozen env); waters within r_lig of a ligand (O) kept; only ions within
ion_shell of a kept atom are used to neutralize (no stranded distant ions; residual
charge accepted). Cut peptide bonds H-capped. Free region = within free_r of the three
reaction atoms. Net charge from FF partial charges (integer per whole residue).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import parmed as pmd
from ase.data import atomic_numbers
from .spec import ReactionSpec, AtomRef


def _resolve(parm, ref: AtomRef, near_xyz=None, xyz=None) -> int:
    """Resolve an AtomRef to a global atom index. If ref.atom in {AUTO, auto_nearest_O},
    pick the O atom of residue ref.resname nearest to `near_xyz` (geometric nucleophile --
    robust when the substrate's atom names are generic, e.g. from a cofold/antechamber)."""
    if ref.atom.upper() in ("AUTO", "AUTO_NEAREST_O"):
        import numpy as np
        cand = [a.idx for res in parm.residues if res.name == ref.resname
                and (ref.resid is None or res.idx == ref.resid)
                for a in res.atoms if a.element == 8]
        if not cand:
            raise KeyError(f"no O atom in residue {ref.resname} for AUTO nucleophile")
        return int(min(cand, key=lambda i: np.linalg.norm(xyz[i] - near_xyz)))
    matches = []
    for res in parm.residues:
        if res.name == ref.resname and (ref.resid is None or res.idx == ref.resid):
            for a in res.atoms:
                if a.name == ref.atom:
                    matches.append(a.idx)
    if len(matches) > 1:
        raise ValueError(f'Ambiguous atom {ref}; specify a residue index')
    if matches:
        return matches[0]
    raise KeyError(f"atom {ref} not found in topology")


def carve(spec: ReactionSpec, prmtop: str, coords: str, enzyme_id: str,
          out_dir: str | Path, r_lig: float | None = None, free_r: float | None = None) -> dict:
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',enzyme_id):
        raise ValueError('enzyme_id must be a simple nonempty file label')
    r_lig = spec.r_lig if r_lig is None else r_lig
    free_r = spec.free_r if free_r is None else free_r
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    from .coordinates import load_system, reimage_molecules
    parm = load_system(prmtop, coords)
    xyz = parm.coordinates
    LIG = {r for r, _ in spec.ligands} | set(spec.metals)
    IONQ = dict(spec.ion_charges)
    WATER = set(spec.water_resnames)
    # electrophile + leaving by name; nucleophile may be geometric (AUTO) -> needs elec xyz
    i_elec = _resolve(parm, spec.electrophile)
    xyz = reimage_molecules(parm, xyz, i_elec)
    for bond in parm.bonds:
        if np.linalg.norm(xyz[bond.atom1.idx]-xyz[bond.atom2.idx]) > 2.2:
            raise ValueError('Long covalent bond before carving; unwrap or review chemistry')
    for resname, expected_charge in spec.qm_residue_charges().items():
        residues = [r for r in parm.residues if r.name == resname]
        if not residues:
            raise ValueError(f'Missing declared ligand {resname}')
        for residue in residues:
            actual = sum(a.charge for a in residue.atoms)
            if abs(actual-expected_charge) > 0.005:
                raise ValueError(f'Ligand {resname} charge {actual} differs from spec {expected_charge}')
    i_leav = _resolve(parm, spec.leaving)
    i_nuc = _resolve(parm, spec.nucleophile, near_xyz=xyz[i_elec], xyz=xyz)
    rc = xyz[[i_nuc, i_elec, i_leav]]
    path_definition = spec.path_definition([a.atomic_number for a in parm.atoms], xyz,
                                           [i_nuc,i_elec,i_leav])

    lig_atoms = np.array([a.idx for res in parm.residues if res.name in LIG for a in res.atoms])
    lig_xyz = xyz[lig_atoms]

    def min_dist_to_ligand(idxs):
        return float(np.min(np.linalg.norm(xyz[idxs][:, None, :] - lig_xyz[None, :, :], axis=2)))

    keep_res = set()
    for res in parm.residues:
        if res.name in LIG:
            keep_res.add(res.idx); continue
        if res.name in IONQ:
            continue
        ai = np.array([a.idx for a in res.atoms])
        if res.name in WATER:
            o = [a.idx for a in res.atoms if a.element == 8]
            if o and min_dist_to_ligand(np.array([o[0]])) <= r_lig:
                keep_res.add(res.idx)
            continue
        if min_dist_to_ligand(ai) <= r_lig:
            keep_res.add(res.idx)

    # --- charge-group completion: never keep a charged sidechain without its salt-bridge
    # partner. Cutting through a salt bridge traps a bare +/-1 at the boundary, which makes
    # the cluster net charge (and the barrier) jump with carve radius -- an artifact, not
    # physics. Pull in any oppositely-charged sidechain whose charged group is within
    # `sb_cutoff` of a kept charged group, iterating until the salt-bridge network is whole.
    POS = {"ARG": ("NH1", "NH2", "NE"), "LYS": ("NZ",), "HIP": ("ND1", "NE2")}
    NEG = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
    sb_cutoff = getattr(spec, "sb_cutoff", 4.5)

    def _cg_atoms(res):
        names = POS.get(res.name) or NEG.get(res.name)
        if not names:
            return None
        idx = [a.idx for a in res.atoms if a.name in names]
        return np.array(idx) if idx else None

    charged = [(r, +1 if r.name in POS else -1, _cg_atoms(r))
               for r in parm.residues if (r.name in POS or r.name in NEG)]
    charged = [(r, sgn, cg) for r, sgn, cg in charged if cg is not None]
    for _ in range(6):
        added = False
        kept_charged = [(r, sgn, cg) for r, sgn, cg in charged if r.idx in keep_res]
        for r, sgn, cg in charged:
            if r.idx in keep_res:
                continue
            for kr, ksgn, kcg in kept_charged:
                if ksgn == sgn:
                    continue
                dmin = np.min(np.linalg.norm(xyz[cg][:, None, :] - xyz[kcg][None, :, :], axis=2))
                if dmin <= sb_cutoff:
                    keep_res.add(r.idx); added = True; break
        if not added:
            break

    kept_nonion = np.array([a.idx for ridx in keep_res for a in parm.residues[ridx].atoms])

    def charge_of(res_ids):
        q = 0.0
        for ridx in res_ids:
            res = parm.residues[ridx]
            q += IONQ[res.name] if res.name in IONQ else sum(a.charge for a in res.atoms)
        return q

    near_ions = []
    for res in parm.residues:
        if res.name not in IONQ:
            continue
        ai = np.array([a.idx for a in res.atoms])
        d = np.min(np.linalg.norm(xyz[ai][:, None, :] - xyz[kept_nonion][None, :, :], axis=2))
        if d <= spec.ion_shell:
            near_ions.append((res.idx, res.name, d))
    near_ions.sort(key=lambda x: x[2])
    for ridx, rn, d in near_ions:
        cur = charge_of(keep_res)
        if abs(cur) < 0.5:
            break
        if (cur > 0 and IONQ[rn] < 0) or (cur < 0 and IONQ[rn] > 0):
            keep_res.add(ridx)
    netq = int(round(charge_of(keep_res)))

    kept_atoms = sorted(a.idx for ridx in keep_res for a in parm.residues[ridx].atoms)
    keepset = set(kept_atoms); local = {gi: li for li, gi in enumerate(kept_atoms)}

    cap_xyz = [];cap_parents=[]
    for b in parm.bonds:
        a1, a2 = b.atom1.idx, b.atom2.idx
        if (a1 in keepset) == (a2 in keepset):
            continue
        kept, cut = (a1, a2) if a1 in keepset else (a2, a1)
        if parm.atoms[cut].element == 1:
            continue
        v = xyz[cut] - xyz[kept]; v = v / np.linalg.norm(v)
        cap_xyz.append(xyz[kept] + 1.09 * v)
        cap_parents.append(local[kept])

    elems = [parm.atoms[i].element_name for i in kept_atoms] + ["H"] * len(cap_xyz)
    coords_arr = (np.vstack([xyz[kept_atoms], np.array(cap_xyz).reshape(-1, 3)])
                  if cap_xyz else xyz[kept_atoms])
    nelec = sum(atomic_numbers[e] for e in elems) - netq
    if (spec.multiplicity < 1 or spec.multiplicity-1 > nelec or
            (nelec-(spec.multiplicity-1)) % 2):
        raise ValueError(f'Incompatible electrons ({nelec}) and multiplicity {spec.multiplicity}')

    frozen = [li for li, gi in enumerate(kept_atoms)
              if np.min(np.linalg.norm(xyz[gi] - rc, axis=1)) > free_r]
    frozen += list(range(len(kept_atoms), len(elems)))

    meta = dict(reaction_id=spec.reaction_id, enzyme_id=enzyme_id, r_lig=r_lig, free_r=free_r,
                net_charge=netq, multiplicity=spec.multiplicity, n_atoms=len(elems),
                n_free=len(elems) - len(frozen), n_frozen=len(frozen), n_caps=len(cap_xyz),
                nelec=int(nelec), i_nuc=local[i_nuc], i_elec=local[i_elec], i_leav=local[i_leav],
                frozen=frozen, prmtop=str(prmtop))
    meta['path_definition'] = path_definition
    meta['cluster_bonds'] = [[local[b.atom1.idx],local[b.atom2.idx]] for b in parm.bonds
                             if b.atom1.idx in keepset and b.atom2.idx in keepset]
    meta['cluster_bonds'] += [[parent,len(kept_atoms)+k] for k,parent in enumerate(cap_parents)]
    amap = [dict(li=li, gi=int(gi), elem=parm.atoms[gi].element_name,
                 charge=float(parm.atoms[gi].charge), res=parm.atoms[gi].residue.name,
                 resid=int(parm.atoms[gi].residue.idx), name=parm.atoms[gi].name)
            for li, gi in enumerate(kept_atoms)]

    tag = enzyme_id
    with open(out / f"cluster_{tag}.xyz", "w") as f:
        f.write(f"{len(elems)}\n{spec.reaction_id} {enzyme_id} q={netq}\n")
        for e, c in zip(elems, coords_arr):
            f.write(f"{e:2s} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}\n")
    json.dump(meta, open(out / f"cluster_{tag}.meta.json", "w"), indent=2)
    json.dump(dict(n_kept=len(kept_atoms), n_caps=len(cap_xyz), prmtop=str(prmtop), atoms=amap),
              open(out / f"cluster_{tag}.map.json", "w"))
    return meta
