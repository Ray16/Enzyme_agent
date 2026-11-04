"""Canonical extractor for processed two-site Ntn complexes.

Preserves the published campaign's atom selection, assembly QC, register and
aggregation definitions. This module has no dependency on archived code or on
study scripts outside structural_modeling_agent. Scope is checked before application.
"""
from __future__ import annotations
import collections
import itertools
import json
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

def parse_model(path):
    """-> (protein dict, ligand dict). protein[chain][resnum] = {atom: xyz, '_r': resname}"""
    prot = collections.defaultdict(lambda: collections.defaultdict(dict))
    lig = collections.defaultdict(list)
    for l in open(path):
        if l.startswith("ATOM"):
            ch, rn, an = l[21], int(l[22:26]), l[12:16].strip()
            prot[ch][rn][an] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            prot[ch][rn]["_r"] = l[17:20].strip()
        elif l.startswith("HETATM"):
            el = (l[76:78].strip() or l[12:16].strip()[0]).upper()
            lig[l[21]].append((el, np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return prot, lig



def amide_carbonyls(atoms, internal_only=True):
    """Ligand amide C=O carbons: a C with an O at ~1.2 A and an N at ~1.35 A.

    The substrate is Ac/NHMe-capped, so 2 of its 5 amides are CAPS (acetyl C and the
    N-methylamide C). Hydrolysing a cap would release acetamide or methylamine -- not the
    L2 linear dimer the assay scores -- so cap carbonyls are excluded by default. This is
    not hypothetical: before the fix, Nyl08 and Nyl09 (whose ligand drifts 7-13 A out of
    the site) had BOTH of their site registers sitting on a cap, i.e. their geometry was
    measured on a chemically irrelevant bond.

    A cap is identified geometrically, not by atom index: the carbonyl C or its amide N
    carries a neighbouring carbon that is itself terminal (a methyl, one heavy neighbour).
    """
    C = [(i, p) for i, (e, p) in enumerate(atoms) if e == "C"]
    O = [(i, p) for i, (e, p) in enumerate(atoms) if e == "O"]
    N = [(i, p) for i, (e, p) in enumerate(atoms) if e == "N"]

    def terminal_methyl_attached(centre_idx):
        cp = atoms[centre_idx][1]
        for j, (e, q) in enumerate(atoms):
            if e != "C" or j == centre_idx:
                continue
            if float(np.linalg.norm(cp - q)) >= 1.75:
                continue
            deg = sum(1 for k, (e2, q2) in enumerate(atoms)
                      if k != j and float(np.linalg.norm(q - q2)) < 1.75)
            if deg == 1:
                return True
        return False

    out = []
    for ci, cp in C:
        oi = [(float(np.linalg.norm(cp - op)), i) for i, op in O
              if float(np.linalg.norm(cp - op)) < 1.35]
        ni = [(float(np.linalg.norm(cp - q)), i) for i, q in N
              if float(np.linalg.norm(cp - q)) < 1.60]
        if not (oi and ni):
            continue
        n_idx = min(ni)[1]
        if internal_only and (terminal_methyl_attached(ci) or terminal_methyl_attached(n_idx)):
            continue                      # cap amide -- wrong chemistry, skip
        out.append((ci, min(oi)[1], n_idx))
    return out



def angle(a, b, c):
    v1, v2 = a - b, c - b
    cs = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(cs, -1, 1))))



def ca_cif(path):
    import gemmi
    st=gemmi.read_structure(str(path)); d=collections.defaultdict(dict)
    for ch in st[0]:
        for r in ch:
            a=r.find_atom("CA","*")
            if a: d[ch.name][r.seqid.num]=np.array([a.pos.x,a.pos.y,a.pos.z])
    return d



def iface(m1,m2,cut=8.0):
    r1,r2=sorted(m1),sorted(m2)
    X1=np.array([m1[k] for k in r1]); X2=np.array([m2[k] for k in r2])
    pr=cKDTree(X1).sparse_distance_matrix(cKDTree(X2),cut).keys()
    return {r1[i] for i,j in pr}



AA3_TO1 = dict(zip('ALA ARG ASN ASP CYS GLN GLU HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
                   'ARNDCQEHILKMFPSTWYV'))
AA3_TO1['GLY'] = 'G'


def pairs(prot, alpha_sequence=None, beta_sequence=None):
    """Pair two declared mature alpha/beta units by heavy-atom contacts.

    Explicit input sequences establish roles when supplied. For the historical
    unequal-fragment NylC layout, the shorter Thr-starting fragment is beta.
    An absolute150-residue threshold incorrectly rejected four large homologs.
    """
    if (alpha_sequence is None) != (beta_sequence is None):
        raise ValueError('supply both alpha and beta sequences')
    if alpha_sequence is not None:
        if alpha_sequence == beta_sequence:
            raise ValueError('alpha/beta sequences do not distinguish roles')
        sequences = {c: ''.join(AA3_TO1.get(r['_r'], 'X') for _, r in sorted(prot[c].items())) for c in prot}
        aa = sorted(c for c in prot if sequences[c] == alpha_sequence)
        bb = sorted(c for c in prot if sequences[c] == beta_sequence)
    else:
        lengths = sorted({len(prot[c]) for c in prot})
        if len(lengths) != 2:
            raise ValueError('chain roles need declared sequences for equal/ambiguous lengths')
        aa = sorted(c for c in prot if len(prot[c]) == lengths[-1])
        bb = sorted(c for c in prot if len(prot[c]) == lengths[0])
    if len(aa) != 2 or len(bb) != 2:
        raise ValueError("expected two alpha and two beta chains")
    if any(prot[c][min(prot[c])]['_r'] != 'THR' for c in bb):
        raise ValueError('beta chain lacks the mature N-terminal Thr')
    xyz = {c: np.array([v for r in prot[c].values() for k, v in r.items() if k != "_r"])
           for c in prot}
    counts = np.array([[cKDTree(xyz[a]).query_ball_point(xyz[b], 8, return_length=True).sum()
                        for b in bb] for a in aa])
    ii, jj = linear_sum_assignment(-counts)
    return [(aa[i], bb[j]) for i, j in zip(ii, jj)]



def merged_ca(prot, pair):
    a, b = pair
    return {**{r: x["CA"] for r, x in prot[a].items() if "CA" in x},
            **{r + len(prot[a]): x["CA"] for r, x in prot[b].items() if "CA" in x}}



def align(x, y):
    """Row-vector rigid transform x -> y, with proper rotation (no reflection)."""
    if x.shape != y.shape or len(x) < 3 or np.linalg.matrix_rank(x-x.mean(0)) < 2:
        raise ValueError("insufficient corresponding beta-domain atoms for alignment")
    cx, cy = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((x - cx).T @ (y - cy))
    corr = np.eye(3)
    corr[-1, -1] = np.linalg.det(u @ vt)
    rot = u @ corr @ vt
    return rot, cy - cx @ rot



def score_pose(path, ref, alpha_sequence=None, beta_sequence=None):
    prot, lig = parse_model(path)
    pp = pairs(prot, alpha_sequence, beta_sequence)
    if len(lig) != 2:
        raise ValueError("expected two ligand copies")
    mm = [merged_ca(prot, p) for p in pp]
    jac = []
    for i in range(2):
        got = iface(mm[i], mm[1-i])
        target = iface(ref[i], ref[1-i])
        jac.append(len(got & target) / max(1, len(got | target)))
    if min(jac) < .5:
        raise ValueError(f"assembly Jaccard below 0.5: {jac}")
    lk = sorted(lig)
    cams = {k: amide_carbonyls(lig[k]) for k in lk}
    if any(len(c) != 3 for c in cams.values()):
        raise ValueError("expected three non-cap amides per ligand")
    nuc = [prot[b][min(prot[b])]["OG1"] for a, b in pp]
    costs = np.array([[min(np.linalg.norm(og - lig[k][ci][1]) for ci, _, _ in cams[k])
                       for k in lk] for og in nuc])
    ii, jj = linear_sum_assignment(costs)
    assignment = dict(zip(ii, jj))
    allxyz, keys = [], []
    for mi, (a, b) in enumerate(pp):
        for sub, ch in enumerate((a, b)):
            for rn, res in prot[ch].items():
                for an, xyz in res.items():
                    if an != "_r":
                        allxyz.append(xyz)
                        keys.append((mi, sub, rn))
    tree = cKDTree(np.array(allxyz))
    sites = []
    for si, (a, b) in enumerate(pp):
        k = lk[assignment[si]]
        atoms = lig[k]
        lx = np.array([p for _, p in atoms])
        distances = [float(np.linalg.norm(nuc[si] - atoms[ci][1])) for ci, _, _ in cams[k]]
        angles = [angle(nuc[si], atoms[ci][1], atoms[oi][1]) for ci, oi, _ in cams[k]]
        close = tree.query_ball_point(lx, 4.)
        # Canonical roles: own monomer/partner, alpha/beta, residue number.
        contacts = {(int(keys[t][0] != si), keys[t][1], keys[t][2]) for hit in close for t in hit}
        alen = len(prot[a])
        # Align catalytic beta domain to its own template. Avoid moving alpha tails.
        nums = [rn for rn in sorted(prot[b]) if "CA" in prot[b][rn] and rn+alen in ref[si]]
        x = np.array([prot[b][rn]["CA"] for rn in nums])
        y = np.array([ref[si][rn+alen] for rn in nums])
        rot, tr = align(x, y)
        sites.append(dict(distances=distances, angles=angles,
                          amides=[list(c) for c in cams[k]], elements=[e for e, _ in atoms],
                          nearest=int(np.argmin(distances)),
                          ligand_xyz=(lx @ rot + tr).tolist(), contacts=sorted(contacts),
                          contact_fraction=float(np.mean([bool(z) for z in close]))))
    cf = json.loads(path.with_name("confidence_" + path.stem + ".json").read_text())
    chunk_match = re.search(r"/chunk(\d+)/", str(path))
    return dict(path=str(path), chunk=int(chunk_match[1]) if chunk_match else 1,
                model=int(re.search(r"_model_(\d+)", path.name)[1]),
                interface_jaccard=min(jac), alpha_len=len(prot[pp[0][0]]),
                ligand_iptm=cf["ligand_iptm"], complex_plddt=cf["complex_plddt"], sites=sites)



def aggregate(name, poses):
    vals = collections.defaultdict(list)
    for si in range(2):
        sites = [p["sites"][si] for p in poses]
        signature = [(s["amides"], s["elements"]) for s in sites]
        if any(s != signature[0] for s in signature):
            raise ValueError("ligand atom identity/order differs across poses")
        near = [s["nearest"] for s in sites]
        modal, count = collections.Counter(near).most_common(1)[0]
        dd = np.array([s["distances"][modal] for s in sites])
        ang = np.array([s["angles"][modal] for s in sites])
        vals["register_purity"].append(count/len(sites))
        vals["engaged_fraction"].append(np.mean([min(s["distances"]) < 4 for s in sites]))
        vals["productive_fraction"].append(np.mean((dd < 3.5) & (ang >= 90) & (ang <= 130)))
        vals["distance_median"].append(np.median(dd))
        vals["distance_iqr"].append(np.subtract(*np.percentile(dd, [75, 25])))
        xyz = np.array([s["ligand_xyz"] for s in sites])
        # RMS displacement about ensemble mean; labelled-atom dispersion, not MD RMSF.
        vals["ligand_dispersion"].append(np.sqrt(np.mean(np.sum((xyz - xyz.mean(0))**2, axis=-1))))
        cc = [set(map(tuple, s["contacts"])) for s in sites]
        vals["contact_consistency"].append(np.mean([len(a & b)/max(len(a | b), 1)
                                                    for a, b in itertools.combinations(cc, 2)]))
        vals["ligand_contact_fraction"].append(np.mean([s["contact_fraction"] for s in sites]))
    out = dict(id=name, n_poses=len(poses), alpha_len=poses[0]["alpha_len"],
               interface_jaccard_min=min(p["interface_jaccard"] for p in poses),
               ligand_iptm=np.mean([p["ligand_iptm"] for p in poses]),
               complex_plddt=np.mean([p["complex_plddt"] for p in poses]))
    out.update({k: float(np.mean(v)) for k, v in vals.items()})
    out["site_distance_spread"] = float(np.ptp(vals["distance_median"]))
    return out
