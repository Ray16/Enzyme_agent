"""Tunnel threading-asymmetry descriptor for the selectivity proxy test.

Per PROTOCOL.md (committed e3259ae). Reuses thread_chain.capacity's bidirectional beam
search: the two arms it threads from the catalytic nucleophile ARE the two tunnel
directions whose asymmetry distinguishes exo (monomer/L1) from endo (dimer/L2) release.

Descriptors per catalytic site:
  L_long, L_short  = max/min of the two threaded arm lengths (forward lf, backward lb)
  asymmetry        = (L_long - L_short) / (L_long + L_short)   [primary]
  distal_capacity  = L_short                                   [secondary 1]
Reads only structure; never the activity CSV (leakage firewall).

Catalytic Thr = first residue after the autocleavage CA-CA break (validated: Nyl50 -> res 227,
matches crystallographic Thr227). No reliance on a hard-coded residue number or terminus.
"""
import sys, json
from pathlib import Path
import numpy as np

TC = Path(__file__).resolve().parents[1] / "fig3_product_length_20261007"
sys.path.insert(0, str(TC))
import thread_chain as tcmod  # noqa: E402


def load_chains(path):
    """Return {chain: [(resseq, resname, atomname, elem, xyz), ...]} for ATOM records."""
    chains = {}
    for ln in open(path):
        if ln[:4] != "ATOM":
            continue
        ch = ln[21]
        rs = int(ln[22:26]); rn = ln[17:20].strip(); an = ln[12:16].strip()
        elem = tcmod.element(an, ln[76:78].strip())
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        chains.setdefault(ch, []).append((rs, rn, an, elem, xyz))
    return chains


def catalytic_thr(chain_atoms):
    """First residue after the autocleavage CA-CA break; return its OG1 xyz or None."""
    ca = {rs: xyz for (rs, rn, an, el, xyz) in chain_atoms if an == "CA"}
    ks = sorted(ca)
    junction = None
    for a, b in zip(ks, ks[1:]):
        if b == a + 1 and np.linalg.norm(ca[b] - ca[a]) > 4.5:
            junction = b
            break
    if junction is None:
        return None, None
    og1 = [(rn, xyz) for (rs, rn, an, el, xyz) in chain_atoms
           if rs == junction and an == "OG1"]
    if not og1 or og1[0][0] != "THR":
        return junction, None
    return junction, og1[0][1]


def prot_tuples(chains):
    """thread_chain expects atoms as indexable tuples with elem at [4], xyz at [5]."""
    out = []
    for ch, atoms in chains.items():
        for (rs, rn, an, el, xyz) in atoms:
            out.append((ch, rs, rn, an, el, xyz))
    return out


def arm_lengths(prot, nuc, clash=1.4, seed=0):
    """Rerun capacity's search but keep the two arm lengths separately."""
    rng = np.random.default_rng(seed)
    coords = np.array([a[5] for a in prot])
    radii = np.array([tcmod.VDW[a[4]] for a in prot])
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    btree = tcmod.bulk_tree(coords, radii, nuc)
    dirs = rng.normal(size=(2000, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    seeds = nuc + 3.0 * dirs
    d, i = tree.query(seeds, workers=-1)
    seeds = seeds[(d - radii[i]) >= clash][:10]
    if not len(seeds):
        return 0, 0
    best = (-1, 0, 0)
    for s in seeds:
        c2 = s + 1.49 * dirs[:200]
        d2, i2 = tree.query(c2, workers=-1)
        c2 = c2[(d2 - radii[i2]) >= clash][:3]
        for b in c2:
            lf, _ = tcmod.grow(tree, radii, s, b, clash, 40, 120, btree, 0.0)
            lb, _ = tcmod.grow(tree, radii, b, s, clash, 40, 120, btree, 0.0)
            if lf + lb > best[0]:
                best = (lf + lb, lf, lb)
    return best[1], best[2]


def score_model(path, clash=1.4):
    chains = load_chains(path)
    prot = prot_tuples(chains)
    sites = []
    for ch, atoms in chains.items():
        junction, nuc = catalytic_thr(atoms)
        if nuc is None:
            continue
        lf, lb = arm_lengths(prot, nuc, clash=clash)
        if lf + lb == 0:
            continue
        L_long, L_short = max(lf, lb), min(lf, lb)
        asym = (L_long - L_short) / (L_long + L_short) if (L_long + L_short) else np.nan
        sites.append(dict(chain=ch, junction=junction, lf=lf, lb=lb,
                          L_long=L_long, L_short=L_short, asymmetry=asym))
    return sites


if __name__ == "__main__":
    import time
    p = sys.argv[1]
    t0 = time.time()
    sites = score_model(p)
    dt = time.time() - t0
    print(json.dumps(dict(model=Path(p).stem, dt_s=round(dt, 1), n_sites=len(sites),
                          sites=sites), default=float, indent=2))
