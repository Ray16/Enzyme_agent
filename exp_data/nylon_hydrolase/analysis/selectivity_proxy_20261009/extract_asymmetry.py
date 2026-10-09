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


# --- single-threaded copies of thread_chain.bulk_tree / grow (workers=1) ---
# The shared tool hardcodes workers=-1 (every cKDTree query grabs all cores); under
# process-level parallelism that oversubscribes a shared node. These copies force
# workers=1 so N processes use exactly N cores. Logic is otherwise identical.
from scipy.spatial import cKDTree  # noqa: E402


def _bulk_tree(coords, radii, centre, box=32.0, spacing=0.7, water=1.4):
    from scipy import ndimage
    lo = centre - box
    n = int(2 * box / spacing) + 1
    ax = lo[:, None] + np.arange(n) * spacing
    G = np.stack(np.meshgrid(ax[0], ax[1], ax[2], indexing="ij"), -1)
    t = cKDTree(coords)
    d, i = t.query(G.reshape(-1, 3), workers=1)
    free = ((d - radii[i]) >= water).reshape(G.shape[:3])
    lab, _ = ndimage.label(free)
    edge = set(lab[0].ravel()) | set(lab[-1].ravel()) | set(lab[:, 0].ravel()) | \
        set(lab[:, -1].ravel()) | set(lab[:, :, 0].ravel()) | set(lab[:, :, -1].ravel())
    edge.discard(0)
    mask = np.isin(lab, list(edge))
    pts = lo + np.argwhere(mask) * spacing
    return cKDTree(pts) if len(pts) else None


def _grow(tree, radii, p0, p1, clash, max_len, beam, btree=None, min_depth=0.0):
    beams = np.array([[p0, p1]], dtype=float)
    best = beams[0]
    for _ in range(max_len):
        B, L, _ = beams.shape
        cand = np.stack([tcmod.candidates(b[-2], b[-1]) for b in beams])
        K = cand.shape[1]
        flat = cand.reshape(-1, 3)
        d, i = tree.query(flat, workers=1)
        clear = (d - radii[i]).reshape(B, K)
        ok = clear >= clash
        if L > 3:
            prev = beams[:, :-3, :]
            dd = np.linalg.norm(cand[:, :, None, :] - prev[:, None, :, :], axis=-1)
            ok &= dd.min(axis=2) >= 2.8
        depth = None
        if btree is not None:
            depth = btree.query(flat, workers=1)[0].reshape(B, K)
            ok &= depth > min_depth
        bi, ki = np.nonzero(ok)
        if not len(bi):
            break
        key = -depth[bi, ki] if btree is not None else clear[bi, ki]
        order = np.argsort(key)[:beam]
        bi, ki = bi[order], ki[order]
        beams = np.concatenate([beams[bi], cand[bi, ki][:, None, :]], axis=1)
        if beams.shape[1] > len(best):
            best = beams[0]
    return len(best), np.array(best)


def arm_lengths(prot, nuc, clash=1.4, seed=0, n_seed=8, n_second=2, beam=60):
    """capacity's bidirectional search, single-threaded, keeping the two arm lengths."""
    rng = np.random.default_rng(seed)
    coords = np.array([a[5] for a in prot])
    radii = np.array([tcmod.VDW[a[4]] for a in prot])
    tree = cKDTree(coords)
    btree = _bulk_tree(coords, radii, nuc)
    dirs = rng.normal(size=(2000, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    seeds = nuc + 3.0 * dirs
    d, i = tree.query(seeds, workers=1)
    seeds = seeds[(d - radii[i]) >= clash][:n_seed]
    if not len(seeds):
        return 0, 0
    best = (-1, 0, 0)
    for s in seeds:
        c2 = s + 1.49 * dirs[:200]
        d2, i2 = tree.query(c2, workers=1)
        c2 = c2[(d2 - radii[i2]) >= clash][:n_second]
        for b in c2:
            lf, _ = _grow(tree, radii, s, b, clash, 40, beam, btree, 0.0)
            lb, _ = _grow(tree, radii, b, s, clash, 40, beam, btree, 0.0)
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
