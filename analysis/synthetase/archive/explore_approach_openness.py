#!/usr/bin/env python3
"""EXPLORATORY (read-only, NOT pipeline): APPROACH-TRAJECTORY OPENNESS of the reactive
center vs measured yield (SfaB panel step-1 cofolds).

Idea (targets the aminolysis RDS WITHOUT docking the amine): the incoming amine must
reach the reactive carbonyl carbon from the face OPPOSITE the leaving group (the
AMP/phosphate). Measure how much of that approach hemisphere is unobstructed by
protein. Cast rays from the reactive carboxyl C into a cone about the approach axis
(= direction pointing AWAY from the nearest ATP phosphate, i.e. rc - P_alpha); a ray
is BLOCKED if a protein (chain A) heavy atom lies within R_BLOCK of the ray line and
in front of rc within length L; OPEN otherwise (reaches bulk solvent).

    openness = fraction of cone rays that are open.

Per pose, mean over gated (docked) poses. Gate: reactive C within GATE A of an ATP P.

Hypothesis: an amine needs an open approach channel -> higher openness -> higher yield.
This may TRADE OFF against the water-exclusion/burial (SASA) finding (buried center =
low SASA = also low openness), so a key question is whether openness adds anything
beyond -SASA.

FREE PARAMETERS: L (ray length), THETA (cone half-angle), R_BLOCK (atom block radius),
N_RAYS. Because the contact-count proxy died from free-parameter (radius) overfitting,
this script does a full robustness sweep over (L, THETA, R_BLOCK) and a N_RAYS
convergence check, and reports whether the signal is stable.

Reuses parsing/reactive-atom logic + spearman/auroc/perm_p from explore_reactive_sasa.py.
"""
import csv, glob, os, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
IPTM = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/protein_substrate_iptm.csv"
MANIFEST = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_manifest.csv"
SASA_CSV = f"{ROOT}/exp_data/amide_synthetase/analysis/exploratory_reactive_sasa.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/exploratory_approach_openness.csv"

GATE = 6.0
N_MIN = 5
# defaults
L0, THETA0, RB0, NRAYS0 = 8.0, 60.0, 1.7, 400
_P = PDBParser(QUIET=True)


# ---- fixed ray set on a cone cap about +z, reused (rotated) for every pose ----
def cone_rays(n, theta_deg):
    """n ~uniform directions within half-angle theta of +z (Fibonacci sphere cap)."""
    theta = np.radians(theta_deg)
    cosmin = np.cos(theta)
    i = np.arange(n) + 0.5
    cz = 1.0 - (i / n) * (1.0 - cosmin)          # uniform in cos over the cap
    phi = np.pi * (1 + 5 ** 0.5) * i             # golden-angle azimuth
    s = np.sqrt(np.maximum(0.0, 1 - cz * cz))
    return np.stack([s * np.cos(phi), s * np.sin(phi), cz], axis=1)  # (n,3) unit


def rot_to(axis):
    """rotation matrix sending +z to unit `axis`."""
    z = np.array([0.0, 0.0, 1.0]); a = axis / (np.linalg.norm(axis) + 1e-12)
    v = np.cross(z, a); c = float(z @ a)
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def openness_from_pose(rc, approach_axis, Aloc, rays_z, L, r_block):
    """fraction of rays (given on +z cone) rotated to approach_axis that are unblocked."""
    R = rot_to(approach_axis)
    dirs = rays_z @ R.T                     # (n,3) ray directions in world frame
    V = Aloc - rc                           # (m,3) protein atom vectors from rc
    if len(V) == 0:
        return 1.0
    T = V @ dirs.T                          # (m,n) projection of each atom on each ray
    # perpendicular distance^2 = |V|^2 - T^2
    v2 = np.einsum("ij,ij->i", V, V)[:, None]     # (m,1)
    perp2 = v2 - T * T                              # (m,n)
    blocked = (T > 0) & (T < L) & (perp2 < r_block * r_block)   # (m,n)
    ray_blocked = blocked.any(axis=0)      # (n,)
    return float((~ray_blocked).mean())


def parse_pose(pdb):
    """Return (rc_coord, approach_axis, protein_heavy_coords_near, dP) or None."""
    m = _P.get_structure("x", pdb)[0]
    if "E" not in m or "B" not in m or "A" not in m:
        return None
    E = list(m["E"].get_atoms()); B = list(m["B"].get_atoms())
    Ps = [a.coord for a in B if a.element == "P"]
    if not Ps:
        return None
    carbs = []
    for a in E:
        if a.element != "C":
            continue
        onb = [b for b in E if b.element == "O" and np.linalg.norm(a.coord - b.coord) < 1.7]
        if len(onb) >= 2:
            carbs.append(a)
    if not carbs:
        return None
    rc_atom = min(carbs, key=lambda a: min(np.linalg.norm(a.coord - p) for p in Ps))
    rc = rc_atom.coord.astype(float)
    Pnear = min(Ps, key=lambda p: np.linalg.norm(rc - p))
    dP = float(np.linalg.norm(rc - Pnear))
    approach_axis = rc - Pnear             # points away from leaving group
    A = np.array([a.coord for a in m["A"].get_atoms() if a.element != "H"], dtype=float)
    # keep only protein atoms within a generous radius for speed (max L in sweep = 12)
    d = np.linalg.norm(A - rc, axis=1)
    Aloc = A[d < 15.0]
    return rc, approach_axis, Aloc, dP


# cache parsed geometry per enzyme so the parameter sweep doesn't re-read PDBs
def load_enzyme(cfgdir):
    base = os.path.basename(cfgdir)
    name = base[len("step1_"):] if base.startswith("step1_") else base
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    poses = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        try:
            g = parse_pose(pdb)
        except Exception:
            g = None
        if g:
            poses.append(g)
    return name, poses


def score_enzyme(poses, rays_z, L, r_block):
    """mean openness over docked poses; returns (n, n_docked, openness or None)."""
    if not poses:
        return 0, 0, None
    dP = np.array([p[3] for p in poses])
    docked = np.where(dP < GATE)[0]
    if len(docked) < N_MIN:
        return len(poses), int(len(docked)), None
    vals = [openness_from_pose(poses[i][0], poses[i][1], poses[i][2], rays_z, L, r_block)
            for i in docked]
    return len(poses), int(len(docked)), float(np.mean(vals))


# ---- stats helpers (same definitions as explore_reactive_sasa.py) ----
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))


def auroc(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    m = ~np.isnan(s); s, l = s[m], l[m]
    pos, neg = s[l == 1], s[l == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def perm_p(scores, labels, n=20000, seed=0):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    m = ~np.isnan(s); s, l = s[m], l[m]
    if len(s) < 4 or l.sum() == 0:
        return float("nan")
    obs = auroc(s, l); rng = np.random.default_rng(seed)
    ge = sum(auroc(s, rng.permutation(l)) >= obs for _ in range(n))
    return (ge + 1) / (n + 1)


def load_col(path, key, val, cast=float):
    d = {}
    for r in csv.DictReader(open(path)):
        v = r.get(val, "")
        try:
            d[r[key]] = cast(v) if v not in ("", None) else None
        except (ValueError, TypeError):
            d[r[key]] = None
    return d


_ENZ = None  # global cache for pool workers


def _score_default(item):
    name, poses = item
    rays = cone_rays(NRAYS0, THETA0)
    n, nd, op = score_enzyme(poses, rays, L0, RB0)
    return name, n, nd, op


def main():
    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    print(f"loading {len(cfgdirs)} enzymes (parsing geometry once, cached for sweep)...")
    with Pool(min(len(cfgdirs), 24)) as p:
        enz = p.map(load_enzyme, cfgdirs)
    enz = [(n, ps) for (n, ps) in enz if ps]

    act = load_col(ACT, "id", "activity_N66L1_ugml_avg")
    iptm = load_col(IPTM, "id", "ps_iptm_mean")
    length = load_col(MANIFEST, "name", "length_aa", cast=lambda v: int(float(v)))
    sasa = load_col(SASA_CSV, "id", "center_sasa")

    # ---- default-parameter run -> CSV ----
    rays = cone_rays(NRAYS0, THETA0)
    rows = []
    for name, poses in enz:
        n, nd, op = score_enzyme(poses, rays, L0, RB0)
        rows.append({"id": name, "n": n, "n_docked": nd,
                     "openness": None if op is None else round(op, 4),
                     "status": "scored" if op is not None else "INCONCLUSIVE",
                     "activity_N66L1": act.get(name)})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "n", "n_docked", "openness", "status", "activity_N66L1"])
        w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["status"] == "scored" and r["activity_N66L1"] is not None]
    y = [r["activity_N66L1"] for r in m]
    labels = [1 if v >= 0.5 else 0 for v in y]
    op = [r["openness"] for r in m]
    print(f"\n=== APPROACH-TRAJECTORY OPENNESS vs N66 yield ===")
    print(f"defaults: L={L0}A theta={THETA0}deg r_block={RB0}A n_rays={NRAYS0}, gate={GATE}A")
    print(f"{len(rows)} enzymes | scored {len(m)} | actives={sum(labels)}")
    ov = np.array(op)
    print(f"openness dist: min {ov.min():.3f}  median {np.median(ov):.3f}  max {ov.max():.3f}")
    print(f"actives openness: {sorted(r['openness'] for r in m if r['activity_N66L1']>=0.5)}\n")
    print("hypothesis A: MORE open -> higher yield ; hypothesis B: MORE open -> lower yield")
    print(f"  Spearman(openness, activity) : {spearman(op, y):+.3f}")
    print(f"  AUROC(open->active)          : {auroc(op, labels):.3f}   perm_p {perm_p(op, labels):.4f}")
    print(f"  AUROC(closed->active)        : {auroc([-v for v in op], labels):.3f}   perm_p {perm_p([-v for v in op], labels):.4f}")

    print("\nactive enzymes' openness standing (rank by openness, 1=most open):")
    ranked = sorted(m, key=lambda r: -r["openness"])
    for r in sorted([r for r in m if r["activity_N66L1"] >= 0.5], key=lambda r: -r["activity_N66L1"]):
        rk = ranked.index(r) + 1
        print(f"  {r['id']:6s} act={r['activity_N66L1']:>5.2f}  openness={r['openness']:.3f}  rank {rk}/{len(ranked)}")

    # ---- confound checks ----
    ids = [r["id"] for r in m]
    def col(d): return [d.get(i, np.nan) for i in ids]
    print("\n--- CONFOUND checks (Spearman of openness vs ...) ---")
    print(f"  vs ps_iptm (docking quality) : {spearman(op, col(iptm)):+.3f}")
    print(f"  vs protein length            : {spearman(op, col(length)):+.3f}")
    sc = col(sasa)
    print(f"  vs center_sasa (SASA expo)   : {spearman(op, sc):+.3f}  (near +1 => openness == SASA, adds nothing)")
    # does openness add beyond -SASA? compare AUROCs
    print(f"  AUROC using -center_sasa     : {auroc([-v if v is not None else np.nan for v in sc], labels):.3f}")

    # ---- robustness sweep over free parameters ----
    print("\n=== ROBUSTNESS SWEEP (is the signal stable across free params?) ===")
    print("reporting Spearman(openness,activity) / AUROC(open->active) per (L,theta,r_block)")
    Ls = [6.0, 8.0, 10.0, 12.0]
    Ths = [45.0, 60.0, 75.0, 90.0]
    RBs = [1.4, 1.7, 2.0]
    print(f"{'L':>4} {'theta':>6} {'rblk':>5} {'rho':>7} {'AUROC':>7} {'scored':>7}")
    stable = []
    for th in Ths:
        raysT = cone_rays(NRAYS0, th)
        for L in Ls:
            for rb in RBs:
                sc_rows = []
                for name, poses in enz:
                    _, _, o = score_enzyme(poses, raysT, L, rb)
                    if o is not None and act.get(name) is not None:
                        sc_rows.append((o, act[name]))
                if len(sc_rows) < 5:
                    continue
                xs = [a for a, _ in sc_rows]; ys = [b for _, b in sc_rows]
                lb = [1 if v >= 0.5 else 0 for v in ys]
                rho = spearman(xs, ys); au = auroc(xs, lb)
                stable.append((rho, au))
                print(f"{L:>4.0f} {th:>6.0f} {rb:>5.1f} {rho:>+7.3f} {au:>7.3f} {len(sc_rows):>7}")
    if stable:
        rr = np.array([s[0] for s in stable]); aa = np.array([s[1] for s in stable])
        print(f"\nSWEEP SUMMARY: rho range [{rr.min():+.3f}, {rr.max():+.3f}] mean {rr.mean():+.3f} sd {rr.std():.3f}")
        print(f"               AUROC range [{aa.min():.3f}, {aa.max():.3f}] mean {aa.mean():.3f} sd {aa.std():.3f}")
        print(f"               sign flips in rho: {'YES (unstable)' if rr.min()<0<rr.max() else 'no (consistent sign)'}")

    # ---- N_RAYS convergence at defaults ----
    print("\n=== N_RAYS convergence (defaults L,theta,r_block) ===")
    for nr in [100, 200, 400, 800]:
        raysN = cone_rays(nr, THETA0)
        xs, ys = [], []
        for name, poses in enz:
            _, _, o = score_enzyme(poses, raysN, L0, RB0)
            if o is not None and act.get(name) is not None:
                xs.append(o); ys.append(act[name])
        lb = [1 if v >= 0.5 else 0 for v in ys]
        print(f"  n_rays={nr:>4}: rho={spearman(xs,ys):+.3f}  AUROC={auroc(xs,lb):.3f}")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
