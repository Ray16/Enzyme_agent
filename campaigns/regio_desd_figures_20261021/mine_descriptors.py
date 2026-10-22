#!/usr/bin/env python3
"""Compute the amendment-6 descriptor battery from the completed folds. NO GPU, no re-folding.

Stage 1 of the declared search: 13 step-1 + 13 step-2 descriptors per ensemble, written to
analysis/descriptors_step1.csv and analysis/descriptors_step2.csv. Stage 2 (the AUROC search and
its selection-corrected null) lives in search_descriptors.py, deliberately in a separate file so
the descriptors cannot be quietly edited in response to a score.

Atom selection reuses the campaign's established rules, because Boltz renames every CCD ligand to
`LIG` and RENUMBERS substrate atoms -- neither residue name nor atom name can be trusted:
  * a carboxylate     = any C with >= 2 O within 1.6 A; those O are the carboxylate oxygens
  * the acyl-adenylate carbonyl = a C with >= 2 bonded O, exactly one of which bridges to a P
    (the anhydride); the terminal O is the carbonyl O, the bridging O is the AMP leaving oxygen
  * the attacking N   = a chain-C nitrogen that is NOT an amide N (not bonded to a C that itself
    carries an O within 1.35 A), nearest the carbonyl C

Chain maps are fixed by the builders and differ between steps -- getting this wrong measures the
Mg as if it were the substrate:
  step 1:  A protein | B ATP       | C Mg      | D acid donor
  step 2:  A protein | B acyl-AMP  | C amine   | D Mg

Courtesy on a shared node: cKDTree is always called with workers=1 and the process pool is capped;
`workers=-1` inside a pool once oversubscribed a node in this project.
"""
import csv, math, os, pathlib, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

CAMP = pathlib.Path(__file__).resolve().parent

GATE = 4.0            # the campaign's validity gate, unchanged
BD_IDEAL = 105.0
HBOND = 3.5
CATION = 4.0
CONTACT = 4.5
POCKET_R = 8.0

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
APOLAR3 = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"}
CATION_ATOMS = {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("LYS", "NZ")}
ACID_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}


# --------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------
def parse(path):
    """-> dict(chain -> list[(atomname, element, resname, resseq, xyz)]), altloc-filtered."""
    ch = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        el = line[76:78].strip().upper()
        if el in ("H", "D", ""):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        ch[line[21]].append((line[12:16].strip(), el, line[17:20].strip(),
                             line[22:26].strip(), xyz))
    return ch


def carboxylate_oxygens(atoms):
    """Carboxylate O by element+bonding, never by name (Boltz renumbers them)."""
    ox = [a[4] for a in atoms if a[1] == "O"]
    out = set()
    for a in atoms:
        if a[1] != "C":
            continue
        bonded = [o for o in ox if math.dist(o, a[4]) < 1.6]
        if len(bonded) >= 2:
            out.update(bonded)
    return [list(o) for o in out]


def protein_arrays(atoms):
    xyz, meta = [], []
    for name, _el, resn, seq, p in atoms:
        if resn in AA3:
            xyz.append(p); meta.append((resn, name, seq))
    return np.asarray(xyz, float), meta


def _count_within(tree, pts, r, keep=None, meta=None):
    if not len(pts):
        return 0
    idx = tree.query_ball_point(np.asarray(pts, float), r=r, workers=1)
    hits = {i for lst in idx for i in lst}
    if keep is None:
        return len(hits)
    return sum(1 for i in hits if keep(meta[i]))


def _min_dist(a, b):
    if not len(a) or not len(b):
        return None
    A = np.asarray(a, float); B = np.asarray(b, float)
    return float(np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)))


def angle(p, vertex, q):
    a = [x - y for x, y in zip(p, vertex)]
    b = [x - y for x, y in zip(q, vertex)]
    na = math.sqrt(sum(v * v for v in a)); nb = math.sqrt(sum(v * v for v in b))
    if na <= 0 or nb <= 0:
        return None
    c = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# --------------------------------------------------------------------------------------------
# step 1 -- adenylation
# --------------------------------------------------------------------------------------------
def pose_step1(path):
    ch = parse(path)
    prot, mg, donor = ch.get("A"), ch.get("C"), ch.get("D")
    atp = ch.get("B")
    if not (prot and atp and donor):
        return None
    nuc = {a[0]: a[4] for a in atp}
    if "PA" not in nuc or "O3A" not in nuc:
        return None
    pa, o3a = nuc["PA"], nuc["O3A"]

    cox = carboxylate_oxygens(donor)
    if not cox:
        return None
    near = min(cox, key=lambda o: math.dist(o, pa))
    d = math.dist(near, pa)
    ang = angle(near, pa, o3a)
    if ang is None:
        return None

    P, meta = protein_arrays(prot)
    tree = cKDTree(P) if len(P) else None
    dxyz = [a[4] for a in donor]

    # Mg: chain C in step 1 (may hold 2 for the sensitivity folds)
    mgs = [a[4] for a in (mg or []) if a[1] == "MG"]
    # ATP phosphate oxygens = O bonded to a P
    ps = [a[4] for a in atp if a[1] == "P"]
    phos_o = [a[4] for a in atp if a[1] == "O" and any(math.dist(a[4], p) < 1.75 for p in ps)]

    hb = cat = cont = 0
    apolar = None
    if tree is not None:
        hb = _count_within(tree, cox, HBOND, lambda m: m[1][0] in ("N", "O"), meta)
        cat = _count_within(tree, cox, CATION, lambda m: (m[0], m[1]) in CATION_ATOMS, meta)
        cont = _count_within(tree, dxyz, CONTACT)
        idx = tree.query_ball_point(np.asarray(pa, float), r=POCKET_R, workers=1)
        res = {(meta[i][0], meta[i][2]) for i in idx}
        if res:
            apolar = sum(1 for rn, _s in res if rn in APOLAR3) / len(res)

    return dict(d=d, ang=ang,
                mg_carboxylate=_min_dist(mgs, cox),
                mg_phosphate=_min_dist(mgs, phos_o),
                carboxylate_hbonds=hb, cation_contacts=cat, donor_contacts=cont,
                pocket_apolar=apolar,
                centroid=list(np.mean(np.asarray(dxyz, float), axis=0)))


# --------------------------------------------------------------------------------------------
# step 2 -- aminolysis
# --------------------------------------------------------------------------------------------
def pose_step2(path):
    ch = parse(path)
    prot, acyl, acc = ch.get("A"), ch.get("B"), ch.get("C")
    if not (prot and acyl and acc):
        return None

    bO = [a[4] for a in acyl if a[1] == "O"]
    bP = [a[4] for a in acyl if a[1] == "P"]
    carbonyl = None
    for a in acyl:
        if a[1] != "C":
            continue
        bonded = [o for o in bO if math.dist(o, a[4]) < 1.6]
        if len(bonded) < 2:
            continue
        bridging = [o for o in bonded if any(math.dist(o, p) < 1.75 for p in bP)]
        terminal = [o for o in bonded if o not in bridging]
        if bridging and terminal:
            carbonyl = (a[4], terminal[0], bridging[0])
            break
    if carbonyl is None:
        return dict(intact=0.0)          # anhydride not resolvable in this pose
    cc, co, cester = carbonyl

    cO = [a[4] for a in acc if a[1] == "O"]
    cCs = [a[4] for a in acc if a[1] == "C"]
    nitro = []
    for a in acc:
        if a[1] != "N":
            continue
        amide = any(math.dist(c, a[4]) < 1.6 and any(math.dist(o, c) < 1.35 for o in cO)
                    for c in cCs)
        if not amide:
            nitro.append(a[4])
    if not nitro:
        return dict(intact=1.0)
    nn = min(nitro, key=lambda x: math.dist(x, cc))
    d = math.dist(nn, cc)
    bd = angle(nn, cc, co)
    anti = angle(nn, cc, cester)
    if bd is None or anti is None:
        return dict(intact=1.0)

    P, meta = protein_arrays(prot)
    tree = cKDTree(P) if len(P) else None
    axyz = [a[4] for a in acc]
    distal = [x for x in nitro if x is not nn]

    oxy = burial = cont = 0
    dist_charge = None
    if tree is not None:
        oxy = _count_within(tree, [co], HBOND, lambda m: m[1][0] in ("N", "O"), meta)
        burial = _count_within(tree, [nn], CONTACT)
        cont = _count_within(tree, axyz, CONTACT)
        if distal:
            acid_xyz = [P[i] for i, m in enumerate(meta) if (m[0], m[1]) in ACID_ATOMS]
            dist_charge = _min_dist(distal, acid_xyz)

    return dict(intact=1.0, d=d, bd=bd, anti=anti, oxyanion_hbonds=oxy, n_burial=burial,
                acceptor_contacts=cont, distal_charge=dist_charge,
                centroid=list(np.mean(np.asarray(axyz, float), axis=0)))


# --------------------------------------------------------------------------------------------
# ensemble aggregation
# --------------------------------------------------------------------------------------------
def _med(vals):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


def _spread(cents):
    if len(cents) < 2:
        return None
    C = np.asarray(cents, float)
    return float(np.mean(np.linalg.norm(C - C.mean(axis=0), axis=1)))


def agg_step1(stem, poses):
    good = [p for p in poses if p and "d" in p]
    if not good:
        return None
    ds = [p["d"] for p in good]; angs = [p["ang"] for p in good]
    n = len(ds)
    gated = [(d, a) for d, a in zip(ds, angs) if d < GATE]
    inv = sum(1.0 / d for d in ds if d > 0)
    return dict(
        stem=stem, n_poses=n,
        s1_A=round(sum(a / d for d, a in zip(ds, angs)) / inv, 3) if inv else None,
        s1_Dh=round(n / inv, 3) if inv else None,
        s1_occupancy=round(len(gated) / n, 4),
        s1_min_d=round(min(ds), 3),
        s1_nac_frac=round(sum(1 for d, a in gated if a > 150) / n, 4),
        s1_best_angle_gated=round(max((a for _d, a in gated), default=0.0), 2),
        s1_mg_carboxylate=_med([p["mg_carboxylate"] for p in good]),
        s1_mg_phosphate=_med([p["mg_phosphate"] for p in good]),
        s1_carboxylate_hbonds=_med([p["carboxylate_hbonds"] for p in good]),
        s1_cation_contacts=_med([p["cation_contacts"] for p in good]),
        s1_donor_contacts=_med([p["donor_contacts"] for p in good]),
        s1_centroid_spread=_spread([p["centroid"] for p in good]),
        s1_pocket_apolar=_med([p["pocket_apolar"] for p in good]),
    )


def agg_step2(stem, poses):
    ok = [p for p in poses if p]
    good = [p for p in ok if "d" in p]
    if not ok:
        return None
    intact = sum(p.get("intact", 0.0) for p in ok) / len(ok)
    if not good:
        return dict(stem=stem, n_poses=len(ok), s2_acyl_intact=round(intact, 4))
    ds = [p["d"] for p in good]; bds = [p["bd"] for p in good]
    n = len(ds)
    gated = [(d, b) for d, b in zip(ds, bds) if d < GATE]
    inv = sum(1.0 / d for d in ds if d > 0)
    wbd = sum(b / d for d, b in zip(ds, bds)) / inv if inv else None
    return dict(
        stem=stem, n_poses=n,
        s2_BD_dev=round(abs(wbd - BD_IDEAL), 3) if wbd is not None else None,
        s2_Dh=round(n / inv, 3) if inv else None,
        s2_occupancy=round(len(gated) / n, 4),
        s2_min_d=round(min(ds), 3),
        s2_nac_frac=round(sum(1 for d, b in gated if 95 <= b <= 115) / n, 4),
        s2_best_bd_gated=round(min((abs(b - BD_IDEAL) for _d, b in gated), default=180.0), 2),
        s2_oxyanion_hbonds=_med([p["oxyanion_hbonds"] for p in good]),
        s2_leaving_anti=_med([p["anti"] for p in good]),
        s2_n_burial=_med([p["n_burial"] for p in good]),
        s2_acceptor_contacts=_med([p["acceptor_contacts"] for p in good]),
        s2_distal_charge=_med([p["distal_charge"] for p in good]),
        s2_centroid_spread=_spread([p["centroid"] for p in good]),
        s2_acyl_intact=round(intact, 4),
    )


def _one(args):
    step, stem, pred = args
    pdbs = sorted(pathlib.Path(pred).glob(f"{stem}_model_*.pdb"))
    if len(pdbs) < 50:
        return step, stem, None
    fn = pose_step1 if step == "step1" else pose_step2
    poses = []
    for p in pdbs:
        try:
            poses.append(fn(p))
        except (ValueError, KeyError, IndexError):
            poses.append(None)
    agg = agg_step1 if step == "step1" else agg_step2
    return step, stem, agg(stem, poses)


S1_COLS = ["stem", "enzyme", "acid_code", "n_mg", "n_poses", "s1_A", "s1_Dh", "s1_occupancy",
           "s1_min_d", "s1_nac_frac", "s1_best_angle_gated", "s1_mg_carboxylate",
           "s1_mg_phosphate", "s1_carboxylate_hbonds", "s1_cation_contacts", "s1_donor_contacts",
           "s1_centroid_spread", "s1_pocket_apolar"]
S2_COLS = ["stem", "enzyme", "donor_code", "acceptor_code", "n_poses", "s2_BD_dev", "s2_Dh",
           "s2_occupancy", "s2_min_d", "s2_nac_frac", "s2_best_bd_gated", "s2_oxyanion_hbonds",
           "s2_leaving_anti", "s2_n_burial", "s2_acceptor_contacts", "s2_distal_charge",
           "s2_centroid_spread", "s2_acyl_intact"]


def main():
    jobs = []
    jm1 = {r["stem"]: r for r in csv.DictReader((CAMP / "cofolds/step1/job_map.csv").open())}
    jm2 = {r["stem"]: r for r in csv.DictReader((CAMP / "cofolds/step2/job_map.csv").open())}
    for step, jm in (("step1", jm1), ("step2", jm2)):
        for stem in sorted(jm):
            pred = CAMP / f"cofolds/{step}/results/{stem}/boltz_results_{stem}/predictions/{stem}"
            if pred.exists():
                jobs.append((step, stem, str(pred)))
    print(f"computing the amendment-6 battery over {len(jobs)} ensembles "
          f"({sum(1 for j in jobs if j[0]=='step1')} step-1, "
          f"{sum(1 for j in jobs if j[0]=='step2')} step-2) x 50 poses")

    nproc = min(12, max(1, (os.cpu_count() or 8) // 2))   # courtesy cap on a shared node
    out = {"step1": [], "step2": []}
    done = 0
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for step, stem, res in ex.map(_one, jobs, chunksize=1):
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
            if res is None:
                print(f"  SKIP {stem}: incomplete or unparseable")
                continue
            meta = (jm1 if step == "step1" else jm2)[stem]
            if step == "step1":
                res.update(enzyme=meta["enzyme"], acid_code=meta["acid_code"], n_mg=meta["n_mg"])
            else:
                res.update(enzyme=meta["enzyme"], donor_code=meta["donor_code"],
                           acceptor_code=meta["acceptor_code"])
            out[step].append(res)

    for step, cols in (("step1", S1_COLS), ("step2", S2_COLS)):
        f = CAMP / f"analysis/descriptors_{step}.csv"
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in sorted(out[step], key=lambda r: r["stem"]):
                w.writerow({c: ("" if r.get(c) is None else r.get(c, "")) for c in cols})
        print(f"wrote {f}  ({len(out[step])} ensembles)")


if __name__ == "__main__":
    sys.exit(main())
