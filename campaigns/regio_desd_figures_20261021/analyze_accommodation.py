#!/usr/bin/env python3
"""ACCOMMODATION analysis -- the paper's own Fig-3 hypothesis, pre-registered in amendment 5.

Tests whether STERIC ACCOMMODATION (can the donor physically get into the pocket) explains the
size-dependent activity pattern that attack geometry could not:
    small diacids (A, G)  -> every enzyme active
    diad MA               -> DesD only
    triad AMA             -> AcsA + DesD only

Descriptors are FIXED by amendment 5; none may be added or re-thresholded after seeing results.
Computed from the existing step-1 folds -- no new GPU.

  donor_heavy_atoms   substrate size, from the input SMILES (exogenous, not model-derived)
  occupancy           n_gated / 50
  D_h                 harmonic mean donor-carboxylate-O to Pa distance
  buried_frac         frac of donor heavy atoms with a protein heavy atom within 4.5 A (median pose)
  n_contacts          median count of protein heavy atoms within 4.5 A of any donor atom
  pocket_apolar_frac  frac of residues lining Pa (any heavy atom within 8 A) that are apolar

Courtesy: cKDTree is called with workers=1 -- workers=-1 inside a pool once oversubscribed a
shared node in this project.
"""
import csv, math, pathlib, sys

import numpy as np
from scipy.spatial import cKDTree

CAMP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP))
from plot_panels import SOURCES, load, spearman  # noqa: E402

APOLAR = set("AVLIMFWPG")
THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
             "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
             "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
FOLD = {"DdaG": "ANL", "SfaB": "ANL", "AsbA": "NIS", "AsbB": "NIS", "AcsA": "NIS", "DesD": "NIS"}
PANEL_ACID = {}
for _n, (_p, _panels, _o) in SOURCES.items():
    PANEL_ACID.update(_panels)


def pose_descriptors(path):
    prot, protres, donor, pa = [], [], [], None
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        try:
            xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
        except ValueError:
            continue
        if line.startswith("ATOM  ") and line[21] == "A":
            prot.append(xyz)
            protres.append((int(line[22:26]), THREE2ONE.get(line[17:20].strip(), "X")))
        elif line[21] == "B" and line[12:16].strip() == "PA":
            pa = xyz
        elif line[21] == "D":
            donor.append(xyz)
    if not prot or not donor:
        return None
    P = np.asarray(prot); D = np.asarray(donor)
    tree = cKDTree(P)
    near = tree.query_ball_point(D, r=4.5, workers=1)          # courtesy: workers=1
    buried = sum(1 for lst in near if lst) / len(D)
    contacts = len({i for lst in near for i in lst})
    apolar = None
    if pa is not None:
        idx = tree.query_ball_point(np.asarray(pa), r=8.0, workers=1)
        resset = {protres[i] for i in idx}
        if resset:
            apolar = sum(1 for _ri, aa in resset if aa in APOLAR) / len(resset)
    return buried, contacts, apolar


def logistic(X, y, iters=200):
    """Newton-Raphson logistic regression with a tiny ridge; returns (beta, se)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = np.diag(p * (1 - p) + 1e-9)
        H = X.T @ W @ X + 1e-6 * np.eye(X.shape[1])
        g = X.T @ (y - p) - 1e-6 * beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    W = np.diag(p * (1 - p) + 1e-9)
    cov = np.linalg.inv(X.T @ W @ X + 1e-6 * np.eye(X.shape[1]))
    return beta, np.sqrt(np.diag(cov))


def ndtr(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def main():
    sc = {}
    for r in csv.DictReader((CAMP / "analysis/step1_scores.csv").open()):
        if int(r["n_mg"]) == 1:
            sc[(r["enzyme"], r["acid_code"])] = r
    jm = {(r["enzyme"], r["acid_code"]): r for r in
          csv.DictReader((CAMP / "cofolds/step1/job_map.csv").open()) if r["n_mg"] == "1"}

    labels = {}
    for _n, (path, _p, _o) in SOURCES.items():
        if path.exists():
            labels.update(load(path))
    # activity label per (enzyme, acid): detected in ANY panel using that donor (prereg section 4)
    act = {}
    for panel, acid in PANEL_ACID.items():
        for e, v in labels.get(panel, {}).items():
            k = (e, acid)
            act[k] = act.get(k, False) or (not v["below_LOD"])

    from rdkit import Chem
    rows = []
    for (enz, acid), s in sorted(sc.items()):
        stem = s["stem"]
        pred = CAMP / f"cofolds/step1/results/{stem}/boltz_results_{stem}/predictions/{stem}"
        pdbs = sorted(pred.glob(f"{stem}_model_*.pdb"))
        if len(pdbs) < 50:
            continue
        buried, contacts, apolar = [], [], []
        for p in pdbs:
            d = pose_descriptors(p)
            if d:
                buried.append(d[0]); contacts.append(d[1])
                if d[2] is not None:
                    apolar.append(d[2])
        if not buried:
            continue
        mol = Chem.MolFromSmiles(jm[(enz, acid)]["smiles"])
        rows.append(dict(
            enzyme=enz, acid=acid, fold=FOLD[enz],
            donor_heavy_atoms=mol.GetNumHeavyAtoms(),
            occupancy=round(int(s["n_gated"]) / 50, 3),
            D_h=float(s["D_h_A"]),
            buried_frac=round(float(np.median(buried)), 4),
            n_contacts=int(np.median(contacts)),
            pocket_apolar_frac=round(float(np.median(apolar)), 4) if apolar else "",
            A_deg=float(s["A_deg"]),
            detected=act.get((enz, acid), ""),
        ))

    out = CAMP / "analysis/accommodation_descriptors.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    lab = [r for r in rows if r["detected"] != ""]
    print(f"{len(rows)} ensembles scored, {len(lab)} with an activity label -> {out}\n")
    print(f"{'enz':6s} {'acid':5s} {'size':>5s} {'occ':>5s} {'D_h':>6s} {'buried':>7s} "
          f"{'contacts':>9s} {'apolar':>7s} {'A':>6s}  active")
    for r in sorted(rows, key=lambda r: (r["donor_heavy_atoms"], r["enzyme"])):
        print(f"{r['enzyme']:6s} {r['acid']:5s} {r['donor_heavy_atoms']:5d} {r['occupancy']:5.2f} "
              f"{r['D_h']:6.2f} {r['buried_frac']:7.3f} {r['n_contacts']:9d} "
              f"{(r['pocket_apolar_frac'] if r['pocket_apolar_frac']!='' else 0):7.3f} "
              f"{r['A_deg']:6.1f}  {r['detected']}")

    # ---- TA1: size x capacity interaction, capacity estimated on SMALL substrates only --------
    small = [r for r in rows if r["acid"] in ("A", "G")]
    cap = {}
    for e in {r["enzyme"] for r in small}:
        vals = [r["buried_frac"] for r in small if r["enzyme"] == e]
        if vals:
            cap[e] = float(np.mean(vals))
    usable = [r for r in lab if r["enzyme"] in cap]
    print(f"\n--- TA1: logistic(detected ~ size + capacity + size x capacity) ---")
    print("capacity per enzyme (mean buried_frac on monomer diacids A/G only, so the predictor "
          "is NOT fitted on the large-substrate cells it must explain):")
    for e, v in sorted(cap.items(), key=lambda t: -t[1]):
        print(f"    {e:6s} {v:.3f}")
    if len({r["detected"] for r in usable}) < 2 or len(usable) < 8:
        print(f"  not evaluable yet: {len(usable)} labelled cells, "
              f"{sum(1 for r in usable if r['detected'])} detected")
        return
    sz = np.array([r["donor_heavy_atoms"] for r in usable], float)
    cp = np.array([cap[r["enzyme"]] for r in usable], float)
    szc = (sz - sz.mean()) / (sz.std() or 1)
    cpc = (cp - cp.mean()) / (cp.std() or 1)
    y = np.array([1.0 if r["detected"] else 0.0 for r in usable])
    beta, se = logistic(np.column_stack([szc, cpc, szc * cpc]), y)
    names = ["intercept", "size", "capacity", "size x capacity"]
    print(f"  n = {len(usable)} cells ({int(y.sum())} detected, {int(len(y)-y.sum())} below LOD)")
    for nm, b, s in zip(names, beta, se):
        z = b / s if s > 0 else 0.0
        print(f"    {nm:16s} beta = {b:+7.3f}  se = {s:5.3f}  z = {z:+6.2f}  p = {2*(1-ndtr(abs(z))):.4f}")
    print("  Bonferroni threshold for this family (6 descriptors x 2 strata = 12 looks): "
          "alpha = 0.0042")
    print("  Clearing 0.05 but not 0.0042 is NOT a finding (amendment 5).")

    # ---- TA2: within-donor and within-enzyme correlation matrix ------------------------------
    print("\n--- TA2: Spearman(descriptor, detected) WITHIN strata ---")
    descs = ["occupancy", "D_h", "buried_frac", "n_contacts", "donor_heavy_atoms"]
    for strat_name, keyf in (("within donor", lambda r: r["acid"]),
                             ("within enzyme", lambda r: r["enzyme"])):
        print(f"  {strat_name}:")
        for d in descs:
            rs = []
            for k in sorted({keyf(r) for r in lab}):
                grp = [r for r in lab if keyf(r) == k]
                if len({r["detected"] for r in grp}) < 2 or len(grp) < 4:
                    continue
                rho, _p = spearman([float(r[d]) for r in grp],
                                   [1.0 if r["detected"] else 0.0 for r in grp])
                if rho is not None:
                    rs.append(rho)
            print(f"    {d:18s} " + (f"mean rho = {np.mean(rs):+.3f} over {len(rs)} strata"
                                     if rs else "no stratum has both classes yet"))


if __name__ == "__main__":
    main()
