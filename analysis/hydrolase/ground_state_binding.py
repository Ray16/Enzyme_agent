#!/usr/bin/env python3
"""Determinant 1 (ground_state_binding) for the nylon-hydrolase panel, instantiated
on the EXISTING nylon-6,6 tetrad cofolds (per tier_awareness: Boltz resolves the
tetrad, not a threaded long chain).

This is DISTINCT from determinant 2 (reactive_conformation / near-attack): it asks
whether the substrate is stably, productively ACCOMMODATED in the active-site cleft
(enclosed, registered, groove-complementary), not whether the attack angle is ideal.
A single conserved fold shares the catalytic triad geometry (det. 2-4) but need not
share a cleft that admits/holds the bulky substrate -- so this is the determinant
most likely to VARY across the panel and the axis the manuscript flags (crystalline
access). instantiate questions answered: which group is presented (the scissile
amide), which end buried vs solvent, is the register consistent.

Proxies (mean over 50 poses), added to analysis/hydrolase_proxies.csv:
  gsb:enclosure     fraction of the 33 ligand atoms with a protein heavy atom <4.5 A
                    (whole-chain enclosure in the cleft)                     dir +
  gsb:n_contact_res number of protein residues contacting the ligand (<4.5 A)
                    (groove breadth)                                         dir +
  gsb:scissile_bury protein heavy atoms within 5 A of the scissile carbonyl C
                    (is the REACTING amide seated, not the chain end)        dir +
  gsb:lig_rmsf      positional RMSF of the ligand centroid across the ensemble,
                    protein-CA superposed (binding-pose / register stability) dir -
  gsb:amide_presented fraction of poses where the ligand atom nearest Thr-Ogamma
                    belongs to an AMIDE (scissile) group, not the terminal
                    carboxylate (productive registration)                    dir +
"""
import csv, glob, os
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/nylon_hydrolase/results/output_configs"
CSV = f"{ROOT}/exp_data/nylon_hydrolase/analysis/hydrolase_proxies.csv"
CONTACT, SCISS = 4.5, 5.0


def parse(path):
    og1 = None; prot = []; pres = []; lig = []; ca = []
    for ln in open(path):
        if ln[:4] != "ATOM" and ln[:6] != "HETATM":
            continue
        ch = ln[21]; an = ln[12:16].strip()
        el = (ln[76:78].strip() or an[0]).upper()
        if el == "H":
            continue
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        if ch == "C":
            lig.append((el, xyz))
        elif ch in ("A", "B"):
            prot.append(xyz); pres.append(ch + ln[22:26])
            if an == "CA":
                ca.append(xyz)
            if ch == "B" and ln[:4] == "ATOM" and int(ln[22:26]) == 1 and an == "OG1":
                og1 = np.array(xyz)
    return og1, np.array(prot, float), pres, lig, np.array(ca, float)


def scissile_C(og1, lig):
    Os = [np.array(x) for e, x in lig if e == "O"]
    Ns = [np.array(x) for e, x in lig if e == "N"]
    Cs = [np.array(x) for e, x in lig if e == "C"]
    best = None
    for cx in Cs:
        if sum(np.linalg.norm(cx - o) < 1.6 for o in Os) == 1 and any(np.linalg.norm(cx - n) < 1.6 for n in Ns):
            d = np.linalg.norm(og1 - cx)
            if best is None or d < best[0]:
                best = (d, cx)
    return None if best is None else best[1]


def amide_atoms(lig):
    """Set of ligand-atom indices that belong to an amide group (C(=O)-N)."""
    coords = [np.array(x) for e, x in lig]
    els = [e for e, x in lig]
    amide = set()
    for i, (e, x) in enumerate(zip(els, coords)):
        if e != "C":
            continue
        o = [j for j, (ej, xj) in enumerate(zip(els, coords)) if ej == "O" and np.linalg.norm(x - xj) < 1.6]
        n = [j for j, (ej, xj) in enumerate(zip(els, coords)) if ej == "N" and np.linalg.norm(x - xj) < 1.6]
        if len(o) == 1 and n:                      # amide carbonyl (not carboxylate)
            amide.update([i, o[0], n[0]])
    return amide


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return lambda x: (x - Pc) @ R.T + Qc


def process(cfgdir):
    name = os.path.basename(cfgdir)
    pred = f"{cfgdir}/boltz_results_{name}/predictions/{name}"
    per = []; cents = []; ca_ref = None
    for pdb in sorted(glob.glob(f"{pred}/{name}_model_*.pdb")):
        og1, prot, pres, lig, ca = parse(pdb)
        if og1 is None or not lig or len(prot) == 0:
            continue
        L = np.array([x for e, x in lig], float)
        D = np.linalg.norm(L[:, None, :] - prot[None, :, :], axis=2)   # 33 x n_prot
        enclosed = (D.min(axis=1) < CONTACT).mean()
        cres = {pres[j] for j in np.where((D < CONTACT).any(axis=0))[0]}
        sc = scissile_C(og1, lig)
        sbury = int((np.linalg.norm(prot - sc, axis=1) < SCISS).sum()) if sc is not None else np.nan
        amide = amide_atoms(lig)
        nearest = int(D.min(axis=1).argmin())
        # atom nearest to OG1:
        near_og1 = int(np.linalg.norm(L - og1, axis=1).argmin())
        presented = int(near_og1 in amide)
        per.append((enclosed, len(cres), sbury, presented))
        cent = L.mean(0)
        if ca_ref is None:
            ca_ref = ca; cents.append(cent)
        elif len(ca) == len(ca_ref):
            cents.append(kabsch(ca, ca_ref)(cent))
    if not per:
        return None
    a = np.array(per, float)
    rmsf = np.nan
    if len(cents) >= 5:
        C = np.array(cents); rmsf = float(np.sqrt(((C - C.mean(0)) ** 2).sum(1).mean()))
    return {"id": name,
            "gsb:enclosure": round(float(np.nanmean(a[:, 0])), 3),
            "gsb:n_contact_res": round(float(np.nanmean(a[:, 1])), 2),
            "gsb:scissile_bury": round(float(np.nanmean(a[:, 2])), 2),
            "gsb:amide_presented": round(float(np.nanmean(a[:, 3])), 3),
            "gsb:lig_rmsf": round(rmsf, 3) if rmsf == rmsf else ""}


def main():
    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 32)) as p:
        add = {r["id"]: r for r in p.map(process, cfgdirs) if r}
    rows = list(csv.DictReader(open(CSV)))
    newcols = ["gsb:enclosure", "gsb:n_contact_res", "gsb:scissile_bury",
               "gsb:amide_presented", "gsb:lig_rmsf"]
    fields = list(rows[0].keys()) + [c for c in newcols if c not in rows[0]]
    for r in rows:
        a = add.get(r["id"], {})
        for c in newcols:
            r[c] = a.get(c, "")
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Added {newcols} to {CSV} ({len(rows)} enzymes).")


if __name__ == "__main__":
    main()
