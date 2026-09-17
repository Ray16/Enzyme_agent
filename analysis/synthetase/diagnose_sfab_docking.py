#!/usr/bin/env python3
"""Diagnose WHY Boltz-2 parked SfaB's adipate outside the active site while it docked
the actives, using the SAME cofold outputs. Boltz names every ligand 'LIG'; ATP is the
LIG with PA+PG, adipate the LIG with a carboxylate. Reports, per best model:
  - adipate carboxyl-O -> ATP Pa closest approach (near-attack needs <4 A)
  - adipate burial (protein heavy atoms within 4 A) : is it in a pocket or on the surface?
  - ATP burial + ATP Pa distance to the ANL adenylation motif (S/TGTTG) : did the
    adenylation machinery itself assemble correctly for SfaB?
If ATP is in-pocket for SfaB but adipate is not, the failure is the ACID subsite for a
non-native substrate, not a global cofold failure.
"""
import glob
import numpy as np

RES = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
MOTIF = "TSGTTG"          # ANL P-loop (…IYTSGTTGLPKGV…); its backbone cradles ATP phosphates


def parse(pdb):
    prot, ligs = [], {}
    seq = []
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        name = ln[12:16].strip(); res = ln[17:20].strip(); ch = ln[21]
        resi = ln[22:26].strip()
        el = (ln[76:78].strip() or name[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ln.startswith("ATOM"):
            prot.append((resi, name, el, xyz))
        else:
            ligs.setdefault((ch, res, resi), []).append((name, el, xyz))
    return prot, ligs


def carboxyl_oxys(lst):
    Cs = [(n, x) for (n, e, x) in lst if e == "C"]
    Os = [x for (n, e, x) in lst if e == "O"]
    out = []
    for _, cx in Cs:
        nb = [ox for ox in Os if np.linalg.norm(cx - ox) < 1.6]
        if len(nb) >= 2:
            out.extend(nb)
    return out


def analyze(e):
    pdb = f"{RES}/step1_{e}/boltz_results_step1_{e}/predictions/step1_{e}/step1_{e}_model_0.pdb"
    prot, ligs = parse(pdb)
    ph = np.array([a[3] for a in prot if a[2] != "H"])
    atp = adip = None
    for key, lst in ligs.items():
        names = {n for (n, e_, x) in lst}
        if {"PA", "PG"} <= names:
            atp = lst
        elif carboxyl_oxys(lst):
            adip = lst
    pa = next(x for (n, e_, x) in atp if n == "PA")
    atp_xyz = np.array([x for (n, e_, x) in atp])
    cb = np.array(carboxyl_oxys(adip))

    def contacts(lig):
        return int(np.sum(np.min(np.linalg.norm(ph[None] - lig[:, None], axis=2), axis=0) < 4.0))

    d_adip_pa = float(np.min(np.linalg.norm(cb - pa[None], axis=1)))
    d_adip_atp = float(np.min(np.linalg.norm(cb[:, None] - atp_xyz[None], axis=2)))
    # nearest protein residue to the adipate (where on the protein did it land?)
    dmin_res = min(prot, key=lambda a: np.min(np.linalg.norm(cb - a[3][None], axis=1)))
    return dict(id=e, d_adip_pa=d_adip_pa, d_adip_atp=d_adip_atp,
                adip_contacts=contacts(cb), atp_contacts=contacts(atp_xyz),
                near_res=f"{dmin_res[0]}")


if __name__ == "__main__":
    print(f"{'enz':6} {'adipO->Pa':>10} {'adipO->ATP':>11} {'adip_contacts':>14} {'ATP_contacts':>13}")
    for e in ["SfaB", "AS9", "AS17", "AS23"]:
        r = analyze(e)
        tag = "buried" if r["adip_contacts"] >= 12 else "SURFACE/loose"
        atag = "in-pocket" if r["atp_contacts"] >= 25 else "loose"
        print(f"{r['id']:6} {r['d_adip_pa']:10.1f} {r['d_adip_atp']:11.1f} "
              f"{r['adip_contacts']:14d} {r['atp_contacts']:13d}   adipate={tag}, ATP={atag}")
