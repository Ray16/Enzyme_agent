#!/usr/bin/env python3
"""Diagnose WHY Boltz-2 parked SfaB's adipate outside the active site while it docked
the actives. Compares ATP + adipate placement (best model) across SfaB and the 3 actives.
Same cofold method for all -> if ATP is in-pocket for SfaB but adipate is not, the failure
is the ACID subsite for a non-native substrate, not a global cofold failure."""
import glob
import numpy as np

RES = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1/results/output_configs"


def parse(pdb):
    prot, het = [], {}
    for line in open(pdb):
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        res = line[17:20].strip()
        an = line[12:16].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        el = line[76:78].strip() or an[0]
        if rec == "ATOM":
            prot.append((res, an, xyz, el))
        else:
            het.setdefault(res, []).append((an, xyz, el))
    return prot, het


def analyze(e):
    pdb = f"{RES}/step1_{e}/boltz_results_step1_{e}/predictions/step1_{e}/step1_{e}_model_0.pdb"
    prot, het = parse(pdb)
    ph = np.array([a[2] for a in prot if a[3] != "H"])           # protein heavy atoms
    atp = het.get("ATP", [])
    pa = np.array([x for an, x, el in atp if an == "PA"])
    adres = [k for k in het if k not in ("ATP", "MG")]
    adip = np.array([x for an, x, el in het.get(adres[0], [])]) if adres else np.empty((0, 3))
    atp_xyz = np.array([x for an, x, el in atp])

    def mind(a, b):
        return np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2))

    def contacts(lig):
        return int(np.sum(np.min(np.linalg.norm(ph[None, :, :] - lig[:, None, :], axis=2), axis=0) < 4.0))

    d_adip_pa = mind(adip, pa) if len(pa) and len(adip) else np.nan
    d_adip_atp = mind(adip, atp_xyz) if len(atp_xyz) and len(adip) else np.nan
    return dict(id=e, het={k: len(v) for k, v in het.items()}, adres=adres,
                d_adip_pa=d_adip_pa, d_adip_atp=d_adip_atp,
                adip_contacts=contacts(adip) if len(adip) else 0,
                atp_contacts=contacts(atp_xyz) if len(atp_xyz) else 0)


if __name__ == "__main__":
    print(f"{'enz':6} {'adip->Pa':>9} {'adip->ATP':>10} {'adip_4A_contacts':>17} {'ATP_4A_contacts':>16}")
    for e in ["SfaB", "AS9", "AS17", "AS23"]:
        r = analyze(e)
        print(f"{r['id']:6} {r['d_adip_pa']:9.1f} {r['d_adip_atp']:10.1f} "
              f"{r['adip_contacts']:17d} {r['atp_contacts']:16d}   adipate={r['adres']}")
