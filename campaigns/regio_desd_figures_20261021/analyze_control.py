#!/usr/bin/env python3
"""POSITIVE CONTROL: does Boltz reproduce the crystallographic DesD-substrate pose?

Everything else in this campaign validates INPUTS. This is the only test of whether the modelled
chemistry lands in the right PLACE, and no `A` ranking should be believed before it reports.

Reference: PDB 7TGM, DesD with
    I3U = hydroxy-N-succinyl-cadaverine (HSC), the native donor  <-- the pose we must reproduce
    LMS = 5'-O-sulfamoyladenosine (AMS), a non-hydrolysable acyl-adenylate mimic
Prediction: cofolds/step1/results/step1_DesD__HSC (DesD + ATP + Mg + HSC, 50 poses)

Metrics, per pose, after superposing the predicted protein onto 7TGM chain A on matched CA atoms:
  * ligand RMSD of predicted HSC vs crystallographic I3U (symmetry-naive, so an upper bound)
  * centroid displacement -- robust to atom-order/symmetry ambiguity
  * pocket contact recovery: Jaccard of residues within 4.5 A of the ligand, predicted vs crystal
  * whether the reactive carboxylate points at the ADENYLATE position (distance from the predicted
    donor carboxyl C to the crystallographic AMS sulfamate S, which marks where P-alpha sits)

Expectation calibrated from this repo: an analogous control reproduced the crystal near-attack pose
in only ~8% of poses, so a LOW pass fraction is still a pass -- what matters is that the correct
pose is populated at all, and that the best pose is close.
"""
import csv, math, pathlib, sys

CAMP = pathlib.Path(__file__).resolve().parent
XTAL = CAMP / "structures/7TGM.pdb"
PRED = CAMP / "cofolds/step1/results/step1_DesD__HSC/boltz_results_step1_DesD__HSC/predictions/step1_DesD__HSC"
CONTACT_R = 4.5

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def parse(path, want_chain=None):
    """-> (ca[(resi, xyz)], residues{resi: [(name, el, xyz)]}, hetero{resname: [(name, el, xyz)]})"""
    ca, res, het = [], {}, {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        ch, resn, name = line[21], line[17:20].strip(), line[12:16].strip()
        el = line[76:78].strip().upper()
        xyz = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        if want_chain and ch != want_chain:
            continue
        if resn in AA3:
            resi = int(line[22:26])
            res.setdefault(resi, []).append((name, el, xyz))
            if name == "CA":
                ca.append((resi, xyz))
        elif resn != "HOH":
            het.setdefault(resn, []).append((name, el, xyz))
    return ca, res, het


def kabsch(P, Q):
    """Rotation+translation mapping P onto Q (both lists of xyz, equal length)."""
    n = len(P)
    cp = [sum(p[i] for p in P) / n for i in range(3)]
    cq = [sum(q[i] for q in Q) / n for i in range(3)]
    P0 = [[p[i] - cp[i] for i in range(3)] for p in P]
    Q0 = [[q[i] - cq[i] for i in range(3)] for q in Q]
    H = [[sum(P0[k][i] * Q0[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    U, S, Vt = _svd3(H)
    d = 1.0 if _det3(_mul3(_T(Vt), _T(U))) > 0 else -1.0
    D = [[1, 0, 0], [0, 1, 0], [0, 0, d]]
    R = _mul3(_T(Vt), _mul3(D, _T(U)))
    return R, cp, cq


def _T(M): return [[M[j][i] for j in range(3)] for i in range(3)]
def _mul3(A, B): return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def _det3(M): return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                      - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                      + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def _svd3(A):
    import numpy as np
    U, S, Vt = np.linalg.svd(np.array(A))
    return U.tolist(), S.tolist(), Vt.tolist()


def apply(R, cp, cq, x):
    v = [x[i] - cp[i] for i in range(3)]
    return tuple(sum(R[i][k] * v[k] for k in range(3)) + cq[i] for i in range(3))


def carboxyl_carbon(atoms):
    oxy = [x for _n, el, x in atoms if el == "O"]
    for name, el, x in atoms:
        if el == "C" and sum(1 for o in oxy if math.dist(o, x) < 1.6) >= 2:
            return x
    return None


def main():
    if not XTAL.exists():
        sys.exit(f"missing reference structure {XTAL}")
    pdbs = sorted(PRED.glob("step1_DesD__HSC_model_*.pdb")) if PRED.exists() else []
    if len(pdbs) < 50:
        sys.exit(f"control not ready: {len(pdbs)}/50 poses folded")

    xca, xres, xhet = parse(XTAL, want_chain="A")
    if "I3U" not in xhet:
        sys.exit("7TGM chain A has no I3U (HSC) ligand")
    xlig = xhet["I3U"]
    xlig_xyz = [x for _n, _e, x in xlig]
    xcen = tuple(sum(c[i] for c in xlig_xyz) / len(xlig_xyz) for i in range(3))
    ams_s = next((x for n, el, x in xhet.get("LMS", []) if el == "S"), None)
    xcontacts = {ri for ri, ats in xres.items()
                 if any(math.dist(a[2], l) <= CONTACT_R for a in ats for l in xlig_xyz)}

    rows = []
    for p in pdbs:
        pca, pres, phet = parse(p, want_chain="A")
        plig = next((v for k, v in parse(p)[2].items() if k == "LIG"), None)
        # chain D is the donor; re-read with chain filter to get it unambiguously
        dl = []
        for line in p.read_text().splitlines():
            if line.startswith("HETATM") and line[21] == "D" and line[16] in (" ", "A"):
                dl.append((line[12:16].strip(), line[76:78].strip().upper(),
                           tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))))
        if not dl or not pca:
            continue
        # superpose on CA atoms present in both (crystal numbering == our mature numbering)
        xmap = dict(xca)
        pairs = [(x, xmap[ri]) for ri, x in pca if ri in xmap]
        if len(pairs) < 100:
            continue
        R, cp, cq = kabsch([a for a, _ in pairs], [b for _, b in pairs])
        ca_rmsd = math.sqrt(sum(math.dist(apply(R, cp, cq, a), b) ** 2
                                for a, b in pairs) / len(pairs))
        moved = [(n, el, apply(R, cp, cq, x)) for n, el, x in dl]
        mxyz = [x for _n, _e, x in moved]
        mcen = tuple(sum(c[i] for c in mxyz) / len(mxyz) for i in range(3))
        centroid_d = math.dist(mcen, xcen)
        lig_rmsd = (math.sqrt(sum(math.dist(a, b) ** 2 for a, b in zip(mxyz, xlig_xyz))
                              / len(mxyz)) if len(mxyz) == len(xlig_xyz) else float("nan"))
        pcontacts = {ri for ri, ats in xres.items()
                     if any(math.dist(a[2], l) <= CONTACT_R for a in ats for l in mxyz)}
        jac = (len(pcontacts & xcontacts) / len(pcontacts | xcontacts)) if (pcontacts | xcontacts) else 0.0
        cc = carboxyl_carbon(moved)
        d_ams = math.dist(cc, ams_s) if (cc and ams_s) else float("nan")
        rows.append(dict(pose=p.stem.rsplit("_", 1)[-1], ca_rmsd=round(ca_rmsd, 2),
                         lig_rmsd=round(lig_rmsd, 2), centroid_d=round(centroid_d, 2),
                         contact_jaccard=round(jac, 3), d_carboxylC_to_AMS_S=round(d_ams, 2)))

    out = CAMP / "analysis/control_7tgm_vs_predicted.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    cen = sorted(r["centroid_d"] for r in rows)
    jac = sorted(r["contact_jaccard"] for r in rows)
    near = sum(1 for r in rows if r["centroid_d"] <= 3.0)
    print(f"POSITIVE CONTROL -- DesD + HSC vs crystal 7TGM (n={len(rows)} poses)")
    print(f"  backbone CA RMSD to crystal : {min(r['ca_rmsd'] for r in rows):.2f} - "
          f"{max(r['ca_rmsd'] for r in rows):.2f} A  (fold reproduced?)")
    print(f"  ligand centroid displacement: best {cen[0]:.2f}, median {cen[len(cen)//2]:.2f} A")
    print(f"  pocket contact Jaccard      : best {jac[-1]:.3f}, median {jac[len(jac)//2]:.3f}")
    print(f"  poses within 3 A of the crystallographic HSC site: {near}/{len(rows)} "
          f"({near/len(rows):.0%})")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
