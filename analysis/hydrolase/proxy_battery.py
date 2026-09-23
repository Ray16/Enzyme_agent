#!/usr/bin/env python3
"""Structural-proxy battery for the nylon-hydrolase (Ntn-hydrolase) panel.

Mirror of the amide-synthetase feature-surfacing battery, adapted to the
DIFFERENT chemistry of amide HYDROLYSIS by an N-terminal-nucleophile (Ntn)
hydrolase (Peptidase S58 / DmpA; the NylC family).

Mechanism modelled by the cofold (chain A = pre-Thr, chain B = mature beta-chain
whose residue 1 is the free catalytic Thr, chain C = nylon-6,6 tetrad ligand):

  nucleophile   = chain B Thr1  OG1  (Ogamma)
  general base  = chain B Thr1  N    (free alpha-amino group -- the Ntn analog of
                  the Ser-His of a classic triad; it deprotonates Ogamma)
  electrophile  = the SCISSILE amide carbonyl C of the ligand nearest OG1
                  (a C with one =O <1.6 A AND one amide N <1.6 A -- excludes the
                  terminal carboxylate, which has two O's)
  -> Ogamma attacks C  ->  sp3 tetrahedral intermediate whose oxyanion O is
     stabilised by an oxyanion hole  ->  C-N cleavage (leaving amine protonated).

Proxies (all computable from the 50-sample Boltz-2 diffusion ENSEMBLE; no MD, no
per-candidate QM/MM), tagged by catalytic determinant, per the KG-informed
shortlist (kg-memory-system, 391 rec/95 papers; see analysis/hydrolase/
STRUCTURAL_PROXIES.md for lit anchors):

  nac:*   (ii  reactive conformation)  joint near-attack gate + Burgi-Dunitz angle
  oxh:*   (iii oxyanion-hole electrostatic TS stabilisation)  H-bond donors to O
  base:*  (iv  nucleophile activation)  Thr alpha-amine <-> OG1 geometry
  preorg:*(vi  dynamic rigidity)        positional RMSF of the catalytic centre
  ti:*    (ii  integrated positioning)  tetrahedral-intermediate template RMSD (v1)
  esp:*   (iii electrostatics)          Coulomb ESP at oxyanion O / carbonyl C
  conf:*  (--  model-confidence NULL)   ligand_iptm etc. (expected non-predictive)

Writes exp_data/nylon_hydrolase/analysis/hydrolase_proxies.csv (one row/enzyme,
columns tag:name) consumed by analysis/hydrolase/feature_surfacing.py.

Parallel over enzymes (each worker parses its 50 PDBs once); verified fast-path
== reference by construction (single pass, deterministic).
"""
import csv, glob, json, os
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/nylon_hydrolase/results/output_configs"
ACT = f"{ROOT}/exp_data/nylon_hydrolase/amide_hydrolase_biorep_activity.csv"
OUT = f"{ROOT}/exp_data/nylon_hydrolase/analysis/hydrolase_proxies.csv"

# ---- geometry / chemistry constants -----------------------------------------
D_JOINT = 3.5           # near-attack distance gate (A)          (kg #3)
BD_LO, BD_HI = 90.0, 110.0   # Burgi-Dunitz productive window (deg) (kg #3, tighter than legacy 130)
BD_IDEAL = 107.0
OXH_CUT = 3.2           # H-bond donor..O distance (A)            (kg #2)
BASE_LO, BASE_HI = 2.5, 3.4   # alpha-amine..OG1 H-bond window (A)  (kg #4)
ESP_CUT = 15.0         # Coulomb ESP cutoff radius (A)           (kg #7)

# sidechain H-bond donors (heavy-atom names) for the oxyanion-hole count
DONOR_SC = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"},
            "TYR": {"OH"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "HIS": {"ND1", "NE2"},
            "TRP": {"NE1"}}
# coarse formal-ish charges for the rank-only Coulomb ESP (charged groups only)
CHG = {("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5, ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
       ("LYS", "NZ"): 1.0, ("ARG", "NH1"): 0.5, ("ARG", "NH2"): 0.5,
       ("HIS", "ND1"): 0.05, ("HIS", "NE2"): 0.05}


# ---- lightweight PDB parse ---------------------------------------------------
def parse(path):
    """Return (thr, prot, lig) for one model.
    thr  = {'OG1','N','CA','CB'} -> xyz (chain B res 1)
    prot = list of (resname, atomname, element, xyz) for protein chains A,B
    lig  = list of (element, atomname, xyz) for chain C
    """
    thr = {}; prot = []; lig = []
    for ln in open(path):
        rec = ln[:6]
        if rec[:4] != "ATOM" and rec[:6] != "HETATM":
            continue
        ch = ln[21]; an = ln[12:16].strip(); rn = ln[17:20].strip()
        elem = (ln[76:78].strip() or an[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ch == "C":
            lig.append((elem, an, xyz))
        elif ch in ("A", "B"):
            prot.append((rn, an, elem, xyz))
            if ch == "B" and ln[:4] == "ATOM" and int(ln[22:26]) == 1 and an in ("OG1", "N", "CA", "CB"):
                thr[an] = xyz
    return thr, prot, lig


def scissile(og1, lig):
    """Nearest SCISSILE amide carbonyl to OG1 -> (C, O, N) xyz, or None.
    amide carbonyl = C with exactly one O<1.6A and >=1 N<1.6A (excludes -COO-)."""
    Os = [x for (e, a, x) in lig if e == "O"]
    Ns = [x for (e, a, x) in lig if e == "N"]
    Cs = [x for (e, a, x) in lig if e == "C"]
    best = None
    for cx in Cs:
        o_nb = [o for o in Os if np.linalg.norm(cx - o) < 1.6]
        n_nb = [n for n in Ns if np.linalg.norm(cx - n) < 1.6]
        if len(o_nb) == 1 and len(n_nb) >= 1:
            d = np.linalg.norm(og1 - cx)
            if best is None or d < best[0]:
                best = (d, cx, o_nb[0], n_nb[0])
    return None if best is None else best[1:]


def angle(a, b, c):
    """angle a-b-c in degrees (b = vertex)."""
    v1 = a - b; v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def kabsch(P, Q):
    """rotation+translation aligning P onto Q; returns function to apply to xyz."""
    Pc = P.mean(0); Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return lambda x: (x - Pc) @ R.T + Qc


# ---- tetrahedral-intermediate template (v1, idealised geometry) --------------
def _ti_template():
    """Ideal local coords of {C, OG1, O_oxyanion, N_leaving} for the sp3 TI.
    C at origin; three real substituents (OG1 forming bond, oxyanion O, leaving N)
    placed at tetrahedral directions with TI-appropriate bond lengths."""
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1]], float)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    C = np.zeros(3)
    OG1 = C + t[0] * 1.9      # partially-formed nucleophile bond
    O = C + t[1] * 1.30       # oxyanion (elongated C-O)
    N = C + t[2] * 1.48       # scissile C-N
    return np.array([C, OG1, O, N])


TI_TMPL = _ti_template()


def ti_rmsd(C, OG1, O, N):
    """RMSD of observed reactive tetrad to the idealised TI template (best fit)."""
    obs = np.array([C, OG1, O, N])
    fit = kabsch(TI_TMPL, obs)          # move template onto observation
    tf = fit(TI_TMPL)
    return float(np.sqrt(((tf - obs) ** 2).sum(1).mean()))


# ---- per-enzyme processing ---------------------------------------------------
def _bfac(path):
    """Mean B-factor (= Boltz pLDDT) of the catalytic Thr (chain B res 1)."""
    vals = []
    for ln in open(path):
        if ln[:4] == "ATOM" and ln[21] == "B" and int(ln[22:26]) == 1:
            try:
                vals.append(float(ln[60:66]))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else float("nan")


def _conf(pred, name, model_i):
    """(ligand_iptm, complex_plddt) from the per-model confidence json, or nan."""
    p = f"{pred}/confidence_{name}_model_{model_i}.json"
    try:
        d = json.load(open(p))
        return float(d.get("ligand_iptm", "nan")), float(d.get("complex_plddt", "nan"))
    except (OSError, ValueError):
        return float("nan"), float("nan")


def process(cfgdir):
    name = os.path.basename(cfgdir)
    pred = f"{cfgdir}/boltz_results_{name}/predictions/{name}"
    pdbs = sorted(glob.glob(f"{pred}/{name}_model_*.pdb"))
    if not pdbs:
        return None

    per = []          # per-model scalar features
    cats = []         # catalytic-atom coords for RMSF (OG1) in each model's frame
    conf = []         # (ligand_iptm, complex_plddt, thr_plddt) per model
    ca_ref = None; frames = []
    for pdb in pdbs:
        mi = int(os.path.basename(pdb).split("_model_")[1].split(".")[0])
        thr, prot, lig = parse(pdb)
        if "OG1" not in thr or "N" not in thr or not lig:
            continue
        og1 = thr["OG1"]; nal = thr["N"]
        sc = scissile(og1, lig)
        if sc is None:
            continue
        C, O, Nlv = sc
        d = float(np.linalg.norm(og1 - C))
        bd = angle(og1, C, O)                       # Burgi-Dunitz
        # oxyanion-hole donors to the carbonyl O
        nd = 0; inplane = 0; mindon = 99.0
        cO = O - C                                   # C=O direction (for in-plane test)
        for (rn, an, el, x) in prot:
            isdon = (an == "N") or (an in DONOR_SC.get(rn, ()))
            if not isdon:
                continue
            r = np.linalg.norm(x - O)
            if r < OXH_CUT:
                nd += 1; mindon = min(mindon, r)
                # in-plane lone-pair approach: O->donor vs O->C angle in [90,150]
                ang = angle(x, O, C)
                if 90.0 <= ang <= 150.0:
                    inplane += 1
        # base positioning (alpha-amine <-> OG1)
        d_base = float(np.linalg.norm(nal - og1))
        # Coulomb ESP at oxyanion O and carbonyl C (rank-only)
        espO = espC = 0.0
        for (rn, an, el, x) in prot:
            q = CHG.get((rn, an))
            if q is None:
                continue
            rO = np.linalg.norm(x - O); rC = np.linalg.norm(x - C)
            if rO < ESP_CUT:
                espO += q / (rO + 1e-6)
            if rC < ESP_CUT:
                espC += q / (rC + 1e-6)
        per.append(dict(
            d=d, bd=bd, joint=int(d <= D_JOINT and BD_LO <= bd <= BD_HI),
            bd_dev=abs(bd - BD_IDEAL),
            nd=nd, inplane=inplane, mindon=mindon, ge2=int(nd >= 2), ge2ip=int(inplane >= 2),
            d_base=d_base, base_hb=int(BASE_LO <= d_base <= BASE_HI),
            espO=espO, espC=espC,
            ti=ti_rmsd(C, og1, O, Nlv)))
        # for RMSF: align this model's protein CA onto the first usable model
        ca = np.array([x for (rn, an, el, x) in prot if an == "CA"])
        if ca_ref is None:
            ca_ref = ca; frames.append(og1)
        elif len(ca) == len(ca_ref):
            frames.append(kabsch(ca, ca_ref)(og1))

    if not per:
        return None
    n = len(per)
    A = lambda k: np.array([p[k] for p in per], float)
    # ensemble RMSF of OG1 (superposed frame); low => preorganised
    rmsf = float("nan")
    if len(frames) >= 5:
        F = np.array(frames); rmsf = float(np.sqrt(((F - F.mean(0)) ** 2).sum(1).mean()))

    row = {"id": name, "n": n}
    # nac (reactive conformation)
    row["nac:joint_frac"] = round(A("joint").mean(), 3)          # +   dist AND BD gate
    gated = A("bd")[A("d") <= D_JOINT]
    row["nac:bd_gated_dev"] = round(float(np.mean(np.abs(gated - BD_IDEAL))), 2) if len(gated) else ""  # -
    row["nac:min_d"] = round(float(A("d").min()), 2)             # +? distance-only (kg NEGATIVE - keep as null)
    row["nac:median_d"] = round(float(np.median(A("d"))), 2)     # distance-only null
    # oxyanion hole (the determinant the synthetase panel lacked)
    row["oxh:ge2_frac"] = round(A("ge2").mean(), 3)              # +
    row["oxh:inplane_frac"] = round(A("ge2ip").mean(), 3)        # +   (in-plane weighting, kg-preferred)
    row["oxh:mean_ndonor"] = round(float(A("nd").mean()), 3)     # +
    row["oxh:min_dON"] = round(float(A("mindon").min()), 2)      # -
    # base positioning / nucleophile activation
    row["base:hb_frac"] = round(A("base_hb").mean(), 3)          # +
    row["base:median_dN_OG1"] = round(float(np.median(A("d_base"))), 2)  # -
    # preorganisation
    row["preorg:og1_rmsf"] = round(rmsf, 3) if rmsf == rmsf else ""      # -
    # tetrahedral-intermediate template RMSD (the star, v1 idealised)
    row["ti:rmsd_min"] = round(float(A("ti").min()), 3)         # -
    row["ti:rmsd_mean"] = round(float(A("ti").mean()), 3)       # -
    # electrostatics (rank-only Coulomb ESP)
    row["esp:oxyanion_O_mean"] = round(float(A("espO").mean()), 4)      # +  stabilise developing negative O
    row["esp:carbonyl_C_mean"] = round(float(A("espC").mean()), 4)      # +
    return row


def load_activity():
    act = {}
    for r in csv.DictReader(open(ACT)):
        if r["type"] != "Powder" or r["enzyme"] in act:
            continue
        def f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        act[r["enzyme"]] = {"L1": f(r["L1_ugml"]), "L2": f(r["L2_ugml"]),
                            "L1sig": r["L1_significant"], "L2sig": r["L2_significant"]}
    return act


def actkey(name):
    return "Nyl" + str(int(name[3:])) if name.startswith("Nyl") and name[3:].isdigit() else name


def main():
    act = load_activity()
    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 32)) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        a = act.get(actkey(r["id"]), {})
        r["L1"] = a.get("L1"); r["L2"] = a.get("L2")
        r["total"] = ((a.get("L1") or 0) + (a.get("L2") or 0)) if a else None
        r["L1sig"] = a.get("L1sig"); r["L2sig"] = a.get("L2sig")
    rows.sort(key=lambda r: -(r["oxh:inplane_frac"] or 0))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {OUT}: {len(rows)} enzymes, {sum(r['n'] for r in rows)} models parsed.")
    print(f"Cols: {[c for c in rows[0] if ':' in c]}")


if __name__ == "__main__":
    main()
