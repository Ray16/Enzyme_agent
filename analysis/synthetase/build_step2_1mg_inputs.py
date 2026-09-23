#!/usr/bin/env python3
"""Build corrected STEP-2 (aminolysis) cofold inputs for the amide-synthetase panel.

Fixes vs the earlier step-2:
  * pocket restraint derived from the CORRECTED 1-Mg step-1 structures (not 2-Mg),
  * pocket = ALL protein residues within 8 A of the reactive-center MIDPOINT
    (adipate proximal-carboxyl-C  <->  ATP Pa), per feedback_step2_pocket_selection,
    NOT the old "15 nearest residues" rule.
Chemistry (unchanged, already correct):
  A=protein(+MSA)  B=adipyl-AMP (acyl-adenylate)  C=HMDA (neutral attacking -NH2)  D=Mg (1)
  soft pocket restraint on the diamine (binder C), max_distance 10 A, force true.
Reuses each enzyme's step-1 sequence + MSA.
"""
import glob, os, re
import numpy as np

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
S1_RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1_1mg/results"
S1_IN = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1_1mg/inputs"
OUT = f"{ROOT}/exp_data/amide_synthetase/boltz2/step2_1mg/inputs"
os.makedirs(OUT, exist_ok=True)
ACYLAMP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OC(=O)CCCCC(=O)[O-])[C@@H](O)[C@H]1O"
HMDA = "NCCCCCC[NH3+]"      # leading -NH2 (neutral nucleophile) + distal -NH3+
POCKET_R = 8.0


def reactive_pocket(name):
    pdb = f"{S1_RES}/{name}/boltz_results_step1_{name}/predictions/step1_{name}/step1_{name}_model_0.pdb"
    if not os.path.exists(pdb):
        return None
    prot = {}          # resnum -> list heavy xyz (protein chain A)
    ligC, ligO, PA = [], [], None
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        nm = ln[12:16].strip(); elem = (ln[76:78].strip() or nm[0]).upper()
        if elem == "H":
            continue
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ln.startswith("ATOM") and ln[21] == "A":
            prot.setdefault(int(ln[22:26]), []).append(xyz)
        elif ln.startswith("HETATM"):
            if nm == "PA":
                PA = xyz
            elif elem == "C":
                ligC.append((nm, xyz))
            elif elem == "O":
                ligO.append(xyz)
    if PA is None or not ligC or not prot:
        return None
    # adipate proximal carboxyl C = ligand C with >=2 O neighbours (<1.6A) nearest to PA
    carbs = [(c, x) for (c, x) in ligC if sum(np.linalg.norm(x - o) < 1.6 for o in ligO) >= 2]
    if not carbs:
        return None
    _, adipC = min(carbs, key=lambda cx: np.linalg.norm(cx[1] - PA))
    mid = (adipC + PA) / 2.0
    pocket = [rn for rn, xs in prot.items() if min(np.linalg.norm(x - mid) for x in xs) < POCKET_R]
    return sorted(pocket)


def field(txt, key):
    m = re.search(rf"^\s*{key}:\s*(.+)$", txt, re.M)
    return m.group(1).strip() if m else None


def main():
    n, miss = 0, []
    for f in sorted(glob.glob(f"{S1_IN}/step1_*.yaml")):
        name = os.path.basename(f)[len("step1_"):-len(".yaml")]
        txt = open(f).read()
        seq, msa = field(txt, "sequence"), field(txt, "msa")
        pocket = reactive_pocket(name)
        if not seq or not msa or not pocket:
            miss.append(name); continue
        contacts = "\n".join(f"        - [A, {r}]" for r in pocket)
        y = ("version: 1\n"
             "# STEP 2 aminolysis (1 Mg): A=prot B=adipyl-AMP C=HMDA(neutral -NH2) D=Mg\n"
             "sequences:\n- protein:\n    id: A\n"
             f"    sequence: {seq}\n    msa: {msa}\n"
             f"- ligand:\n    id: B\n    smiles: '{ACYLAMP}'\n"
             f"- ligand:\n    id: C\n    smiles: '{HMDA}'\n"
             "- ligand:\n    id: D\n    ccd: MG\n"
             "constraints:\n  - pocket:\n      binder: C\n"
             f"      contacts:\n{contacts}\n      max_distance: 10.0\n      force: true\n")
        open(f"{OUT}/step2_{name}.yaml", "w").write(y)
        n += 1
    print(f"wrote {n} step-2 (1-Mg pocket) inputs to {OUT}; missing: {miss}")


if __name__ == "__main__":
    main()
