#!/usr/bin/env python3
"""Inject the step-2 pocket constraint from step-1 cofold structures (AS9 method).

For each enzyme, read its step-1 cofold model (protein + ATP + 2 Mg + adipate),
find protein residues whose heavy atoms lie within --cutoff A of any
ligand/cofactor atom, take the first --nres by residue number, and append a soft
pocket restraint on the diamine (binder C) to the step-2 YAML:

    constraints:
      - pocket: {binder: C, contacts: [[A, r], ...], max_distance: 10.0, force: true}

Naming: step-2 skeletons are step2_<name>.yaml; step-1 job = step1_<name>.
Step-1 chain layout: A=protein, B=ATP, C=MG, D=MG, E=adipate -> lig chains B,C,D,E.

Run after step-1 completes (point --s1_out at the predictions tree):
  python inject_pocket.py --s1_out /path/to/step1_out \
      --skel data/cofold_adipic_panel/step2 [--cutoff 5.0] [--nres 15]
"""
import argparse, glob, os, numpy as np
from Bio.PDB import PDBParser, MMCIFParser

LIG_CHAINS = ["B", "C", "D", "E"]  # ATP, MG, MG, adipate in step-1

def load_model(path):
    parser = MMCIFParser(QUIET=True) if path.endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    return parser.get_structure("s", path)[0]

def binding_residues(model, cutoff, nres):
    lig = []
    for ch in LIG_CHAINS:
        if ch in model:
            lig += [a.get_coord() for a in model[ch].get_atoms()]
    lig = np.array(lig)
    hits = []
    for res in model["A"]:
        if res.id[0] != " ":
            continue
        rc = np.array([a.get_coord() for a in res if a.element != "H"])
        if rc.size and np.linalg.norm(rc[:, None, :] - lig[None, :, :], axis=2).min() <= cutoff:
            hits.append(res.id[1])
    return sorted(hits)[:nres]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_out", required=True)
    ap.add_argument("--skel", default="data/cofold_adipic_panel/step2")
    ap.add_argument("--pdb_glob",
                    default="{s1_out}/{job}/boltz_results_{job}/predictions/{job}/{job}_model_0.pdb")
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--nres", type=int, default=15)
    args = ap.parse_args()

    done, miss = 0, []
    for fn in sorted(glob.glob(os.path.join(args.skel, "step2_*.yaml"))):
        name = os.path.basename(fn)[len("step2_"):-len(".yaml")]
        job = f"step1_{name}"
        pdb = args.pdb_glob.format(s1_out=args.s1_out, job=job)
        if not os.path.exists(pdb):
            miss.append(name); continue
        res = binding_residues(load_model(pdb), args.cutoff, args.nres)
        body = open(fn).read().rstrip("\n").split("\nconstraints:")[0]
        contacts = "\n".join(f"        - [A, {r}]" for r in res)
        body += ("\nconstraints:\n  - pocket:\n      binder: C\n"
                 f"      contacts:\n{contacts}\n      max_distance: 10.0\n      force: true\n")
        open(fn, "w").write(body)
        done += 1
    print(f"Injected pocket into {done} step-2 YAMLs; missing step-1 PDB: {miss}")

if __name__ == "__main__":
    main()
