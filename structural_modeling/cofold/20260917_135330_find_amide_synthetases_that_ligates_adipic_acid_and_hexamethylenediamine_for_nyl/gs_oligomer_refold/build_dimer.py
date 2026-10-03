#!/usr/bin/env python3
"""Turn a step-1 MONOMER cofold YAML into an adjacent-pair DIMER.
Two identical protein chains (same seq + MSA) + the same ligand set (ATP + 2 MG + adipate),
affinity on adipate. This reconstitutes the GS inter-subunit active site that a monomer
leaves open (see project_synthetase_audit_oct2: PuuA neighbor loop res47-53 sits 3.5A from
the docked adipate). Full oligomers are VRAM-infeasible; the dimer is the minimal sufficient unit."""
import sys, os, shutil, yaml, hashlib
mono_yaml, out_dir, msa_out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
acc = os.path.basename(mono_yaml).replace("step1_","").replace(".yaml","")
d = yaml.safe_load(open(mono_yaml))
prot = next(s["protein"] for s in d["sequences"] if "protein" in s)
seq, msa = prot["sequence"], prot["msa"]
# resolve MSA (archived path may be stale -> fall back to the archive copy by hash)
if not os.path.exists(msa):
    h = os.path.basename(msa)
    cands = [
        f"archive/cofold_20260830_125748_superseded_synthetase_panel/msa_cache/{h}",
        f"structural_modeling/cofold/20260917_135330_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl/msa_cache/{h}",
    ]
    msa = next((c for c in cands if os.path.exists(c)), None)
    if msa is None: raise FileNotFoundError(f"MSA not found for {acc}: {h} in {cands}")
# copy MSA into task msa_cache, reference by absolute path
msa_local = os.path.join(msa_out_dir, os.path.basename(msa))
shutil.copy(msa, msa_local)
msa_abs = os.path.abspath(msa_local)
dimer = {"version":1, "sequences":[
    {"protein":{"id":"A","sequence":seq,"msa":msa_abs}},
    {"protein":{"id":"B","sequence":seq,"msa":msa_abs}},
    {"ligand":{"id":"C","ccd":"ATP"}},
    {"ligand":{"id":"D","ccd":"MG"}},
    {"ligand":{"id":"E","ccd":"MG"}},
    {"ligand":{"id":"F","smiles":"[O-]C(=O)CCCCC(=O)[O-]"}},
], "properties":[{"affinity":{"binder":"F"}}]}
out = os.path.join(out_dir, f"dimer_{acc}.yaml")
yaml.safe_dump(dimer, open(out,"w"), sort_keys=False, default_flow_style=False)
print(f"{acc}: {len(seq)}aa dimer -> {out}  (MSA {os.path.basename(msa)})")
