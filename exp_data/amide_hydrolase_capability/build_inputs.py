#!/usr/bin/env python
"""Build Boltz-2 YAML inputs for the amide-hydrolysis capability benchmark.
One protein chain (A, mature sequence) + one ligand (B, N-methylacetamide).
Signal peptides with an explicit range are stripped; propeptides without a
range are kept (full sequence) and recorded as such.
"""
import csv, re, os, json

BASE = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_hydrolase_capability"
INP = os.path.join(BASE, "inputs")
os.makedirs(INP, exist_ok=True)

# read fasta
seqs = {}
cur = None
for line in open(os.path.join(BASE, "sequences.fasta")):
    line = line.strip()
    if line.startswith(">"):
        cur = line[1:].split()[0]
        seqs[cur] = ""
    elif cur:
        seqs[cur] += line

records = []
with open(os.path.join(BASE, "benchmark_set.csv")) as f:
    for row in csv.DictReader(f):
        records.append(row)

log = []
for row in records:
    _id = row["id"]
    notes = (row.get("notes") or "").strip()
    seq = seqs[_id]
    full_len = len(seq)
    action = []
    mature = seq
    # strip explicit signal peptide range signal:1..N
    m = re.search(r"signal:\s*1\.\.(\d+)", notes)
    if m:
        n = int(m.group(1))
        if n < len(mature):
            mature = mature[n:]
            action.append(f"stripped signal 1..{n}")
    # propeptide without explicit range -> keep full, record
    if re.search(r"propeptide", notes, re.I):
        pm = re.search(r"propeptide:\s*1?\.?\.?(\d+)", notes)
        if pm and ".." in notes[pm.start():pm.start()+30]:
            action.append("propeptide range present but not auto-stripped (kept)")
        else:
            action.append("propeptide:present (no range -> kept full, ambiguous)")
    if not action:
        action.append("full sequence (no signal/propeptide note)")

    yaml_txt = (
        "version: 1\n"
        "sequences:\n"
        "- protein:\n"
        "    id: A\n"
        f"    sequence: {mature}\n"
        "- ligand:\n"
        "    id: B\n"
        "    smiles: 'CC(=O)NC'\n"
    )
    with open(os.path.join(INP, f"{_id}.yaml"), "w") as fo:
        fo.write(yaml_txt)
    log.append({"id": _id, "label": row["label"], "full_len": full_len,
                "mature_len": len(mature), "action": "; ".join(action)})

with open(os.path.join(BASE, "input_build_log.json"), "w") as f:
    json.dump(log, f, indent=2)
for e in log:
    print(f"{e['id']:6s} {e['label']:10s} len {e['full_len']}->{e['mature_len']:5d}  {e['action']}")
print(f"\nWrote {len(log)} YAMLs to {INP}")
