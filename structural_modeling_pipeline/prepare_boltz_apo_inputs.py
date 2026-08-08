#!/usr/bin/env python3
"""Prepare Boltz-2 YAML inputs for apo structure prediction of ALL enzymes.

Generates one YAML per protein for:
  - 29 amide synthetases (amide_synthetase/all_sequences.fasta, excluding pls/B5BR95)
  - 93 amide hydrolases  (amide_hydrolase/all_sequences.fasta)
  - 1  NylC reference    (amide_hydrolase/ref_structure/Q1EPR4.fasta)

No msa: empty — use --use_msa_server at predict time for proper MSA.

Output dirs:
  amide_synthetase/boltz_input/
  amide_hydrolase/boltz_input/
"""

import os
import shutil
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDE = {"B5BR95"}


def load_fasta(path, extract_uniprot=False):
    seqs = {}
    uid = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if uid and buf:
                    seqs[uid] = "".join(buf)
                raw_id = line.split()[0][1:]
                if extract_uniprot and "|" in raw_id:
                    uid = raw_id.split("|")[1]
                else:
                    uid = raw_id
                buf = []
            else:
                buf.append(line)
    if uid and buf:
        seqs[uid] = "".join(buf)
    return seqs


def write_yamls(seqs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for uid, seq in sorted(seqs.items()):
        if uid in EXCLUDE:
            print(f"  Skipping {uid}")
            continue
        data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": seq}}
            ],
        }
        path = os.path.join(out_dir, f"{uid}.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        written += 1
    print(f"  Wrote {written} YAML files to {out_dir}")


def main():
    # Remove old hydrolase boltz dirs
    for d in ["boltz_input", "boltz_output"]:
        old = os.path.join(BASE_DIR, "amide_hydrolase", d)
        if os.path.exists(old):
            shutil.rmtree(old)
            print(f"Removed {old}")

    # Synthetases
    syn_seqs = load_fasta(os.path.join(BASE_DIR, "amide_synthetase", "all_sequences.fasta"))
    syn_out = os.path.join(BASE_DIR, "amide_synthetase", "boltz_input")
    print(f"\nSynthetases ({len(syn_seqs)} sequences):")
    write_yamls(syn_seqs, syn_out)

    # Hydrolases
    hyd_seqs = load_fasta(os.path.join(BASE_DIR, "amide_hydrolase", "all_sequences.fasta"))
    nylc_seqs = load_fasta(os.path.join(BASE_DIR, "amide_hydrolase", "ref_structure", "Q1EPR4.fasta"), extract_uniprot=True)
    hyd_seqs.update(nylc_seqs)
    hyd_out = os.path.join(BASE_DIR, "amide_hydrolase", "boltz_input")
    print(f"\nHydrolases + NylC ({len(hyd_seqs)} sequences):")
    write_yamls(hyd_seqs, hyd_out)

    print(f"\nTotal: {len(syn_seqs) + len(hyd_seqs)} YAML inputs ready")


if __name__ == "__main__":
    main()
