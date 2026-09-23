#!/usr/bin/env python3
"""Active-site / autocleavage confidence proxies for the Ntn-hydrolase panel.

Motivation (why active-site pLDDT is NOT just global foldability here): Peptidase
S58 / NylC hydrolases are N-terminal-nucleophile enzymes that require AUTOCLEAVAGE
to expose the catalytic N-terminal Thr. We cofold the pre-split construct (chain A
+ chain B, cut at the Thr), so the model's confidence AROUND the active site and at
the A/B junction encodes whether it reconstructs the correctly self-processed,
catalytically-poised mature state -- a mechanistically loaded quantity, distinct
from global complex pLDDT.

Adds three columns to exp_data/nylon_hydrolase/analysis/hydrolase_proxies.csv:
  as:shell_plddt  = mean pLDDT (PDB B-factor) of protein residues within 8 A of the
                    catalytic Thr Ogamma, averaged over the 50-pose ensemble.
  conf:ab_iptm    = pair_chains_iptm[A][B] = confidence in the relative placement of
                    the two chains split at the catalytic Thr (autocleavage-junction
                    / mature-state confidence).
  conf:bchain_ptm = chains_ptm[B] (nucleophile-bearing beta-chain fold confidence).
"""
import csv, glob, json, os
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/nylon_hydrolase/results/output_configs"
CSV = f"{ROOT}/exp_data/nylon_hydrolase/analysis/hydrolase_proxies.csv"
SHELL = 8.0


def parse_shell(path):
    """(mean shell pLDDT within 8A of Thr1 OG1) for one model, or nan."""
    og1 = None
    atoms = []  # (reskey, bfac, xyz)
    for ln in open(path):
        if ln[:4] != "ATOM":
            continue
        ch = ln[21]; an = ln[12:16].strip()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        try:
            b = float(ln[60:66])
        except ValueError:
            b = np.nan
        if ch == "B" and int(ln[22:26]) == 1 and an == "OG1":
            og1 = xyz
        if ch in ("A", "B"):
            atoms.append((ch + ln[22:26], b, xyz))
    if og1 is None:
        return np.nan
    res = {}
    for rk, b, x in atoms:
        if np.linalg.norm(x - og1) <= SHELL:
            res.setdefault(rk, []).append(b)   # residue in shell if ANY atom within 8A
    if not res:
        return np.nan
    return float(np.nanmean([np.nanmean(v) for v in res.values()]))


def process(cfgdir):
    name = os.path.basename(cfgdir)
    pred = f"{cfgdir}/boltz_results_{name}/predictions/{name}"
    pdbs = sorted(glob.glob(f"{pred}/{name}_model_*.pdb"))
    if not pdbs:
        return None
    shell, ab, bptm = [], [], []
    for pdb in pdbs:
        mi = os.path.basename(pdb).split("_model_")[1].split(".")[0]
        shell.append(parse_shell(pdb))
        try:
            d = json.load(open(f"{pred}/confidence_{name}_model_{mi}.json"))
            ab.append(float(d["pair_chains_iptm"]["0"]["1"]))
            bptm.append(float(d["chains_ptm"]["1"]))
        except (OSError, KeyError, ValueError):
            ab.append(np.nan); bptm.append(np.nan)
    return {"id": name,
            "as:shell_plddt": round(float(np.nanmean(shell)), 2),
            "conf:ab_iptm": round(float(np.nanmean(ab)), 3),
            "conf:bchain_ptm": round(float(np.nanmean(bptm)), 3)}


def main():
    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 32)) as p:
        add = {r["id"]: r for r in p.map(process, cfgdirs) if r}
    rows = list(csv.DictReader(open(CSV)))
    newcols = ["as:shell_plddt", "conf:ab_iptm", "conf:bchain_ptm"]
    fields = list(rows[0].keys())
    for c in newcols:
        if c not in fields:
            fields.append(c)
    for r in rows:
        a = add.get(r["id"], {})
        for c in newcols:
            r[c] = a.get(c, "")
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Added {newcols} to {CSV} ({len(rows)} enzymes).")


if __name__ == "__main__":
    main()
