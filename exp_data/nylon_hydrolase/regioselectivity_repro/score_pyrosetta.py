#!/usr/bin/env python3
"""Scoring Part B (pyrosetta env): Rosetta REF2015 interface_delta between the protein
(chain A) and the ligand (chain X) for each GPU-minimized complex, then compare
Delta(MAXA) vs Delta(XAMA) per enzyme against the paper (Nyl50->XAMA, Nyl12->MAXA,
Nyl10->tie). Reads rosetta/<system>.params + rosetta/<system>_complex.pdb from prep.
"""
import os, glob
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

HERE = os.path.dirname(os.path.abspath(__file__))
ROS = f"{HERE}/rosetta"
SYSTEMS = ["Nyl10__MAXA_precursor", "Nyl10__XAMA_precursor",
           "Nyl12__MAXA_precursor", "Nyl12__XAMA_precursor",
           "Nyl50__MAXA_precursor", "Nyl50__XAMA_precursor"]


def score(system):
    params = f"{ROS}/{system}.params"; cpx = f"{ROS}/{system}_complex.pdb"
    if not (os.path.exists(params) and os.path.exists(cpx)):
        return None
    pyrosetta.init(f"-extra_res_fa {params} -mute all -ignore_zero_occupancy false", silent=True)
    pose = pose_from_pdb(cpx)
    iam = InterfaceAnalyzerMover("A_X")
    iam.set_pack_separated(True)
    iam.apply(pose)
    return float(iam.get_interface_dG())


def main():
    res = {}
    for s in SYSTEMS:
        try:
            d = score(s)
        except Exception as e:
            d = None; print(f"{s}: ERROR {str(e)[:120]}")
        if d is not None:
            res[s] = d; print(f"{s}: interface_delta = {d:.1f} REU")
    print("\n=== regioselectivity: Delta(MAXA) vs Delta(XAMA) (more negative = preferred) ===")
    paper = {"Nyl10": "tie", "Nyl12": "MAXA", "Nyl50": "XAMA"}
    print(f"{'enzyme':8}{'d(MAXA)':>10}{'d(XAMA)':>10}{'pred':>8}{'paper':>8}{'match':>7}")
    for enz in ["Nyl10", "Nyl12", "Nyl50"]:
        dm = res.get(f"{enz}__MAXA_precursor"); dx = res.get(f"{enz}__XAMA_precursor")
        if dm is None or dx is None:
            print(f"{enz:8}{'--':>10}{'--':>10}   (incomplete)"); continue
        gap = dx - dm
        pred = "MAXA" if gap > 2 else ("XAMA" if gap < -2 else "tie")
        match = "Y" if pred == paper[enz] else ("~" if paper[enz] == "tie" else "N")
        print(f"{enz:8}{dm:>10.1f}{dx:>10.1f}{pred:>8}{paper[enz]:>8}{match:>7}")


if __name__ == "__main__":
    main()
