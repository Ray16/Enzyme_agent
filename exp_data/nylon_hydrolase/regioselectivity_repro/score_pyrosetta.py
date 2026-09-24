#!/usr/bin/env python3
"""Scoring Part B (pyrosetta env): Rosetta REF2015 interface_delta for ONE system
(fresh process per system, since each ligand's params share resname LIG). Appends
{system: interface_delta} to rosetta/interface_delta.json.
Usage: score_pyrosetta.py <system>
"""
import json, os, sys
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

HERE = os.path.dirname(os.path.abspath(__file__))
ROS = f"{HERE}/rosetta"
OUTJSON = f"{ROS}/interface_delta.json"


def main(system):
    params = f"{ROS}/{system}.params"; cpx = f"{ROS}/{system}_complex.pdb"
    pyrosetta.init(f"-extra_res_fa {params} -mute all -ignore_zero_occupancy false")
    pose = pose_from_pdb(cpx)
    sf = pyrosetta.create_score_function("ref2015")
    # ligand is the last residue; analyze the interface across its chain
    lig_chain = pose.chain(pose.total_residue())
    fixed = pyrosetta.rosetta.std.set_int_t()
    fixed.add(int(lig_chain))
    # (fixed_chains, tracer, sf, compute_packstat, pack_input, pack_separated)
    iam = InterfaceAnalyzerMover(fixed, False, sf, False, False, True)
    iam.apply(pose)
    dG = float(iam.get_interface_dG())
    res = {}
    if os.path.exists(OUTJSON):
        try: res = json.load(open(OUTJSON))
        except Exception: res = {}
    res[system] = round(dG, 2)
    json.dump(res, open(OUTJSON, "w"), indent=2)
    print(f"{system}: interface_delta = {dG:.1f} REU")


if __name__ == "__main__":
    main(sys.argv[1])
