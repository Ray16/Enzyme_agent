#!/usr/bin/env python3
"""GPU OpenMM minimization (the 'optimization' step) reading the Amber prmtop/inpcrd
already built by the CPU parametrization (antechamber+tleap). FF19SB + GAFF v1, OBC2
implicit solvent. Starts from the original Boltz pose (inpcrd) and minimizes on CUDA.
Run under gpu_reserve so CUDA_VISIBLE_DEVICES is set.
Usage: gpu_minimize.py <system>
"""
import sys, os, glob
from openmm import app, unit, LangevinIntegrator, Platform

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = f"{HERE}/minimized"


def main(system):
    wds = sorted(glob.glob(f"{OUTD}/min_{system}_*"), key=os.path.getmtime)
    wd = next((w for w in reversed(wds)
               if os.path.exists(f"{w}/complex.prmtop") and os.path.exists(f"{w}/complex.inpcrd")), None)
    if wd is None:
        raise SystemExit(f"{system}: no work dir with complex.prmtop/inpcrd yet")
    prm = app.AmberPrmtopFile(f"{wd}/complex.prmtop")
    crd = app.AmberInpcrdFile(f"{wd}/complex.inpcrd")
    system_mm = prm.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds)
    integ = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    # prefer GPU; CUDA build may be too new for the node driver (PTX), so fall back to
    # OpenCL (still GPU) then CPU
    order = [os.environ.get("OPENMM_PLATFORM", "CUDA"), "OpenCL", "CPU"]
    sim = None; used = None
    for name in dict.fromkeys(order):
        try:
            sim = app.Simulation(prm.topology, system_mm, integ,
                                 Platform.getPlatformByName(name))
            sim.context.setPositions(crd.positions)
            sim.context.getState(getEnergy=True).getPotentialEnergy()   # force ctx build
            used = name; break
        except Exception:
            integ = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            sim = None
    if sim is None:
        raise SystemExit(f"{system}: no working OpenMM platform")
    print(f"{system}: platform={used}")
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    sim.minimizeEnergy(maxIterations=5000)
    e1 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    out = f"{OUTD}/{system}.min.pdb"
    with open(out, "w") as f:
        app.PDBFile.writeFile(prm.topology,
                              sim.context.getState(getPositions=True).getPositions(), f)
    print(f"{system}: CUDA min E {e0:.1f} -> {e1:.1f} kcal/mol (dev {os.environ.get('CUDA_VISIBLE_DEVICES','?')}) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
