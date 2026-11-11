"""Full-system MLIP barrier: UMA over every atom, one consistent environment.

An alternative to the embedded DFT/MM backend (embedded.py) for barriers. There
is no QM/MM partition, no link atoms, no point-charge embedding, hence no
boundary or MM-remainder artifact: a single UMA potential (uma-s-1p2, the OMol25
= wB97M-V/def2-TZVPD target level) is evaluated over the whole solvated system.

Why it is robust for a barrier:
  * The reaction path is built on ONE environment (jobs.consistent_path_positions):
    each point keeps its own active-site geometry and shares the reference frame's
    distant bulk. UMA energy is a sum of local atomic contributions, so every atom
    whose local environment is identical across points contributes an identical
    energy that cancels exactly in dE -- the unchanged bulk drops out.
  * Every frame is structurally validated (jobs.validate_geometry) before any GPU
    time: integer charge, intact covalent bonds (except the reacting bond), no
    steric clashes.

Caveat (stated, not hidden): UMA has a finite receptive field, so true long-range
polarization beyond its cutoff is not captured -- but a finite explicit cluster
has the same limit and the full system is strictly more complete. This is a
reference-potential energy, NOT a validated activation free energy.

CLI: python -m enzyme_barrier.fullsystem <path_dir> <out.json> [model]
"""
import json
import time
from pathlib import Path
import numpy as np

UMA_COMMON = "/nfs/lambda_stor_01/homes/rzhu/uma_enzyme_ts/common"


def _uma():
    import sys
    if UMA_COMMON not in sys.path:
        sys.path.insert(0, UMA_COMMON)
    import uma_helper
    return uma_helper


def full_system_barrier(path_dir, out, *, model="uma-s-1p2", inference_settings="default",
                        order=("R", "path_max", "TS", "P"), validate=True):
    """Build consistent frames, validate each, run full-system UMA single points.

    Returns the result dict (also written to `out`). dE_act is top-vs-reactant,
    dE_rxn is product-vs-reactant, both in kcal/mol.
    """
    from ase import Atoms
    from .jobs import consistent_path_positions, validate_geometry
    uma = _uma()
    EV2KCAL = uma.KCAL_MOL_PER_EV

    parm, region, ref, frames = consistent_path_positions(path_dir)
    qm = np.array(sorted(set(region["indices"])))
    N = len(parm.atoms)
    transplant = set(region["consistent_env_transplant"])
    frozen = [i for i in range(N) if i not in transplant]
    reaction_atoms = region.get("reaction_atoms", ())

    reports = {}
    xref = frames[ref]
    for nm, x in frames.items():
        moved = int((np.linalg.norm(x - xref, axis=1)[frozen] > 1e-9).sum()) if frozen else 0
        if moved:
            raise ValueError(f"Frame {nm!r}: {moved} frozen-bulk atoms moved -- "
                             "environment is not consistent; aborting.")
        if validate:
            reports[nm] = validate_geometry(parm, x, reaction_atoms=reaction_atoms)
        print(f"  {nm}: QM max move={np.linalg.norm(x-xref,axis=1)[qm].max():.3f} A  "
              f"frozen-bulk moved=0  geometry_ok={reports.get(nm,{}).get('passed','skipped')}",
              flush=True)

    qtot = int(round(sum(a.charge for a in parm.atoms)))
    numbers = [a.atomic_number for a in parm.atoms]
    print(f"full system: N={N} total_charge={qtot} spin=1 model={model} ({inference_settings})",
          flush=True)
    calc = uma.get_calculator("omol", model, inference_settings=inference_settings)

    present = [k for k in order if k in frames]
    E = {}
    for nm in present:
        a = Atoms(numbers=numbers, positions=frames[nm])
        a.info.update(charge=qtot, spin=1)
        a.calc = calc
        t = time.monotonic()
        E[nm] = float(a.get_potential_energy())
        print(f"  {nm}: E={E[nm]:.6f} eV  ({time.monotonic()-t:.1f}s)", flush=True)

    top = next((k for k in ("path_max", "TS") if k in E), None)
    dEa = (E[top] - E[ref]) * EV2KCAL if top else None
    dErxn = (E["P"] - E[ref]) * EV2KCAL if "P" in E else None
    res = dict(path_dir=str(path_dir), method=f"full-system MLIP ({model}, omol)",
               level="UMA OMol25 = wB97M-V/def2-TZVPD target", n_atoms=N,
               reference=ref, n_transplant=len(transplant), total_charge=qtot,
               geometry_reports=reports, E_eV=E,
               dE_act_kcal=dEa, dE_rxn_kcal=dErxn,
               quantity="reference_potential_energy_gaps_on_scan_geometries",
               validated_activation_free_energy=False,
               note="No QM/MM split / embedding / boundary. One consistent "
                    "environment; frozen bulk cancels in dE. Geometry = scan path, "
                    "not a DFT-stationary TS.")
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    if dEa is not None:
        print(f"\nΔE‡ = {dEa:.1f} kcal/mol   ΔE_rxn = {dErxn:+.1f} kcal/mol  -> {out}", flush=True)
    return res


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Full-system MLIP barrier on a consistent environment")
    p.add_argument("path_dir")
    p.add_argument("out")
    p.add_argument("model", nargs="?", default="uma-s-1p2")
    p.add_argument("--no-validate", action="store_true")
    a = p.parse_args(argv)
    full_system_barrier(a.path_dir, a.out, model=a.model, validate=not a.no_validate)


if __name__ == "__main__":
    main()
